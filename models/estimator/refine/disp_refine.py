import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================
#  유틸: 좌→우 워핑(스테레오)
# =========================

def _make_base_grid(B, H, W, device, dtype):
    # 픽셀 좌표계 (x: 0..W-1, y: 0..H-1)
    ys, xs = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing='ij'
    )
    # [1, H, W], 배치로 확장
    xs = xs.unsqueeze(0).expand(B, -1, -1)
    ys = ys.unsqueeze(0).expand(B, -1, -1)
    return xs, ys

def _to_normalized_grid(x_pix, y_pix, H, W):
    # grid_sample용 정규화 좌표로 변환(align_corners=True 가정)
    gx = 2.0 * x_pix / (W - 1) - 1.0
    gy = 2.0 * y_pix / (H - 1) - 1.0
    return torch.stack([gx, gy], dim=-1)  # [B, H, W, 2]

def warp_right_to_left(img_right, disp_left, padding_mode='border'):
    """
    img_right: [B,3,H,W], disp_left: [B,1,H,W] (px)
    반환: warped_right, valid_mask (in-bounds 샘플 위치)
    """
    B, C, H, W = img_right.shape
    device, dtype = img_right.device, img_right.dtype

    xs, ys = _make_base_grid(B, H, W, device, dtype)  # [B,H,W]
    x_src = xs - disp_left.squeeze(1)                 # 우영상에서 샘플링할 x 좌표(픽셀)
    y_src = ys

    grid = _to_normalized_grid(x_src, y_src, H, W)    # [B,H,W,2]
    warped = F.grid_sample(img_right, grid, mode='bilinear',
                           padding_mode=padding_mode, align_corners=True)

    # 유효(in-bounds) 마스크
    valid_x = (grid[..., 0] >= -1.0) & (grid[..., 0] <= 1.0)
    valid_y = (grid[..., 1] >= -1.0) & (grid[..., 1] <= 1.0)
    valid = (valid_x & valid_y).unsqueeze(1).float()  # [B,1,H,W]
    return warped, valid

def warp_right_disp_to_left(disp_right, disp_left, padding_mode='border'):
    """
    우측 disparity를 좌표계로 워핑.
    disp_right: [B,1,H,W], disp_left: [B,1,H,W]
    반환: disp_right_warped_to_left, valid_mask
    """
    B, C, H, W = disp_right.shape
    device, dtype = disp_right.device, disp_right.dtype

    xs, ys = _make_base_grid(B, H, W, device, dtype)
    x_src = xs - disp_left.squeeze(1)
    y_src = ys

    grid = _to_normalized_grid(x_src, y_src, H, W)
    warped = F.grid_sample(disp_right, grid, mode='bilinear',
                           padding_mode=padding_mode, align_corners=True)
    valid_x = (grid[..., 0] >= -1.0) & (grid[..., 0] <= 1.0)
    valid_y = (grid[..., 1] >= -1.0) & (grid[..., 1] <= 1.0)
    valid = (valid_x & valid_y).unsqueeze(1).float()
    return warped, valid

# =========================
#  SSIM(간단 구현, 3x3 평균창)
# =========================

class SSIM(nn.Module):
    def __init__(self, channels=3, kernel_size=3, C1=0.01**2, C2=0.03**2):
        super().__init__()
        self.C1 = C1
        self.C2 = C2
        self.pad = kernel_size // 2
        # 평균 풀링을 사용하여 지역 통계 추정
        self.pool = nn.AvgPool2d(kernel_size, stride=1, padding=self.pad, count_include_pad=False)

    def forward(self, x, y):
        # x,y: [B,C,H,W], 값 범위 [0,1] 권장
        mu_x = self.pool(x)
        mu_y = self.pool(y)
        sigma_x  = self.pool(x * x) - mu_x * mu_x
        sigma_y  = self.pool(y * y) - mu_y * mu_y
        sigma_xy = self.pool(x * y) - mu_x * mu_y

        ssim_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        ssim_d = (mu_x * mu_x + mu_y * mu_y + self.C1) * (sigma_x + sigma_y + self.C2)
        ssim = ssim_n / (ssim_d + 1e-12)
        ssim = torch.clamp((1 - ssim) / 2, 0, 1)  # dissimilarity(0=완전 동일, 1=완전 상이)
        return ssim

# =========================
#  Charbonnier (부드러운 L1)
# =========================

def charbonnier(x, eps=1e-3):
    return torch.sqrt(x * x + eps * eps)

# =========================
#  좌우/포토 손실 모듈
# =========================

class StereoRegularizationLoss(nn.Module):
    """
    Photometric(SSIM+Charbonnier) + Left-Right Consistency Loss
    - d_right가 주어지면 LR 손실 활성화.
    - Occlusion은 in-bounds 마스크 + (옵션) LR 임계 기반으로 제거.
    """
    def __init__(self, alpha_ssim=0.85, w_photo=1.0, w_lr=0.2, occlusion_tau=1.0,
                 use_fb_occlusion=True):
        super().__init__()
        self.alpha = alpha_ssim
        self.w_photo = w_photo
        self.w_lr = w_lr
        self.occlusion_tau = occlusion_tau
        self.use_fb_occ = use_fb_occlusion
        self.ssim = SSIM()

    def forward(self, left, right, d_left, d_right=None):
        """
        left, right: [B,3,H,W] in [0,1]
        d_left: [B,1,H,W]   (px, full-res)
        d_right: [B,1,H,W]  (optional, px)
        반환: dict(loss, photo, lr, valid_ratio)
        """
        # 우영상 워핑
        right_warp, valid = warp_right_to_left(right, d_left, padding_mode='border')
        photometric_l1 = charbonnier(torch.abs(left - right_warp))  # [B,3,H,W]
        photometric_ssim = self.ssim(left, right_warp)              # [B,3,H,W]

        # 채널 평균
        l1_map   = photometric_l1.mean(1, keepdim=True)
        ssim_map = photometric_ssim.mean(1, keepdim=True)

        photo_map = self.alpha * ssim_map + (1 - self.alpha) * l1_map

        # Occlusion 마스크(옵션: LR 기준)
        if (d_right is not None) and self.use_fb_occ:
            d_right_warp, valid_r = warp_right_disp_to_left(d_right, d_left, padding_mode='border')
            lr_res = torch.abs(d_left - d_right_warp)
            occ = (lr_res > self.occlusion_tau).float()
            valid = valid * valid_r * (1.0 - occ)

        # Photometric loss
        # 마스크 평균으로 정규화
        denom = valid.sum() + 1e-6
        photo_loss = (photo_map * valid).sum() / denom

        # Left-Right Consistency loss (옵션)
        if d_right is not None:
            d_right_warp, valid_r = warp_right_disp_to_left(d_right, d_left, padding_mode='border')
            lr_map = charbonnier(torch.abs(d_left - d_right_warp))
            valid_lr = valid * valid_r  # 동일 유효영역 교집합
            lr_loss = (lr_map * valid_lr).sum() / (valid_lr.sum() + 1e-6)
        else:
            lr_loss = torch.tensor(0.0, device=left.device, dtype=left.dtype)

        total = self.w_photo * photo_loss + self.w_lr * lr_loss
        return {
            'loss': total,
            'photo': photo_loss.detach(),
            'lr': lr_loss.detach(),
            'valid_ratio': (valid.mean().detach())
        }

# =========================
#  Refinement Head
# =========================

class ResBlock(nn.Module):
    def __init__(self, ch, dilation=1):
        super().__init__()
        pad = dilation
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=pad, dilation=dilation, bias=False)
        self.bn1   = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=pad, dilation=dilation, bias=False)
        self.bn2   = nn.BatchNorm2d(ch)
        self.relu  = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(x + out)

class DisparityRefinement(nn.Module):
    """
    입력: left, right, disp_init(full-res, px)
    출력: disp_refined, aux(dict: warped_right, valid_mask, delta)
    - 입력 특징: [left, right_warped_by_disp, disp_init, |left-right_warped|] 를 concat
    - 얕은 ResNet으로 Δd 예측, tanh 스케일로 안정적 잔차(픽셀 단위) 제한
    """
    def __init__(self, base_ch=64, num_blocks=5, use_error_map=True, max_residual=1.5):
        super().__init__()
        self.use_error_map = use_error_map
        self.max_residual = max_residual

        in_ch = 3 + 3 + 1 + (1 if use_error_map else 0)  # L, R_warp, d, |L-Rw|
        mid_ch = base_ch

        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, mid_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
        )

        # 다양한 수용영역 확보를 위한 팁: 중간 몇 개 블록에 팽창(dilation) 사용
        dilations = [1, 1, 2, 4, 1][:num_blocks]
        blocks = []
        for d in dilations:
            blocks.append(ResBlock(mid_ch, dilation=d))
        self.blocks = nn.Sequential(*blocks)

        self.head = nn.Sequential(
            nn.Conv2d(mid_ch, mid_ch // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch // 2, 1, 3, padding=1, bias=True)
        )
        # Δd의 폭을 제한하기 위한 초기화(작은 값)
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    @torch.no_grad()
    def _safe_cast(self, x, ref):
        # AMP/정밀도 안전 캐스트
        if x.dtype != ref.dtype:
            x = x.to(ref.dtype)
        return x

    def forward(self, left, right, disp_init):
        """
        left, right: [B,3,H,W] (0~1)
        disp_init: [B,1,H,W]  (px)
        """
        right_warp, valid = warp_right_to_left(right, disp_init, padding_mode='border')
        if self.use_error_map:
            err = torch.abs(left - right_warp).mean(1, keepdim=True)  # [B,1,H,W]
            feat_in = torch.cat([left, right_warp, disp_init, err], dim=1)
        else:
            feat_in = torch.cat([left, right_warp, disp_init], dim=1)

        f = self.stem(feat_in)
        f = self.blocks(f)
        delta = self.head(f)  # [B,1,H,W], 초기엔 거의 0

        # 안정성: 한 스텝 잔차를 적정 범위로 제한(과보정 방지)
        delta = self.max_residual * torch.tanh(delta)

        disp_refined = disp_init + delta
        aux = {
            'warped_right': right_warp,
            'valid_mask': valid,
            'delta': delta
        }
        return disp_refined, aux
