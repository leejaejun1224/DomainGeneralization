import math
from typing import Optional, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.estimator.submodules import DomainNorm

# =========================
# 1) 유틸
# =========================

def l2_normalize_channel(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # x: (B, C, H, W)
    return x / (x.pow(2).sum(dim=1, keepdim=True).sqrt() + eps)

def charbonnier(x: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    return (x.pow(2) + eps**2).sqrt()

def make_base_grid(B: int, H: int, W: int, device) -> Tuple[torch.Tensor, torch.Tensor]:
    # 정규화 전 좌표 (픽셀): x in [0,W-1], y in [0,H-1]
    yy, xx = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing="ij"
    )
    xx = xx.float()[None].repeat(B, 1, 1)  # (B,H,W)
    yy = yy.float()[None].repeat(B, 1, 1)
    return xx, yy

def to_grid_sample_coords(x_pix: torch.Tensor, y_pix: torch.Tensor, H: int, W: int) -> torch.Tensor:
    # 픽셀 좌표 -> grid_sample 정규화 좌표([-1,1])
    gx = 2.0 * (x_pix / max(W - 1, 1)) - 1.0
    gy = 2.0 * (y_pix / max(H - 1, 1)) - 1.0
    grid = torch.stack([gx, gy], dim=-1)  # (B,H,W,2)
    return grid

# =========================
# 2) 모듈: 경량 HR 인코더
# =========================

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dilation=1):
        super().__init__()
        padding = dilation
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=padding, dilation=dilation, bias=False)
        # self.bn = nn.BatchNorm2d(out_ch)
        self.bn = DomainNorm(out_ch, l2=True)
        self.act = nn.SiLU(inplace=True)
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class HRFeatureEncoder(nn.Module):
    """
    원본 해상도에서 얕은 특징 추출 (stride=1, dilated conv 혼합)
    """
    def __init__(self, in_ch=3, c=64, dilations=(1,2,4,8)):
        super().__init__()
        layers = []
        ch = in_ch
        for d in dilations:
            layers.append(ConvBlock(ch, c, dilation=d))
            ch = c
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)  # (B,c,H,W)

# =========================
# 3) 1/4 특징 업샘플/정렬
# =========================

class QuarterAdapter(nn.Module):
    """
    1/4 특징(Fq)을 HxW로 업샘플 + 채널 정렬.
    - mmcv.ops.CARAFEPack 사용 가능하면 CARAFE 업샘플
    - 없으면 bilinear 업샘플 -> 1x1 conv
    """
    def __init__(self, in_ch, out_ch, use_carafe: bool = True, scale=4):
        super().__init__()
        self.scale = scale
        self.use_carafe = use_carafe
        self.proj = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        # self.bn = nn.BatchNorm2d(out_ch)
        self.bn = DomainNorm(out_ch, l2=True)
        self.act = nn.SiLU(inplace=True)

        self.carafe = None
        if use_carafe:
            try:
                from mmcv.ops import CARAFEPack
                # 커널 5~7 권장. 여기선 5.
                self.carafe = CARAFEPack(channels=in_ch, up_kernel=5, up_group=1, scale_factor=scale)
            except Exception:
                self.carafe = None  # 폴백
                self.use_carafe = False

    def forward(self, fq, H, W):
        if fq is None:
            return None
        if self.carafe is not None:
            x = self.carafe(fq)  # (B,C,H,W)
        else:
            x = F.interpolate(fq, size=(H, W), mode="bilinear", align_corners=True)
        x = self.act(self.bn(self.proj(x)))
        return x  # (B, out_ch, H, W)

# =========================
# 4) 1D 국소 상관 볼륨(±r)
# =========================

class LocalCorrelation1D(nn.Module):
    """
    정규화 내적 상관 기반 1D 비용 볼륨 (±search_range).
    phi(1x1)로 임베딩 후 L2 정규화하여 코사인 유사도로 계산.
    """
    def __init__(self, in_ch, emb_ch=64, search_range=4):
        super().__init__()
        self.phi = nn.Conv2d(in_ch, emb_ch, 1, bias=False)
        self.search_range = int(search_range)

    def forward(self, F_L: torch.Tensor, F_R: torch.Tensor, d0: torch.Tensor):
        """
        F_L, F_R: (B,C,H,W)
        d0: (B,1,H,W)  - 픽셀 단위 disparity
        반환: cost_vol (B, K, H, W), deltas(list of int)
        """
        B, C, H, W = F_L.shape
        K = self.search_range * 2 + 1
        deltas = list(range(-self.search_range, self.search_range + 1))

        # 임베딩 + 정규화
        FL = l2_normalize_channel(self.phi(F_L))
        FR = l2_normalize_channel(self.phi(F_R))

        # 베이스 좌표
        xx, yy = make_base_grid(B, H, W, F_L.device)

        cost_list = []
        for d in deltas:
            # 오른쪽 특징의 샘플 위치: x' = x - (d0 + d)
            xprime = xx - (d0[:, 0] + float(d))
            grid = to_grid_sample_coords(xprime, yy, H, W)
            FR_s = F.grid_sample(FR, grid, mode="bilinear", padding_mode="border", align_corners=True)
            # 코사인 유사도 = 채널 내적
            cost = (FL * FR_s).sum(dim=1, keepdim=False)  # (B,H,W)
            cost_list.append(cost)

        cost_vol = torch.stack(cost_list, dim=1)  # (B,K,H,W)
        return cost_vol, deltas

# =========================
# 5) 재귀 업데이트 헤드 (Conv-GRU)
# =========================

class ConvGRUCell(nn.Module):
    def __init__(self, in_ch, hid_ch, k=3):
        super().__init__()
        p = k // 2
        self.convz = nn.Conv2d(in_ch + hid_ch, hid_ch, k, padding=p)
        self.convr = nn.Conv2d(in_ch + hid_ch, hid_ch, k, padding=p)
        self.convq = nn.Conv2d(in_ch + hid_ch, hid_ch, k, padding=p)

    def forward(self, x, h):
        if h is None:
            h = torch.zeros(x.size(0), self.convr.out_channels, x.size(2), x.size(3), device=x.device)
        z = torch.sigmoid(self.convz(torch.cat([x, h], dim=1)))
        r = torch.sigmoid(self.convr(torch.cat([x, h], dim=1)))
        q = torch.tanh(self.convq(torch.cat([x, r * h], dim=1)))
        h = (1 - z) * h + z * q
        return h

class RefineHead(nn.Module):
    """
    입력: cost_vol(B,K,H,W) + context(B,Cc,H,W)
    출력: logits(B,K,H,W), delta_res(B,1,H,W)
    """
    def __init__(self, K: int, ctx_ch: int = 64, hid: int = 64, iters: int = 2):
        super().__init__()
        self.iters = iters
        in_ch = K + ctx_ch
        self.pre = nn.Sequential(
            nn.Conv2d(in_ch, hid, 3, padding=1),
            nn.SiLU(True)
        )
        self.gru = ConvGRUCell(hid, hid, k=3)
        self.head_logits = nn.Conv2d(hid, K, 3, padding=1)
        self.head_delta  = nn.Conv2d(hid, 1, 3, padding=1)

    def forward(self, cost_vol, context):
        x = torch.cat([cost_vol, context], dim=1)
        x = self.pre(x)
        h = None
        for _ in range(self.iters):
            h = self.gru(x, h)
        logits = self.head_logits(h)
        delta_res = self.head_delta(h)
        return logits, delta_res

# =========================
# 6) 상보성(Barlow/VICReg) 손실
# =========================

def _flatten_hw(x: torch.Tensor) -> torch.Tensor:
    # (B,C,H,W) -> (N,C) with N=B*H*W
    B, C, H, W = x.shape
    return x.permute(0, 2, 3, 1).reshape(B * H * W, C)

def barlow_cross_loss(Fh: torch.Tensor, Fq: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    두 분기 간 교차 상관행렬을 0으로 유도 (대각 포함): sum(C^2).
    """
    assert Fh.shape == Fq.shape, "Channel/size must match."
    Za = _flatten_hw(Fh)
    Zb = _flatten_hw(Fq)
    # 표준화
    Za = (Za - Za.mean(0)) / (Za.std(0) + eps)
    Zb = (Zb - Zb.mean(0)) / (Zb.std(0) + eps)
    N = Za.size(0)
    C = (Za.t() @ Zb) / N  # (C,C)
    return (C ** 2).mean()

def vicreg_variance_covariance(x: torch.Tensor, eps: float = 1e-4) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    분산 하한(>=1), 공분산 off-diag 억제(두 항목 반환).
    """
    Z = _flatten_hw(x)  # (N,C)
    N, C = Z.shape
    # 분산 하한
    std = Z.std(dim=0) + eps
    var_loss = torch.mean(F.relu(1.0 - std))  # 각 채널 std가 1보다 작으면 패널티
    # 공분산
    Zc = Z - Z.mean(dim=0)
    cov = (Zc.t() @ Zc) / (N - 1)  # (C,C)
    off_diag = cov - torch.diag(torch.diag(cov))
    cov_loss = (off_diag ** 2).sum() / (C * (C - 1))
    return var_loss, cov_loss

# =========================
# 7) 최상위: Stereo Refinement 모듈
# =========================

class StereoRefiner(nn.Module):
    """
    - HR 원본 해상도 특징 + (선택) 1/4 보조 특징(업샘플)
    - 국소 1D 비용 볼륨(±r)
    - Conv-GRU 기반 정밀화 헤드 (분류+잔차)
    """
    def __init__(self,
                 img_in_ch: int = 3,
                 hr_ch: int = 64,
                 ctx_ch: int = 64,
                 emb_ch: int = 64,
                 search_range: int = 4,
                 gru_iters: int = 2,
                 use_quarter: bool = True,
                 quarter_in_ch: int = 32,
                 quarter_use_carafe: bool = True):
        super().__init__()
        self.search_range = int(search_range)
        self.K = self.search_range * 2 + 1
        self.use_quarter = use_quarter

        # HR 특징
        self.hr_enc = HRFeatureEncoder(in_ch=img_in_ch, c=hr_ch, dilations=(1,2,4,8))

        # 1/4 보조 특징 업샘플/정렬
        if use_quarter:
            self.q_adapt_L = QuarterAdapter(quarter_in_ch, out_ch=hr_ch, use_carafe=quarter_use_carafe, scale=4)
            self.q_adapt_R = QuarterAdapter(quarter_in_ch, out_ch=hr_ch, use_carafe=quarter_use_carafe, scale=4)
        else:
            self.q_adapt_L, self.q_adapt_R = None, None

        # 컨텍스트 구성 (HR + 보조를 concat 후 축소)
        ctx_in = hr_ch + (hr_ch if use_quarter else 0)
        self.ctx_proj = nn.Sequential(
            nn.Conv2d(ctx_in, ctx_ch, 3, padding=1, bias=False),
            # nn.BatchNorm2d(ctx_ch),
            DomainNorm(ctx_ch, l2=True),
            nn.SiLU(True)
        )

        # 국소 상관 볼륨
        self.cost = LocalCorrelation1D(in_ch=hr_ch, emb_ch=emb_ch, search_range=search_range)

        # 정밀화 헤드
        self.refine = RefineHead(K=self.K, ctx_ch=ctx_ch, hid=96, iters=gru_iters)

    @torch.no_grad()
    def _build_context(self, Fh: torch.Tensor, Fq_up: Optional[torch.Tensor]) -> torch.Tensor:
        if Fq_up is None:
            x = Fh
        else:
            x = torch.cat([Fh, Fq_up], dim=1)
        return self.ctx_proj(x)

    def forward(self,
                left_img: torch.Tensor,
                right_img: torch.Tensor,
                d0: torch.Tensor,
                Fq_left: Optional[torch.Tensor] = None,
                Fq_right: Optional[torch.Tensor] = None
                ) -> Dict[str, torch.Tensor]:
        """
        left_img/right_img: (B,3,H,W)
        d0: (B,1,H,W)
        Fq_left/right: (B,Cq,H/4,W/4) or None
        """
        B, _, H, W = left_img.shape

        # HR features
        Fh_L = self.hr_enc(left_img)   # (B,hr,H,W)
        Fh_R = self.hr_enc(right_img)  # 공유 가중치가 아니라면 별도 enc를 두는 것이 일반적이나,
                                       # 경량 구현을 위해 weight sharing (동일 모듈 재사용)도 가능.
                                       # 필요 시 좌/우 별도의 인코더로 분리 가능.

        # Quarter upsample/adapt
        if self.use_quarter:
            FqL_up = self.q_adapt_L(Fq_left, H, W) if Fq_left is not None else None
            FqR_up = self.q_adapt_R(Fq_right, H, W) if Fq_right is not None else None
        else:
            FqL_up, FqR_up = None, None

        # Context (좌측 기준)
        ctx_L = self._build_context(Fh_L, FqL_up)

        # Local cost volume (±r)
        cost_vol, deltas = self.cost(Fh_L, Fh_R, d0)  # (B,K,H,W)

        # Refine head (logits over K, + residual)
        logits, delta_res = self.refine(cost_vol, ctx_L)  # (B,K,H,W), (B,1,H,W)

        # Expectation over deltas (softmax)
        probs = F.softmax(logits, dim=1)  # (B,K,H,W)
        delta_vals = torch.arange(-self.search_range, self.search_range + 1, device=logits.device, dtype=logits.dtype)
        delta_vals = delta_vals.view(1, self.K, 1, 1)
        delta_exp = (probs * delta_vals).sum(dim=1, keepdim=True)  # (B,1,H,W)

        d_ref = d0 + delta_exp + delta_res  # 최종 정밀화

        # return {
        #     "d_ref": d_ref,
        #     "logits": logits,
        #     "probs": probs,
        #     "delta_res": delta_res,
        #     "Fh_L": Fh_L, "Fh_R": Fh_R,
        #     "FqL_up": FqL_up, "FqR_up": FqR_up,
        #     "ctx_L": ctx_L,
        #     "cost_vol": cost_vol
        # }

        return d_ref, _

# =========================
# 8) 손실 함수 세트 (Supervised, Photometric 금지)
# =========================

class RefineLoss(nn.Module):
    """
    총손실 = EPE(Charb) + CE(±r 내부) + L1(residual) + 상보성(Barlow/VICReg)
    """
    def __init__(self,
                 search_range: int = 4,
                 w_epe: float = 1.0,
                 w_ce: float = 0.5,
                 w_res: float = 0.05,
                 w_barlow: float = 0.05,
                 w_vic_var: float = 0.01,
                 w_vic_cov: float = 0.01,
                 label_smoothing: float = 0.0):
        super().__init__()
        self.r = int(search_range)
        self.w_epe = w_epe
        self.w_ce = w_ce
        self.w_res = w_res
        self.w_barlow = w_barlow
        self.w_vic_var = w_vic_var
        self.w_vic_cov = w_vic_cov
        self.label_smoothing = label_smoothing

    def _ce_local(self, logits, d_gt, d0, valid):
        """
        ±r 내부에서만 분류 CE 적용.
        """
        with torch.no_grad():
            delta_gt = d_gt - d0  # (B,1,H,W)
            # 최근접 정수 라벨
            tgt = delta_gt.round().clamp(-self.r, self.r).long() + self.r  # 0..K-1
            in_win = (delta_gt.abs() <= self.r + 1e-6)
            mask = valid & in_win

        if mask.any():
            # label smoothing 옵션
            K = logits.size(1)
            if self.label_smoothing > 0:
                # one-hot -> smoothed target 분포
                target = torch.zeros_like(logits).float()
                target.scatter_(1, tgt.clamp(0, K-1), 1.0)
                target = (1 - self.label_smoothing) * target + self.label_smoothing / K
                logp = F.log_softmax(logits, dim=1)
                loss = -(target * logp).sum(dim=1)  # (B,H,W)
            else:
                loss = F.cross_entropy(logits, tgt.squeeze(1), reduction='none')  # (B,H,W)

            loss = loss[mask.squeeze(1)].mean()
        else:
            loss = torch.tensor(0.0, device=logits.device)
        return loss

    def forward(self,
                outputs: Dict[str, torch.Tensor],
                d_gt: torch.Tensor,
                d0: torch.Tensor,
                valid_mask: Optional[torch.Tensor] = None,
                features_for_comp: str = "left"  # "left" or "both"
                ) -> Dict[str, torch.Tensor]:
        """
        d_gt, d0: (B,1,H,W)
        valid_mask: (B,1,H,W) - 유효 GT (예: KITTI noc/all). None이면 d_gt>0로 설정.
        """
        d_ref = outputs["d_ref"]
        logits = outputs["logits"]
        delta_res = outputs["delta_res"]

        # 유효 마스크
        if valid_mask is None:
            valid = torch.isfinite(d_gt) & (d_gt > 0)
        else:
            valid = valid_mask.bool() & torch.isfinite(d_gt)

        # 1) EPE(Charbonnier)
        epe = charbonnier((d_ref - d_gt).abs())[valid].mean() if valid.any() else torch.tensor(0.0, device=d_ref.device)

        # 2) CE (±r 내부)
        ce = self._ce_local(logits, d_gt, d0, valid)

        # 3) residual L1 (과도 수정 억제)
        res_l1 = delta_res.abs()[valid].mean() if valid.any() else torch.tensor(0.0, device=d_ref.device)

        # 4) 상보성(Barlow/VICReg)
        barlow = torch.tensor(0.0, device=d_ref.device)
        vic_var = torch.tensor(0.0, device=d_ref.device)
        vic_cov = torch.tensor(0.0, device=d_ref.device)

        def _accum_branch(Fh, Fq):
            b = barlow_cross_loss(Fh, Fq)
            v1, c1 = vicreg_variance_covariance(Fh)
            v2, c2 = vicreg_variance_covariance(Fq)
            return b, (v1 + v2) * 0.5, (c1 + c2) * 0.5

        if outputs.get("FqL_up") is not None and outputs.get("Fh_L") is not None:
            b, v, c = _accum_branch(outputs["Fh_L"], outputs["FqL_up"])
            barlow = barlow + b
            vic_var = vic_var + v
            vic_cov = vic_cov + c
        if features_for_comp == "both" and outputs.get("FqR_up") is not None and outputs.get("Fh_R") is not None:
            b, v, c = _accum_branch(outputs["Fh_R"], outputs["FqR_up"])
            barlow = barlow + b
            vic_var = vic_var + v
            vic_cov = vic_cov + c

        total = (self.w_epe * epe +
                 self.w_ce * ce +
                 self.w_res * res_l1 +
                 self.w_barlow * barlow +
                 self.w_vic_var * vic_var +
                 self.w_vic_cov * vic_cov)

        return {
            "loss_total": total,
            "loss_epe": epe.detach(),
            "loss_ce": ce.detach(),
            "loss_res": res_l1.detach(),
            "loss_barlow": barlow.detach(),
            "loss_vic_var": vic_var.detach(),
            "loss_vic_cov": vic_cov.detach()
        }

# =========================
# 9) 메트릭 (EPE 전/후 비교)
# =========================

def compute_epe_map(pred: torch.Tensor, gt: torch.Tensor, valid_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, float]:
    """
    반환: (pix-wise EPE, 평균 EPE)
    """
    if valid_mask is None:
        valid_mask = (gt > 0) & torch.isfinite(gt)
    epe_map = (pred - gt).abs()
    if valid_mask.any():
        epe_mean = epe_map[valid_mask].mean().item()
    else:
        epe_mean = float('nan')
    return epe_map, epe_mean
