import torch
import torch.nn.functional as F
from torch import nn
from typing import Dict, Tuple
import cv2

import torch
import torch.nn.functional as F



# --------- 유틸 ---------
def _normalize_vp(vp_in, B, device, dtype=torch.float32):
    # list/tuple/torch 모두 허용, [2] / [1,2] / [B,2] 전부 OK
    if isinstance(vp_in, (list, tuple)):
        vp = torch.tensor(vp_in, dtype=dtype, device=device)
    else:
        vp = vp_in.to(device=device, dtype=dtype)

    if vp.ndim == 1:
        assert vp.numel() == 2, f"vanishing_points must be 2 numbers, got {vp.numel()}"
        vp = vp.view(1, 2)
    elif vp.ndim == 2:
        assert vp.size(1) == 2, f"vanishing_points must be shape [*,2], got {vp.shape}"
    else:
        raise ValueError(f"vanishing_points shape not supported: {vp.shape}")

    if vp.size(0) == 1 and B > 1:
        vp = vp.expand(B, 2).clone()
    elif vp.size(0) != B:
        # 길이 불일치 자동 보정(반복/절단)
        if vp.size(0) > B:
            vp = vp[:B]
        else:
            rep = (B + vp.size(0) - 1) // vp.size(0)
            vp = vp.repeat(rep, 1)[:B]
    return vp  # [B,2]

def _to_grid_coords(xp, yp, H, W):
    """
    xp, yp : [B,1,H,W] 또는 [B,H,W]
    return : [B,H,W,2]  (grid_sample 2D 포맷)
    """
    if xp.ndim == 4 and xp.size(1) == 1:
        xp = xp[:, 0]
        yp = yp[:, 0]
    gx = (xp / max(W-1, 1)) * 2.0 - 1.0
    gy = (yp / max(H-1, 1)) * 2.0 - 1.0
    return torch.stack((gx, gy), dim=-1)  # [B,H,W,2]

def _samp2d(img, grid, mode='bilinear'):
    """
    img : [B,1,H,W]  (float)
    grid: [B,H,W,2]  (float)
    """
    if img.ndim == 3:  # [B,H,W] -> [B,1,H,W]
        img = img.unsqueeze(1)
    img  = img.to(torch.float32).contiguous()
    grid = grid.to(torch.float32).contiguous()
    return F.grid_sample(img, grid, mode=mode, padding_mode='border', align_corners=True)

# --------- 메인 로스 ---------
def vp_smooth_loss(
    data_batch,  # [B,2] 또는 [2] 등 (disp 해상도 기준 좌표)
    step=1.0,           # 소실점 방향 스텝(픽셀)
    tau=3.0,            # 인접 차이 >= tau 면 스무딩 배제
    use_pseudo_for_gate=True,
    reduction='mean',
    eps=1e-6
):  
    disp_pred = data_batch['tgt_pred_disp_s_for_loss'].unsqueeze(1)  # [B,1,H,W]
    disp_pseudo = data_batch['pseudo_disp'][0].unsqueeze(1) 
    valid_mask = data_batch['tgt_mask_pred_s'] > 0.8
    vanishing_points = torch.tensor(
        [[1248/2, 384/2]] * 1,     # 배치 모두 같은 VP
        dtype=torch.float32, device=disp_pred.device
    )
    B, _, H, W = disp_pred.shape
    device = disp_pred.device

    # 1) VP 정규화
    vp = _normalize_vp(vanishing_points, B, device)  # [B,2]
    x_v = vp[:, 0].view(B,1,1,1)
    y_v = vp[:, 1].view(B,1,1,1)

    # 2) 좌표/방향 필드
    ys, xs = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing='ij'
    )
    xs = xs[None, None].expand(B, 1, H, W)
    ys = ys[None, None].expand(B, 1, H, W)

    dx = x_v - xs
    dy = y_v - ys
    norm = torch.sqrt(dx*dx + dy*dy + eps)
    rx = dx / norm
    ry = dy / norm

    dx1 = step * rx;  dy1 = step * ry
    dx2 = 2.0 * dx1;  dy2 = 2.0 * dy1

    # 3) 그리드 생성 (2D 포맷으로)
    g0 = _to_grid_coords(xs,          ys,          H, W)  # [B,H,W,2]
    g1 = _to_grid_coords(xs + dx1,    ys + dy1,    H, W)
    g2 = _to_grid_coords(xs + dx2,    ys + dy2,    H, W)

    # 4) 샘플링
    d0 = _samp2d(disp_pred, g0, mode='bilinear')  # [B,1,H,W]
    d1 = _samp2d(disp_pred, g1, mode='bilinear')
    d2 = _samp2d(disp_pred, g2, mode='bilinear')

    base = disp_pseudo if use_pseudo_for_gate else disp_pred
    b0 = _samp2d(base,       g0, mode='bilinear')
    b1 = _samp2d(base,       g1, mode='bilinear')
    b2 = _samp2d(base,       g2, mode='bilinear')

    if valid_mask.ndim == 3:
        valid_mask = valid_mask.unsqueeze(1)
    m0 = _samp2d(valid_mask, g0, mode='nearest')   # [B,1,H,W]
    m1 = _samp2d(valid_mask, g1, mode='nearest')
    m2 = _samp2d(valid_mask, g2, mode='nearest')
    m012 = (m0 > 0.5) & (m1 > 0.5) & (m2 > 0.5)

    # 5) 게이팅(이미 급변이면 제외)
    gate_small_jump = ((b1 - b0).abs() < tau) & ((b2 - b1).abs() < tau)
    gate = (m012 & gate_small_jump).float()  # [B,1,H,W]

    # 6) 2차차분 + Charbonnier
    curv = d0 - 2.0*d1 + d2
    loss_map = torch.sqrt(curv*curv + eps*eps) * gate

    denom = gate.sum() + eps
    if reduction == 'sum':
        return loss_map.sum() / denom
    else:  # 'mean'
        return loss_map.sum() / denom




def compute_photometric_error(data_batch, threshold):
    """
    img_left, img_right: (B, C, H, W), normalized [0,1]
    disp:             (B, 1, H, W), disparities for right->left warp
    returns:          (B, 1, H, W) photometric L1 error between img_left and warped img_right
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1).to(device='cuda:0')
    std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1).to(device='cuda:0')

    disp = data_batch['pseudo_disp'][0]  # (B, 1, H, W)
    img_left = (data_batch['tgt_left'] * std + mean).clamp(0,1)
    img_right = (data_batch['tgt_right'] * std + mean).clamp(0,1)
    
    B, C, H, W = img_left.shape

    # meshgrid for sampling
    y, x = torch.meshgrid(
        torch.arange(H, device=img_left.device),
        torch.arange(W, device=img_left.device),
        indexing='ij'
    )
    # build sampling coords for grid_sample
    x = x.unsqueeze(0).expand(B, -1, -1).float()
    y = y.unsqueeze(0).expand(B, -1, -1).float()
    # compute source x coords: x_src = x - disp
    x_src = x - disp.squeeze(1)
    # normalize to [-1,1] for grid_sample
    x_norm = 2.0 * (x_src / (W - 1)) - 1.0
    y_norm = 2.0 * (y     / (H - 1)) - 1.0
    grid = torch.stack((x_norm, y_norm), dim=3)  # (B, H, W, 2)

    # warp right image into left view
    warped_right = F.grid_sample(img_right, grid, mode='bilinear', padding_mode='border', align_corners=True)
    # photometric L1
    photo_err = torch.abs(img_left - warped_right).mean(dim=1, keepdim=True)
    
    valid_mask = (photo_err <= threshold).float()
    data_batch['valid_disp'] = disp*valid_mask
    

def _meshgrid(h: int, w: int, device):
    y, x = torch.meshgrid(
        torch.arange(h, device=device),
        torch.arange(w, device=device),
        indexing='ij'
    )
    return y.float(), x.float()


def warp_image(src_img: torch.Tensor,
               disp: torch.Tensor,
               direction: str = 'R->L') -> torch.Tensor:
    assert direction in ('R->L', 'L->R')
    B, C, H, W = src_img.shape
    y, x = _meshgrid(H, W, src_img.device)
    if direction == 'R->L':     # right → left 로 가져오기
        x_src = x.unsqueeze(0) - disp.squeeze(1)   # x_R = x_L – d
    else:                       # left → right
        x_src = x.unsqueeze(0) + disp.squeeze(1)   # x_L = x_R + d
    y_src = y.unsqueeze(0).expand_as(x_src)

    # 정규화 좌표계 [-1,1]
    x_norm = 2.0 * x_src / (W - 1) - 1.0
    y_norm = 2.0 * y_src / (H - 1) - 1.0
    grid   = torch.stack((x_norm, y_norm), dim=3)      # [B,H,W,2]

    return F.grid_sample(src_img, grid,
                         mode='bilinear',
                         padding_mode='border',
                         align_corners=True)




def ssim(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    mu_x   = F.avg_pool2d(x, 3, 1, 1)
    mu_y   = F.avg_pool2d(y, 3, 1, 1)
    sigma_x  = F.avg_pool2d(x * x, 3, 1, 1) - mu_x ** 2
    sigma_y  = F.avg_pool2d(y * y, 3, 1, 1) - mu_y ** 2
    sigma_xy = F.avg_pool2d(x * y, 3, 1, 1) - mu_x * mu_y
    ssim_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    ssim_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)
    return torch.clamp((1 - ssim_n / ssim_d) / 2, 0, 1)

def consistency_photometric_loss(
        data_batch,
        w_photo=0.8,        # photometric loss 가중치
        w_consist=0.1,      # consistency loss 가중치
        w_smooth_loss=0.1,
        w_div_loss=0.05,
        disp_thresh_px=3.0,  # occlusion 임계값 (논문에서 δ=3)
        photo_thresh=0.3
):
    import torch
    
    # 예측 시차
    disp_l = data_batch['tgt_pred_disp_s'][0]
    disp_r = data_batch['tgt_pred_disp_s_reverse'][0]
    
    # 배치 차원 추가 (필요시)
    if disp_l.dim() == 2:
        disp_l = disp_l.unsqueeze(0)
        disp_r = disp_r.unsqueeze(0)
    
    # 이미지 복원
    device = disp_l.device
    mean = torch.tensor([0.485, 0.456, 0.406], device=device)[:, None, None]
    std  = torch.tensor([0.229, 0.224, 0.225], device=device)[:, None, None]
    img_l = (data_batch['tgt_left'] * std + mean).clamp(0, 1)
    img_r = (data_batch['tgt_right'] * std + mean).clamp(0, 1)
    
    # 재투영
    reproj_l = warp_image(img_r, disp_l, 'R->L')
    reproj_r = warp_image(img_l, disp_r, 'L->R')
    
    # Occlusion 마스크 생성 (FCStereo 방식)
    occlusion_mask = add_occlusion_mask(disp_l, disp_r, thresh=disp_thresh_px)
    
    # data_batch에 마스크 추가
    data_batch['occlusion_mask_l'] = occlusion_mask
    data_batch['occlusion_mask_r'] = occlusion_mask  # 대칭적으로 사용
    
    # Photometric loss (마스크 적용)
    ssim_l = ssim(img_l, reproj_l)
    ssim_r = ssim(img_r, reproj_r)
    l1_l = (img_l - reproj_l).abs()
    l1_r = (img_r - reproj_r).abs()
    
    # 마스크를 적용한 photometric loss
    mask_expanded = occlusion_mask.unsqueeze(1)  # [B, 1, H, W]
    
    ssim_l_masked = (ssim_l * mask_expanded).sum() / (mask_expanded.sum() + 1e-7)
    ssim_r_masked = (ssim_r * mask_expanded).sum() / (mask_expanded.sum() + 1e-7)
    l1_l_masked = (l1_l * mask_expanded).sum() / (mask_expanded.sum() + 1e-7)
    l1_r_masked = (l1_r * mask_expanded).sum() / (mask_expanded.sum() + 1e-7)
    
    photometric_loss = 0.85 * (ssim_l_masked + ssim_r_masked) / 2.0 + \
                      0.15 * (l1_l_masked + l1_r_masked) / 2.0
    
    # Consistency loss (마스크 적용)
    disp_r_warp = warp_image(disp_r.unsqueeze(1), disp_l.unsqueeze(1), 'R->L')
    consistency_diff = (disp_l.unsqueeze(1) - disp_r_warp).abs()
    
    # 마스크를 적용한 consistency loss
    consistency_loss = (consistency_diff * mask_expanded).sum() / (mask_expanded.sum() + 1e-7)
    
    # Smoothness losses
    smooth_loss = disparity_smoothness_loss(data_batch)
    div_loss = disparity_div_smoothness_loss(data_batch)
    
    # 전체 loss 조합
    total_loss = (w_photo * photometric_loss + 
                  w_consist * consistency_loss +
                  w_smooth_loss * smooth_loss + 
                  w_div_loss * div_loss)
    
    return {
        "loss_total": total_loss,
        "loss_photo": photometric_loss.detach(),
        "loss_consist": consistency_loss.detach(),
        "loss_smooth": smooth_loss.detach(),
        "loss_div": div_loss.detach(),
        "occlusion_ratio": (1.0 - occlusion_mask.mean()).detach()  # 추가 정보
    }

def add_occlusion_mask(disp_l, disp_r, thresh=3.0):
    """
    FCStereo 논문의 left-right geometric consistency check 구현
    
    Args:
        disp_l: 왼쪽 disparity map [B, H, W]
        disp_r: 오른쪽 disparity map [B, H, W]  
        thresh: 일관성 임계값 (논문에서 δ=3 사용)
    
    Returns:
        mask: occlusion mask [B, H, W] (1: valid, 0: occluded)
    """
    import torch
    
    B, H, W = disp_l.shape
    device = disp_l.device
    
    # 격자 좌표 생성
    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, device=device), 
        torch.arange(W, device=device),
        indexing='ij'
    )
    grid_x = grid_x.float()
    grid_y = grid_y.float()
    
    # 오른쪽 disparity를 왼쪽으로 warp
    src_x = grid_x - disp_r[0]  # disparity만큼 이동
    src_y = grid_y
    
    # normalized 좌표 생성 ([-1,1] 범위)
    src_x_norm = 2.0 * (src_x / (W - 1)) - 1.0
    src_y_norm = 2.0 * (src_y / (H - 1)) - 1.0
    
    grid_norm = torch.stack((src_x_norm, src_y_norm), dim=2)  # [H, W, 2]
    grid_norm = grid_norm.unsqueeze(0).repeat(B, 1, 1, 1)  # [B, H, W, 2]
    
    # grid_sample을 사용한 warping
    disp_r_1c = disp_r.unsqueeze(1)  # [B, 1, H, W]
    disp_r_warped = torch.nn.functional.grid_sample(
        disp_r_1c, grid_norm, 
        mode='bilinear', 
        padding_mode='border', 
        align_corners=True
    ).squeeze(1)  # [B, H, W]
    
    # 재투영 오차 계산 (논문의 R 값)
    reprojection_error = torch.abs(disp_l - disp_r_warped)
    
    # 마스크 생성 (임계값 이하면 valid)
    mask = (reprojection_error < thresh).float()
    
    return mask

# def consistency_photometric_loss(data_batch, w_photo = 0.5, w_consist = 0.5):
    
#     disp_l = data_batch['tgt_pred_disp_s'][0]
#     disp_r = data_batch['tgt_pred_disp_s_reverse'][0]

#     mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1).to(device='cuda:0')
#     std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1).to(device='cuda:0')

#     img_l = (data_batch['tgt_left'] * std + mean).clamp(0,1)
#     img_r = (data_batch['tgt_right'] * std + mean).clamp(0,1)


#     warp_to_left = warp_image(img_r, disp_l, 'R->L')
#     warp_to_right = warp_image(img_l, disp_r, 'L->R')

#     l1_l = (img_l - warp_to_left).abs().mean()
#     l1_r = (img_r - warp_to_right).abs().mean()

#     ssim_l = ssim(img_l, warp_to_left).mean() 
#     ssim_r = ssim(img_r, warp_to_right).mean()

#     photometric_loss = 0.85 * (ssim_l + ssim_r) / 2.0 + 0.15 * (l1_l + l1_r) / 2.0

#     disp_warp = warp_image(disp_r.unsqueeze(1), disp_l.unsqueeze(1), 'R->L')
#     consistency_loss = (disp_l - disp_warp).abs().mean()

#     total_loss = w_photo * photometric_loss + w_consist * consistency_loss

#     return {
#         'loss_total':      total_loss,
#         'loss_photo':      photometric_loss.detach(),
#         'loss_lr':         consistency_loss.detach(),
#     }

# def consistency_photometric_loss(data_batch, w_photo=1.0, w_consist=0.0, w_smooth=0.2):
#     # 예측된 left/right disparity
#     disp_l = data_batch['tgt_pred_disp_s'][0]               # (B, H, W)

#     # 정규화 값 복원
#     mean = torch.tensor([0.485, 0.456, 0.406], device=disp_l.device)[:, None, None]
#     std  = torch.tensor([0.229, 0.224, 0.225], device=disp_l.device)[:, None, None]

#     img_l = (data_batch['tgt_left']  * std + mean).clamp(0, 1)   # (B,3,H,W)
#     img_r = (data_batch['tgt_right'] * std + mean).clamp(0, 1)

#     # photometric loss: 오직 left 기준으로만 계산
#     # → 오른쪽 disparity disp_r는 photometric term에서 제거
#     # R→L warp: img_r 을 disp_l 로 왼쪽 좌표로 되돌림
#     warp_to_left = warp_image(img_r, disp_l, 'R->L')

#     # L1 & SSIM (left only)
#     l1_l   = (img_l - warp_to_left).abs().mean()
#     ssim_l = ssim(img_l, warp_to_left).mean()

#     photometric_loss = 0.85 * ssim_l + 0.15 * l1_l
#     smooth_loss = disparity_smoothness_loss(data_batch)
#     # consistency loss (좌-우 일관성) 은 그대로 유지
#     # disp_warp shape → (B,1,H,W), squeeze back to (B,H,W)

#     total_loss = w_photo * photometric_loss + w_smooth * smooth_loss

#     return {
#         'loss_total': total_loss,
#         'loss_photo': photometric_loss.detach()
#     }

def disparity_smoothness_loss(data_batch):


    mean = torch.tensor([0.485, 0.456, 0.406], device='cuda:0')[:, None, None]
    std  = torch.tensor([0.229, 0.224, 0.225], device='cuda:0')[:, None, None]

    img_l = (data_batch['tgt_left']  * std + mean).clamp(0, 1)# -> unnormalize필요
    disp = data_batch['tgt_pred_disp_s'][0].unsqueeze(1)

    # img -> [batch, 3, h, w]
    dx_img = img_l[:,:,:,:-1] - img_l[:,:,:,1:]
    dy_img = img_l[:,:,:-1,:] - img_l[:,:,1:,:]

    # disp -> 
    dx_disp = disp[:,:,:,:-1] - disp[:,:,:,1:]
    dy_disp = disp[:,:,:-1,:] - disp[:,:,1:,:]


    wx = torch.exp(-dx_img)
    wy = torch.exp(-dy_img)

    smooth_x = torch.abs(dx_disp)*wx
    smooth_y = torch.abs(dy_disp)*wy

    loss = (smooth_x.sum() + smooth_y.sum())/(smooth_x.numel() + smooth_y.numel() + 1e-8)

    return loss
def disparity_div_smoothness_loss(
        data_batch,
        alpha: float = 10.0,
        clip_value: float = 2.0,
        eps: float = 1e-8,
):
    """
    DIV-D : 클램프된 2차(edge-aware) 평활 손실
    """
    device = data_batch['tgt_left'].device
    mean = torch.tensor([0.485, 0.456, 0.406], device=device)[:, None, None]
    std  = torch.tensor([0.229, 0.224, 0.225], device=device)[:, None, None]

    # ─── 0. 입력 복원 ───────────────────────────────────────────
    img  = (data_batch['tgt_left'] * std + mean).clamp(0, 1)        # [B,3,H,W]
    disp = data_batch['tgt_pred_disp_s'][0].unsqueeze(1)            # [B,1,H,W]

    # ─── 1. 2차 차분(중앙 차분) ────────────────────────────────
    dxx = disp[:, :, :, :-2] - 2.0 * disp[:, :, :, 1:-1] + disp[:, :, :, 2:]   # [B,1,H,W-2]
    dyy = disp[:, :, :-2, :] - 2.0 * disp[:, :, 1:-1, :] + disp[:, :, 2:, :]   # [B,1,H-2,W]

    dxx = torch.clamp(dxx, -clip_value, clip_value)
    dyy = torch.clamp(dyy, -clip_value, clip_value)

    # ─── 2. 엣지-가중치(크기 정합) ─────────────────────────────
    # 1-차 밝기 gradient
    gx = torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1])             # [B,3,H,W-1]
    gy = torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :])             # [B,3,H-1,W]

    # dxx 위치(열 1~W-2)의 엣지값 = 인접 두 gradient 평균
    gx_c = 0.5 * (gx[:, :, :, :-1] + gx[:, :, :, 1:])               # [B,3,H,W-2]
    gy_c = 0.5 * (gy[:, :, :-1, :] + gy[:, :, 1:, :])               # [B,3,H-2,W]

    wx = torch.exp(-alpha * gx_c.mean(1, keepdim=True))             # [B,1,H,W-2]
    wy = torch.exp(-alpha * gy_c.mean(1, keepdim=True))             # [B,1,H-2,W]

    # ─── 3. 손실 계산 ──────────────────────────────────────────
    loss_num = (torch.abs(dxx) * wx).sum() + (torch.abs(dyy) * wy).sum()
    loss_den = dxx.numel() + dyy.numel() + eps
    return loss_num / loss_den