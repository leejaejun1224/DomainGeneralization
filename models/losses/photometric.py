import torch
import torch.nn.functional as F
from torch import nn
from typing import Dict, Tuple
import cv2

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