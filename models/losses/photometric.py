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

def consistency_photometric_loss(data_batch, w_photo=0.5, w_consist=0.3, w_smooth_loss=0.3):
    # 1) Disparities
    disp_l = data_batch['tgt_pred_disp_s'][0]          # [B, H, W]
    disp_r = data_batch['tgt_pred_disp_s_reverse'][0]  # [B, H, W]

    # 2) Denormalize images
    device = disp_l.device
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1,3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1,3,1,1)
    img_l = (data_batch['tgt_left']  * std + mean).clamp(0,1)  # [B,3,H,W]
    img_r = (data_batch['tgt_right'] * std + mean).clamp(0,1)

    # 3) Reprojection (predicted disparity)
    reproj_l = warp_image(img_r, disp_l, 'R->L')  # [B,3,H,W]
    reproj_r = warp_image(img_l, disp_r, 'L->R')

    # 4) Photometric error maps
    l1_l   = (img_l - reproj_l).abs().mean(1, keepdim=True)      # [B,1,H,W]
    l1_r   = (img_r - reproj_r).abs().mean(1, keepdim=True)
    ssim_l = 1 - ssim(img_l, reproj_l)       # [B,1,H,W]
    ssim_r = 1 - ssim(img_r, reproj_r)
    reproj_loss_map_l = 0.85 * ssim_l + 0.15 * l1_l
    reproj_loss_map_r = 0.85 * ssim_r + 0.15 * l1_r

    # 5) Identity reprojection (disp=0)
    zeros = torch.zeros_like(disp_l)
    id_l = warp_image(img_r, zeros, 'R->L')
    id_r = warp_image(img_l, zeros, 'L->R')
    id_l1   = (img_l - id_l).abs().mean(1, keepdim=True)
    id_r1   = (img_r - id_r).abs().mean(1, keepdim=True)
    id_ssim_l = 1 - ssim(img_l, id_l)
    id_ssim_r = 1 - ssim(img_r, id_r)
    id_loss_map_l = 0.85 * id_ssim_l + 0.15 * id_l1
    id_loss_map_r = 0.85 * id_ssim_r + 0.15 * id_r1

    # 6) Auto-masking via min(reproj, identity)
    combined_l = torch.cat([reproj_loss_map_l, id_loss_map_l], dim=1)  # [B,2,H,W]
    combined_r = torch.cat([reproj_loss_map_r, id_loss_map_r], dim=1)
    min_l, _ = torch.min(combined_l, dim=1, keepdim=True)  # [B,1,H,W]
    min_r, _ = torch.min(combined_r, dim=1, keepdim=True)

    photometric_loss = (min_l.mean() + min_r.mean()) * 0.5

    # 7) Left-right consistency
    disp_warp = warp_image(disp_r.unsqueeze(1), disp_l.unsqueeze(1), 'R->L')
    consistency_loss = (disp_l.unsqueeze(1) - disp_warp).abs().mean()
    smooth_loss = disparity_smoothness_loss(data_batch)
    # 8) Total
    total_loss = w_photo * photometric_loss + w_consist * consistency_loss + w_smooth_loss * smooth_loss
    
    return {
        'loss_total': total_loss,
        'loss_photo': photometric_loss.detach(),
        'loss_lr':    consistency_loss.detach(),
    }

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