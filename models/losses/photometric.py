import torch
import torch.nn.functional as F
from torch import nn
from typing import Dict, Tuple
import cv2

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
    """direction = 'R->L'  (오른쪽→왼쪽 재구성) 또는 'L->R'."""
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


def consistency_photometric_loss(data_batch, w_photo = 0.5, w_consist = 0.5):
    
    disp_l = data_batch['src_pred_disp_s'][0]
    disp_r = data_batch['src_pred_disp_s_reverse'][0]

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1).to(device='cuda:0')
    std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1).to(device='cuda:0')

    img_l = (data_batch['src_left']*std + mean).clamp(0,1)
    img_r = (data_batch['src_right']*std + mean).clamp(0,1)


    warp_to_left = warp_image(img_r, disp_l, 'R->L')
    warp_to_right = warp_image(img_l, disp_r, 'L->R')

    l1_l = (img_l - warp_to_left).abs().mean()
    l1_r = (img_r - warp_to_right).abs().mean()

    ssim_l = ssim(img_l, warp_to_left).mean() 
    ssim_r = ssim(img_r, warp_to_right).mean()

    photometric_loss = 0.85 * (ssim_l + ssim_r) / 2.0 + 0.15 * (l1_l + l1_r) / 2.0

    disp_warp = warp_image(disp_r.unsqueeze(1), disp_l.unsqueeze(1), 'R->L')
    consistency_loss = (disp_l - disp_warp).abs().mean()

    total_loss = w_photo * photometric_loss + w_consist * consistency_loss

    return {
        'loss_total':      total_loss,
        'loss_photo':      photometric_loss.detach(),
        'loss_lr':         consistency_loss.detach(),
    }

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