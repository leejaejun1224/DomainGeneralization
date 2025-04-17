import torch
import torch.nn.functional as F

def ssim(img1, img2, C1=0.01**2, C2=0.03**2):
    mu1, mu2 = img1.mean([2,3], keepdim=True), img2.mean([2,3], keepdim=True)
    sigma1 = (img1 - mu1).pow(2).mean([2,3], keepdim=True)
    sigma2 = (img2 - mu2).pow(2).mean([2,3], keepdim=True)
    sigma12 = ((img1 - mu1)*(img2 - mu2)).mean([2,3], keepdim=True)
    ssim_map = ((2*mu1*mu2 + C1)*(2*sigma12 + C2)) / ((mu1**2 + mu2**2 + C1)*(sigma1 + sigma2 + C2))
    return ssim_map.clamp(0,1)

def warp_right_to_left(I_R, disparity_L):
    """
    I_R         : (B, 3, H, W)
    disparity_L : (B, H, W)   or (B, 1, H, W)
    returns I_R warped into the left view
    """
    # ensure disparity is [B, H, W]
    if disparity_L.dim() == 4:
        disparity = disparity_L.squeeze(1)
    else:
        disparity = disparity_L

    B, _, H, W = I_R.shape

    # build a meshgrid of pixel coordinates
    x = torch.linspace(0, W - 1, W, device=I_R.device)   # (W,)
    y = torch.linspace(0, H - 1, H, device=I_R.device)   # (H,)

    # expand to [B, H, W]
    x_grid = x.view(1, 1, W).expand(B, H, W)
    y_grid = y.view(1, H, 1).expand(B, H, W)

    # apply disparity offset (positive disparity shifts right→left)
    x_warp = x_grid - disparity

    # normalize to [-1, 1]
    x_norm = 2.0 * (x_warp / (W - 1)) - 1.0
    y_norm = 2.0 * (y_grid / (H - 1)) - 1.0

    # stack to [B, H, W, 2]
    grid = torch.stack((x_norm, y_norm), dim=3)

    # sample
    return F.grid_sample(I_R, grid, mode='bilinear',
                         padding_mode='border',
                         align_corners=True)


def photometric_loss(data_batch, ssim_w=0.85):
    I_L = data_batch['src_left']
    I_R = data_batch['tgt_right']
    disparity_L = data_batch['depth_map_s_up']

    I_hat = warp_right_to_left(I_R, disparity_L)
    l1     = torch.abs(I_L - I_hat).mean(1, True)
    ssim_v = ssim(I_L, I_hat)
    photometric = ssim_w * (1 - ssim_v)/2 + (1-ssim_w) * l1
    return photometric.mean()          # scalar

def photometric_loss_low(data_batch, ssim_w=0.85):
    I_L = data_batch['src_left_low']
    I_R = data_batch['tgt_right_low']
    disparity_L = data_batch['depth_map_s']

    I_hat = warp_right_to_left(I_R, disparity_L)
    l1     = torch.abs(I_L - I_hat).mean(1, True)
    ssim_v = ssim(I_L, I_hat)
    photometric = ssim_w * (1 - ssim_v)/2 + (1-ssim_w) * l1
    return photometric.mean()          # scalar

def photometric_loss_half(data_batch, ssim_w=0.85):
    I_L = data_batch['src_left_half']
    I_R = data_batch['tgt_right_half']
    disparity_L = data_batch['depth_map_s_half']  

    I_hat = warp_right_to_left(I_R, disparity_L)
    l1     = torch.abs(I_L - I_hat).mean(1, True)
    ssim_v = ssim(I_L, I_hat)
    photometric = ssim_w * (1 - ssim_v)/2 + (1-ssim_w) * l1
    return photometric.mean()          # scalar