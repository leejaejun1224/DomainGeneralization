import torch
import torch.nn as nn
import torch.nn.functional as F

class SiLogLoss(nn.Module):
    def __init__(self, lambd=0.5):
        super().__init__()
        self.lambd = lambd

    def forward(self, pred, target, valid_mask):
        valid_mask = valid_mask.detach()
        diff_log = torch.log(target[valid_mask]) - torch.log(pred[valid_mask])
        loss = torch.sqrt(torch.pow(diff_log, 2).mean() -
                          self.lambd * torch.pow(diff_log.mean(), 2))
        return loss

def norm_grid(v_grid):
    _, _, H, W = v_grid.size()

    # scale grid to [-1,1]
    v_grid_norm = torch.zeros_like(v_grid)
    v_grid_norm[:, 0, :, :] = 2.0 * v_grid[:, 0, :, :] / (W - 1) - 1.0
    v_grid_norm[:, 1, :, :] = 2.0 * v_grid[:, 1, :, :] / (H - 1) - 1.0
    return v_grid_norm.permute(0, 2, 3, 1)  # BHW2

def mesh_grid(B, H, W):
    # mesh grid
    x_base = torch.arange(0, W).repeat(B, H, 1)  # BHW
    y_base = torch.arange(0, H).repeat(B, W, 1).transpose(1, 2)  # BHW

    base_grid = torch.stack([x_base, y_base], 1)  # B2HW
    return base_grid

def disp_warp(x, disp, r2l=False, pad='border', mode='bilinear'):
    B, _, H, W = x.size()
    offset = -1
    if r2l:
        offset = 1

    base_grid = mesh_grid(B, H, W).type_as(x)  # B2HW
    v_grid = norm_grid(base_grid + torch.cat((offset*disp,torch.zeros_like(disp)),1))  # BHW2
    x_recons = nn.functional.grid_sample(x, v_grid, mode=mode, padding_mode=pad)
    return x_recons
def SSIM(x: torch.Tensor, y: torch.Tensor, md=1):
    assert x.size() == y.size(), f"xsize: {x.size()}, ysize: {y.size()}"

    patch_size = 2 * md + 1
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    refl = nn.ReflectionPad2d(md)

    x = refl(x)
    y = refl(y)
    mu_x = nn.AvgPool2d(patch_size, 1, 0)(x)
    mu_y = nn.AvgPool2d(patch_size, 1, 0)(y)
    mu_x_mu_y = mu_x * mu_y
    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)

    sigma_x = nn.AvgPool2d(patch_size, 1, 0)(x * x) - mu_x_sq
    sigma_y = nn.AvgPool2d(patch_size, 1, 0)(y * y) - mu_y_sq
    sigma_xy = nn.AvgPool2d(patch_size, 1, 0)(x * y) - mu_x_mu_y

    SSIM_n = (2 * mu_x_mu_y + C1) * (2 * sigma_xy + C2)
    SSIM_d = (mu_x_sq + mu_y_sq + C1) * (sigma_x + sigma_y + C2)
    SSIM = SSIM_n / SSIM_d
    dist = torch.clamp((1 - SSIM) / 2, 0, 1)
    return dist

def loss_photometric(im1_scaled, im1_recons):
    loss = []

    loss += [0.15 * (im1_scaled - im1_recons).abs().mean(1, True)]
    loss += [0.85 * SSIM(im1_recons, im1_scaled).mean(1, True)]
    return sum([l for l in loss])
def gradient(data):
    D_dy = data[:, :, 1:] - data[:, :, :-1]
    D_dx = data[:, :, :, 1:] - data[:, :, :, :-1]
    return D_dx, D_dy
def smooth_grad(disp, image, alpha, order=1):
    img_dx, img_dy = gradient(image)
    weights_x = torch.exp(-torch.mean(torch.abs(img_dx), 1, keepdim=True) * alpha)
    weights_y = torch.exp(-torch.mean(torch.abs(img_dy), 1, keepdim=True) * alpha)

    dx, dy = gradient(disp)
    if order == 2:
        dx2, dxdy = gradient(dx)
        dydx, dy2 = gradient(dy)
        dx, dy = dx2, dy2

    loss_x = weights_x[:, :, :, 1:] * dx[:, :, :, 1:].abs()
    loss_y = weights_y[:, :, 1:, :] * dy[:, :, 1:, :].abs()

    return loss_x.mean() / 2. + loss_y.mean() / 2.

def loss_smooth(disp, im1_scaled):
#    if 'smooth_2nd' in self.cfg and self.cfg.smooth_2nd:
#    func_smooth = smooth_grad_2nd
#    else:
    func_smooth = smooth_grad
    loss = []
    loss += [func_smooth(disp, im1_scaled, 1, order=1)]
    return sum([l.mean() for l in loss])



class MonoDepthLoss(nn.Module):
    """
    Self-supervised Monodepth loss:
      1) Photometric reconstruction (L1 + SSIM)
      2) Edge-aware smoothness
      3) Optional scale-invariant log loss if GT is available
    """
    def __init__(self,
                 photometric_weight: float = 1.0,
                 smooth_weight: float = 0.001,
                 silog_weight: float = 1.0,
                 silog_lambda: float = 0.85,
                 l1_weight: float = 0.8):
        super().__init__()
        self.photometric_weight = photometric_weight
        self.smooth_weight = smooth_weight
        self.silog_weight = silog_weight
        self.l1_weight = l1_weight
        # 기존에 정의된 SILogLoss 사용
        self.silog = SiLogLoss(lambd=silog_lambda)

        
    def forward(self,
                disp: torch.Tensor,
                im1: torch.Tensor,
                im2: torch.Tensor,
                mask: torch.Tensor = None,
                disp_gt: torch.Tensor = None) -> torch.Tensor:
        # """
        # Args:
        #   disp:  predicted disparity [B,1,H,W]
        #   im1:   left image [B,3,H,W]
        #   im2:   right image [B,3,H,W]
        #   mask:  optional valid‑pixel mask [B,1,H,W]
        #   disp_gt: optional GT disparity for SILog term
        # Returns:
        #   scalar total loss
        # """
        # # 1) Photometric: warp im2→im1
        # im1_recon = disp_warp(im2, disp)                   # [B,3,H,W]
        # photometric = loss_photometric(im1, im1_recon)     # [B,1,H,W]
        # # if mask is not None:
        # #     photometric = photometric * mask
        # photometric_loss = photometric.mean()

        # # 2) Edge-aware smoothness
        # smooth_loss = loss_smooth(disp, im1)               # scalar

        # # 3) Optional SILog if GT 제공 시
        # silog_loss = torch.tensor(0., device=disp.device)
        # if disp_gt is not None and self.silog_weight > 0:
        #     valid_gt = disp_gt > 0
        #     silog_loss = self.silog(disp, disp_gt, valid_gt)
        #     l1_loss = F.smooth_l1_loss(disp[valid_gt], disp_gt[valid_gt], size_average=True)

        # # Combine with weights
        # total = (
        #     self.l1_weight * l1_loss +
        #     self.photometric_weight * photometric_loss +
        #     self.smooth_weight * smooth_loss +
        #     self.silog_weight   * silog_loss
        # )

        valid_mask = disp_gt > 0
        l1_loss = F.smooth_l1_loss(disp[valid_mask], disp_gt[valid_mask], size_average=True)
        total = l1_loss
        
        return total

# — 사용 예시 —
# loss_fn = MonoDepthLoss(photometric_weight=1.0, smooth_weight=0.1, silog_weight=0.2)
# loss = loss_fn(pred_disp, img_L, img_R, mask=lr_mask(pred_disp, pred_disp), disp_gt=gt_disp)