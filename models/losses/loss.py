import torch 
import torch.nn as nn
import torch.nn.functional as F


def calc_entropy_loss(source_entropy, target_entropy, mask):
    # entropy_loss = F.smooth_l1_loss(source_entropy[mask], target_entropy[mask], size_average=True)
    target = torch.zeros_like(source_entropy)
    entropy_loss = F.smooth_l1_loss(source_entropy[mask], target[mask], size_average=True) * 100
    return entropy_loss




def get_loss(disp_ests, disp_gts, img_masks, weights):

    all_losses = []

    for disp_est, disp_gt, img_mask, weight in zip(disp_ests, disp_gts, img_masks, weights):
        
        all_losses.append(weight * F.smooth_l1_loss(disp_est[img_mask], disp_gt[img_mask], size_average=True))
    return sum(all_losses)

# dont touch 
def calc_supervised_train_loss(data_batch, model='s'):
    key = 'src_pred_disp_' + model
    pred_disp, gt_disp, gt_disp_low = data_batch[key], data_batch['src_disparity'], data_batch['src_disparity_low']
    
    mask = (gt_disp > 0) & (gt_disp < 256)
    data_batch['mask'] = mask
    mask_low = (gt_disp_low > 0) & (gt_disp_low < 256)
    masks = [mask, mask_low, mask, mask_low]
    gt_disps = [gt_disp, gt_disp_low, gt_disp, gt_disp_low]
    # scale별 weight 예시
    weights = [1.0, 0.3, 0.5, 0.3]
    loss = get_loss(pred_disp, gt_disps, masks, weights)
    return loss


def calc_supervised_val_loss(data_batch, model='s'):
    key = 'src_pred_disp_' + model
    pred_disp, gt_disp = data_batch[key], data_batch['src_disparity']

    mask = (gt_disp > 0) & (gt_disp < 256)
    data_batch['mask'] = mask
    masks = [mask]
    gt_disps = [gt_disp]
    # scale별 weight 예시
    weights = [1.0]
    loss = get_loss(pred_disp, gt_disps, masks, weights)
    return loss


def calc_pseudo_loss(data_batch, threshold, model='s'):
    key = 'tgt_pred_disp_' + model
    pred_disp, pseudo_disp, confidence_map = data_batch[key], data_batch['pseudo_disp'][1], data_batch['confidence_map']
    # print("confidence_map", confidence_map.shape)
    # print("pseudo_disp", pseudo_disp.shape)
    # print("pred_disp", pred_disp[1].shape)


    ### only calculate loss for the index number one of output
    ### not the batch size!!!!!!!!!!!!!!! dont be confuse
    pred_disp = pred_disp[1]


    ## oh shit it only think about the first index of the output 
    ## no consider the batch size
    mask = (pseudo_disp > 0) & (pseudo_disp < 256) & (confidence_map >= threshold.unsqueeze(1).unsqueeze(2).cuda())
    data_batch['pseudo_mask'] = mask
    mask = mask.tolist()
    weights = [1.0]
    confidence_mask = confidence_map >= threshold.unsqueeze(1).unsqueeze(2).cuda()
    true_count = confidence_mask.sum(dim=(0,1,2)) 
    total_pixels = confidence_mask.numel()
    true_ratio = true_count.float() / total_pixels

    pseudo_label_loss = get_loss(pred_disp, pseudo_disp, mask, weights)
    return pseudo_label_loss, true_ratio



def calc_pseudo_entropy_top1_loss(data_batch, model='s'):

    entropy_mask = data_batch['tgt_entropy_map_t'] > 0
    
    true_count = entropy_mask.sum() 
    total_pixels = entropy_mask.numel()
    true_ratio = true_count.float() / total_pixels

    weights = [1.0]
    entropy_mask = [entropy_mask]


    pseudo_label_loss = get_loss([data_batch['tgt_pred_disp_' + model][1].unsqueeze(1)], 
                                 [data_batch['tgt_entropy_map_idx_t']], 
                                 entropy_mask, 
                                 weights)

    return pseudo_label_loss, true_ratio


def calc_pseudo_entropy_loss(data_batch, min_ent=0.00088, model='s'):

    entropy_map = data_batch['tgt_entropy_map_' + model]
    target_entropy = torch.clamp(entropy_map - min_ent, min=0)
    entropy_loss = nn.L1Loss(reduction='sum')(entropy_map, target_entropy)

    entropy_mask = entropy_map < 0.00089

    true_count = entropy_mask.sum() 
    total_pixels = entropy_mask.numel()
    true_ratio = true_count.float() / total_pixels

    # 디버깅 출력
    return entropy_loss, true_ratio



def calc_pseudo_soft_loss(data_batch, threshold, model='s'):
    key = 'tgt_pred_disp_' + model
    pred_disp, pseudo_disp = data_batch[key], data_batch['pseudo_disp'][1]

    mask = (pseudo_disp > 0) & (pseudo_disp < 256)
    data_batch['pseudo_mask'] = mask
    mask = mask.tolist()
    weights = [1.0]
    pseudo_soft_loss = get_loss(pred_disp, pseudo_disp, mask, weights)
    return pseudo_soft_loss


def calc_reconstruction_loss(data_batch, domain='tgt', model='s', alpha=0.85):

    disp_key = f'{domain}_pred_disp_{model}' if domain == 'src' else 'pseudo_disp'
    left_key = f'{domain}_left'
    right_key = f'{domain}_right'

    mask = data_batch[disp_key][2] > 0
    left_masked = data_batch[left_key] * mask.unsqueeze(1)
    B, C, H, W = data_batch[left_key].shape

    # Create base coordinate grids
    y_base, x_base = torch.meshgrid(
        torch.arange(H, device=data_batch[left_key].device),
        torch.arange(W, device=data_batch[left_key].device),
        indexing='ij'
    )
    x_base = x_base.unsqueeze(0).expand(B, -1, -1)  # (B,H,W)
    y_base = y_base.unsqueeze(0).expand(B, -1, -1)  # (B,H,W)

    # Warp coordinates using disparity
    x_warped = x_base + data_batch[disp_key][0].squeeze(1)

    # Normalize coordinates to [-1,1] range for grid_sample
    x_norm = 2.0 * x_warped / (W-1) - 1.0
    y_norm = 2.0 * y_base / (H-1) - 1.0
    
    grid = torch.stack((x_norm, y_norm), dim=-1)  # (B,H,W,2)

    # Warp left image to right view
    left_warped = F.grid_sample(
        left_masked, grid,
        mode='bilinear',
        padding_mode='border',
        align_corners=True
    )

    # Calculate reconstruction loss
    mask = left_warped > 0
    data_batch[f"{domain}_left_right_difference"] = torch.abs(left_warped - data_batch[right_key]*mask)
    reconstruction_loss = F.smooth_l1_loss(left_warped, data_batch[right_key]*mask, reduction='mean')
    
    return reconstruction_loss


def compute_ssim(img1, img2, window_size=3, channel=3):
    window = torch.ones(1, channel, window_size, window_size) / (window_size**2)
    window = window.type_as(img1)

    mean1 = F.conv2d(img1, window, padding=window_size//2, groups=1)
    mean2 = F.conv2d(img2, window, padding=window_size//2, groups=1)

    mean1_sq = mean1.pow(2)
    mean2_sq = mean2.pow(2)
    mean12 = mean1 * mean2

    sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=1) - mean1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=1) - mean2_sq
    sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=1) - mean12

    C1 = 1e-5
    C2 = 1e-5

    ssim_map = ((2*mean1*mean2 + C1)*(2*sigma12 + C2)) / ((mean1_sq + mean2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
    
    return ssim_map.mean(dim=(1,2,3))