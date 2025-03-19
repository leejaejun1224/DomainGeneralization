import torch 
import torch.nn as nn
import torch.nn.functional as F


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


def calc_pseudo_soft_loss(data_batch, threshold, model='s'):
    key = 'tgt_pred_disp_' + model
    pred_disp, pseudo_disp = data_batch[key], data_batch['pseudo_disp'][1]

    mask = (pseudo_disp > 0) & (pseudo_disp < 256)
    data_batch['pseudo_mask'] = mask
    mask = mask.tolist()
    weights = [1.0]
    pseudo_soft_loss = get_loss(pred_disp, pseudo_disp, mask, weights)
    return pseudo_soft_loss


def calc_reconstruction_loss(data_batch, alpha=0.85):
    bs, _, height, width = data_batch['tgt_left'].shape
    

    x_base = torch.linspace(0, width-1, width).repeat(bs, height, 1).type_as(data_batch['tgt_left'])
    y_base = torch.linspace(0, height-1, height).repeat(bs, width, 1).transpose(1, 2).type_as(data_batch['tgt_left'])

    x_warped = x_base - data_batch['pseudo_disp'][0].squeeze(1)

    x_norm = 2.0 * x_warped / (width-1) - 1.0   
    y_norm = 2.0 * y_base / (height-1) - 1.0

    flow_field = torch.stack((x_norm, y_norm), dim=1).permute(0, 2, 3, 1)

    ### [B, C, H, W]
    img_right_reconstructed = F.grid_sample(data_batch['tgt_left'], flow_field, mode='bilinear', padding_mode='zeros', align_corners=True)

    ssim = compute_ssim(data_batch['tgt_right'], img_right_reconstructed)
    ssim_loss = (1 - ssim)/2

    l1_loss = F.l1_loss(data_batch['tgt_right'], img_right_reconstructed, reduction='none').mean(dim=(1,2,3))

    # reconstruction_loss = alpha * ssim_loss + (1 - alpha) * l1_loss
    reconstruction_loss = (1 - alpha) * l1_loss
    return reconstruction_loss.mean()


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
