import torch 
import torch.nn as nn
import torch.nn.functional as F


def get_loss(disp_ests, disp_gts, img_masks, weights):
    all_losses = []

    for disp_est, disp_gt, img_mask, weight in zip(disp_ests, disp_gts, img_masks, weights):
        
        all_losses.append(weight * F.smooth_l1_loss(disp_est[img_mask], disp_gt[img_mask], size_average=True))
    return sum(all_losses)


def calc_supervised_train_loss(data_batch):
    pred_disp, gt_disp, gt_disp_low = data_batch['src_pred_disp'], data_batch['src_disparity'], data_batch['src_disparity_low']
    mask = (gt_disp > 0) & (gt_disp < 192)
    data_batch['mask'] = mask
    mask_low = (gt_disp_low > 0) & (gt_disp_low < 192)
    masks = [mask, mask_low, mask, mask_low]
    gt_disps = [gt_disp, gt_disp_low, gt_disp, gt_disp_low]
    # scale별 weight 예시
    weights = [1.0, 0.3, 0.5, 0.3]
    loss = get_loss(pred_disp, gt_disps, masks, weights)
    return loss


def calc_supervised_val_loss(data_batch):
    pred_disp, gt_disp = data_batch['src_pred_disp'], data_batch['src_disparity']
    mask = (gt_disp > 0) & (gt_disp < 192)
    data_batch['mask'] = mask
    masks = [mask]
    gt_disps = [gt_disp]
    # scale별 weight 예시
    weights = [1.0]
    loss = get_loss(pred_disp, gt_disps, masks, weights)
    return loss


def calc_pseudo_loss(data_batch, cfg):
    threshold = cfg['uda']['threshold']
    pred_disp, pseudo_disp, confidence_map = data_batch['tgt_pred_disp'], data_batch['pseudo_disp'], data_batch['confidence_map']
    pred_disp = pred_disp[1]
    mask = (pseudo_disp[0] > 0) & (pseudo_disp[0] < 192) & (confidence_map.float() >= threshold)
    data_batch['pseudo_mask'] = mask
    mask = mask.tolist()
    weights = [1.0]
    confidence_mask = confidence_map.float() >= threshold
    true_count = confidence_mask.sum(dim=(0,1,2)) 
    total_pixels = confidence_mask.numel() // confidence_mask.shape[0]
    true_ratio = true_count.float() / total_pixels

    pseudo_label_loss = get_loss(pred_disp, pseudo_disp, mask, weights)
    return pseudo_label_loss * true_ratio

