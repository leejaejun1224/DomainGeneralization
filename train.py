import torch
import torch.nn as nn
from model.loss import get_loss



def calc_supervised_loss(pred_disp, gt_disp, gt_disp_low):
    mask = (gt_disp > 0) & (gt_disp < 192)
    mask_low = (gt_disp_low > 0) & (gt_disp_low < 192)
    masks = [mask, mask_low, mask, mask_low]
    gt_disps = [gt_disp, gt_disp_low, gt_disp, gt_disp_low]
    # scale별 weight 예시
    weights = [1.0, 0.3, 0.5, 0.3]
    loss = get_loss(pred_disp, gt_disps, masks, weights)
    print("supervised loss : ", loss)
    return loss

def calc_pseudo_loss(pred_disp, pseudo_disp, confidence_map, threshold):
    pred_disp = pred_disp[1]
    mask = (pseudo_disp[0] > 0) & (pseudo_disp[0] < 192) & (confidence_map.float() >= threshold)
    mask = mask.tolist()
    weights = [1.0]

    confidence_mask = confidence_map.float() >= threshold
    true_count = confidence_mask.sum(dim=(0,1,2)) 
    total_pixels = confidence_mask.numel() // confidence_mask.shape[0]
    true_ratio = true_count.float() / total_pixels
    print("true_ratio : ", true_ratio)
    pseudo_label_loss = get_loss(pred_disp, pseudo_disp, mask, weights)
    print("pseudo_label_loss : ", pseudo_label_loss)
    return pseudo_label_loss*true_ratio


def compute_uda_loss(model, data_batch, cfg):
    src_left = data_batch['src_left']
    src_right = data_batch['src_right']
    src_disp_gt = data_batch['src_disparity']
    src_disp_gt_low = data_batch['src_disparity_low']

    tgt_left = data_batch['tgt_left']
    tgt_right = data_batch['tgt_right']

    threshold = cfg['uda']['threshold']

    # print(model.__class__.__name__)
    src_pred = model(src_left, src_right)
    supervised_loss = calc_supervised_loss(src_pred, src_disp_gt, src_disp_gt_low)

    with torch.no_grad():
        pseudo_disp, confidence_map = model.ema_forward(
            tgt_left, tgt_right, return_confidence=True
        )

    tgt_pred = model(tgt_left, tgt_right)

    pseudo_loss = calc_pseudo_loss(tgt_pred, pseudo_disp, confidence_map, threshold)
    total_loss = supervised_loss + pseudo_loss
    # 로그용
    log_vars = {
        'loss': total_loss.item(),
        'supervised_loss': supervised_loss.item(),
        'pseudo_loss': pseudo_loss.item()
    }
    return total_loss, log_vars

    
