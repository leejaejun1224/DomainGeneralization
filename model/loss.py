import torch 
import torch.nn as nn
import torch.nn.functional as F

def get_loss(disp_ests, disp_gts, img_masks, weights):
    all_losses = []

    for disp_est, disp_gt, img_mask, weight in zip(disp_ests, disp_gts, img_masks, weights):
        
        all_losses.append(weight * F.smooth_l1_loss(disp_est[img_mask], disp_gt[img_mask], size_average=True))
    return sum(all_losses)