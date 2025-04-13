import torch
import torch.nn.functional as F

def calc_entropy(data_batch, threshold=0.00089, k=12, temperature=0.5, eps=1e-6):
    for model in ['s', 't']:
        if model == 's':
            temperature = 1.0
        key = 'tgt_corr_volume_' + model
        vol = data_batch[key].squeeze(1) # [B, D, H, W]
        B = vol.shape[0]
        topk_vals, _ = torch.topk(vol, k=k, dim=1)
        top_one, top_one_idx = torch.topk(vol, k=1, dim=1)
        scaled_topk_vals = topk_vals / temperature

        exp_vals = torch.exp(scaled_topk_vals)
        sum_exp = exp_vals.sum(dim=1, keepdim=True) + eps

        p = exp_vals / sum_exp
        p = torch.clamp(p, eps, 1.0)

        H = -(p * p.log()).sum(dim=1) - 2.484
        H = H.unsqueeze(1)
        
        # Handle batch-specific thresholds
        if isinstance(threshold, torch.Tensor) and threshold.shape[0] == B:
            # Create per-batch threshold masks
            batch_masks = []
            for i in range(B):
                batch_mask = H[i:i+1] < threshold[i]
                batch_masks.append(batch_mask)
            mask = torch.cat(batch_masks, dim=0).cuda()
        else:
            mask = H < threshold
        
        # H = H * mask
        top_one_idx = top_one_idx * mask
        
        disparity_cleaned = replace_above_threshold_with_local_mean(top_one_idx)

        data_batch['tgt_entropy_mask_' + model] = mask
        data_batch['tgt_entropy_map_' + model] = H
        data_batch['tgt_entropy_map_idx_' + model] = disparity_cleaned



def replace_above_threshold_with_local_mean(disparity: torch.Tensor,
                                              kernel_size: int = 7,
                                              threshold_value: float = 40.0) -> torch.Tensor:

    if disparity.dim() != 4 or disparity.size(1) != 1:
        raise ValueError("disparity 텐서는 (B, 1, H, W) 형태여야 합니다.")
    
    B, C, H, W = disparity.shape
    orig_dtype = disparity.dtype
    # 계산을 위해 float32 타입으로 변환
    disparity_f = disparity.to(torch.float32)
    patches = F.unfold(disparity_f, kernel_size=kernel_size, padding=kernel_size // 2)
    patches_reshaped = patches.view(B, C, kernel_size * kernel_size, H * W)
    valid_mask = patches_reshaped != 0  # bool tensor, shape: (B, 1, 49, H*W)
    count_valid = valid_mask.sum(dim=2, keepdim=True).float()
    sum_valid = (patches_reshaped * valid_mask.float()).sum(dim=2, keepdim=True)
    local_mean = sum_valid / (count_valid + 1e-6)  # (B, 1, 1, H*W)
    local_mean_map = local_mean.view(B, C, H, W)
    condition = disparity_f > threshold_value
    disparity_cleaned = torch.where(condition, local_mean_map, disparity_f)
    
    return disparity_cleaned.to(orig_dtype)

