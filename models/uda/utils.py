import torch
import torch.nn.functional as F




def calc_entropy(data_batch, threshold, k=12, temperature=0.5, eps=1e-6):
    for model in ['s', 't']:
        if model == 's':
            temperature = 1.0
        key = 'tgt_corr_volume_' + model
        vol = data_batch[key].squeeze(1) # [B, D, H, W]
        B = vol.shape[0]
        topk_vals, _ = torch.topk(vol, k=k, dim=1)
        top_one, top_one_idx = torch.topk(vol, k=1, dim=1)
        # Calculate the max value of the top one pixels
        top_one_max = top_one.max().item()
        top_one_min = top_one.min().item()
        top_one_mean = top_one.mean().item()
        

        scaled_topk_vals = topk_vals / temperature

        exp_vals = torch.exp(scaled_topk_vals)
        sum_exp = exp_vals.sum(dim=1, keepdim=True) + eps

        p = exp_vals / sum_exp
        p = torch.clamp(p, eps, 1.0)

        # H = -(p * p.log()).sum(dim=1) - 2.484
        H = -(p * p.log()).sum(dim=1)
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
        
        # Count the number of true pixels in mask
        num_true_pixels = mask.sum().item()

        disparity_cleaned = replace_above_threshold_with_local_mean(top_one_idx)

        data_batch['tgt_entropy_mask_' + model] = mask
        data_batch['tgt_entropy_map_' + model] = H
        data_batch['tgt_entropy_map_idx_' + model] = disparity_cleaned



def replace_above_threshold_with_local_mean(disparity: torch.Tensor,
                                              kernel_size: int = 7,
                                              threshold_value: float = 35.0) -> torch.Tensor:

    if disparity.dim() != 4 or disparity.size(1) != 1:
        raise ValueError("disparity 텐서는 (B, 1, H, W) 형태여야 합니다.")
    
    B, C, H, W = disparity.shape
    orig_dtype = disparity.dtype
    # 계산을 위해 float32 타입으로 변환
    disparity_f = disparity.to(torch.float32)
    
    # Check which values are above threshold
    condition = disparity_f > threshold_value
    
    # Count the number of pixels above threshold
    num_above_threshold = condition.sum().item()
    
    # Count per batch
    batch_counts = condition.view(B, -1).sum(dim=1)  # Shape: [B]
    
    # Simply set values above threshold to zero
    disparity_cleaned = torch.where(condition, torch.zeros_like(disparity_f), disparity_f)
    
    return disparity_cleaned.to(orig_dtype)