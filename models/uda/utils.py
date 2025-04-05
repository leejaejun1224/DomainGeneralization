import torch


def calc_entropy(data_batch, threshold=0.00089, model='t', k=12, temperature=0.5, eps=1e-6):
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
        # Use single threshold for all batches
        mask = H < threshold
        
    H = H * mask
    top_one_idx = top_one_idx * mask
    # top_one_idx = top_one_idx.unsqueeze(1)
    
    data_batch['tgt_entropy_map_' + model] = H
    data_batch['tgt_entropy_map_idx_' + model] = top_one_idx * 4