import torch
import torch.nn.functional as F


def refine_disparity(data_batch, mask=None, threshold=3.0):
    top_one = data_batch['tgt_entropy_map_idx_t_2']*4.0
    top_one = F.interpolate(top_one, scale_factor=4,mode='nearest')
    pred_disp = data_batch['pseudo_disp'][0].unsqueeze(1)
    
    if mask is None:
        mask = (pred_disp > 0) & (pred_disp < 192)
    top_one = top_one * mask 
    pred_disp = pred_disp * mask

    diff = torch.abs(top_one - pred_disp)
    diff_mask = diff <= threshold
    # top_one = pred_disp * diff_mask

    diff_mask2 = diff > threshold
    pred_disp = pred_disp * diff_mask

    # result = top_one + pred_disp
    result = pred_disp
    result = torch.clamp(result, 0, 192-1)
    # print(result.max())
    return result, diff_mask



def calc_directional_loss(data_batch):
    
    C, H, W = data_batch['tgt_confidence_map_s'].shape
    mask = torch.zeros(data_batch['tgt_confidence_map_s'].shape, device=data_batch['tgt_confidence_map_s'].device)
    mask[ : , H//2 : H , : ] = 1
    
    sign_diff = data_batch['tgt_confidence_map_s']
    target_diff = torch.sign(sign_diff)
    
    masked_loss = torch.abs(sign_diff - target_diff) * mask
    
    directional_loss = masked_loss.sum() / mask.sum()
    
    return directional_loss




def calc_confidence_entropy(data_batch,threshold, k=12, temperature=0.2):
    # Get the cost volume from the student model
    last_confidence_map = data_batch['tgt_mask_pred_t']
    pred = data_batch['pseudo_disp'][1].unsqueeze(1)  # [B, 1, H, W]
    
    # Sort values in descending order and get top-k indices
    _, ind = last_confidence_map.sort(1, True)
    pool_ind = ind[:, :k]
    
    # Gather top-k values from the cost volume
    cost_topk = torch.gather(last_confidence_map, 1, pool_ind)
    
    # Apply temperature scaling
    cost_topk = cost_topk / temperature
    
    # Convert to probability distribution using softmax
    prob = F.softmax(cost_topk, 1)
    
    # Calculate entropy: -sum(p * log(p))
    # Add small epsilon to avoid log(0)
    entropy = -torch.sum(prob * torch.log(prob + 1e-10), dim=1)
    
    # Store the entropy map in the data batch
    data_batch['confidence_entropy_map_s'] = entropy
    mask = entropy < threshold
    data_batch['tgt_confidence_entropy_disp_t'] = pred * mask.float()
    return data_batch


def calc_entropy(data_batch, threshold, k=12, temperature=0.5, eps=1e-6):
    for data in ['src', 'tgt']:
        for model in ['s', 't']:
            for i in range(1, 3):
                # ─────────────────────────────────────────────
                # 1) cost volume 가져오기
                # ─────────────────────────────────────────────
                key = f"{data}_corr_volume_{model}_{i}"
                if key not in data_batch:
                    continue
                vol = data_batch[key].squeeze(1)      # [B, D, H, W]
                B, D, H, W = vol.shape

                # ─────────────────────────────────────────────
                # 2) temperature scaling & softmax
                # ─────────────────────────────────────────────
                temp = 0.5 if model == 's' else temperature
                p_vol = F.softmax(vol / temp, dim=1)  # [B, D, H, W]

                # ─────────────────────────────────────────────
                # 3) top‑k 확률 정규화 후 entropy map
                # ─────────────────────────────────────────────
                topk_vals, _ = torch.topk(p_vol, k=k, dim=1)       # [B,k,H,W]
                sum_topk = topk_vals.sum(dim=1, keepdim=True).clamp(min=eps)
                topk_p = topk_vals / sum_topk
                H_map = -(topk_p * torch.log(topk_p)).sum(dim=1, keepdim=True)  # [B,1,H,W]

                # ─────────────────────────────────────────────
                # 4) threshold mask
                # ─────────────────────────────────────────────
                thr = threshold.view(B,1,1,1) if torch.is_tensor(threshold) and threshold.shape[0]==B else float(threshold)
                mask_bool = (H_map < thr)            # [B,1,H,W]
                mask = mask_bool.float()

                # ─────────────────────────────────────────────
                # 5) Straight‑Through Hard Top‑1 + mask
                # ─────────────────────────────────────────────
                top1_idx = torch.argmax(p_vol, dim=1, keepdim=True)     # [B,1,H,W]
                hard_onehot = F.one_hot(top1_idx.squeeze(1), num_classes=D) \
                                .permute(0,3,1,2).float()              # [B,D,H,W]
                p_hard = hard_onehot - p_vol.detach() + p_vol
                disp_vals = torch.arange(D, device=vol.device).view(1,D,1,1).float()
                hard_disp = (p_hard * disp_vals).sum(dim=1, keepdim=True)  # [B,1,H,W]
                refined_disp = hard_disp * mask                             # [B,1,H,W]

                # ─────────────────────────────────────────────
                # ★ 6) 추가 블록 ★  (i == 1일 때만) - 누적 평균 업데이트
                # ─────────────────────────────────────────────


                data_batch[f"{data}_entropy_mask_{model}_{i}"] = mask_bool
                data_batch[f"{data}_entropy_map_{model}_{i}"]  = H_map
                data_batch[f"{data}_entropy_map_idx_{model}_{i}"] = refined_disp

    return data_batch





def replace_above_threshold_with_local_mean(disparity: torch.Tensor,
                                              kernel_size: int = 7,
                                              threshold_value: float = 40.0) -> torch.Tensor:

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