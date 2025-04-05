import torch
import torch.nn.functional as F

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
    
    disparity_cleaned = replace_above_threshold_with_local_mean(top_one_idx * 4)
    
    data_batch['tgt_entropy_map_' + model] = H
    data_batch['tgt_entropy_map_idx_' + model] = disparity_cleaned

def replace_above_threshold_with_local_mean(disparity: torch.Tensor,
                                              kernel_size: int = 7,
                                              threshold_value: float = 160.0) -> torch.Tensor:
    """
    (B, 1, H, W) 형태의 disparity 텐서에서,
    각 픽셀의 7×7 커널 내(0인 값은 제외)의 평균을 계산한 후,
    해당 픽셀의 값이 threshold_value (기본 160)를 넘으면 그 픽셀을
    커널의 평균값으로 대체한 새로운 텐서를 반환합니다.
    
    Args:
        disparity (torch.Tensor): (B, 1, H, W) 형태의 입력 텐서.
        kernel_size (int): 국소 영역 크기 (기본값 7, 즉 7x7 영역).
        threshold_value (float): 픽셀 값이 이 값을 초과하면 주변 평균으로 대체.
        
    Returns:
        torch.Tensor: 이상치가 대체된 새로운 disparity 텐서 (disparity_cleaned).
    """
    if disparity.dim() != 4 or disparity.size(1) != 1:
        raise ValueError("disparity 텐서는 (B, 1, H, W) 형태여야 합니다.")
    
    B, C, H, W = disparity.shape
    orig_dtype = disparity.dtype
    # 계산을 위해 float32 타입으로 변환
    disparity_f = disparity.to(torch.float32)
    
    # 각 픽셀의 7x7 영역 패치를 추출 (unfold)
    patches = F.unfold(disparity_f, kernel_size=kernel_size, padding=kernel_size // 2)
    # patches shape: (B, 49, H*W)
    patches_reshaped = patches.view(B, C, kernel_size * kernel_size, H * W)  # (B, 1, 49, H*W)
    
    # 커널 내에서 0인(유효하지 않은) 픽셀은 제외하기 위한 마스크
    valid_mask = patches_reshaped != 0  # bool tensor, shape: (B, 1, 49, H*W)
    
    # 유효한 값들의 개수 (0인 값은 제외)
    count_valid = valid_mask.sum(dim=2, keepdim=True).float()
    
    # 유효한 값들의 합과 제곱합 계산 (여기서는 단순 합만 사용)
    sum_valid = (patches_reshaped * valid_mask.float()).sum(dim=2, keepdim=True)
    
    # 각 커널의 유효한 값에 대한 평균 계산
    local_mean = sum_valid / (count_valid + 1e-6)  # (B, 1, 1, H*W)
    
    # (B, 1, H*W) → (B, 1, H, W)로 재구성
    local_mean_map = local_mean.view(B, C, H, W)
    
    # 조건: 원래 픽셀 값이 threshold_value (예: 160)를 넘으면 true
    condition = disparity_f > threshold_value
    
    # 조건에 맞는 픽셀은 local_mean_map의 값으로 대체, 나머지는 원래 값을 유지
    disparity_cleaned = torch.where(condition, local_mean_map, disparity_f)
    
    return disparity_cleaned.to(orig_dtype)

