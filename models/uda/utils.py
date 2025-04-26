import torch
import torch.nn.functional as F


def refine_disparity(data_batch, threshold):
    top_one = data_batch['tgt_entropy_map_idx_t_1']
    pred_disp = data_batch['pseudo_disp'][1].unsqueeze(1) / 4.0
    mask = (top_one > 0) & (top_one < 256)

    top_one = top_one * mask 
    pred_disp = pred_disp * mask

    diff = torch.abs(top_one - pred_disp)
    diff_mask = diff <= threshold
    top_one = top_one * diff_mask
    diff_mask2 = diff > threshold
    pred_disp = pred_disp * diff_mask2

    result = top_one + pred_disp
    result = torch.clamp(result, 0, 192-1)
    # print(result.max())
    return result, diff_mask
    


def calc_entropy(data_batch, threshold, k=12, temperature=0.5, eps=1e-6):
    for model in ['s', 't']:
        for i in range(1,3):
            # 1) cost volume 가져오기
            key = 'tgt_corr_volume_' + model + '_' + str(i) 
            vol = data_batch[key].squeeze(1)       # [B, D, H, W]
            B, D, H, W = vol.shape

            # 2) temperature scaling & softmax
            temp = (0.5 if model == 's' else temperature)
            p_vol = F.softmax(vol / temp, dim=1)   # [B, D, H, W]

            # 3) top-k 확률만 골라 정규화 → topk_p
            topk_vals, _ = torch.topk(p_vol, k=k, dim=1)            # [B, k, H, W]
            sum_topk = topk_vals.sum(dim=1, keepdim=True).clamp(min=eps)   # [B,1,H,W]
            topk_p = topk_vals / sum_topk                                   # [B, k, H, W]

            H_map = -(topk_p * torch.log(topk_p)).sum(dim=1, keepdim=True)  # [B,1,H,W]
            H_map = H_map - 2.484

            # 5) threshold mask
            if isinstance(threshold, torch.Tensor) and threshold.shape[0] == B:
                thr = threshold.view(B,1,1,1)
            else:
                thr = float(threshold)
            mask_bool = (H_map < thr)           # [B,1,H,W]
            mask = mask_bool.float()

            # 6) Straight-Through Hard Top-1
            top1_idx = torch.argmax(p_vol, dim=1, keepdim=True)            # [B,1,H,W]
            hard_onehot = F.one_hot(top1_idx.squeeze(1), num_classes=D)    \
                            .permute(0,3,1,2).float()                      # [B,D,H,W]
            p_hard = hard_onehot - p_vol.detach() + p_vol                  # [B,D,H,W]
            disp_vals = torch.arange(D, device=vol.device).view(1,D,1,1).float()
            hard_disp = (p_hard * disp_vals).sum(dim=1, keepdim=True)       # [B,1,H,W]

            # 7) hard_disp에 mask 직접 적용
            refined_disp = hard_disp * mask                                 # [B,1,H,W]

            disp_vals = torch.arange(D, device=vol.device, dtype=vol.dtype) \
                            .view(1, D, 1, 1)
            disp_map = (F.softmax(vol, dim=1) * disp_vals).sum(dim=1, keepdim=True)


            # 8) 결과 저장
            data_batch['tgt_entropy_mask_'    + model + '_' + str(i)] = mask_bool
            data_batch['tgt_entropy_map_'     + model + '_' + str(i)] = H_map
            data_batch['tgt_entropy_map_idx_' + model + '_' + str(i)] = refined_disp

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