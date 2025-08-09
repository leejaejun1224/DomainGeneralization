import torch
import torch.nn.functional as F
import math

def confidence_band_weight_from_logits(logits, low=0.6, high=0.95, sharp=20.0):
    # logits: [B,D,H,W]
    p = F.softmax(logits, dim=1)
    p_max = p.max(dim=1).values
    w_lo = torch.sigmoid((p_max - low)  * sharp)
    w_hi = torch.sigmoid((high - p_max) * sharp)
    return w_lo * w_hi  # [B,H,W]

def lr_cycle_consistency_mask(disp_L, disp_R, tol=1.0):
    """
    disp_L: [B,1,H,W] (L->R disparity), disp_R: [B,1,H,W] (R->L disparity)
    cycle: d_L(x) + d_R(x - d_L(x)) ≈ 0
    """
    B, _, H, W = disp_L.shape
    # sample R disparity at warped coords
    # grid x' = x - d_L(x)
    xs = torch.linspace(-1, 1, W, device=disp_L.device).view(1,1,1,W).expand(B,1,H,W)
    ys = torch.linspace(-1, 1, H, device=disp_L.device).view(1,1,H,1).expand(B,1,H,W)
    # px coords in [-1,1], disp normalized to grid needs scaling by 2/W
    grid_x = xs - 2.0 * disp_L / (W - 1)
    grid = torch.stack([grid_x.squeeze(1), ys.squeeze(1)], dim=-1)  # [B,H,W,2]
    dR_warp = F.grid_sample(disp_R, grid, mode='bilinear', padding_mode='border', align_corners=True)
    err = torch.abs(disp_L + dR_warp)  # [B,1,H,W]
    return (err < tol).float().squeeze(1)  # [B,H,W]

def variance_weight_from_pseudos(pseudo_list, sigma=2.0):
    """
    pseudo_list: list of [B,1,H,W] disparities from teacher under different augs
    returns w_cons in [0,1] using exp(-var / sigma^2)
    """
    # Stack to [N,B,1,H,W] -> [B,1,H,W,N]
    stk = torch.stack([p[:,0:1] for p in pseudo_list], dim=-1)  # [B,1,H,W,N]
    var = stk.var(dim=-1, unbiased=False)  # [B,1,H,W]
    w = torch.exp(-var / (sigma**2))
    return w.squeeze(1)  # [B,H,W]

def down_or_up(mask_hw, target_hw):
    # mask_hw: [B,H,W] float/bool, resize to target_hw (Ht,Wt)
    Ht, Wt = target_hw
    return F.interpolate(mask_hw.unsqueeze(1).float(), size=(Ht, Wt), mode='nearest').squeeze(1)

def filter_mask(data_batch):
    logits_s = data_batch['tgt_attn_weights_s'].squeeze(1)   # [B,D,Hs,Ws]
    logits_t = data_batch['tgt_attn_weights_t'].squeeze(1)   # [B,D,Ht,Wt]
    if logits_t.shape[-2:] != logits_s.shape[-2:]:
        logits_t = F.interpolate(logits_t, size=logits_s.shape[-2:], mode='bilinear', align_corners=False)

    # ==== 2) 구성 요소별 신뢰 가중치 ====
    # (a) 엔트로피 밴드
    w_ent = confidence_band_weight_from_logits(logits_t, low=0.6, high=0.95, sharp=20.0)  # [B,Hs,Ws]



    # (c) 좌우 사이클 일관성
    # 학생 순방향/역방향이 이미 있음: tgt_pred_disp_s_for_loss, tgt_pred_disp_s_reverse[0]
    dL = data_batch['tgt_pred_disp_s_for_loss'].unsqueeze(1)        # [B,1,H,W]
    dR = data_batch['tgt_pred_disp_s_reverse'][0].unsqueeze(1)      # [B,1,H,W]
    w_cycle = lr_cycle_consistency_mask(dL, dR, tol=1.0)              # [B,H,W]
    w_cycle = down_or_up(w_cycle, logits_s.shape[-2:])                # [B,Hs,Ws]

    # (d) 증강 합의(teacher 여러 번) -> 분산 작을수록 신뢰↑
    pseudo_list = [data_batch['pseudo_disp_for_loss']] + \
                [data_batch[f'pseudo_disp_random_{i+1}'][0] for i in range(9)]  # 모두 [B,1,H,W]
    w_cons = variance_weight_from_pseudos(pseudo_list, sigma=2.0)     # [B,H,W]
    w_cons = down_or_up(w_cons, logits_s.shape[-2:])                   # [B,Hs,Ws]

    # (e) 학생-교사 근접성 (trust-region)
    d_s = data_batch['tgt_pred_disp_s_for_loss']   # [B,1,H,W]
    d_t = data_batch['pseudo_disp_for_loss']       # [B,1,H,W]
    delta = torch.abs(d_s - d_t).squeeze(1)        # [B,H,W]
    w_delta = (delta < 3.0).float()                # 3px 내 근접 시만 KD
    w_delta = down_or_up(w_delta, logits_s.shape[-2:])  # [B,Hs,Ws]

    # 최종 신뢰 가중치
    trust_w = (w_ent * w_cycle * w_cons * w_delta).clamp(0,1)  # [B,Hs,Ws]
    print(trust_w)
    return trust_w.unsqueeze(1)  # [B,1,Hs,Ws] (채널 차원 추가)
