import math
import torch
import torch.nn.functional as F

def entropy_from_logits(logits, tau=2.0, normalize=True, eps=1e-8):
    """
    logits: [B, D, H, W]  (disparity 축 = dim=1)
    tau>1: 분포 평탄화 → 엔트로피/기울기 회복
    """
    p_tau = F.softmax(logits / tau, dim=1)
    H = -(p_tau * torch.log(p_tau.clamp_min(eps))).sum(dim=1)  # [B,H,W]
    if normalize:
        H = H / math.log(logits.size(1))  # [0,1] 정규화
    return H, p_tau

def entropy_sharpen_loss(logits, mask_conf, tau=3.0, lam=1.0):
    """
    mask_conf: [B,H,W] 또는 [B,1,H,W] (0~1 소프트 가중치)
    tau 보상: 최종 손실에 tau 곱해 gradient 약화 상쇄
    """
    H, _ = entropy_from_logits(logits, tau=tau, normalize=True)
    if mask_conf.dim() == 4:
        mask_conf = mask_conf.squeeze(1)
    denom = mask_conf.sum().clamp_min(1.0)
    loss = (H * mask_conf).sum() / denom
    return lam * tau * loss

def top2_margin_loss(logits, margin=2.0, weight=None):
    """
    'z_top1 - z_top2 >= margin'을 유도하는 soft hinge
    logits: [B,D,H,W]
    """
    top2 = torch.topk(logits, k=2, dim=1)
    z1, z2 = top2.values[:, 0], top2.values[:, 1]  # [B,H,W]
    per_px = F.softplus(margin - (z1 - z2))        # [B,H,W]
    if weight is not None:
        if weight.dim() == 4:
            weight = weight.squeeze(1)
        denom = weight.sum().clamp_min(1.0)
        return (per_px * weight).sum() / denom
    return per_px.mean()

def kd_consistency_loss(logits_student, logits_teacher, T_t=4.0, T_s=1.0, weight=None):
    """
    Teacher→Student KL (teacher는 stopgrad 외부에서 보장)
    logits_*: [B,D,H,W]
    """
    with torch.no_grad():
        q = F.softmax(logits_teacher / T_t, dim=1)
    log_p = F.log_softmax(logits_student / T_s, dim=1)
    per_px = F.kl_div(log_p, q, reduction='none').sum(dim=1)  # [B,H,W]
    if weight is not None:
        if weight.dim() == 4:
            weight = weight.squeeze(1)
        denom = weight.sum().clamp_min(1.0)
        loss = (per_px * weight).sum() / denom
    else:
        loss = per_px.mean()
    return (T_t ** 2) * loss  # KD 스케일링

def confidence_band_weight(p, low=0.6, high=0.95, sharp=20.0):
    """
    p: [B,D,H,W]의 확률분포 → p_max로 게이팅
    밴드 [low,high]에서 가중치↑, 그 외↓
    """
    p_max = p.max(dim=1).values  # [B,H,W]
    w_lo = torch.sigmoid((p_max - low) * sharp)
    w_hi = torch.sigmoid((high - p_max) * sharp)
    return w_lo * w_hi  # [B,H,W]
