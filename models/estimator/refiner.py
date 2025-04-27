import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------------------------
# 1. Entropy Mask Predictor
# ----------------------------------------------------------
class MaskHead(nn.Module):
    def __init__(self, in_ch: int, mid_ch: int = 32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, mid_ch, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, 1, 3, padding=1, bias=False),
        )
    def forward(self, x):
        return torch.sigmoid(self.conv(x))          # [B,1,H,W]  (prob. of high‑entropy)

# ----------------------------------------------------------
# 2. High↔Low Co‑Attention with positional bias
# ----------------------------------------------------------
class HiLoAttention(nn.Module):
    def __init__(self, in_ch: int, emb: int = 64, pos_sigma: float = 0.05):
        super().__init__()
        self.q = nn.Conv2d(in_ch, emb, 1, bias=False)
        self.k = nn.Conv2d(in_ch, emb, 1, bias=False)
        self.v = nn.Conv2d(in_ch, emb, 1, bias=False)
        self.proj = nn.Linear(emb, in_ch, bias=False)
        self.scale = emb ** -0.5
        self.pos_sigma = pos_sigma

    def forward(self, feat, mask_hi):
        # feat: [B,C,H,W], mask_hi: [B,1,H,W] bool or 0/1
        B,C,H,W = feat.shape
        Q = self.q(feat)    # [B,E,H,W]
        K = self.k(feat)
        V = self.v(feat)

        # flatten
        Qf = Q.view(B, -1, H*W).permute(0,2,1)     # [B,N,E]
        Kf = K.view(B, -1, H*W).permute(0,2,1)     # [B,N,E]
        Vf = V.view(B, -1, H*W).permute(0,2,1)     # [B,N,E]

        mask_flat = mask_hi.view(B, -1)            # [B,N]

        # pre‑compute positional coordinates (normalized)
        y = torch.linspace(0, 1, H, device=feat.device)
        x = torch.linspace(0, 1, W, device=feat.device)
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        pos = torch.stack([yy, xx], dim=-1).view(1, -1, 2)  # [1,N,2]

        out_feat = feat.clone()

        for b in range(B):
            hi_idx  = mask_flat[b].nonzero(as_tuple=False).squeeze(1)
            lo_idx  = (~mask_flat[b]).nonzero(as_tuple=False).squeeze(1)
            if hi_idx.numel()==0 or lo_idx.numel()==0:
                continue

            q = Qf[b, hi_idx]                       # [Nh,E]
            k = Kf[b, lo_idx]                       # [Nl,E]
            v = Vf[b, lo_idx]                       # [Nl,E]

            # similarity + positional bias
            sim = (q @ k.T) * self.scale            # [Nh,Nl]

            pos_q = pos[:, hi_idx].squeeze(0)       # [Nh,2]
            pos_k = pos[:, lo_idx].squeeze(0)       # [Nl,2]
            d2 = torch.cdist(pos_q, pos_k, p=2).pow(2)  # [Nh,Nl]
            pos_bias = -d2 / (2*self.pos_sigma**2)

            attn = F.softmax(sim + pos_bias, dim=-1)        # [Nh,Nl]
            agg  = attn @ v                                  # [Nh,E]

            # scatter back
            delta = self.proj(agg) # [Nh,C]
            feat_flat = out_feat[b].view(C, -1)
            feat_flat[:, hi_idx] = feat_flat[:, hi_idx] + delta.T
            out_feat[b] = feat_flat.view(C, H, W)

        return out_feat

# ----------------------------------------------------------
# 3. RefineCostVolume Module
# ----------------------------------------------------------
class RefineCostVolume(nn.Module):
    def __init__(self, feat_ch: int, max_disp: int,
                 emb: int = 64, pos_sigma: float = 0.05, tau: float = 0.0):
        super().__init__()
        self.mask_head = MaskHead(feat_ch)
        # self.propagation = RelativePropagation(feat_ch, K=20)
        self.propagation = RefineNet(in_channels=32, num_heads=4, window_size=11, tau=1.0)
        self.max_disp  = max_disp//4
        self.tau = tau        # threshold for teacher mask

    def build_cost(self, fL, fR):
        B,C,H,W = fL.shape
        cost = fL.new_zeros(B, self.max_disp, H, W)
        for d in range(self.max_disp):
            if d>0:
                cost[:,d,:,d:] = (fL[:,:,:,d:] * fR[:,:,:,:-d]).mean(1)
            else:
                cost[:,d] = (fL * fR).mean(1)
        return cost

    def forward(self, featL, featR, teacher_entropy=None):
        """
        featL / featR : [B,C,H,W]
        teacher_entropy: [B,1,H,W] normed entropy map (0~1)   (optional, training only)
        returns:
          cost_refined, mask_pred, (optional) mask_loss
        """
        # 1. predict high‑entropy mask
        mask_pred_L= self.mask_head(featL)   # prob ∈ (0,1)
        mask_pred_R = self.mask_head(featR)
        # teacher mask supervision (training)
        mask_loss = None
        if teacher_entropy is not None:
            # teacher_entropy: low →0, high→1
            mask_loss = F.binary_cross_entropy(mask_pred_L, teacher_entropy.float())

        # Binarize for attention (inference / forward)
        mask_bin_L = mask_pred_L < 0.5
        mask_bin_R = mask_pred_R < 0.5
        # 2. Hi‑Lo attention refinement
        featL_ref = self.propagation(featL, mask_bin_L)
        featR_ref = self.propagation(featR, mask_bin_R)
        # 3. cost volume
        # cost = self.build_cost(featL_ref, featR_ref).unsqueeze(1)
        return featL_ref,featR_ref, mask_pred_L, mask_pred_R, mask_loss



class RelativePropagation(nn.Module):

    def __init__(self, feat_ch: int, K: int = 8):
        super().__init__()
        self.K = K
        self.offset_conv = nn.Conv2d(feat_ch, 2 * K, kernel_size=1)
        self.weight_conv = nn.Conv2d(feat_ch, K, kernel_size=1)

    def forward(self, feat: torch.Tensor, mask_high: torch.Tensor) -> torch.Tensor:
        B, C, H, W = feat.shape
        K = self.K

        # 1) predict offsets and weights
        offsets = self.offset_conv(feat)           # [B, 2K, H, W]
        weights = self.weight_conv(feat)           # [B, K, H, W]
        weights = F.softmax(weights, dim=1)        # normalize weights

        # 2) base sampling grid in normalized coords [-1,1]
        xs = torch.linspace(-1, 1, W, device=feat.device)
        ys = torch.linspace(-1, 1, H, device=feat.device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')  # [H, W]
        base_grid = torch.stack((grid_x, grid_y), dim=-1)       # [H, W, 2]
        base_grid = base_grid.unsqueeze(0).expand(B, H, W, 2)    # [B, H, W, 2]

        # 3) propagate features
        residual = torch.zeros_like(feat)
        for k in range(K):
            # extract dx, dy and normalize
            dx = offsets[:, 2*k    , :, :]  # [B, H, W]
            dy = offsets[:, 2*k + 1, :, :]  # [B, H, W]
            dx_norm = dx / ((W - 1) / 2)
            dy_norm = dy / ((H - 1) / 2)

            # vectorized offset addition
            offset_grid = torch.stack((dx_norm, dy_norm), dim=-1)  # [B, H, W, 2]
            grid_k = base_grid + offset_grid                     # [B, H, W, 2]

            # sample and accumulate
            sampled = F.grid_sample(
                feat, grid_k, mode='bilinear', align_corners=True
            )  # [B, C, H, W]
            w = weights[:, k:k+1, :, :]                            # [B, 1, H, W]
            residual = residual + sampled * w

        # 4) apply only on high-entropy positions
        refined_feat = feat + residual * mask_high.float()
        return refined_feat
    
class RefineNet(nn.Module):
    def __init__(self,
                 in_channels: int = 32,
                 num_heads: int = 2,
                 window_size: int = 11,
                 tau: float = 1.0):
        super().__init__()
        assert in_channels % num_heads == 0
        self.C    = in_channels
        self.h    = num_heads
        self.Hd   = in_channels // num_heads
        self.win  = window_size
        self.scale = self.Hd ** -0.5

        self.to_qkv   = nn.Conv2d(in_channels, in_channels * 3, 1, bias=False)
        self.proj     = nn.Conv2d(in_channels, in_channels, 1, bias=True)
        self.pos_bias = nn.Parameter(torch.zeros(num_heads, window_size * window_size))
        self.unfold   = nn.Unfold(kernel_size=window_size,
                                  padding=window_size // 2)
        self.tau      = tau

    def forward(self, feat: torch.Tensor, mask_high: torch.Tensor):
        B, C, H, W = feat.shape
        L = H * W

        # 1) Q,K,V 분리 → (B, h, Hd, H, W)
        qkv = self.to_qkv(feat).view(B, 3, self.h, self.Hd, H, W)
        q, k, v = qkv[:,0], qkv[:,1], qkv[:,2]

        # 2) confidence mask → (B,1,H,W)
        w_mask = mask_high.float()  # already {0,1}

        # 3) 채널(=h*Hd) 차원만 flat
        #    k_flat,v_flat: (B, h*Hd, H, W)
        k_flat = k.flatten(1,2)
        v_flat = v.flatten(1,2)
        w_flat = w_mask       # (B, 1, H, W)

        # 4) unfold → (B, channels*win², L)
        k_unf = self.unfold(k_flat)
        v_unf = self.unfold(v_flat)
        w_unf = self.unfold(w_flat)
        # 5) view back → (B, h, Hd, win², L) / (B,1,1,win²,L)
        k_unf = k_unf.view(B, self.h, self.Hd, self.win*self.win, L)
        v_unf = v_unf.view(B, self.h, self.Hd, self.win*self.win, L)
        w_unf = w_unf.repeat(1, self.h, 1, 1, 1)

        # 6) confidence-gating
        k_unf = k_unf * (w_unf / self.tau)
        v_unf = v_unf * (w_unf / self.tau)

        # 7) Q 준비 → (B, h, Hd, L)
        q_flat = q.flatten(3).unsqueeze(3)  # (B, h, Hd, 1, L)
        # 8) Attention score → (B, h, win², L)
        attn = (q_flat * k_unf).sum(2) * self.scale

        bias = self.pos_bias.unsqueeze(0).unsqueeze(-1)
        attn = attn + bias
        attn = F.softmax(attn, dim=2)
        # 9) Weighted sum → (B, h, Hd, L)
        out_flat = (attn.unsqueeze(2) * v_unf).sum(3)

        # 10) 최종 reshape → (1, 32, 64, 128)
        out = out_flat.view(B, self.C, H, W)

        feat_res = self.proj(out)

        return feat + feat_res
