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
            # teacher_entropy: low →1, high→0
            mask_loss = F.binary_cross_entropy(mask_pred_L, teacher_entropy.float())

        # Binarize for attention (inference / forward)

        ## 여기서 true인 부분은 내가 refine을 해야하는 부분
        ### output이 sigmoid 결과임. 고로 1에 가까울수록 신뢰픽셀이라는 뜻임.
        mask_bin_L = mask_pred_L > 0.5
        mask_bin_R = mask_pred_R > 0.5
        # 2. Hi‑Lo attention refinement
        featL_ref = self.propagation(featL, mask_bin_L)
        featR_ref = self.propagation(featR, mask_bin_R)

        mask_pred_ref_L = self.mask_head(featL_ref)
        if teacher_entropy is not None:
            mask_loss += F.binary_cross_entropy(mask_pred_ref_L, teacher_entropy.float())
        # 3. cost volume
        # cost = self.build_cost(featL_ref, featR_ref).unsqueeze(1)
        return featL_ref,featR_ref, mask_pred_L, mask_pred_R, mask_loss



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
        self.gate = nn.Parameter(torch.zeros(1,self.C,1,1))



    def forward(self, feat: torch.Tensor, mask_high: torch.Tensor):
        B, C, H, W = feat.shape
        L = H * W

        # ─── Step 1: Q/K/V 준비 ─────────────────────────────────────────
        # to_qkv → (B, 3, h, Hd, H, W)
        qkv = self.to_qkv(feat).view(B, 3, self.h, self.Hd, H, W)
        q, k, v = qkv[:,0], qkv[:,1], qkv[:,2]          # 각각 (B, h, Hd, H, W)
        k_flat = k.flatten(1, 2)                        # (B, h*Hd, H, W)
        v_flat = v.flatten(1, 2)                        # (B, h*Hd, H, W)

        # ─── Step 2: K/V를 “신뢰 픽셀”만 뽑아내도록 gating ─────────────
        # mask_high: 1=신뢰, 0=불신 → unfold → (B, win², H*W)
        w_unf = self.unfold(mask_high.float())                   # (B, win², L)
        w_unf = w_unf.view(B, 1, 1, self.win*self.win, L)        # (B,1,1,win²,L)

        # k/v unfold → (B, h, Hd, win², L)
        k_unf = self.unfold(k_flat).view(B, self.h, self.Hd, self.win*self.win, L)
        v_unf = self.unfold(v_flat).view(B, self.h, self.Hd, self.win*self.win, L)

        # **여기서 곱하기** → K/V 는 오직 “신뢰 픽셀” 정보만 남김
        k_unf = k_unf * w_unf      # (B, h, Hd, win², L)
        v_unf = v_unf * w_unf      # (B, h, Hd, win², L)

        # ─── Step 3: “불신 픽셀”만 쿼리로 뽑아서 attention ────────────
        q_flat = q.flatten(3)
        mask_low = (mask_high < 0.5).view(B, L)

        # 결과를 담을 텐서
        out_flat = feat.new_zeros(B, self.h*self.Hd, L)

        for b in range(B):
            low_idx = mask_low[b].nonzero(as_tuple=False).squeeze(1)
            if low_idx.numel() == 0:
                continue

            # — Q: 불신 픽셀만
            q_low = q_flat[b, :, :, low_idx]           # (h, Hd, N_low)

            # — K/V: 이미 “신뢰 픽셀”만 gating 된 k_unf/v_unf
            k_low = k_unf[b, :, :, :, low_idx]         # (h, Hd, win², N_low)
            v_low = v_unf[b, :, :, :, low_idx]         # (h, Hd, win², N_low)

            # attention score 계산
            attn = (q_low.unsqueeze(2) * k_low).sum(1) * self.scale   # (h, win², N_low)
            attn = attn + self.pos_bias[..., None]                    # positional bias 더하기
            attn = F.softmax(attn, dim=1)

            # weighted sum
            out_low = (attn.unsqueeze(1) * v_low).sum(3)              # (h, Hd, N_low)

            # scatter back: 불신 픽셀 위치에만 값 채우기
            out_flat[b, :, low_idx] = out_low.view(self.h*self.Hd, -1)

        # reshape & 1×1 conv
        out = out_flat.view(B, self.C, H, W)
        feat_ref = self.proj(out)

        # residual은 오직 불신 픽셀에만
        mask_low_map = mask_low.view(B, 1, H, W).float()
        refined = feat + feat_ref * mask_low_map

        return refined



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
    
    