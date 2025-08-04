# file: local_mha_refiner.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalMHARefiner(nn.Module):
    def __init__(
        self,
        attn_channels: int = 8,
        feat_channels: int = 80,
        embed_dim: int = 64,
        num_heads: int = 4,
        window_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.win = window_size
        self.emb = embed_dim
        self.heads = num_heads
        head_dim = embed_dim // num_heads
        assert embed_dim % num_heads == 0, "embed_dim % num_heads != 0"

        # Q / K / V 1×1 projection
        self.q_proj = nn.Conv2d(feat_channels, embed_dim, 1, bias=False)
        self.k_proj = nn.Conv2d(feat_channels, embed_dim, 1, bias=False)
        self.v_proj = nn.Conv2d(feat_channels, embed_dim, 1, bias=False)

        # output projection → Δattention map
        self.out_proj = nn.Sequential(
            nn.Conv2d(embed_dim, attn_channels, 1, bias=False),
            nn.Dropout(dropout),
        )

        self.scale = head_dim ** -0.5
        self.unfold = nn.Unfold(kernel_size=window_size, padding=window_size // 2)

        # 초기화
        for m in (self.q_proj, self.k_proj, self.v_proj):
            nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(self.out_proj[0].weight)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _split_heads(tensor, num_heads: int):
        # tensor: [B, N, ?, E]
        B, N, *extra, E = tensor.shape
        head_dim = E // num_heads
        tensor = tensor.view(B, N, *extra, num_heads, head_dim)
        # → [B, num_heads, N, ?, head_dim]
        permute_order = (0, -2, 1) + tuple(range(2, 2 + len(extra))) + (-1,)
        return tensor.permute(*permute_order).contiguous()

    # ------------------------------------------------------------------ #
    def forward(
        self,
        attn_vol: torch.Tensor,   # [B, C_att, D, H_att, W_att]
        feat_map: torch.Tensor,   # [B, C_feat, H_feat, W_feat]
        disp_gap: torch.Tensor,   # [B, H_feat, W_feat]   (≥2 → mask)
    ) -> torch.Tensor:
        B, C_att, D, H_att, W_att = attn_vol.shape
        _, _, H_feat, W_feat = feat_map.shape
        N_feat = H_feat * W_feat                      # 토큰 수

        # ① Q / K / V
        Q = self.q_proj(feat_map)                     # [B, E, H_feat, W_feat]
        K = self.k_proj(feat_map)
        V = self.v_proj(feat_map)

        # ② Unfold K,V : local window
        # K_unfold : [B, E*win², H_feat*W_feat]
        K_unfold = self.unfold(K).transpose(1, 2)
        V_unfold = self.unfold(V).transpose(1, 2)

        # reshape → [B, N_feat, win², E]
        win2 = self.win * self.win
        K_unfold = K_unfold.view(B, N_feat, win2, self.emb)
        V_unfold = V_unfold.view(B, N_feat, win2, self.emb)

        # ③ Split heads
        Q_h = self._split_heads(Q.flatten(2).transpose(1, 2), self.heads)        # [B,h,N_feat,head]
        K_h = self._split_heads(K_unfold, self.heads)                             # [B,h,N_feat,win²,head]
        V_h = self._split_heads(V_unfold, self.heads)

        # ④ Attention
        attn_scores = (Q_h.unsqueeze(3) * K_h).sum(-1) * self.scale              # [B,h,N_feat,win²]
        attn_probs  = F.softmax(attn_scores, dim=-1)

        context = (attn_probs.unsqueeze(-1) * V_h).sum(3)                        # [B,h,N_feat,head]
        # 병합 heads → [B, N_feat, E]
        context = context.permute(0, 2, 1, 3).contiguous()           # [B, N, h, head_dim]
        B, N_tok, h, d = context.shape                               # N_tok = H_feat * W_feat
        context = context.view(B, N_tok, h * d) 
        # ⑤ Δattention 2-D map
        delta_2d = context.transpose(1, 2).view(B, self.emb, H_feat, W_feat)     # [B,E,H_feat,W_feat]
        delta_2d = self.out_proj(delta_2d)                                       # [B,C_att,H_feat,W_feat]

        # ⑥ 해상도 차이 보정
        if (H_feat, W_feat) != (H_att, W_att):
            delta_2d = F.interpolate(delta_2d, size=(H_att, W_att),
                                     mode="bilinear", align_corners=False)

        # ⑦ Disparity dimension broadcast
        delta_attn = delta_2d.unsqueeze(2).expand(-1, -1, D, -1, -1)             # [B,C_att,D,H_att,W_att]

        # ⑧ Mask: disp_gap ≥ 2 → Δ=0
        mask = (disp_gap >= 2).view(B, 1, 1, H_feat, W_feat)
        if (H_feat, W_feat) != (H_att, W_att):
            mask = F.interpolate(mask.float(), size=(H_att, W_att), mode="nearest").bool()
        delta_attn = delta_attn.masked_fill(mask, 0.0)

        # ⑨ Residual add
        refined_attn = attn_vol + delta_attn
        return refined_attn