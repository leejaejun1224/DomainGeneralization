import torch
import torch.nn as nn
import torch.nn.functional as F

class SemanticCostCrossAtt(nn.Module):
    """
    비용 볼륨(cost: B, Ccv, D, H4, W4)에 SegFormer stage4 세만틱 특징
    (feat_s4: B, 256, Hs, Ws)을 크로스-어텐션으로 주입.
    - Q: cost (3D 1x1x1) -> (heads*dim)
    - K,V: feat_s4 (2D 1x1) -> (heads*dim), (heads*dv)
    - 어텐션은 disparity slice 별(=D축 묶음)로 공간 토큰(H4*W4) vs 세만틱 토큰(Ns) 간에 수행
    """
    def __init__(self, cv_chan: int, feat_chan: int = 256,
                 heads: int = 2, dim_qk: int = 16, dim_v: int = 16,
                 max_tokens: int = 256, dropout: float = 0.0):
        super().__init__()
        assert (dim_qk % 1 == 0) and (dim_v % 1 == 0)
        self.cv_chan = cv_chan
        self.heads = heads
        self.dim_qk = dim_qk
        self.dim_v = dim_v
        self.max_tokens = max_tokens

        # Projections
        self.q_proj = nn.Conv3d(cv_chan, heads * dim_qk, kernel_size=1, bias=False)
        self.k_proj = nn.Conv2d(feat_chan, heads * dim_qk, kernel_size=1, bias=False)
        self.v_proj = nn.Conv2d(feat_chan, heads * dim_v,  kernel_size=1, bias=False)
        self.out_proj = nn.Conv3d(heads * dim_v, cv_chan, kernel_size=1, bias=False)

        self.norm_in  = nn.GroupNorm(1, cv_chan)          # LN 유사
        self.norm_out = nn.GroupNorm(1, cv_chan)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("scale", torch.tensor(dim_qk ** -0.5))

        # 잔차 스케일(훈련 안정)
        self.gamma = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

    def _token_pool(self, f: torch.Tensor) -> torch.Tensor:
        """세만틱 토큰 수를 max_tokens 이하로 줄이기 위해 Adaptive AvgPool 사용."""
        B, C, Hs, Ws = f.shape
        Ns = Hs * Ws
        if Ns <= self.max_tokens:
            return f
        ratio = (Ns / float(self.max_tokens)) ** 0.5
        out_h = max(1, int(round(Hs / ratio)))
        out_w = max(1, int(round(Ws / ratio)))
        return F.adaptive_avg_pool2d(f, (out_h, out_w))

    def forward(self, cost: torch.Tensor, feat_s4: torch.Tensor) -> torch.Tensor:

        B, Ccv, D, H, W = cost.shape
        cost_in = self.norm_in(cost)

        # Q from cost (per disparity slice)
        q = self.q_proj(cost_in)          # (B, h*dk, D, H, W)
        q = q.view(B, self.heads, self.dim_qk, D, H, W)

        # K,V from semantic feature (pooled)
        feat_s4 = self._token_pool(feat_s4)   # (B, 256, Hs', Ws')
        k = self.k_proj(feat_s4)              # (B, h*dk, Hs', Ws')
        v = self.v_proj(feat_s4)              # (B, h*dv, Hs', Ws')

        Bk, Hk, Wk = k.shape[0], k.shape[2], k.shape[3]
        Ns = Hk * Wk

        k = k.view(Bk, self.heads, self.dim_qk, Ns)   # (B, h, dk, Ns)
        v = v.view(Bk, self.heads, self.dim_v,  Ns)   # (B, h, dv, Ns)

        # disparity slice 별로 attention: (B*D*h, HW, dk) x (B*D*h, dk, Ns)
        q_flat = q.permute(0, 3, 1, 4, 5, 2).contiguous()  # (B, D, h, H, W, dk)
        q_flat = q_flat.view(B * D * self.heads, H * W, self.dim_qk)

        k_flat = k.permute(0, 1, 2, 3).contiguous()        # (B, h, dk, Ns)
        k_flat = k_flat.view(B * self.heads, self.dim_qk, Ns)
        k_flat = k_flat.repeat_interleave(D, dim=0)        # (B*D*h, dk, Ns)

        v_flat = v.permute(0, 1, 3, 2).contiguous()        # (B, h, Ns, dv)
        v_flat = v_flat.view(B * self.heads, Ns, self.dim_v)
        v_flat = v_flat.repeat_interleave(D, dim=0)        # (B*D*h, Ns, dv)

        attn_logits = torch.bmm(q_flat, k_flat) * self.scale  # (B*D*h, HW, Ns)
        attn = F.softmax(attn_logits, dim=-1)
        attn = self.dropout(attn)

        out = torch.bmm(attn, v_flat)                      # (B*D*h, HW, dv)
        out = out.view(B, D, self.heads, H, W, self.dim_v).permute(0, 2, 5, 1, 3, 4)
        # (B, h, dv, D, H, W) -> (B, h*dv, D, H, W)
        out = out.contiguous().view(B, self.heads * self.dim_v, D, H, W)

        out = self.out_proj(out)                           # (B, Ccv, D, H, W)
        out = self.norm_out(cost + self.gamma * out)       # residual
        return out
