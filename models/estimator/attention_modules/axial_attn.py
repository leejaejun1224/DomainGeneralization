import torch
import torch.nn as nn
import torch.nn.functional as F

# ==================================
# NEW: AxialRowColAttention3D
# ==================================
class AxialRowColAttention3D(nn.Module):
    """
    3D cost volume (B, C, D, H, W)에 대해 Row(H-축), Col(W-축) 1D 어텐션을 각각 수행하고 합산.
    - Q,K,V: 1x1x1 3D conv로 heads*dim 임베딩
    - Row:  (B*h*D*W, H, dim) self-attn
    - Col:  (B*h*D*H, W, dim) self-attn
    """
    def __init__(self, channels: int, heads: int = 2, dim: int = 16, dropout: float = 0.0):
        super().__init__()
        self.channels = channels
        self.heads = heads
        self.dim = dim

        self.q_proj = nn.Conv3d(channels, heads * dim, kernel_size=1, bias=False)
        self.k_proj = nn.Conv3d(channels, heads * dim, kernel_size=1, bias=False)
        self.v_proj = nn.Conv3d(channels, heads * dim, kernel_size=1, bias=False)
        self.out_proj = nn.Conv3d(heads * dim, channels, kernel_size=1, bias=False)

        self.norm_in  = nn.GroupNorm(1, channels)
        self.norm_out = nn.GroupNorm(1, channels)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("scale", torch.tensor(dim ** -0.5))
        self.gamma = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

    def _attn_1d(self, q, k, v):
        # q: (Bxh, Lq, dim), k: (Bxh, Lk, dim), v: (Bxh, Lk, dim)
        attn = torch.bmm(q, k.transpose(1, 2)) * self.scale  # (Bxh, Lq, Lk)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = torch.bmm(attn, v)                              # (Bxh, Lq, dim)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, D, H, W)
        """
        B, C, D, H, W = x.shape
        xin = self.norm_in(x)

        q = self.q_proj(xin).view(B, self.heads, self.dim, D, H, W)
        k = self.k_proj(xin).view(B, self.heads, self.dim, D, H, W)
        v = self.v_proj(xin).view(B, self.heads, self.dim, D, H, W)

        # Row attention (H-axis)
        q_row = q.permute(0,1,3,5,4,2).contiguous().view(B*self.heads*D*W, H, self.dim)  # (B*h*D*W, H, dim)
        k_row = k.permute(0,1,3,5,4,2).contiguous().view(B*self.heads*D*W, H, self.dim)
        v_row = v.permute(0,1,3,5,4,2).contiguous().view(B*self.heads*D*W, H, self.dim)
        out_row = self._attn_1d(q_row, k_row, v_row).view(B, self.heads, D, W, H, self.dim)
        out_row = out_row.permute(0,1,5,2,4,3).contiguous()  # (B,h,dim,D,H,W)

        # Col attention (W-axis)
        q_col = q.permute(0,1,3,4,5,2).contiguous().view(B*self.heads*D*H, W, self.dim)  # (B*h*D*H, W, dim)
        k_col = k.permute(0,1,3,4,5,2).contiguous().view(B*self.heads*D*H, W, self.dim)
        v_col = v.permute(0,1,3,4,5,2).contiguous().view(B*self.heads*D*H, W, self.dim)
        out_col = self._attn_1d(q_col, k_col, v_col).view(B, self.heads, D, H, W, self.dim)
        out_col = out_col.permute(0,1,5,2,3,4).contiguous()  # (B,h,dim,D,H,W)

        out = out_row + out_col                               # (B,h,dim,D,H,W)
        out = out.view(B, self.heads * self.dim, D, H, W)
        out = self.out_proj(out)                              # (B,C,D,H,W)
        out = self.norm_out(x + self.gamma * out)             # residual
        return out
