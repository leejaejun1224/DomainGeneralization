import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================
# NEW (memory-lite): SemanticCostCrossAttLite
# =========================
class SemanticCostCrossAttLite(nn.Module):
    """
    비용 볼륨(cost: B, Ccv, D, H, W)에 SegFormer stage4 세만틱 특징
    (feat_s4: B, 256, Hs, Ws)을 크로스-어텐션으로 주입.
    - 공간 다운샘플(쿼리) + 쿼리 축 분할 계산으로 메모리 절감
    - D축은 루프(=K/V 복제 없음)
    """
    def __init__(self,
                 cv_chan: int,
                 feat_chan: int = 256,
                 heads: int = 1,
                 dim_qk: int = 12,
                 dim_v: int = 12,
                 max_tokens: int = 128,
                 spatial_down: int = 2,
                 q_chunk: int = 1024,
                 dropout: float = 0.0):
        super().__init__()
        assert spatial_down in [1, 2, 4], "spatial_down은 1/2/4 중 선택"

        self.cv_chan = cv_chan
        self.heads = heads
        self.dim_qk = dim_qk
        self.dim_v  = dim_v
        self.max_tokens = max_tokens
        self.spatial_down = spatial_down
        self.q_chunk = q_chunk

        # Projections
        self.q_proj = nn.Conv3d(cv_chan, heads * dim_qk, kernel_size=1, bias=False)
        self.k_proj = nn.Conv2d(feat_chan, heads * dim_qk, kernel_size=1, bias=False)
        self.v_proj = nn.Conv2d(feat_chan, heads * dim_v,  kernel_size=1, bias=False)
        self.out_proj = nn.Conv3d(heads * dim_v, cv_chan, kernel_size=1, bias=False)

        self.norm_in  = nn.GroupNorm(1, cv_chan)
        self.norm_out = nn.GroupNorm(1, cv_chan)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("scale", torch.tensor(dim_qk ** -0.5))
        self.gamma = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

    def _token_pool(self, f: torch.Tensor) -> torch.Tensor:
        # 세만틱 토큰 수를 max_tokens 이하로 축소
        B, C, Hs, Ws = f.shape
        Ns = Hs * Ws
        if Ns <= self.max_tokens:
            return f
        ratio = (Ns / float(self.max_tokens)) ** 0.5
        out_h = max(1, int(round(Hs / ratio)))
        out_w = max(1, int(round(Ws / ratio)))
        return F.adaptive_avg_pool2d(f, (out_h, out_w))

    def _downsample_cost(self, x: torch.Tensor) -> torch.Tensor:
        # (D,H,W) 중 H,W만 다운샘플
        if self.spatial_down == 1:
            return x
        return F.avg_pool3d(
            x, kernel_size=(1, self.spatial_down, self.spatial_down),
            stride=(1, self.spatial_down, self.spatial_down))

    def forward(self, cost: torch.Tensor, feat_s4: torch.Tensor) -> torch.Tensor:
        """
        cost:   (B, Ccv, D, H, W)   (H=W_full/4)
        feat_s4:(B, Csf=256, Hs, Ws)  (SegFormer stage4)
        return: (B, Ccv, D, H, W)
        """
        B, Ccv, D, H, W = cost.shape
        x_in = self.norm_in(cost)

        # 쿼리(코스트) 공간 다운샘플
        xq = self._downsample_cost(x_in)      # (B, Ccv, D, Hq, Wq)
        _, _, _, Hq, Wq = xq.shape

        # Q from cost
        q = self.q_proj(xq)                   # (B, h*dk, D, Hq, Wq)

        # K,V from semantic feature (토큰 축소)
        feat_s4 = self._token_pool(feat_s4)   # (B, 256, Hs', Ws')
        k = self.k_proj(feat_s4)              # (B, h*dk, Hs', Ws')
        v = self.v_proj(feat_s4)              # (B, h*dv, Hs', Ws')

        Hk, Wk = k.shape[2], k.shape[3]
        Ns = Hk * Wk
        # (B, h, dk, Ns), (B, h, dv, Ns)
        k = k.view(B, self.heads, self.dim_qk, Ns)
        v = v.view(B, self.heads, self.dim_v,  Ns)
        # (B*h, dk, Ns), (B*h, Ns, dv)
        k_bh = k.view(B * self.heads, self.dim_qk, Ns).contiguous()
        v_bh = v.permute(0,1,3,2).contiguous().view(B * self.heads, Ns, self.dim_v)

        # 출력 버퍼: (B, h, dv, D, Hq, Wq)
        out_all = cost.new_zeros((B, self.heads, self.dim_v, D, Hq, Wq))

        HW = Hq * Wq
        chunk = self.q_chunk

        # D축 루프 (K/V 복제 없음)
        for d in range(D):
            # q_d: (B, h*dk, Hq, Wq) -> (B, h, dk, Hq, Wq) -> (B*h, HW, dk)
            q_d = q[:, :, d, :, :].view(B, self.heads, self.dim_qk, Hq, Wq)
            q_bh = q_d.permute(0,1,3,4,2).contiguous().view(B * self.heads, HW, self.dim_qk)

            # 쿼리 축 분할 계산
            out_bh = cost.new_empty((B * self.heads, HW, self.dim_v))
            start = 0
            while start < HW:
                end = min(start + chunk, HW)
                q_chunk = q_bh[:, start:end, :]                        # (B*h, Ck)
                attn_logits = torch.bmm(q_chunk, k_bh) * self.scale    # (B*h, chunk, Ns)
                attn = F.softmax(attn_logits, dim=-1)
                attn = self.dropout(attn)
                out_chunk = torch.bmm(attn, v_bh)                      # (B*h, chunk, dv)
                out_bh[:, start:end, :] = out_chunk
                start = end

            # (B*h, HW, dv) -> (B, h, dv, Hq, Wq)
            out_bh = out_bh.view(B, self.heads, Hq, Wq, self.dim_v).permute(0,1,4,2,3).contiguous()
            out_all[:, :, :, d, :, :] = out_bh

        # (B, h*dv, D, Hq, Wq) -> 1x1x1 proj -> (B, Ccv, D, Hq, Wq)
        out = out_all.view(B, self.heads * self.dim_v, D, Hq, Wq)
        out = self.out_proj(out)

        # 원래 해상도(H, W)로 업샘플 후 잔차
        if (Hq != H) or (Wq != W):
            
            out = out.to(torch.float32)  # 또는 torch.float16 (GPU에서 권장)
            out = F.interpolate(out, size=(D, H, W), mode='trilinear', align_corners=False)
            out = out.to(torch.bfloat16)
            
        out = self.norm_out(cost + self.gamma * out)
        return out
