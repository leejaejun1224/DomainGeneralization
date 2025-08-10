import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SubModule(nn.Module):
    def __init__(self):
        super(SubModule, self).__init__()


    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                
class RowStripAttention3D(SubModule):
    def __init__(self, cv_chan: int, feat_chan: int = 0,
                 heads: int = 2, dim: int = 8,           # 메모리 절약 기본값
                 min_range: int = 16, local_bias_init: float = -2.0,
                 use_feat: bool = True,
                 memory_efficient: bool = True,          # NEW
                 block_size: int = 64,                   # NEW
                 kv_stride: int = 1):                    # NEW
        super().__init__()
        self.cv_chan = cv_chan
        self.feat_chan = feat_chan
        self.heads = heads
        self.dim = dim
        self.min_range = min_range
        self.use_feat = use_feat and (feat_chan > 0)
        self.memory_efficient = memory_efficient
        self.block_size = block_size
        self.kv_stride = kv_stride

        # cv 전용 투영
        self.q_proj = nn.Conv1d(cv_chan, heads * dim, kernel_size=1, bias=False)
        self.k_proj = nn.Conv1d(cv_chan, heads * dim, kernel_size=1, bias=False)
        self.v_proj = nn.Conv1d(cv_chan, heads * dim, kernel_size=1, bias=False)
        self.out_proj = nn.Conv1d(heads * dim, cv_chan, kernel_size=1, bias=False)
        # feat 전용 투영
        self.k_proj_feat = nn.Conv1d(self.feat_chan, self.heads * self.dim, kernel_size=1, bias=False)
        self.v_proj_feat = nn.Conv1d(self.feat_chan, self.heads * self.dim, kernel_size=1, bias=False)

        self.local_bias = nn.Parameter(torch.tensor(local_bias_init, dtype=torch.float32))
        self.res_scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.weight_init()

    def _shape_head(self, x, B, D, H, W):
        # [B*D*H, h*dim, W] -> [N_h, W, dim], N_h = B*D*H*h
        x = x.view(B*D*H, self.heads, self.dim, W).permute(0,1,3,2).contiguous()
        Nh = x.shape[0]*x.shape[1]
        return x.view(Nh, W, self.dim), Nh  # [Nh, W, dim], Nh

    def _unshape_head(self, x, B, D, H, W):
        # [Nh, W, dim] -> [B*D*H, h*dim, W]
        x = x.view(B*D*H, self.heads, W, self.dim).permute(0,1,3,2).contiguous()
        return x.view(B*D*H, self.heads*self.dim, W)

    def forward(self, cv: torch.Tensor, feat2d: torch.Tensor = None):
        B, C, D, H, W = cv.shape
        cv_1d = cv.permute(0,2,3,1,4).contiguous().view(B*D*H, C, W)

        q = self.q_proj(cv_1d)     # [B*D*H, h*dim, W]
        k_cv = self.k_proj(cv_1d)
        v_cv = self.v_proj(cv_1d)

        if self.use_feat and (feat2d is not None):
            if feat2d.size(-2) != H or feat2d.size(-1) != W:
                feat2d = F.interpolate(feat2d, size=(H, W), mode='bilinear', align_corners=False)
            feat1d = feat2d.permute(0,2,1,3).contiguous().view(B*H, self.feat_chan, W)  # [B*H, F, W]
            k_feat = self.k_proj_feat(feat1d)  # [B*H, h*dim, W]
            v_feat = self.v_proj_feat(feat1d)
            # D축 반복
            k_feat = k_feat.repeat_interleave(D, dim=0)  # [B*D*H, h*dim, W]
            v_feat = v_feat.repeat_interleave(D, dim=0)
            k = k_cv + k_feat
            v = v_cv + v_feat
        else:
            k, v = k_cv, v_cv

        scale = 1.0 / math.sqrt(self.dim)

        # 헤드 차원 분리
        qh, Nh = self._shape_head(q, B, D, H, W)  # [Nh, W, dim]
        kh, _  = self._shape_head(k, B, D, H, W)  # [Nh, W, dim]
        vh, _  = self._shape_head(v, B, D, H, W)  # [Nh, W, dim]

        # K/V 다운샘플 (옵션)
        if self.kv_stride > 1:
            kh = kh[:, ::self.kv_stride, :]   # [Nh, W', dim]
            vh = vh[:, ::self.kv_stride, :]   # [Nh, W', dim]
        Wk = kh.size(1)

        if not self.memory_efficient:
            # 원래 방식 (OOM 위험)
            logits = torch.einsum('n i d, n j d -> n i j', qh, kh) * scale  # [Nh, W, Wk]
            if self.min_range > 0:
                qi = torch.arange(W, device=cv.device).view(1, W, 1)
                kj = torch.arange(Wk, device=cv.device).view(1, 1, Wk) * self.kv_stride
                dist = (qi - kj).abs()
                logits = logits + self.local_bias * (dist < self.min_range)
            attn = F.softmax(logits, dim=-1)
            if self.min_range > 0:
                long_mass_mean = (attn * (dist >= self.min_range)).sum(-1).mean()
            else:
                long_mass_mean = attn[...,0].new_tensor(1.0)
            out = torch.einsum('n i j, n j d -> n i d', attn, vh)  # [Nh, W, dim]
        else:
            # 메모리 효율 스트리밍 소프트맥스
            block = self.block_size
            # 러닝 통계
            m = torch.full((Nh, W, 1), -float('inf'), device=cv.device)
            l = torch.zeros((Nh, W, 1), device=cv.device)
            out = torch.zeros((Nh, W, self.dim), device=cv.device)

            # 장거리 질량 추정(합/전체합)
            long_sum = torch.zeros((Nh, W, 1), device=cv.device)

            qi = torch.arange(W, device=cv.device).view(1, W, 1)  # [1,W,1]

            for j0 in range(0, Wk, block):
                j1 = min(j0 + block, Wk)
                kh_blk = kh[:, j0:j1, :]            # [Nh, J, dim]
                vh_blk = vh[:, j0:j1, :]            # [Nh, J, dim]
                # logits_blk: [Nh, W, J]
                logits_blk = torch.einsum('n i d, n j d -> n i j', qh, kh_blk) * scale

                # 로컬 억제 바이어스
                if self.min_range > 0:
                    kj = (torch.arange(j0, j1, device=cv.device).view(1, 1, -1) * self.kv_stride)  # [1,1,J]
                    dist_blk = (qi - kj).abs()  # [1, W, J]
                    logits_blk = logits_blk + self.local_bias * (dist_blk < self.min_range)

                # 러닝 소프트맥스 결합 (m: max, l: sum exp)
                m_blk = logits_blk.max(dim=-1, keepdim=True).values           # [Nh, W, 1]
                m_new = torch.maximum(m, m_blk)
                # scale factors
                exp_m  = torch.exp(m - m_new)
                exp_mb = torch.exp(logits_blk - m_new)                        # [Nh, W, J]
                # 합 업데이트
                l = l * exp_m + exp_mb.sum(dim=-1, keepdim=True)              # [Nh, W, 1]
                # 출력 업데이트
                out = out * exp_m + torch.einsum('n i j, n j d -> n i d', exp_mb, vh_blk)
                m = m_new

                # 장거리 질량(정규화 전 합)도 누적
                if self.min_range > 0:
                    long_mask_blk = (dist_blk >= self.min_range).float()
                    long_sum = long_sum * exp_m + (exp_mb * long_mask_blk).sum(dim=-1, keepdim=True)

            # 정규화
            out = out / (l + 1e-8)                                           # [Nh, W, dim]
            if self.min_range > 0:
                long_mass_mean = (long_sum / (l + 1e-8)).mean()
            else:
                long_mass_mean = out.new_tensor(1.0)

        # heads 합치기 -> conv1d proj -> residual
        out = self._unshape_head(out, B, D, H, W)                            # [B*D*H, h*dim, W]
        out = self.out_proj(out)                                             # [B*D*H, C, W]
        out = cv_1d + self.res_scale * out
        out = out.view(B, D, H, C, W).permute(0,3,1,2,4).contiguous()        # [B,C,D,H,W]

        stats = {'long_mass_mean': long_mass_mean}
        return out, stats


# ===========================================================================

# === NEW: 장거리 규제 손실(원하는 '장거리 질량'을 확보하게 강제) =============
def attn_range_loss(long_mass_mean: torch.Tensor, target_mass: float = 0.30):
    """
    long_mass_mean: RowStripAttention3D가 반환한 장거리 질량 평균(스칼라 텐서)
    target_mass:    이 이상은 장거리에서 질량이 나오도록 유도(예: 0.30)
    """
    # target_mass - long_mass_mean 이 양수일 때만 벌점
    return F.relu(target_mass - long_mass_mean)
# ===========================================================================



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