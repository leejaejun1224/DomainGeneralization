import math
import torch
import torch.nn as nn
import torch.nn.functional as F


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

class RefineCostVolume(nn.Module):
    def __init__(self, feat_ch: int, max_disp: int,
                 emb: int = 32, pos_sigma: float = 0.05, tau: float = 0.0):
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

        mask_pred_L= self.mask_head(featL) 
        mask_pred_R = self.mask_head(featR)
        mask_loss = None
        if teacher_entropy is not None:
            mask_loss = F.binary_cross_entropy(mask_pred_L, teacher_entropy.float())

        ## 여기서 true인 부분은 내가 refine을 해야하는 부분
        ### output이 sigmoid 결과임. 고로 1에 가까울수록 신뢰픽셀이라는 뜻임.
        mask_bin_L = mask_pred_L > 0.5
        mask_bin_R = mask_pred_R > 0.5
        # 2. Hi‑Lo attention refinement
        featL_ref = self.propagation(featL, mask_pred_L)
        featR_ref = self.propagation(featR, mask_pred_R)

        mask_pred_ref_L = self.mask_head(featL_ref)
        # if teacher_entropy is not None:
        #     mask_loss += F.binary_cross_entropy(mask_pred_ref_L, teacher_entropy.float())

        # cost는 weighted cost volume으로 다시 제작
        # cost = self.build_cost(featL_ref, featR_ref).unsqueeze(1)
        return featL_ref,featR_ref, mask_pred_L, mask_pred_R



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
        # self.gate = nn.Parameter(torch.zeros(1,self.C,1,1))



    def forward(self, feat: torch.Tensor, mask_high: torch.Tensor):
        B, C, H, W = feat.shape
        L = H * W


        qkv = self.to_qkv(feat).view(B, 3, self.h, self.Hd, H, W)

        ## shape은 [B, h, Hd, H, W]
        q, k, v = qkv[:,0], qkv[:,1], qkv[:,2]
        
        ## 플래튼 하면 크기는 [B, h*Hd, H, W]
        k_flat = k.flatten(1, 2)
        v_flat = v.flatten(1, 2)

        ## mask_high는 신뢰영역 = 1, 아니면 0
        w_unf = self.unfold(mask_high.float()) 
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

            ## 신뢰픽셀만 모임
            k_low = k_unf[b, :, :, :, low_idx]         # (h, Hd, win², N_low)
            v_low = v_unf[b, :, :, :, low_idx]         # (h, Hd, win², N_low)
     

            ## 크기는 [h, win², N_low]
            attn = (q_low.unsqueeze(2) * k_low).sum(1) * self.scale
            ## pos bias
            attn = attn + self.pos_bias[..., None]
            attn = F.softmax(attn, dim=1)
            # 여기까지 하면 attn shape은 [h, win제곱, low mask 개수]

            ## 잊지 말것 여기서 v_low의 shape은 [h, Hd, win제곱, mask 갯수]
            out_low = (attn.unsqueeze(1) * v_low).sum(dim=2)
            ## 이러면 out_low 크기는 [h, Hd, N_low]

            
            ## low conf픽셀에만 값 채우고
            out_flat[b, :, low_idx] = out_low.view(self.h*self.Hd, -1)

        # reshape & 1×1 conv
        out = out_flat.view(B, self.C, H, W)
        feat_ref = self.proj(out)

        # residual은 오직 불신 픽셀에만
        mask_low_map = mask_low.view(B, 1, H, W).float()
        refined = feat + feat_ref * mask_low_map
        # refined = feat + feat_ref

        return refined



# --------------------------------------------------------------
# confmap.py
# --------------------------------------------------------------
import torch, torch.nn.functional as F
import numpy as np
from skimage.segmentation import slic           # pip install scikit-image
import cv2

# -------- Sobel 커널 (HWC → CHW 편의를 위해 1×1 conv weight로 사용) ----
_k = torch.tensor([[1,  0, -1],
                   [2,  0, -2],
                   [1,  0, -1]], dtype=torch.float32)
sobel_kernel = _k.view(1,1,3,3).repeat(1,1,1,1)  # [1,1,3,3]



# ------------------------------------------
def guided_filter(I, p, radius=7, eps=1e-3):
    """
    I: H×W×3 (0–1 float), p: H×W (0–1 float)
    radius, eps: guidedFilter 파라미터
    """
    # OpenCV는 uint8 / [0,255] 를 기대하므로 변환
    guide_uint8 = (I * 255).astype(np.uint8)
    p_uint8     = (p * 255).astype(np.uint8)

    try:
        # opencv-contrib-python 설치 시
        gf = cv2.ximgproc.guidedFilter(guide=guide_uint8,
                                       src=p_uint8,
                                       radius=radius,
                                       eps=eps)
        # 다시 [0,1] float32
        return (gf.astype(np.float32) / 255.0)
    except Exception:
        # fallback: bilateralFilter 로 근사
        # d=diameter, sigmaColor=eps*255, sigmaSpace=radius
        bf = cv2.bilateralFilter(p_uint8,
                                 d=radius*2+1,
                                 sigmaColor=eps*255,
                                 sigmaSpace=radius)
        return (bf.astype(np.float32) / 255.0)
# ------------------------------------------
def compute_confidence(pred_up: torch.Tensor,
                       prob:     torch.Tensor,
                       rgb:      torch.Tensor,
                       device    = "cuda"):
    """
    pred_up : [B,1,H,W], prob:[B,k,H,W], rgb:[B,3,H,W]  (0–1 range)
    returns conf_final : [B,1,H,W]
    """
    B, _, H, W = pred_up.shape
    k = prob.size(1)
    # 1) gap + entropy
    # p1, p2 = prob.topk(2, dim=1).values        
    p1 = prob[:,0,:,:]
    p2 = prob[:,1,:,:]               # [B,2,H,W]
    gap     = (p1-p2).clamp_(0,1)                     # [B,H,W]
    entropy = -(prob*prob.clamp_min(1e-8).log()).sum(1)         # [B,H,W]
    conf0   = torch.sigmoid(  6*(gap-0.10) ) * torch.exp(-entropy)
    conf0   = conf0.unsqueeze(1)                                # [B,1,H,W]

    # 2) disparity gradient down-weight
    grad = F.conv2d(pred_up, sobel_kernel.to(device),
                    padding=1).abs().mean(1, keepdim=True)      # [B,1,H,W]
    w_edge = torch.exp( -(grad/4.0)**2 )

    conf1  = conf0 * w_edge                                     # [B,1,H,W]

    # 3) superpixel 평균 + guided filter
    conf_out = torch.zeros_like(conf1)
    rgb_np = rgb.permute(0,2,3,1).cpu().numpy()                 # B,H,W,3
    for b in range(B):
        # SLIC
        lbl = slic(rgb_np[b], n_segments=2000, compactness=10,
                   start_label=0)                               # [H,W] int
        lbl_t = torch.from_numpy(lbl).to(device).long()

        # superpixel 평균
        conf_sp = torch.zeros_like(conf1[b,0])
        conf_sum = torch.zeros_like(conf_sp)
        conf_cnt = torch.zeros_like(conf_sp)
        conf_flat = conf1[b,0].flatten()
        lbl_flat  = lbl_t.flatten()
        conf_sum = conf_sum.flatten()
        conf_cnt = conf_cnt.flatten()

        conf_sum.index_add_(0, lbl_flat, conf_flat)
        conf_cnt.index_add_(0, lbl_flat,
                            torch.ones_like(conf_flat))
        conf_mean = conf_sum / (conf_cnt + 1e-6)                # [N_sp]
        conf_sp = conf_mean[lbl_t]                              # [H,W]

        # guided filter (OpenCV requires CPU + numpy)
        cf_sp_np = conf_sp.cpu().numpy()
        cf_gf = guided_filter(rgb_np[b], cf_sp_np,
                              radius=15, eps=1e-3)               # [H,W]
        conf_out[b,0] = torch.from_numpy(cf_gf).to(device)

    conf_final = conf_out.clamp_(0,1)
    return conf_final
