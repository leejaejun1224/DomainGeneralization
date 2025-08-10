# === [ADD] import 보강 ===
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional

class SemanticDisparityFiller:
    """
    disp의 0-홀을 semantic segmentation(seg) 라벨 내부에서만 채웁니다.

    Parameters
    ----------
    ignore_label : int
        무시 라벨(라벨 외곽/패딩)
    kernels : List[int]
        국소 보간에서 사용할 정사각 커널 크기(홀수만). 작은 것부터 순차 확대
    min_valid_for_plane : int
        평면 적합을 시도하기 위한 라벨 내부 유효 픽셀 최소 개수
    huber_delta : float
        평면 적합 IRLS 과정에서 Huber 로버스트 가중치의 델타
    plane_iters : int
        IRLS 반복 횟수
    smooth_iters : int
        마지막 라플라시안 스무딩 반복 횟수(라벨 내부, 홀로 채운 픽셀만 업데이트)
    clip_margin : float
        평면 예측값 클리핑 마진(px). [min-clip, max+clip]로 제한
    """
    def __init__(
        self,
        ignore_label: int = 255,
        kernels: List[int] = (3, 5, 7, 11, 15),
        min_valid_for_plane: int = 128,
        huber_delta: float = 1.0,
        plane_iters: int = 3,
        smooth_iters: int = 8,
        clip_margin: float = 1.0
    ):
        assert all(k % 2 == 1 for k in kernels), "kernels는 모두 홀수여야 합니다."
        self.ignore_label = int(ignore_label)
        self.kernels = list(kernels)
        self.min_valid_for_plane = int(min_valid_for_plane)
        self.huber_delta = float(huber_delta)
        self.plane_iters = int(plane_iters)
        self.smooth_iters = int(smooth_iters)
        self.clip_margin = float(clip_margin)
        # 4-이웃 라플라시안 보정용 커널
        self.kernel4 = torch.tensor([[0., 1., 0.],
                                     [1., 0., 1.],
                                     [0., 1., 0.]]).view(1,1,3,3)

    @staticmethod
    def _to_bchw(disp: torch.Tensor) -> Tuple[torch.Tensor, bool]:
        """[B,1,H,W] 또는 [B,H,W] → [B,1,H,W], 그리고 원래 채널 여부 반환"""
        if disp.dim() == 3:     # [B,H,W]
            return disp.unsqueeze(1), True
        elif disp.dim() == 4:   # [B,1,H,W] or [B,C,H,W]
            if disp.size(1) != 1:
                raise ValueError("disp는 채널 1개만 지원합니다. shape={}".format(disp.shape))
            return disp, False
        else:
            raise ValueError("disp shape 지원 불가: {}".format(disp.shape))

    @staticmethod
    def _same_pad(k: int) -> int:
        return k // 2

    @staticmethod
    def _ones_kernel(k: int, device, dtype):
        return torch.ones((1,1,k,k), device=device, dtype=dtype)

    def _local_fill_label(
        self, d: torch.Tensor, label_mask: torch.Tensor, orig_zero: torch.Tensor
    ) -> torch.Tensor:
        """
        라벨 내부에서 0-홀을 국소 평균 보간으로 채움.
        d: [1,H,W] float, label_mask/orig_zero: [H,W] bool
        반환: 라벨 내부에서 가능한 만큼 채워진 d
        """
        device, dtype = d.device, d.dtype
        valid = (d[0] > 0) & label_mask
        hole  = (~valid) & label_mask & orig_zero
        if hole.sum() == 0:
            return d

        for k in self.kernels:
            if hole.sum() == 0:
                break
            pad = self._same_pad(k)
            w = self._ones_kernel(k, device, dtype)

            valid_f = valid.float().unsqueeze(0).unsqueeze(0)     # [1,1,H,W]
            d_f     = d * valid_f                                  # [1,1,H,W]

            denom = F.conv2d(valid_f, w, padding=pad)              # 이웃 유효 개수
            num   = F.conv2d(d_f,     w, padding=pad)              # 이웃 유효값 합
            avg   = num / (denom + 1e-6)

            fillable = hole & (denom[0,0] > 0)
            if fillable.any():
                d[0][fillable] = avg[0,0][fillable]
                valid = (d[0] > 0) & label_mask
                hole  = (~valid) & label_mask & orig_zero

        return d

    def _plane_fill_label(
        self, d: torch.Tensor, label_mask: torch.Tensor, seed_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        라벨 내부에서 남은 홀을 평면 적합으로 채움(유효 시드가 충분할 때).
        d: [1,H,W] float, label_mask/seed_mask: [H,W] bool (seed_mask=True는 원래부터 유효값)
        """
        device, dtype = d.device, d.dtype
        H, W = d.shape[-2:]
        valid = seed_mask & label_mask
        hole  = (d[0] == 0) & label_mask

        if hole.sum() == 0 or valid.sum() < self.min_valid_for_plane:
            return d

        # 좌표 및 관측 벡터 구성
        ys, xs = torch.nonzero(valid, as_tuple=True)
        z = d[0, ys, xs].view(-1, 1).to(dtype)

        X = torch.stack([xs.to(dtype), ys.to(dtype), torch.ones_like(xs, dtype=dtype)], dim=1)  # [N,3]
        w = torch.ones((X.size(0), 1), device=device, dtype=dtype)

        for _ in range(self.plane_iters):
            # 가중 최소제곱: sqrt(w)를 곱해주면 일반 lstsq로 해결 가능
            sw = torch.sqrt(w).clamp_min(1e-6)
            Xw = X * sw
            zw = z * sw
            # lstsq 해
            beta = torch.linalg.lstsq(Xw, zw, rcond=None).solution  # [3,1]
            # 잔차
            r = (X @ beta - z).abs()
            # Huber 가중치 업데이트
            delta = self.huber_delta
            w = torch.where(r <= delta, torch.ones_like(r), (delta / (r + 1e-6)))

        # 평면으로 홀 예측
        ys_h, xs_h = torch.nonzero(hole, as_tuple=True)
        Xh = torch.stack([xs_h.to(dtype), ys_h.to(dtype), torch.ones_like(xs_h, dtype=dtype)], dim=1)
        zh = (Xh @ beta).view(-1)

        # 클리핑(라벨 내부 시드 분포 범위)
        zmin = d[0][valid].min()
        zmax = d[0][valid].max()
        zh = zh.clamp(zmin - self.clip_margin, zmax + self.clip_margin)

        d[0, ys_h, xs_h] = zh
        return d

    def _laplacian_smooth_label(
        self, d: torch.Tensor, label_mask: torch.Tensor, seed_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        라벨 내부에서만 라플라시안 스무딩.
        seed_mask=True 위치는 고정(업데이트 안 함).
        """
        if self.smooth_iters <= 0:
            return d
        device, dtype = d.device, d.dtype
        k4 = self.kernel4.to(device=device, dtype=dtype)
        lab = label_mask.float().unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
        seed = seed_mask.bool()

        for _ in range(self.smooth_iters):
            # 이웃 합과 개수(라벨 내부 이웃만)
            sum_nb = F.conv2d(d * lab, k4, padding=1)
            cnt_nb = F.conv2d(lab,     k4, padding=1)
            avg_nb = sum_nb / (cnt_nb + 1e-6)

            # 업데이트 대상: 라벨 내부 & 시드가 아닌 위치
            upd = label_mask & (~seed)
            d[0][upd] = avg_nb[0,0][upd]

        return d

    def fill(
        self, disp: torch.Tensor, seg: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        disp : [B,1,H,W] 또는 [B,H,W] float tensor (0 = 노이즈 제거 구간)
        seg  : [B,H,W] long tensor (semantic label)

        Returns
        -------
        disp_filled : disp와 동일 shape
        filled_mask : [B,H,W] bool (이번에 새로 채워진 픽셀)
        """
        disp_bchw, squeezed = self._to_bchw(disp)
        assert seg.dim() == 3 and seg.shape == disp_bchw.shape[::1][0:1] + disp_bchw.shape[-2:], \
            f"seg shape 불일치: seg={seg.shape}, disp={disp_bchw.shape}"

        B, _, H, W = disp_bchw.shape
        device, dtype = disp_bchw.device, disp_bchw.dtype

        out = disp_bchw.clone()
        filled_all = torch.zeros((B, H, W), device=device, dtype=torch.bool)

        for b in range(B):
            d = out[b:b+1]                  # [1,1,H,W] 슬라이스
            d = d.squeeze(1)                # [1,H,W] 로 단순화
            s = seg[b]                      # [H,W]
            orig_zero = (d[0] == 0)

            labels = torch.unique(s)
            for lbl in labels.tolist():
                if lbl == self.ignore_label:
                    continue
                label_mask = (s == lbl)
                if label_mask.sum() == 0:
                    continue

                seed_mask = (d[0] > 0) & label_mask  # 원래부터 유효한 시드

                # 1) 국소 보간
                d = self._local_fill_label(d, label_mask, orig_zero)

                # 2) 평면 적합(시드 충분할 때만)
                if ((d[0] == 0) & label_mask).sum() > 0:
                    d = self._plane_fill_label(d, label_mask, seed_mask)

                # 3) 라플라시안 스무딩(시드 고정)
                d = self._laplacian_smooth_label(d, label_mask, seed_mask)

            # 배치 결과 기록
            newly_filled = orig_zero & (d[0] > 0)
            filled_all[b] = newly_filled
            out[b:b+1] = d.unsqueeze(1)

        # 입력 shape로 복구
        disp_filled = out.squeeze(1) if squeezed else out
        return disp_filled, filled_all

# === [ADD] same-label smoothness loss ===
def seg_disp_hinge_loss(
    disp: torch.Tensor,
    seg: torch.Tensor,
    tau: float = 1.0,
    ignore_label: int = 255,
    neighbor_mode: str = '8',   # '4' 또는 '8'
    radius: int = 1,
    reduction: str = 'mean'
):
    """
    disp: [B,1,H,W] 또는 [B,H,W] (시차 단위: px 기준 권장)
    seg : [B,H,W] (Long; 동일 세그먼트인 픽셀 쌍에만 제약)
    tau : 허용 임계값(픽셀). |Δd| <= tau이면 페널티 0
    neighbor_mode: '4' → 좌우/상하, '8' → + 대각선
    radius: 이웃 거리(1 이상). r이 커질수록 1/r 가중치 부여
    ignore_label: 무시할 라벨(패딩 포함)
    reduction: 'mean' | 'sum' | 'none'(=합,분모 튜플 반환)
    """
    if disp.dim() == 3:
        disp = disp.unsqueeze(1)  # [B,1,H,W]
    if seg.dim() == 4:
        seg = seg.squeeze(1)      # [B,H,W]

    B, C, H, W = disp.shape
    assert C == 1, "disp는 [B,1,H,W] 형식을 권장합니다."
    assert seg.shape == (B, H, W), f"seg shape 불일치: {seg.shape} != {(B,H,W)}"

    seg = seg.to(disp.device)
    tau = torch.as_tensor(tau, dtype=disp.dtype, device=disp.device)

    total = disp.new_tensor(0.0)
    denom = disp.new_tensor(0.0)

    def accumulate(dy, dx, w=1.0):
        nonlocal total, denom
        y1s, y1e = max(0, dy), H - max(0, -dy)
        x1s, x1e = max(0, dx), W - max(0, -dx)
        y2s, y2e = max(0, -dy), H - max(0, dy)
        x2s, x2e = max(0, -dx), W - max(0, dx)
        if y1e <= y1s or x1e <= x1s:
            return

        d1 = disp[:, :, y1s:y1e, x1s:x1e]
        d2 = disp[:, :, y2s:y2e, x2s:x2e]
        s1 = seg[:,      y1s:y1e, x1s:x1e]
        s2 = seg[:,      y2s:y2e, x2s:x2e]

        valid = (s1 == s2) & (s1 != ignore_label)
        if not valid.any():
            return

        delta = (d1 - d2).abs().squeeze(1)  # [B,h,w]
        penalty = (delta - tau).clamp_min(0.0) * valid.float() * float(w)
        total = total + penalty.sum()
        denom = denom + valid.float().sum()

    for r in range(1, radius + 1):
        w = 1.0 / float(r)
        # 좌우 / 상하
        accumulate(0,  r, w)
        accumulate(r,  0, w)
        if neighbor_mode in ('8', 'diag'):
            # 대각선(가중치 보정)
            diag_w = w / 1.41421356237
            accumulate(r,  r,  diag_w)
            accumulate(r, -r,  diag_w)

    if reduction == 'mean':
        return total / denom.clamp_min(1.0)
    elif reduction == 'sum':
        return total
    else:
        # 'none' 대용: (합계, 유효쌍개수) 반환하여 바깥에서 가중 평균 가능
        return total, denom


def calc_semantic_loss(pred_disp, data_batch):
    lambda_seg = 0.1  # 가중치는 상황에 맞게 조정

    seg_gt = data_batch["tgt_seg"].long().to(pred_disp[0].device)  # DataLoader가 np→tensor 변환
    loss_seg = seg_disp_hinge_loss(
        pred_disp,
        seg_gt,
        tau=1.0,                # 0.5~1.5 사이에서 탐색 권장
        ignore_label=255,       # 위 로더 기본값과 일치
        neighbor_mode='8',
        radius=1,
        reduction='mean'
    )

    # 멀티스케일을 함께 쓰는 경우(선택):
    # pred_disp_half, pred_disp_low가 있을 때
    # if "seg_half" in data_batch and "seg_low" in data_batch:
    #     loss_seg_half = seg_disp_hinge_loss(pred_disp_half, batch["seg_half"].long().to(pred_disp.device),
    #                                         tau=1.0, ignore_label=255, neighbor_mode='8', radius=1)
    #     loss_seg_low  = seg_disp_hinge_loss(pred_disp_low,  batch["seg_low"].long().to(pred_disp.device),
    #                                         tau=1.0, ignore_label=255, neighbor_mode='8', radius=1)
    #     loss_seg = loss_seg + 0.5 * loss_seg_half + 0.25 * loss_seg_low

    total_loss =  lambda_seg * loss_seg
    return total_loss