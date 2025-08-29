import os
import argparse
from pathlib import Path

import numpy as np
import cv2
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# I/O
# ------------------------------------------------------------
def load_rgb(path):
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

def load_gray(path):
    bgr = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if bgr is None:
        raise FileNotFoundError(path)
    return bgr

def read_pfm(file_path):
    with open(file_path, "rb") as f:
        header = f.readline().decode("utf-8").rstrip()
        if header not in ("PF", "Pf"):
            raise ValueError("Not a PFM file.")
        dims = f.readline().decode("utf-8").strip()
        while dims.startswith("#"):
            dims = f.readline().decode("utf-8").strip()
        w, h = map(int, dims.split())
        scale = float(f.readline().decode("utf-8").strip())
        endian = "<" if scale < 0 else ">"
        count = w*h*(3 if header=="PF" else 1)
        data = np.fromfile(f, endian+"f", count=count)
        data = data.reshape((h, w, 3)) if header=="PF" else data.reshape((h, w))
        return np.flipud(data), abs(scale)

def load_disp_left(path):
    ext = Path(path).suffix.lower()
    if ext == ".pfm":
        arr, _ = read_pfm(path)
        if arr.ndim == 3:
            arr = arr[...,0]
        return arr.astype(np.float32)
    elif ext == ".npy":
        arr = np.load(path)
        if arr.ndim == 3:
            arr = arr[...,0]
        return arr.astype(np.float32)
    elif ext == ".npz":
        npz = np.load(path)
        key = "disp" if "disp" in npz else ("disparity" if "disparity" in npz else list(npz.keys())[0])
        arr = npz[key]
        if arr.ndim == 3:
            arr = arr[...,0]
        return arr.astype(np.float32)
    else:
        raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if raw is None:
            raise FileNotFoundError(path)
        if raw.ndim == 3:
            raw = raw[...,0]
        if raw.dtype == np.uint16:
            # KITTI: disp_px * 256
            return (raw.astype(np.float32) / 256.0)
        return raw.astype(np.float32)

def resize_disp_to(img_hw, disp):
    Ht, Wt = img_hw
    H0, W0 = disp.shape
    if (Ht, Wt) == (H0, W0):
        return disp
    disp_rs = cv2.resize(disp, (Wt, Ht), interpolation=cv2.INTER_LINEAR)
    scale_x = Wt / float(W0)
    return disp_rs * scale_x

# ------------------------------------------------------------
# Task 1: Forward warping (Left -> Right) + Z-buffer
# ------------------------------------------------------------
def forward_warp_left_to_right(left_rgb, disp_L, min_disp=0.0):
    """Return: right_rgb_warped, hit_mask(0/255), disp_R_init (on right grid)"""
    H, W = disp_L.shape
    if left_rgb.shape[:2] != (H, W):
        raise ValueError("left_rgb and disp shape mismatch.")
    right_rgb = np.zeros_like(left_rgb)
    right_disp = np.full((H, W), np.nan, dtype=np.float32)
    hit_mask = np.zeros((H, W), dtype=np.uint8)

    for y in range(H):
        xl = np.arange(W, dtype=np.int32)
        d = disp_L[y]
        valid = np.isfinite(d) & (d > min_disp)
        if not np.any(valid):
            continue
        xl = xl[valid]
        d = d[valid]
        xr = np.rint(xl.astype(np.float32) - d).astype(np.int32)
        inb = (xr >= 0) & (xr < W)
        if not np.any(inb):
            continue
        xl, xr, d = xl[inb], xr[inb], d[inb]

        # 같은 xr에 여러 소스 -> 큰 disparity(가까움) 우선
        order = np.lexsort((-d, xr))
        xr_sorted = xr[order]
        xl_sorted = xl[order]
        d_sorted  = d[order]
        uniq_xr, idx_first = np.unique(xr_sorted, return_index=True)
        xl_best = xl_sorted[idx_first]
        d_best  = d_sorted[idx_first]

        right_rgb[y, uniq_xr] = left_rgb[y, xl_best]
        right_disp[y, uniq_xr] = d_best
        hit_mask[y, uniq_xr] = 255

    return right_rgb, hit_mask, right_disp

def apply_black_mask(rgb, hit_mask):
    out = rgb.copy()
    out[hit_mask == 0] = 0
    return out

# ------------------------------------------------------------
# 좌-우 일관성(LR) 체크 (선택)
# ------------------------------------------------------------
def lr_consistency_mask(disp_L, disp_R, tol=1.0):
    """Left 기준 유효성 마스크(True=일관) 및 Right로의 투영 마스크"""
    H, W = disp_L.shape
    X = np.tile(np.arange(W), (H,1)).astype(np.float32)
    Y = np.tile(np.arange(H).reshape(H,1), (1,W)).astype(np.float32)

    xr = np.rint(X - disp_L).astype(np.int32)
    yr = np.rint(Y).astype(np.int32)
    inb = (xr >= 0) & (xr < W)

    valid_L = np.zeros((H, W), dtype=bool)
    diff = np.full((H, W), np.inf, dtype=np.float32)
    sel = inb & np.isfinite(disp_L)
    if np.any(sel):
        dr = np.full_like(disp_L, np.nan, dtype=np.float32)
        dr[sel] = disp_R[yr[sel], xr[sel]]
        diff[sel] = np.abs(disp_L[sel] - dr[sel])
        valid_L = (diff <= tol)

    # Left에서 invalid로 판정된 픽셀을 Right 좌표로 보냄(가려짐/오류 영역)
    invalid_L = (~valid_L) & np.isfinite(disp_L)
    hit_R = np.zeros_like(invalid_L, dtype=np.uint8)
    # forward warp of invalid mask using Z-buffer rule: 큰 disp 우선
    for y in range(H):
        xl = np.arange(W, dtype=np.int32)
        d = disp_L[y]
        inv = invalid_L[y]
        sel2 = inv & np.isfinite(d)
        if not np.any(sel2):
            continue
        xl_v = xl[sel2]; d_v = d[sel2]
        xr = np.rint(xl_v.astype(np.float32) - d_v).astype(np.int32)
        inb = (xr >= 0) & (xr < W)
        if not np.any(inb):
            continue
        xr = xr[inb]; d_v = d_v[inb]
        order = np.lexsort((-d_v, xr))
        xr_s = xr[order]
        uniq_xr, idx_first = np.unique(xr_s, return_index=True)
        hit_R[y, uniq_xr] = 255
    return valid_L, hit_R  # hit_R: Right에서 '문제 가능' 위치

# ------------------------------------------------------------
# Task 2: AD-Census 기반 국소 탐색(±search_px)로 disparity 재평가
# ------------------------------------------------------------
def census5x5_u8x3(img_gray_u8):
    """5x5 Census(24bit) -> 3채널 uint8(상위바이트,중간,하위)"""
    img = img_gray_u8
    H, W = img.shape
    r = 2
    pad = cv2.copyMakeBorder(img, r, r, r, r, cv2.BORDER_REPLICATE)
    center = pad[r:r+H, r:r+W]
    desc32 = np.zeros((H, W), dtype=np.uint32)
    for dy in range(-r, r+1):
        for dx in range(-r, r+1):
            if dy == 0 and dx == 0: 
                continue
            nei = pad[r+dy:r+dy+H, r+dx:r+dx+W]
            # >= 로 정의(양쪽 동일 정의이면 상관 없음)
            bit = (nei >= center).astype(np.uint32)
            desc32 = (desc32 << 1) | bit
    b0 = ((desc32 >> 16) & 0xFF).astype(np.uint8)
    b1 = ((desc32 >> 8) & 0xFF).astype(np.uint8)
    b2 = (desc32 & 0xFF).astype(np.uint8)
    return np.dstack([b0, b1, b2])  # (H,W,3) uint8

_LUT256 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)

def hamming_bytes3(a_u8x3, b_u8x3):
    """a,b: (H,W,3) uint8. 반환: (H,W) uint16 해밋 거리"""
    xor = cv2.bitwise_xor(a_u8x3, b_u8x3)
    # LUT으로 바이트별 bitcount 후 합산
    h = _LUT256[xor[...,0]] + _LUT256[xor[...,1]] + _LUT256[xor[...,2]]
    return h.astype(np.float32)  # max 24

def refine_disp_right_local_search_ADcensus(left_gray, right_gray, disp_R_init, hit_mask_R,
                                            search_px=4, win=5, w_ad=0.5, w_census=0.5):
    """
    오른쪽 좌표계 disparity(disp_R_init)를 중심으로 ±search_px 탐색.
    비용 = w_ad * AD(win평균) + w_census * (Hamming/24)
    반환: disp_R_refined (float32, NaN=invalid)
    """
    assert win % 2 == 1
    r = win // 2
    H, W = right_gray.shape
    valid = (hit_mask_R > 0) & np.isfinite(disp_R_init) & (disp_R_init > 0)
    disp_ref = np.full((H, W), np.nan, np.float32)
    if not np.any(valid):
        return disp_ref

    R = right_gray.astype(np.float32)
    L = left_gray.astype(np.float32)

    # Census(5x5) 미리 계산
    cR = census5x5_u8x3(right_gray)
    cL = census5x5_u8x3(left_gray)

    # Right 픽셀 좌표 그리드
    X = np.tile(np.arange(W, dtype=np.float32), (H,1))
    Y = np.tile(np.arange(H, dtype=np.float32).reshape(H,1), (1,W))
    base_mapx = X + disp_R_init  # Left에서 샘플할 x좌표(기준 d)
    base_mapy = Y

    min_cost = np.full((H, W), np.inf, np.float32)
    best_d   = np.zeros((H, W), np.float32)

    # 후보 Δ ∈ [-search_px, ..., +search_px]
    for delta in range(-search_px, search_px + 1):
        mapx = base_mapx + delta
        mapy = base_mapy

        # Left 영상/서술자를 Right 격자에 샘플링
        L_samp = cv2.remap(L, mapx, mapy, interpolation=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REPLICATE)
        cL_samp = cv2.remap(cL, mapx, mapy, interpolation=cv2.INTER_NEAREST,
                            borderMode=cv2.BORDER_REPLICATE)

        # AD 비용 (윈도우 평균)
        ad = np.abs(L_samp - R)
        ad = cv2.boxFilter(ad, ddepth=-1, ksize=(win, win), normalize=True)

        # Census 해밋 거리 (0~24)
        ham = hamming_bytes3(cL_samp, cR)  # float32
        ham_norm = ham / 24.0

        cost = w_ad * (ad / 255.0) + w_census * ham_norm
        cost[~valid] = np.inf

        better = cost < min_cost
        best_d = np.where(better, disp_R_init + delta, best_d)
        min_cost = np.where(better, cost, min_cost)

    disp_ref[valid] = best_d[valid]
    return disp_ref

# ------------------------------------------------------------
# 시각화
# ------------------------------------------------------------
def auto_vmin_vmax(disp):
    v = disp[np.isfinite(disp) & (disp > 0)]
    if v.size == 0:
        return 0.0, 1.0
    vmin = np.percentile(v, 1)
    vmax = np.percentile(v, 99)
    if vmax <= vmin: vmax = vmin + 1.0
    return float(vmin), float(vmax)

def save_and_viz(out_dir, right_rgb, warped_rgb, masked_rgb, hole_mask,
                 disp_R_init, disp_R_ref):
    os.makedirs(out_dir, exist_ok=True)
    plt.figure(figsize=(14,10))

    plt.subplot(2,3,1); plt.imshow(right_rgb); plt.title("Right image"); plt.axis("off")
    plt.subplot(2,3,2); plt.imshow(warped_rgb); plt.title("Warped L→R (Z-buffer)"); plt.axis("off")
    plt.subplot(2,3,3); plt.imshow(masked_rgb); plt.title("Warped + Black Mask"); plt.axis("off")

    plt.subplot(2,3,4); plt.imshow(hole_mask, cmap="gray"); plt.title("Hole mask (0=hole)"); plt.axis("off")

    vmin, vmax = auto_vmin_vmax(disp_R_init)
    plt.subplot(2,3,5); im1 = plt.imshow(disp_R_init, cmap="magma", vmin=vmin, vmax=vmax)
    plt.title("Disp_R init"); plt.axis("off"); plt.colorbar(im1, fraction=0.046, pad=0.04)

    vmin2, vmax2 = auto_vmin_vmax(disp_R_ref)
    plt.subplot(2,3,6); im2 = plt.imshow(disp_R_ref, cmap="magma", vmin=vmin2, vmax=vmax2)
    plt.title("Disp_R refined"); plt.axis("off"); plt.colorbar(im2, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "viz_panel.png"), dpi=200)
    plt.close()

    cv2.imwrite(os.path.join(out_dir, "warped_right.png"),
                cv2.cvtColor(warped_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(out_dir, "warped_right_masked.png"),
                cv2.cvtColor(masked_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(out_dir, "hole_mask.png"), hole_mask)

    # disparity 저장(시각화용 컬러맵 PNG와 수치 npy 모두)
    np.save(os.path.join(out_dir, "disp_right_init.npy"), disp_R_init)
    np.save(os.path.join(out_dir, "disp_right_refined.npy"), disp_R_ref)

    # 컬러맵 이미지는 그대로 viz_panel 참고

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--left", default="/home/jaejun/dataset/kitti_2015/training/image_2/000020_10.png")
    ap.add_argument("--right", default="/home/jaejun/dataset/kitti_2015/training/image_3/000020_10.png")
    ap.add_argument("--disp_left", default="/home/jaejun/DomainGeneralization/log/2025-08-19_21_56_st/pseudo_disp/000020_10.png")
    ap.add_argument("--out_dir", default="./log/outputs")
    ap.add_argument("--min_disp", type=float, default=0.0)
    ap.add_argument("--search_px", type=int, default=4, help="±국소 탐색 범위 (픽셀)")
    ap.add_argument("--win", type=int, default=5, help="AD 집계 윈도우(홀수)")
    ap.add_argument("--lr_tol", type=float, default=1.0, help="LR consistency 허용 오차")
    args = ap.parse_args()

    left_rgb  = load_rgb(args.left)
    right_rgb = load_rgb(args.right)
    H, W = right_rgb.shape[:2]

    disp_L = load_disp_left(args.disp_left)
    disp_L = resize_disp_to((left_rgb.shape[0], left_rgb.shape[1]), disp_L)

    # Task 1: forward warp (Left->Right)
    warped_rgb, hit_mask, disp_R_init = forward_warp_left_to_right(left_rgb, disp_L, min_disp=args.min_disp)
    masked_rgb = apply_black_mask(warped_rgb, hit_mask)
    hole_mask = (hit_mask == 0).astype(np.uint8) * 255

    # 선택: LR consistency에서 '문제 가능' 영역 표시(참고)
    validL, problem_R = lr_consistency_mask(disp_L, disp_R_init, tol=args.lr_tol)
    # 필요하다면 hole_mask와 합칠 수도 있음: hole_mask = cv2.bitwise_or(hole_mask, problem_R)

    # Task 2: 국소 탐색으로 disparity 재평가(오른쪽 좌표계)
    left_gray  = cv2.cvtColor(left_rgb, cv2.COLOR_RGB2GRAY)
    right_gray = cv2.cvtColor(right_rgb, cv2.COLOR_RGB2GRAY)
    disp_R_ref = refine_disp_right_local_search_ADcensus(
        left_gray, right_gray, disp_R_init, hit_mask,
        search_px=args.search_px, win=args.win, w_ad=0.5, w_census=0.5
    )

    save_and_viz(args.out_dir, right_rgb, warped_rgb, masked_rgb, hole_mask, disp_R_init, disp_R_ref)

if __name__ == "__main__":
    main()
