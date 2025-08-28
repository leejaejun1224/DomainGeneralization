import os
import sys
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple

# ------------------------------
# I/O: disparity 로더 (PNG uint16, pfm, npy/npz 지원)
# ------------------------------
def read_pfm(file_path):
    """
    PFM(Portable Float Map) 로더.
    반환: (ndarray, scale). PFM의 좌표계는 보통 아래->위 이므로 이후 flipud 필요.
    """
    with open(file_path, "rb") as f:
        header = f.readline().decode("utf-8").rstrip()
        if header == "PF":
            color = True
        elif header == "Pf":
            color = False
        else:
            raise ValueError("Not a PFM file.")

        # 주석 줄 건너뜀
        dims = f.readline().decode("utf-8").strip()
        while dims.startswith("#"):
            dims = f.readline().decode("utf-8").strip()
        w, h = map(int, dims.split())

        scale = float(f.readline().decode("utf-8").strip())
        endian = "<" if scale < 0 else ">"
        count = w * h * (3 if color else 1)
        data = np.fromfile(f, endian + "f", count=count)
        if color:
            data = data.reshape((h, w, 3))
        else:
            data = data.reshape((h, w))
        return data, abs(scale)

def load_disparity(disp_path: str) -> np.ndarray:
    """
    disparity를 float32 (px)로 반환.
    - KITTI 16-bit PNG: 값 = disp(px) * 256 -> /256 적용
    - PFM: float 그대로, 상하반전 보정
    - NPY/NPZ: 저장된 float 그대로
    """
    disp_path = str(disp_path)
    ext = os.path.splitext(disp_path)[1].lower()

    if ext == ".pfm":
        data, _ = read_pfm(disp_path)
        disp = np.flipud(data).astype(np.float32)
        if disp.ndim == 3:
            disp = disp[..., 0].astype(np.float32)
        return disp

    if ext == ".npy":
        disp = np.load(disp_path).astype(np.float32)
        if disp.ndim == 3:
            disp = disp[..., 0].astype(np.float32)
        return disp

    if ext == ".npz":
        npz = np.load(disp_path)
        # 흔한 키 우선
        key = "disp" if "disp" in npz else ("disparity" if "disparity" in npz else list(npz.keys())[0])
        disp = npz[key].astype(np.float32)
        if disp.ndim == 3:
            disp = disp[..., 0].astype(np.float32)
        return disp

    # 이미지 파일 (예: KITTI 16-bit PNG)
    raw = cv2.imread(disp_path, cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise FileNotFoundError(f"Cannot read disparity: {disp_path}")

    if raw.ndim == 3:
        raw = raw[..., 0]  # 채널이 있다면 첫 채널 사용

    if raw.dtype == np.uint16:
        disp = raw.astype(np.float32) / 256.0
    else:
        disp = raw.astype(np.float32)  # 이미 px 단위라고 가정
    return disp

def load_left_rgb(img_path: str) -> np.ndarray:
    bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb

# ------------------------------
# 워핑 (Left -> Right), Z-buffer
# ------------------------------
def warp_left_to_right_zbuffer(left_rgb: np.ndarray, disp: np.ndarray, valid_disp_min: float = 0.0):
    """
    forward warping + Z-buffer(큰 disparity 우선)로 오른쪽 시점 합성.
    반환: warped_rgb (H,W,3), hit_mask (H,W) 255=채움, 0=hole
    """
    H, W = disp.shape
    if left_rgb.shape[:2] != (H, W):
        raise ValueError("left image and disparity must have the same size.")

    warped = np.zeros_like(left_rgb)
    hit_mask = np.zeros((H, W), dtype=np.uint8)

    # 행 단위로 처리
    for y in range(H):
        xl = np.arange(W, dtype=np.int32)
        drow = disp[y]
        # 유효 disparity
        valid = (drow > valid_disp_min) & np.isfinite(drow)
        if not np.any(valid):
            continue

        xl_v = xl[valid]
        d_v = drow[valid]
        xr_f = xl_v.astype(np.float32) - d_v
        xr_i = np.rint(xr_f).astype(np.int32)

        in_bounds = (xr_i >= 0) & (xr_i < W)
        if not np.any(in_bounds):
            continue

        xl_v = xl_v[in_bounds]
        xr_i = xr_i[in_bounds]
        d_v = d_v[in_bounds]

        # 같은 xr_i에 여러 소스가 매핑되면 disparity가 큰(가까운) 것을 남김
        # xr_i 오름차순, disparity 내림차순으로 정렬 후 첫 항목 선택
        order = np.lexsort((-d_v, xr_i))
        xr_sorted = xr_i[order]
        xl_sorted = xl_v[order]
        # unique로 같은 목적지 픽셀 중 첫 항목(=max disp)을 취함
        uniq_xr, idx_first = np.unique(xr_sorted, return_index=True)
        xl_best = xl_sorted[idx_first]
        xr_best = uniq_xr

        warped[y, xr_best] = left_rgb[y, xl_best]
        hit_mask[y, xr_best] = 255

    return warped, hit_mask

def inpaint_holes_rgb(warped_rgb: np.ndarray, hit_mask: np.ndarray, radius: int = 3) -> np.ndarray:
    """
    OpenCV Telea로 hole inpaint.
    hit_mask==0 인 영역을 메움.
    """
    bgr = cv2.cvtColor(warped_rgb, cv2.COLOR_RGB2BGR)
    hole = (hit_mask == 0).astype(np.uint8) * 255
    # 가장자리 경계 디테일 확보를 위해 hole을 소폭 팽창
    kernel = np.ones((3, 3), np.uint8)
    hole = cv2.dilate(hole, kernel, iterations=1)
    inpainted = cv2.inpaint(bgr, hole, radius, cv2.INPAINT_TELEA)
    return cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB)

# ------------------------------
# 도우미: 크기 불일치 시 disparity 스케일 보정
# ------------------------------
def resize_disparity_to_match(disp: np.ndarray, target_HW: Tuple[int, int]) -> np.ndarray:
    """
    disparity는 'px' 단위 값이므로 가로 스케일 변화량에 맞게 값도 함께 스케일해야 함.
    """
    Ht, Wt = target_HW
    H0, W0 = disp.shape
    if (H0, W0) == (Ht, Wt):
        return disp
    disp_rs = cv2.resize(disp, (Wt, Ht), interpolation=cv2.INTER_LINEAR)
    scale_x = Wt / float(W0)
    return disp_rs * scale_x

# ------------------------------
# 시각화
# ------------------------------
def auto_vmin_vmax(disp: np.ndarray):
    v = disp[np.isfinite(disp) & (disp > 0)]
    if v.size == 0:
        return 0.0, 1.0
    vmin = np.percentile(v, 1)
    vmax = np.percentile(v, 99)
    if vmax <= vmin:
        vmax = vmin + 1.0
    return float(vmin), float(vmax)

def visualize_and_save(left_rgb, disp, warped_rgb, filled_rgb, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    vmin, vmax = auto_vmin_vmax(disp)

    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.imshow(left_rgb)
    plt.title("Left image")
    plt.axis("off")

    plt.subplot(2, 2, 2)
    im = plt.imshow(disp, cmap="magma", vmin=vmin, vmax=vmax)
    plt.title("Disparity (px)")
    plt.axis("off")
    plt.colorbar(im, fraction=0.046, pad=0.04)

    plt.subplot(2, 2, 3)
    plt.imshow(warped_rgb)
    plt.title("Warped L→R (Z-Buffer)")
    plt.axis("off")

    plt.subplot(2, 2, 4)
    plt.imshow(filled_rgb)
    plt.title("Warped + Inpaint")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "viz_left2right.png"), dpi=200)
    # 결과 이미지 저장
    cv2.imwrite(os.path.join(out_dir, "warped_right.png"),
                cv2.cvtColor(warped_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(out_dir, "warped_right_inpaint.png"),
                cv2.cvtColor(filled_rgb, cv2.COLOR_RGB2BGR))
    print(f"[Saved] {os.path.join(out_dir, 'viz_left2right.png')}")
    print(f"[Saved] {os.path.join(out_dir, 'warped_right.png')}")
    print(f"[Saved] {os.path.join(out_dir, 'warped_right_inpaint.png')}")
    plt.show()

# ------------------------------
# 메인
# ------------------------------
def main():
    parser = argparse.ArgumentParser(description="Warp KITTI left image to right view using disparity.")
    parser.add_argument("--left", type=str, default="/home/jaejun/dataset/kitti_2015/training/image_2/000020_10.png")
    parser.add_argument("--disp", type=str, default="/home/jaejun/DomainGeneralization/log/2025-08-19_21_56_st/pseudo_disp/000020_10.png")
    parser.add_argument("--out_dir", type=str, default="./outputs", help="Directory to save results")
    parser.add_argument("--min_disp", type=float, default=0.0, help="Minimum valid disparity (px). Use >0 to filter noise.")
    args = parser.parse_args()

    left_rgb = load_left_rgb(args.left)
    disp = load_disparity(args.disp)

    # 크기 맞추기 + 스케일 보정
    H, W = left_rgb.shape[:2]
    disp = resize_disparity_to_match(disp, (H, W))

    warped_rgb, hit_mask = warp_left_to_right_zbuffer(left_rgb, disp, valid_disp_min=args.min_disp)
    inpaint_rgb = inpaint_holes_rgb(warped_rgb, hit_mask, radius=3)

    visualize_and_save(left_rgb, disp, warped_rgb, inpaint_rgb, args.out_dir)

if __name__ == "__main__":
    # 예: python warp_kitti_left2right.py --left 000000_10.png --disp 000000_10.png --out_dir ./outputs
    main()
