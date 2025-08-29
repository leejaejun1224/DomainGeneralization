import os
import argparse
from pathlib import Path

import numpy as np
import cv2
import matplotlib.pyplot as plt

# torch는 선택 사항 (없으면 OpenCV remap 사용)
try:
    import torch
    import torch.nn.functional as F
    _TORCH_AVAILABLE = True
except Exception:
    _TORCH_AVAILABLE = False

# ------------------------------------------------------------
# I/O
# ------------------------------------------------------------
def load_rgb(path):
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

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
        if arr.ndim == 3: arr = arr[...,0]
        return arr.astype(np.float32)
    elif ext == ".npy":
        arr = np.load(path)
        if arr.ndim == 3: arr = arr[...,0]
        return arr.astype(np.float32)
    elif ext == ".npz":
        npz = np.load(path)
        key = "disp" if "disp" in npz else ("disparity" if "disparity" in npz else list(npz.keys())[0])
        arr = npz[key]
        if arr.ndim == 3: arr = arr[...,0]
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
# Forward warping (Left -> Right) + Z-buffer
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
        right_disp[y, uniq_xr] = d_best       # 주의: 오른쪽 좌표계로 옮겨진 disparity (≈ d_R)
        hit_mask[y, uniq_xr] = 255

    return right_rgb, hit_mask, right_disp

def apply_black_mask(rgb, mask255):
    out = rgb.copy()
    out[mask255 == 0] = 0
    return out

# ------------------------------------------------------------
# Grid sampling 기반 L->R (disp_R 사용)
#   - PyTorch grid_sample 우선 사용
#   - 미설치 시 OpenCV remap으로 대체
# ------------------------------------------------------------
def _warp_left_to_right_grid_torch(left_rgb, disp_R, padding_mode='border'):
    """
    left_rgb: (H,W,3) uint8, disp_R: (H,W) float32 (오른쪽 좌표계 disparity)
    return: warped_rgb (H,W,3) uint8, valid_mask (H,W) uint8(0/255)
    """
    assert _TORCH_AVAILABLE, "PyTorch not available"
    H, W = disp_R.shape

    # to torch
    img = torch.from_numpy(left_rgb).permute(2,0,1).unsqueeze(0).float() / 255.0  # [1,3,H,W]
    dR  = torch.from_numpy(disp_R).unsqueeze(0).unsqueeze(0).float()              # [1,1,H,W]

    device = torch.device('cpu')
    img = img.to(device)
    dR  = dR.to(device)

    # base grid (pixel coords) -> 둘 다 [1,H,W]로 맞춘다
    ys, xs = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing='ij'
    )
    xs = xs.unsqueeze(0)  # [1,H,W]
    ys = ys.unsqueeze(0)  # [1,H,W]

    # 오른쪽 격자에서 왼쪽 소스 좌표: x_src = x_r + d_R
    x_src = xs + dR.squeeze(1)   # [1,H,W]
    y_src = ys                   # [1,H,W]

    # 정규화(-1~1), align_corners=True에 맞춘 식
    grid_x = 2.0 * (x_src / (W - 1.0)) - 1.0   # [1,H,W]
    grid_y = 2.0 * (y_src / (H - 1.0)) - 1.0   # [1,H,W]
    grid = torch.stack([grid_x, grid_y], dim=-1)  # [1,H,W,2]

    warped = F.grid_sample(img, grid, mode='bilinear',
                           padding_mode=padding_mode, align_corners=True)  # [1,3,H,W]

    valid_x = (grid[..., 0] >= -1.0) & (grid[..., 0] <= 1.0)
    valid_y = (grid[..., 1] >= -1.0) & (grid[..., 1] <= 1.0)
    valid = (valid_x & valid_y).unsqueeze(1).float()  # [1,1,H,W]

    warped_np = (warped.squeeze(0).permute(1,2,0).cpu().numpy() * 255.0).clip(0,255).astype(np.uint8)
    valid_np = (valid.squeeze(0).squeeze(0).cpu().numpy() * 255).astype(np.uint8)
    return warped_np, valid_np


def _warp_left_to_right_grid_cv2(left_rgb, disp_R):
    """
    OpenCV remap으로 L->R 역매핑.
    """
    H, W = disp_R.shape
    X = np.tile(np.arange(W, dtype=np.float32), (H,1))
    Y = np.tile(np.arange(H, dtype=np.float32).reshape(H,1), (1,W))
    mapx = X + disp_R.astype(np.float32)
    mapy = Y.astype(np.float32)
    warped = cv2.remap(left_rgb, mapx, mapy, interpolation=cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_REPLICATE)
    valid = (mapx >= 0) & (mapx <= (W-1)) & (mapy >= 0) & (mapy <= (H-1))
    return warped, (valid.astype(np.uint8) * 255)

def warp_left_to_right_grid(left_rgb, disp_R, padding_mode='border'):
    if _TORCH_AVAILABLE:
        return _warp_left_to_right_grid_torch(left_rgb, disp_R, padding_mode)
    else:
        return _warp_left_to_right_grid_cv2(left_rgb, disp_R)

# ------------------------------------------------------------
# 유틸: 시각화/저장
# ------------------------------------------------------------
def auto_vmin_vmax(disp):
    v = disp[np.isfinite(disp) & (disp > 0)]
    if v.size == 0:
        return 0.0, 1.0
    vmin = np.percentile(v, 1)
    vmax = np.percentile(v, 99)
    if vmax <= vmin: vmax = vmin + 1.0
    return float(vmin), float(vmax)

def l1_diff_gray(a_rgb, b_rgb):
    a = a_rgb.astype(np.float32)
    b = b_rgb.astype(np.float32)
    diff = np.abs(a - b).mean(axis=2)  # 채널 평균 L1
    # 보기 좋게 99퍼센타일 기준 정규화
    mx = np.percentile(diff, 99) if np.isfinite(diff).any() else 1.0
    mx = max(mx, 1e-6)
    return (np.clip(diff / mx, 0, 1) * 255).astype(np.uint8)

def save_and_viz_compare(out_dir,
                         right_rgb,
                         fw_rgb, fw_mask, disp_R_init,
                         grid_rgb, grid_mask):
    os.makedirs(out_dir, exist_ok=True)

    # 차이맵
    diff_fw  = l1_diff_gray(right_rgb, fw_rgb)
    diff_grid= l1_diff_gray(right_rgb, grid_rgb)

    plt.figure(figsize=(16,12))
    # 1행: GT / Forward / Grid / Disp_R
    plt.subplot(3,4,1); plt.imshow(right_rgb); plt.title("Right (GT)"); plt.axis("off")
    plt.subplot(3,4,2); plt.imshow(fw_rgb);    plt.title("Forward warp L→R (Z-buffer)"); plt.axis("off")
    plt.subplot(3,4,3); plt.imshow(grid_rgb);  plt.title("Grid sampling L→R (disp_R)"); plt.axis("off")
    vmin, vmax = auto_vmin_vmax(disp_R_init)
    im = plt.subplot(3,4,4); im = plt.imshow(disp_R_init, cmap="magma", vmin=vmin, vmax=vmax)
    plt.title("disp_R_init (from forward)"); plt.axis("off"); plt.colorbar(im, fraction=0.046, pad=0.04)

    # 2행: mask들 + masked 결과
    plt.subplot(3,4,5); plt.imshow(fw_mask, cmap="gray");   plt.title("Forward mask (hit 255)"); plt.axis("off")
    plt.subplot(3,4,6); plt.imshow(apply_black_mask(fw_rgb, fw_mask)); plt.title("Forward masked"); plt.axis("off")
    plt.subplot(3,4,7); plt.imshow(grid_mask, cmap="gray"); plt.title("Grid valid mask (255 in)"); plt.axis("off")
    plt.subplot(3,4,8); plt.imshow(apply_black_mask(grid_rgb, grid_mask)); plt.title("Grid masked"); plt.axis("off")

    # 3행: 차이맵 비교
    plt.subplot(3,4,9);  plt.imshow(diff_fw,   cmap="gray"); plt.title("|Right - Forward| (L1 avg)"); plt.axis("off")
    plt.subplot(3,4,10); plt.imshow(diff_grid, cmap="gray"); plt.title("|Right - Grid| (L1 avg)");    plt.axis("off")
    # 두 방법 차이
    method_gap = l1_diff_gray(fw_rgb, grid_rgb)
    plt.subplot(3,4,11); plt.imshow(method_gap, cmap="gray"); plt.title("|Forward - Grid|"); plt.axis("off")

    # 빈칸 하나는 설명
    plt.subplot(3,4,12); plt.axis("off"); plt.text(0.0, 0.5,
        "비고:\n- Forward: 소스 다수→타겟 1의 충돌을 Z-buffer로 해결\n- Grid: 타겟 1→소스 1 역매핑(bilinear)\n- occlusion/holes 처리가 상이",
        fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "compare_forward_vs_grid.png"), dpi=200)
    plt.close()

    # 개별 저장
    cv2.imwrite(os.path.join(out_dir, "right_gt.png"),
                cv2.cvtColor(right_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(out_dir, "forward_rgb.png"),
                cv2.cvtColor(fw_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(out_dir, "grid_rgb.png"),
                cv2.cvtColor(grid_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(out_dir, "forward_mask.png"), fw_mask)
    cv2.imwrite(os.path.join(out_dir, "grid_mask.png"), grid_mask)
    np.save(os.path.join(out_dir, "disp_right_init.npy"), disp_R_init)

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--left", default="/home/jaejun/dataset/kitti_2015/training/image_2/000020_10.png")
    ap.add_argument("--right", default="/home/jaejun/dataset/kitti_2015/training/image_3/000020_10.png")
    ap.add_argument("--disp_left", default="/home/jaejun/DomainGeneralization/log/2025-08-19_21_56_st/pseudo_disp/000020_10.png")
    ap.add_argument("--out_dir", default="./log/outputs_compare")
    ap.add_argument("--min_disp", type=float, default=0.0)
    args = ap.parse_args()

    left_rgb  = load_rgb(args.left)
    right_rgb = load_rgb(args.right)

    disp_L = load_disp_left(args.disp_left)
    disp_L = resize_disp_to((left_rgb.shape[0], left_rgb.shape[1]), disp_L)

    # 1) Forward warping (L→R, Z-buffer) 및 disp_R_init 생성
    fw_rgb, fw_hit_mask, disp_R_init = forward_warp_left_to_right(left_rgb, disp_L, min_disp=args.min_disp)

    # 2) Grid sampling으로 L→R (disp_R_init 사용)
    grid_rgb, grid_valid_mask = warp_left_to_right_grid(left_rgb, disp_R_init, padding_mode='border')

    # 3) 비교 시각화 저장
    save_and_viz_compare(args.out_dir,
                         right_rgb,
                         fw_rgb, fw_hit_mask, disp_R_init,
                         grid_rgb, grid_valid_mask)

    print(f"[OK] Saved comparison to: {args.out_dir}")

if __name__ == "__main__":
    main()
