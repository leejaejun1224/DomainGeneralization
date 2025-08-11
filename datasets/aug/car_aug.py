#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
조건부 하단(0.7h~1.0h) glossy 패치 증강 스크립트
- 입력: 좌영상 디렉토리, 우영상 디렉토리, 좌 disparity(PFM) 디렉토리
- 조건:
  1) 패치 중심 y가 [0.7h, 1.0h] 범위에 위치
  2) 패치 크기: 높이=150±(size_jitter), 폭=300±(size_jitter)
  3) 패치 disparity: 70±10 (무작위)
  4) 선택한 위치의 배경 좌 disparity 중 '패치 disparity 보다 큰' 값이 존재하면 그 쌍은 PASS(증강하지 않음)
  5) 조건 만족 시에만 좌/우 이미지, 좌 disparity PFM, 좌 disparity PNG 저장

출력 구조( --out-root=/path/to/out ):
  /path/to/out/
    left/            (좌 이미지 증강)
    right/           (우 이미지 증강)
    disp_left/       (좌 disparity PFM 증강)
    disp_left_png/   (좌 disparity PNG 증강, 16-bit, disp*256)
"""

import os
import argparse
import glob
import math
from pathlib import Path
import numpy as np
import cv2
import random

# ------------------------- PFM IO -------------------------
def read_pfm(path):
    with open(path, "rb") as f:
        header = f.readline().decode("utf-8").rstrip()
        if header not in ("PF", "Pf"):
            raise ValueError(f"Not a PFM file: {path}")
        dims = f.readline().decode("utf-8")
        # skip comments
        while dims.startswith("#"):
            dims = f.readline().decode("utf-8")
        w, h = map(int, dims.strip().split())
        scale = float(f.readline().decode("utf-8").strip())
        endian = "<" if scale < 0 else ">"
        data = np.fromfile(f, endian + "f")
        if header == "PF":
            img = np.reshape(data, (h, w, 3))
        else:
            img = np.reshape(data, (h, w))
        img = np.flipud(img).astype(np.float32)
        return img

def write_pfm(path, image, scale=1.0):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    image = np.flipud(image).astype(np.float32)
    color = image.ndim == 3 and image.shape[2] == 3
    with open(path, "wb") as f:
        f.write(("PF\n" if color else "Pf\n").encode("utf-8"))
        f.write(f"{image.shape[1]} {image.shape[0]}\n".encode("utf-8"))
        f.write(f"-{abs(scale)}\n".encode("utf-8"))  # little-endian
        image.tofile(f)

def write_disp_png16(path, disp):
    """16-bit PNG로 저장(KITTI 관행: value = disp*256)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out = np.clip(disp * 256.0, 0, 65535).astype(np.uint16)
    cv2.imwrite(path, out)

# ----------------------- Utilities -----------------------
def gaussian_blur_field(field, sigma):
    k = max(1, int(sigma * 3) | 1)
    return cv2.GaussianBlur(field, (k, k), sigmaX=sigma, sigmaY=sigma,
                            borderType=cv2.BORDER_REFLECT)

def rounded_rect_alpha_opaque(h, w, radius_ratio=0.12, edge_feather_px=6):
    """내부 1.0(불투명), 테두리만 소프트 페더."""
    r = int(radius_ratio * min(h, w))
    r = max(2, r)
    alpha = np.zeros((h, w), np.float32)
    cv2.rectangle(alpha, (r, 0), (w - 1 - r, h - 1), 1, -1)
    cv2.rectangle(alpha, (0, r), (w - 1, h - 1 - r), 1, -1)
    for cx, cy in [(r, r), (w - 1 - r, r), (r, h - 1 - r), (w - 1 - r, h - 1 - r)]:
        cv2.circle(alpha, (cx, cy), r, 1, -1)
    if edge_feather_px > 0:
        k = int(edge_feather_px * 4) | 1
        soft = cv2.GaussianBlur(alpha, (k, k), edge_feather_px)
        inner = cv2.erode(alpha, cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * edge_feather_px + 1, 2 * edge_feather_px + 1)), 1)
        alpha = np.where(inner > 0.5, 1.0, soft)
    return np.clip(alpha, 0.0, 1.0).astype(np.float32)

def _make_low_freq_noise(h, w, rng, cells=2):
    """아주 저주파 노이즈(셀 수가 작을수록 굴곡 개수가 적음)."""
    if cells <= 0:
        return np.zeros((h, w, 1), np.float32)
    gh = max(2, int(max(2, h // (cells * 2))))
    gw = max(2, int(max(2, w // (cells * 2))))
    small = rng.randn(gh, gw).astype(np.float32)
    noise = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)[..., None]
    return noise

# ----------------------- Patch Synth ----------------------
def glossy_car_like_patch(w, h,
                          palette="gray",
                          base_gray=185,
                          base_bgr=(160, 170, 185),
                          seed=42,
                          noise_level=0.02,
                          noise_cells=2):
    """회색조, 매끈·광택 패치 생성."""
    rng = np.random.RandomState(seed)
    if palette == "gray":
        base = np.full((h, w, 3), float(base_gray), np.float32)
    else:
        base = np.tile(np.array(base_bgr, np.float32)[None, None, :], (h, w, 1))

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    nx = (xx - cx) / (0.65 * w)
    ny = (yy - cy) / (0.80 * h)

    radial = np.clip(1.0 - (nx * nx + 1.2 * ny * ny), 0.2, 1.0)
    radial = radial[..., None]

    angle = math.radians(20.0)
    band = ((xx - 0.15 * w) * math.cos(angle) + (yy - 0.25 * h) * math.sin(angle)) / w
    band = np.clip(1.0 - np.abs(band) * 2.0, 0.0, 1.0)
    band = gaussian_blur_field(band, max(w, h) * 0.06)[..., None]

    hlx, hly = 0.32 * w, 0.25 * h
    spec = np.exp(-(((xx - hlx) ** 2) / (0.08 * w * w) + ((yy - hly) ** 2) / (0.10 * h * h)))[..., None]

    if noise_level > 0:
        noise = _make_low_freq_noise(h, w, rng, cells=max(1, noise_cells))
        noise = gaussian_blur_field(noise, max(w, h) * 0.12)
        noise = noise * (noise_level * 255.0 * 0.04)
    else:
        noise = 0.0

    intensity = (0.78 + 0.20 * radial) + 0.18 * band + 0.22 * spec
    if intensity.ndim == 2:
        intensity = intensity[..., None]
    if not np.isscalar(noise) and noise.ndim == 2:
        noise = noise[..., None]

    img = base * intensity + noise
    img = np.clip(img, 0, 255).astype(np.uint8)

    alpha = rounded_rect_alpha_opaque(h, w, radius_ratio=0.14,
                                      edge_feather_px=max(3, int(min(h, w) * 0.02)))
    return img, alpha

def make_disparity_patch(w, h, d0=70.0, jitter=10.0, seed=42):
    rng = np.random.RandomState(seed + 1)
    var = rng.randn(h, w).astype(np.float32)
    var = gaussian_blur_field(var, max(w, h) * 0.10)
    var = (var - var.mean()) / (var.std() + 1e-6)
    var = var * (jitter * 0.6)

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    nx = (xx - (w - 1) / 2) / (w / 2)
    ny = (yy - (h - 1) / 2) / (h / 2)
    curvature = 0.15 * jitter * (0.6 * nx * nx + 0.4 * ny * ny)

    disp = d0 + var + curvature
    disp = np.clip(disp, d0 - jitter, d0 + jitter).astype(np.float32)
    return disp

# --------------------- Compositing ------------------------
def alpha_blend(dst, src, alpha, y0, x0):
    H, W = dst.shape[:2]
    h, w = src.shape[:2]
    y1, x1 = min(H, y0 + h), min(W, x0 + w)
    yy0, xx0 = max(0, y0), max(0, x0)
    sy0, sx0 = yy0 - y0, xx0 - x0
    if y1 <= yy0 or x1 <= xx0:
        return
    roi = dst[yy0:y1, xx0:x1, :].astype(np.float32)
    sroi = src[sy0:sy0 + (y1 - yy0), sx0:sx0 + (x1 - xx0), :].astype(np.float32)
    aroi = alpha[sy0:sy0 + (y1 - yy0), sx0:sx0 + (x1 - xx0)].astype(np.float32)[..., None]
    blended = (1 - aroi) * roi + aroi * sroi
    dst[yy0:y1, xx0:x1, :] = np.clip(blended, 0, 255).astype(np.uint8)


def forward_warp_to_right_canvas_splat(patch_bgr, alpha, disp, canvas_h, canvas_w, y0, x0):
    h, w = patch_bgr.shape[:2]
    canvas = np.zeros((canvas_h, canvas_w, 3), np.float32)
    wsum   = np.zeros((canvas_h, canvas_w), np.float32)

    for y in range(h):
        yl = y0 + y
        if yl < 0 or yl >= canvas_h:
            continue
        xl = x0 + np.arange(w, dtype=np.float32)
        xr = xl - disp[y]                  # 부동소수 타깃 x
        x0i = np.floor(xr).astype(np.int32)
        frac = xr - x0i                    # 0~1

        for off, weight in ((0, 1.0 - frac), (1, frac)):
            xp = x0i + off
            valid = (xp >= 0) & (xp < canvas_w) & (alpha[y] > 1e-3) & (weight > 1e-6)
            if not np.any(valid):
                continue
            a = (alpha[y, valid] * weight[valid]).astype(np.float32)
            c = patch_bgr[y, valid, :].astype(np.float32)
            idx = xp[valid]
            canvas[yl, idx, :] += (a[:, None] * c)
            wsum[yl, idx]      += a

    eps = 1e-6
    canvas = np.where(wsum[..., None] > eps, canvas / (wsum[..., None] + eps), 0.0)
    amask  = np.clip(wsum, 0.0, 1.0).astype(np.float32)

    # 경계만 약하게 부드럽게
    k = max(1, int(0.004 * max(canvas_h, canvas_w)) | 1)
    if k >= 3:
        amask = cv2.GaussianBlur(amask, (k, k), 0)

    return canvas.astype(np.uint8), amask, wsum


# ----------------------- Dataset helpers ------------------
IMG_EXTS = {".png", ".jpg", ".jpeg"}

def list_left_images(left_dir):
    files = []
    for ext in IMG_EXTS:
        files += glob.glob(os.path.join(left_dir, "**", f"*{ext}"), recursive=True)
    files = sorted(files)
    return files

def make_out_path(out_root, subdir, in_root, in_path, new_ext=None):
    rel = os.path.relpath(in_path, in_root)
    if new_ext is not None:
        rel = os.path.splitext(rel)[0] + new_ext
    out_path = os.path.join(out_root, subdir, rel)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    return out_path

# ---------------------------- Main ------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--left_dir", default="/home/jaejun/dataset/flyingthing/FlyingThings3D_subset_image_clean/FlyingThings3D_subset/train/image_clean/left_real")
    ap.add_argument("--right_dir", default="/home/jaejun/dataset/flyingthing/FlyingThings3D_subset_image_clean/FlyingThings3D_subset/train/image_clean/right_real")
    ap.add_argument("--disp_left_dir",default="/home/jaejun/dataset/flyingthing/FlyingThings3D_subset_disparity/FlyingThings3D_subset/train/disparity/left")
    ap.add_argument("--out_root", default="./augmented_data",)

    # 배치/난수
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--trials_per_image", type=int, default=20,
                    help="각 이미지에서 유효 위치를 찾기 위한 최대 시도 횟수")

    # 패치 위치/크기/모양
    ap.add_argument("--ymin_ratio", type=float, default=0.70, help="패치 중심 y 하한(비율)")
    ap.add_argument("--ymax_ratio", type=float, default=1.00, help="패치 중심 y 상한(비율)")
    ap.add_argument("--height_base", type=int, default=150)
    ap.add_argument("--width_base", type=int, default=300)
    ap.add_argument("--size_jitter", type=float, default=0.20, help="±비율(예: 0.2 ⇒ ±20%)")

    # disparity 조건
    ap.add_argument("--disp_mean", type=float, default=70.0)
    ap.add_argument("--disp_jitter", type=float, default=0.0)
    ap.add_argument("--zbuffer_margin", type=float, default=0.5,
                    help="배경 최대 disparity < (패치 disparity - margin) 이어야 통과")
    ap.add_argument("--disp_valid_min", type=float, default=0.1,
                    help="유효한 disparity 하한(0 이하면 무시)")

    # 패치 외관(회색, 매끈; 필요 시 조정)
    ap.add_argument("--palette", choices=["gray", "blue"], default="gray")
    ap.add_argument("--base_gray", type=int, default=215)
    ap.add_argument("--noise_level", type=float, default=0.0)
    ap.add_argument("--noise_cells", type=int, default=0)

    args = ap.parse_args()
    rng = np.random.RandomState(args.seed)

    left_files = list_left_images(args.left_dir)
    processed, skipped = 0, 0

    standard = 0.9
    for i, li in enumerate(left_files):
        rand_int = random.random()
        
        if rand_int < standard:
            continue
        
        rel = os.path.relpath(li, args.left_dir)
        ri = os.path.join(args.right_dir, rel)
        di = os.path.join(args.disp_left_dir, os.path.splitext(rel)[0] + ".pfm")

        if not (os.path.exists(ri) and os.path.exists(di)):
            skipped += 1
            continue

        L = cv2.imread(li, cv2.IMREAD_COLOR)
        R = cv2.imread(ri, cv2.IMREAD_COLOR)
        if L is None or R is None:
            skipped += 1
            continue
        H, W = L.shape[:2]
        # 읽기 직후 (대체)
        
        dL_raw = read_pfm(di)
        if dL_raw.ndim == 3:
            dL_raw = dL_raw[..., 0]

        # 유효 픽셀에서 부호 감지
        valid = np.isfinite(dL_raw) & (np.abs(dL_raw) > args.disp_valid_min)
        orig_sign = -1.0 if (np.median(dL_raw[valid]) < 0) else 1.0

        # 내부 표현은 항상 양수로 사용
        dL = dL_raw * orig_sign


        # 패치 크기 샘플
        h_base = args.height_base
        w_base = args.width_base
        jitter = args.size_jitter
        ph = int(round(h_base * rng.uniform(1 - jitter, 1 + jitter)))
        pw = int(round(w_base * rng.uniform(1 - jitter, 1 + jitter)))
        ph = max(20, min(ph, H // 2))         # 너무 크지 않게
        pw = max(40, min(pw, W - 20))

        # disparity 샘플
        d0 = float(rng.uniform(args.disp_mean - args.disp_jitter,
                               args.disp_mean + args.disp_jitter)) if hasattr(args, 'disp_mean') else float(rng.uniform(args.disp_mean - args.disp_jitter, args.disp_mean + args.disp_jitter))  # guard
        # 위 한 줄 가드(일부 셸에서 '-'가 파싱될 경우 대비)
        d0 = float(rng.uniform(args.disp_mean - args.disp_jitter,
                               args.disp_mean + args.disp_jitter))

        # 위치 찾기: 중심 y in [0.7h, 1.0h], x 임의. 조건(배경 disparity < d0 - margin) 만족 시 채택
        success = False
        for _ in range(args.trials_per_image):
            yc = int(rng.uniform(args.ymin_ratio * H, args.ymax_ratio * H))
            y0 = max(0, min(H - ph, yc - ph // 2))
            x0 = int(rng.uniform(0, max(1, W - pw)))
            # 배경 disparity 검사
            sub = dL[y0:y0 + ph, x0:x0 + pw]
            valid = np.isfinite(sub) & (sub > args.disp_valid_min)
            if not np.any(valid):
                continue
            max_bg = float(sub[valid].max())
            if max_bg < d0 - args.zbuffer_margin:                
                success = True
                break

        if not success:
            skipped += 1
            continue
        
        
        # 패치/마스크/disp 생성
        patch, alpha = glossy_car_like_patch(
            pw, ph,
            palette=args.palette,
            base_gray=args.base_gray,
            seed=int(rng.randint(0, 1 << 31)),
            noise_level=args.noise_level,
            noise_cells=args.noise_cells
        )
        disp_patch = make_disparity_patch(
            pw, ph, d0=d0, jitter=args.disp_jitter, seed=int(rng.randint(0, 1 << 31))
        )

        # 좌 합성 + disparity 갱신
        L_out = L.copy()
        alpha_blend(L_out, patch, alpha, y0, x0)

        dL_out = dL.copy()
        m = alpha > 1e-3
        yy0, xx0 = y0, x0
        yy1, xx1 = y0 + ph, x0 + pw
        sub_d = dL_out[yy0:yy1, xx0:xx1]
        sub_d[m] = disp_patch[m]
        dL_out[yy0:yy1, xx0:xx1] = sub_d

        # 우 합성(워프)
        canvasR, amaskR, _ = forward_warp_to_right_canvas_splat(
            patch, alpha, disp_patch, H, W, y0, x0
        )
        a3 = amaskR[..., None]
        R_out = np.clip((1 - a3) * R.astype(np.float32) + a3 * canvasR.astype(np.float32), 0, 255).astype(np.uint8)


        # 저장 경로
        out_left_path = make_out_path(args.out_root, "left", args.left_dir, li, new_ext=None)
        out_right_path = make_out_path(args.out_root, "right", args.right_dir, ri, new_ext=None)
        out_pfm_path = make_out_path(args.out_root, "disp_left", args.disp_left_dir, di, new_ext=".pfm")
        out_png_path = make_out_path(args.out_root, "disp_left_png", args.disp_left_dir,
                                     di, new_ext=".png")

        # 저장
        cv2.imwrite(out_left_path, L_out)
        cv2.imwrite(out_right_path, R_out)
        # 좌 disparity PFM: 원래 부호로 되돌려 저장
        write_pfm(out_pfm_path, (dL_out * orig_sign).astype(np.float32))
        # 좌 disparity PNG: 항상 양(절댓값)으로 저장
        write_disp_png16(out_png_path, np.abs(dL_out))

        processed += 1

    print(f"Done. processed={processed}, skipped={skipped}")

if __name__ == "__main__":
    main()