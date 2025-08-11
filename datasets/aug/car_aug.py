#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
조건부 하단(0.7h~1.0h) glossy 패치 증강 스크립트
- 입력: 좌영상 디렉토리, 우영상 디렉토리, 좌 disparity(PFM) 디렉토리
- 조건:
  1) 패치 중심 y가 [0.7h, 1.0h] 범위에 위치
  2) 패치 크기: 높이=150±(size_jitter), 폭=300±(size_jitter)
  3) 패치 disparity: 70±10 (무작위)
  4) 선택한 위치의 배경 좌 disparity 중 '패치 disparity 보다 큰' 값이 존재하면 PASS
  5) 통과 시에만 좌/우 이미지, 좌 disparity PFM, 좌 disparity PNG 저장

추가:
- 회색 밝기 30~250 랜덤
- 모양: rounded/ellipse/capsule/trapezoid 랜덤
- 각도: -18~+18° 랜덤 회전
- 위치: 좌/우 코너 무작위
"""

import os
import argparse
import glob
import math
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
        while dims.startswith("#"):
            dims = f.readline().decode("utf-8")
        w, h = map(int, dims.strip().split())
        scale = float(f.readline().decode("utf-8").strip())
        endian = "<" if scale < 0 else ">"
        data = np.fromfile(f, endian + "f")
        img = np.reshape(data, (h, w, 3)) if header == "PF" else np.reshape(data, (h, w))
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
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out = np.clip(disp * 256.0, 0, 65535).astype(np.uint16)
    cv2.imwrite(path, out)

# ----------------------- Utilities -----------------------
def gaussian_blur_field(field, sigma):
    k = max(1, int(sigma * 3) | 1)
    return cv2.GaussianBlur(field, (k, k), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT)

def opaque_from_binary(bin_mask, edge_feather_px=6):
    """0/1 바이너리 마스크 -> 내부 1.0, 가장자리만 페더링된 알파"""
    bin_u8 = (bin_mask > 0).astype(np.uint8) * 255
    if edge_feather_px > 0:
        k = int(edge_feather_px * 4) | 1
        soft = cv2.GaussianBlur(bin_u8.astype(np.float32) / 255.0, (k, k), edge_feather_px)
        inner = cv2.erode(bin_u8, cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * edge_feather_px + 1, 2 * edge_feather_px + 1)), 1)
        alpha = np.where(inner > 0, 1.0, soft)
    else:
        alpha = bin_u8.astype(np.float32) / 255.0
    return np.clip(alpha.astype(np.float32), 0.0, 1.0)

def make_shape_alpha_opaque(h, w, shape_kind="rounded", rng=None, edge_feather_px=6):
    if rng is None:
        rng = np.random.RandomState(0)
    bm = np.zeros((h, w), np.uint8)
    if shape_kind == "ellipse":
        cv2.ellipse(bm, (w//2, h//2), (w//2-1, h//2-1), 0, 0, 360, 255, -1)
    elif shape_kind == "capsule":
        r = max(2, int(min(h, w) * rng.uniform(0.18, 0.28)))
        cv2.rectangle(bm, (r, 0), (w-1-r, h-1), 255, -1)
        cv2.circle(bm, (r, h//2), r, 255, -1)
        cv2.circle(bm, (w-1-r, h//2), r, 255, -1)
    elif shape_kind == "trapezoid":
        top_ratio = rng.uniform(0.6, 0.95)
        dx = int((1 - top_ratio) * w * 0.5)
        poly = np.array([[dx, 0], [w-1-dx, 0], [w-1, h-1], [0, h-1]], np.int32)
        cv2.fillPoly(bm, [poly], 255)
    else:  # rounded
        r = int(rng.uniform(0.08, 0.22) * min(h, w))
        r = max(2, r)
        cv2.rectangle(bm, (r, 0), (w - 1 - r, h - 1), 255, -1)
        cv2.rectangle(bm, (0, r), (w - 1, h - 1 - r), 255, -1)
        for cx, cy in [(r, r), (w - 1 - r, r), (r, h - 1 - r), (w - 1 - r, h - 1 - r)]:
            cv2.circle(bm, (cx, cy), r, 255, -1)
    return opaque_from_binary(bm, edge_feather_px=edge_feather_px)

def rotate_triplet(patch, alpha, disp, angle_deg):
    """패치/알파/disp를 동일 각도로 회전하여 새 크기로 반환."""
    h, w = patch.shape[:2]
    c = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(c, angle_deg, 1.0)
    cos = abs(M[0, 0]); sin = abs(M[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)
    M[0, 2] += new_w / 2.0 - c[0]
    M[1, 2] += new_h / 2.0 - c[1]
    patch_r = cv2.warpAffine(patch, M, (new_w, new_h), flags=cv2.INTER_LINEAR, borderValue=0)
    alpha_r = cv2.warpAffine(alpha, M, (new_w, new_h), flags=cv2.INTER_LINEAR, borderValue=0)
    disp_r  = cv2.warpAffine(disp,  M, (new_w, new_h), flags=cv2.INTER_LINEAR, borderValue=0)
    # 회전 후 내부가 연해지는 문제를 보완(내부를 1.0에 가깝게)
    hard = (alpha_r > 0.5).astype(np.uint8) * 255
    inner = cv2.erode(hard, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), 1)
    soft  = cv2.GaussianBlur((hard / 255.0).astype(np.float32), (5, 5), 1.0)
    alpha_r = np.where(inner > 0, 1.0, soft).astype(np.float32)
    alpha_r = np.clip(alpha_r, 0.0, 1.0)
    return patch_r, alpha_r, disp_r

# ----------------------- Patch Synth ----------------------
def glossy_car_like_patch(w, h,
                          palette="gray",
                          base_gray=185,
                          base_bgr=(160, 170, 185),
                          seed=42,
                          noise_level=0.02,
                          noise_cells=2,
                          band_angle_deg=20.0,
                          shape_kind="rounded"):
    """회색조, 매끈·광택 패치 + 모양 선택."""
    rng = np.random.RandomState(seed)
    if palette == "gray":
        base = np.full((h, w, 3), float(base_gray), np.float32)
    else:
        base = np.tile(np.array(base_bgr, np.float32)[None, None, :], (h, w, 1))

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    nx = (xx - cx) / (0.65 * w)
    ny = (yy - cy) / (0.80 * h)

    radial = np.clip(1.0 - (nx * nx + 1.2 * ny * ny), 0.2, 1.0)[..., None]
    angle = math.radians(band_angle_deg)
    band = ((xx - 0.15 * w) * math.cos(angle) + (yy - 0.25 * h) * math.sin(angle)) / w
    band = np.clip(1.0 - np.abs(band) * 2.0, 0.0, 1.0)
    band = gaussian_blur_field(band, max(w, h) * 0.06)[..., None]

    hlx, hly = 0.32 * w, 0.25 * h
    spec = np.exp(-(((xx - hlx) ** 2) / (0.08 * w * w) + ((yy - hly) ** 2) / (0.10 * h * h)))[..., None]

    if noise_level > 0:
        gh = max(2, int(max(2, h // (max(1, noise_cells) * 2))))
        gw = max(2, int(max(2, w // (max(1, noise_cells) * 2))))
        small = rng.randn(gh, gw).astype(np.float32)
        noise = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)[..., None]
        noise = gaussian_blur_field(noise, max(w, h) * 0.12)
        noise = noise * (noise_level * 255.0 * 0.04)
    else:
        noise = 0.0

    intensity = (0.78 + 0.20 * radial) + 0.18 * band + 0.22 * spec
    if intensity.ndim == 2: intensity = intensity[..., None]
    if not np.isscalar(noise) and noise.ndim == 2: noise = noise[..., None]
    img = np.clip(base * intensity + noise, 0, 255).astype(np.uint8)

    edge_feather_px = max(3, int(min(h, w) * 0.02))
    if shape_kind == "random":
        shape_kind = random.choice(["rounded", "ellipse", "capsule", "trapezoid"])
    alpha = make_shape_alpha_opaque(h, w, shape_kind=shape_kind, rng=rng, edge_feather_px=edge_feather_px)
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
    disp = np.clip(d0 + var + curvature, d0 - jitter, d0 + jitter).astype(np.float32)
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
    """좌->우 forward warp: xR = xL - d (splat, 구멍 최소화)."""
    h, w = patch_bgr.shape[:2]
    canvas = np.zeros((canvas_h, canvas_w, 3), np.float32)
    wsum   = np.zeros((canvas_h, canvas_w), np.float32)

    for y in range(h):
        yl = y0 + y
        if yl < 0 or yl >= canvas_h:
            continue
        xl = x0 + np.arange(w, dtype=np.float32)
        xr = xl - disp[y]
        x0i = np.floor(xr).astype(np.int32)
        frac = xr - x0i
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
    return sorted(files)

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
    ap.add_argument("--trials_per_image", type=int, default=20)
    ap.add_argument("--aug_prob", type=float, default=0.1, help="이미지별 증강 적용 확률")

    # 패치 위치/크기/모양
    ap.add_argument("--ymin_ratio", type=float, default=0.70)
    ap.add_argument("--ymax_ratio", type=float, default=1.00)
    ap.add_argument("--height_base", type=int, default=150)
    ap.add_argument("--width_base",  type=int, default=300)
    ap.add_argument("--size_jitter", type=float, default=0.20)

    # 모양/각도/코너
    ap.add_argument("--shape", choices=["random","rounded","ellipse","capsule","trapezoid"], default="random")
    ap.add_argument("--rotate_deg_min", type=float, default=-18.0)
    ap.add_argument("--rotate_deg_max", type=float, default=18.0)
    ap.add_argument("--corner", choices=["random","left","right"], default="random")

    # 회색 밝기 범위
    ap.add_argument("--base_gray_min", type=int, default=30)
    ap.add_argument("--base_gray_max", type=int, default=250)

    # 반사 밴드 각도 범위
    ap.add_argument("--band_angle_min", type=float, default=-35.0)
    ap.add_argument("--band_angle_max", type=float, default=35.0)

    # disparity 조건
    ap.add_argument("--disp_mean", type=float, default=70.0)
    ap.add_argument("--disp_jitter", type=float, default=10.0)
    ap.add_argument("--zbuffer_margin", type=float, default=0.5)
    ap.add_argument("--disp_valid_min", type=float, default=0.1)

    # 외관(팔레트/노이즈)
    ap.add_argument("--palette", choices=["gray","blue"], default="gray")
    ap.add_argument("--noise_level", type=float, default=0.0)
    ap.add_argument("--noise_cells", type=int, default=0)

    args = ap.parse_args()
    rng = np.random.RandomState(args.seed)
    left_files = list_left_images(args.left_dir)
    processed, skipped = 0, 0

    for li in left_files:
        if random.random() >= args.aug_prob:
            continue

        rel = os.path.relpath(li, args.left_dir)
        ri = os.path.join(args.right_dir, rel)
        di = os.path.join(args.disp_left_dir, os.path.splitext(rel)[0] + ".pfm")
        if not (os.path.exists(ri) and os.path.exists(di)):
            skipped += 1; continue

        L = cv2.imread(li, cv2.IMREAD_COLOR)
        R = cv2.imread(ri, cv2.IMREAD_COLOR)
        if L is None or R is None:
            skipped += 1; continue
        H, W = L.shape[:2]

        # disparity 부호 자동 감지 → 내부 양수 사용
        dL_raw = read_pfm(di)
        if dL_raw.ndim == 3: dL_raw = dL_raw[..., 0]
        valid_sign = np.isfinite(dL_raw) & (np.abs(dL_raw) > args.disp_valid_min)
        if np.any(valid_sign):
            orig_sign = -1.0 if (np.median(dL_raw[valid_sign]) < 0) else 1.0
        else:
            orig_sign = 1.0
        dL = dL_raw * orig_sign

        success = False
        for _ in range(args.trials_per_image):
            # 크기, 밝기, 모양, 밴드각, 회전각, 코너 샘플
            ph = int(round(args.height_base * rng.uniform(1 - args.size_jitter, 1 + args.size_jitter)))
            pw = int(round(args.width_base  * rng.uniform(1 - args.size_jitter, 1 + args.size_jitter)))
            ph = max(20, min(ph, H // 2)); pw = max(40, min(pw, W - 20))
            base_gray_val = int(rng.randint(args.base_gray_min, args.base_gray_max + 1))
            band_angle = float(rng.uniform(args.band_angle_min, args.band_angle_max))
            shape_kind = args.shape if args.shape != "random" else "random"

            patch, alpha = glossy_car_like_patch(
                pw, ph,
                palette=args.palette,
                base_gray=base_gray_val,
                seed=int(rng.randint(0, 1 << 31)),
                noise_level=args.noise_level,
                noise_cells=args.noise_cells,
                band_angle_deg=band_angle,
                shape_kind=shape_kind
            )
            d0 = float(rng.uniform(args.disp_mean - args.disp_jitter, args.disp_mean + args.disp_jitter))
            disp_patch = make_disparity_patch(pw, ph, d0=d0, jitter=args.disp_jitter, seed=int(rng.randint(0, 1 << 31)))

            angle_deg = float(rng.uniform(args.rotate_deg_min, args.rotate_deg_max))
            patch_r, alpha_r, disp_r = rotate_triplet(patch, alpha, disp_patch, angle_deg)
            rh, rw = patch_r.shape[:2]

            # 코너 선택 및 y 범위 보정
            m = int(0.03 * min(H, W))
            side = args.corner
            if side == "random":
                side = "left" if rng.rand() < 0.5 else "right"
            x0 = m if side == "left" else W - rw - m
            # y중심이 범위 안에 들도록 yc를 직접 샘플
            y_low  = max(args.ymin_ratio * H, rh / 2 + m)
            y_high = min(args.ymax_ratio * H, H - rh / 2 - m)
            if y_low >= y_high:
                continue
            yc = rng.uniform(y_low, y_high)
            y0 = int(round(yc - rh / 2))
            if x0 < 0 or x0 + rw > W or y0 < 0 or y0 + rh > H:
                continue

            # 조건 4: 배경 disparity 체크(알파>0.2인 영역만)
            sub = dL[y0:y0 + rh, x0:x0 + rw]
            mask = (alpha_r > 0.2) & np.isfinite(sub) & (sub > args.disp_valid_min)
            if not np.any(mask):
                continue
            max_bg = float(sub[mask].max())
            if max_bg >= d0 - args.zbuffer_margin:
                continue  # 더 가까운 배경이 있으므로 PASS

            # 통과 → 합성
            L_out = L.copy()
            alpha_blend(L_out, patch_r, alpha_r, y0, x0)

            dL_out = dL.copy()
            sub_d = dL_out[y0:y0 + rh, x0:x0 + rw]
            sub_d[mask] = disp_r[mask]
            dL_out[y0:y0 + rh, x0:x0 + rw] = sub_d

            canvasR, amaskR, _ = forward_warp_to_right_canvas_splat(patch_r, alpha_r, disp_r, H, W, y0, x0)
            a3 = amaskR[..., None]
            R_out = np.clip((1 - a3) * R.astype(np.float32) + a3 * canvasR.astype(np.float32), 0, 255).astype(np.uint8)

            # 저장 경로
            out_left_path = make_out_path(args.out_root, "left", args.left_dir, li, new_ext=None)
            out_right_path = make_out_path(args.out_root, "right", args.right_dir, ri, new_ext=None)
            out_pfm_path = make_out_path(args.out_root, "disp_left", args.disp_left_dir, di, new_ext=".pfm")
            out_png_path = make_out_path(args.out_root, "disp_left_png", args.disp_left_dir, di, new_ext=".png")

            cv2.imwrite(out_left_path, L_out)
            cv2.imwrite(out_right_path, R_out)
            # PFM은 원부호로 복원, PNG는 abs로 저장
            write_pfm(out_pfm_path, (dL_out * orig_sign).astype(np.float32))
            write_disp_png16(out_png_path, np.abs(dL_out))

            processed += 1
            success = True
            break

        if not success:
            skipped += 1

    print(f"Done. processed={processed}, skipped={skipped}")

if __name__ == "__main__":
    main()

    
    
    
    
    