# car_patch_module.py
# -*- coding: utf-8 -*-
import math
import numpy as np
import cv2

# ----------------------- 저수준 유틸 -----------------------
def _gaussian_blur_field(field, sigma):
    k = max(1, int(sigma * 3) | 1)
    return cv2.GaussianBlur(field, (k, k), sigmaX=sigma, sigmaY=sigma,
                            borderType=cv2.BORDER_REFLECT)

def _opaque_from_binary(bin_mask, edge_feather_px=6):
    bm = (bin_mask > 0).astype(np.uint8) * 255
    if edge_feather_px > 0:
        k = int(edge_feather_px * 4) | 1
        soft = cv2.GaussianBlur(bm.astype(np.float32) / 255.0, (k, k), edge_feather_px)
        inner = cv2.erode(bm, cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * edge_feather_px + 1, 2 * edge_feather_px + 1)), 1)
        alpha = np.where(inner > 0, 1.0, soft)
    else:
        alpha = bm.astype(np.float32) / 255.0
    return np.clip(alpha.astype(np.float32), 0.0, 1.0)

def _make_shape_alpha_opaque(h, w, shape_kind="rounded", rng=None, edge_feather_px=6):
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
    return _opaque_from_binary(bm, edge_feather_px=edge_feather_px)

def _rotate_triplet(patch, alpha, disp, angle_deg):
    """패치/알파/disp를 동일 회전."""
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
    # 회전 후 내부 진하게
    hard = (alpha_r > 0.5).astype(np.uint8) * 255
    inner = cv2.erode(hard, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), 1)
    soft  = cv2.GaussianBlur((hard / 255.0).astype(np.float32), (5, 5), 1.0)
    alpha_r = np.where(inner > 0, 1.0, soft).astype(np.float32)
    return patch_r, alpha_r, disp_r

def _alpha_blend(dst, src, alpha, y0, x0):
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

def _forward_warp_splat_to_right(patch_bgr, alpha, disp, canvas_h, canvas_w, y0, x0):
    """좌->우 splat 워핑: xR = xL - d. 구멍/점선 최소화."""
    h, w = patch_bgr.shape[:2]
    canvas = np.zeros((canvas_h, canvas_w, 3), np.float32)
    wsum   = np.zeros((canvas_h, canvas_w), np.float32)
    for y in range(h):
        yl = y0 + y
        if yl < 0 or yl >= canvas_h:
            continue
        xl = x0 + np.arange(w, dtype=np.float32)
        xr = xl - disp[y]                # 부동소수
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
    # 경계만 약한 블러
    k = max(1, int(0.004 * max(canvas_h, canvas_w)) | 1)
    if k >= 3:
        amask = cv2.GaussianBlur(amask, (k, k), 0)
    return canvas.astype(np.uint8), amask

# ----------------------- 패치 합성용 생성기 -----------------------
def _glossy_patch(w, h, base_gray=185, band_angle_deg=20.0,
                  noise_level=0.0, noise_cells=0, shape_kind="rounded", rng=None):
    rng = rng or np.random.RandomState(42)
    base = np.full((h, w, 3), float(base_gray), np.float32)

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    nx = (xx - cx) / (0.65 * w)
    ny = (yy - cy) / (0.80 * h)

    radial = np.clip(1.0 - (nx * nx + 1.2 * ny * ny), 0.2, 1.0)[..., None]
    angle = math.radians(band_angle_deg)
    band = ((xx - 0.15 * w) * math.cos(angle) + (yy - 0.25 * h) * math.sin(angle)) / w
    band = np.clip(1.0 - np.abs(band) * 2.0, 0.0, 1.0)
    band = _gaussian_blur_field(band, max(w, h) * 0.06)[..., None]

    hlx, hly = 0.32 * w, 0.25 * h
    spec = np.exp(-(((xx - hlx) ** 2) / (0.08 * w * w) + ((yy - hly) ** 2) / (0.10 * h * h)))[..., None]

    if noise_level > 0:
        gh = max(2, int(max(2, h // (max(1, noise_cells) * 2))))
        gw = max(2, int(max(2, w // (max(1, noise_cells) * 2))))
        small = rng.randn(gh, gw).astype(np.float32)
        noise = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)[..., None]
        noise = _gaussian_blur_field(noise, max(w, h) * 0.12)
        noise = noise * (noise_level * 255.0 * 0.04)
    else:
        noise = 0.0

    intensity = (0.78 + 0.20 * radial) + 0.18 * band + 0.22 * spec
    if intensity.ndim == 2: intensity = intensity[..., None]
    if not np.isscalar(noise) and noise.ndim == 2: noise = noise[..., None]
    img = np.clip(base * intensity + noise, 0, 255).astype(np.uint8)

    edge_feather_px = max(3, int(min(h, w) * 0.02))
    alpha = _make_shape_alpha_opaque(h, w, shape_kind=shape_kind, rng=rng, edge_feather_px=edge_feather_px)
    return img, alpha

def _make_disp_patch(w, h, d0=70.0, jitter=10.0, rng=None):
    rng = rng or np.random.RandomState(0)
    var = rng.randn(h, w).astype(np.float32)
    var = _gaussian_blur_field(var, max(w, h) * 0.10)
    var = (var - var.mean()) / (var.std() + 1e-6)
    var = var * (jitter * 0.6)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    nx = (xx - (w - 1) / 2) / (w / 2)
    ny = (yy - (h - 1) / 2) / (h / 2)
    curvature = 0.15 * jitter * (0.6 * nx * nx + 0.4 * ny * ny)
    disp = np.clip(d0 + var + curvature, d0 - jitter, d0 + jitter).astype(np.float32)
    return disp

# ----------------------- 외부용 클래스 -----------------------
class CarPatchAugmenter:
    """
    FlyingThings3D 좌기준 스테레오쌍에 glossy 저텍스처 패치를 조건부 삽입.
    반환 시 입력 disparity의 '원래 부호'를 유지.
    """
    def __init__(self,
                 aug_prob=0.1,
                 ymin_ratio=0.70, ymax_ratio=1.00,
                 height_base=150, width_base=300, size_jitter=0.20,
                 disp_mean=70.0, disp_jitter=10.0,
                 zbuffer_margin=0.5, disp_valid_min=0.1,
                 base_gray_range=(30, 250),
                 shape='random',                 # 'random'|'rounded'|'ellipse'|'capsule'|'trapezoid'
                 rotate_deg_range=(-18.0, 18.0),
                 corner='random',                # 'random'|'left'|'right'
                 noise_level=0.0, noise_cells=0,
                 seed=42):
        self.aug_prob = float(aug_prob)
        self.ymin_ratio = float(ymin_ratio)
        self.ymax_ratio = float(ymax_ratio)
        self.height_base = int(height_base)
        self.width_base  = int(width_base)
        self.size_jitter = float(size_jitter)
        self.disp_mean = float(disp_mean)
        self.disp_jitter = float(disp_jitter)
        self.zbuffer_margin = float(zbuffer_margin)
        self.disp_valid_min = float(disp_valid_min)
        self.base_gray_range = (int(base_gray_range[0]), int(base_gray_range[1]))
        self.shape = shape
        self.rotate_deg_range = (float(rotate_deg_range[0]), float(rotate_deg_range[1]))
        self.corner = corner
        self.noise_level = float(noise_level)
        self.noise_cells = int(noise_cells)
        self.rng = np.random.RandomState(seed)

    def __call__(self, imgL, imgR, disp):
        """
        입력:
          imgL, imgR: np.uint8 (H,W,3), RGB
          disp: np.float32 (H,W), 좌기준 dispariy (부호 상관없음)
        출력:
          imgL_out, imgR_out, disp_out (disp_out은 입력 부호를 유지)
        """
        if self.rng.rand() >= self.aug_prob:
            return imgL, imgR, disp  # 미적용

        H, W = imgL.shape[:2]
        if H < 32 or W < 64:
            return imgL, imgR, disp

        # 부호 자동 감지 → 내부는 양수로 변환
        d_raw = disp.astype(np.float32)
        valid = np.isfinite(d_raw) & (np.abs(d_raw) > self.disp_valid_min)
        orig_sign = -1.0 if (np.any(valid) and np.median(d_raw[valid]) < 0) else 1.0
        dL = d_raw * orig_sign  # 내부 양수

        # 패치 크기/밝기/모양/회전 샘플
        ph = int(round(self.height_base * self.rng.uniform(1 - self.size_jitter, 1 + self.size_jitter)))
        pw = int(round(self.width_base  * self.rng.uniform(1 - self.size_jitter, 1 + self.size_jitter)))
        ph = max(20, min(ph, H // 2)); pw = max(40, min(pw, W - 20))
        base_gray = int(self.rng.randint(self.base_gray_range[0], self.base_gray_range[1] + 1))
        band_angle = float(self.rng.uniform(-35.0, 35.0))
        shape_kind = self.shape if self.shape != "random" else self.rng.choice(["rounded","ellipse","capsule","trapezoid"])
        d0 = float(self.rng.uniform(self.disp_mean - self.disp_jitter, self.disp_mean + self.disp_jitter))
        rot_angle = float(self.rng.uniform(self.rotate_deg_range[0], self.rotate_deg_range[1]))

        patch, alpha = _glossy_patch(pw, ph, base_gray=base_gray,
                                     band_angle_deg=band_angle,
                                     noise_level=self.noise_level,
                                     noise_cells=self.noise_cells,
                                     shape_kind=shape_kind,
                                     rng=self.rng)
        disp_patch = _make_disp_patch(pw, ph, d0=d0, jitter=self.disp_jitter, rng=self.rng)
        patch_r, alpha_r, disp_r = _rotate_triplet(patch, alpha, disp_patch, rot_angle)
        rh, rw = patch_r.shape[:2]

        # 좌/우 코너 및 y 범위 샘플
        m = int(0.03 * min(H, W))
        side = self.corner
        if side == "random":
            side = "left" if self.rng.rand() < 0.5 else "right"
        x0 = m if side == "left" else W - rw - m
        y_low  = max(self.ymin_ratio * H, rh / 2 + m)
        y_high = min(self.ymax_ratio * H, H - rh / 2 - m)
        if not (y_low < y_high) or x0 < 0 or x0 + rw > W:
            return imgL, imgR, disp  # 범위 불가 → 미적용
        yc = self.rng.uniform(y_low, y_high)
        y0 = int(round(yc - rh / 2))

        # 조건 4: 선택 위치에 더 큰 disparity(가까운 배경)가 있으면 PASS
        sub = dL[y0:y0 + rh, x0:x0 + rw]
        mask = (alpha_r > 0.2) & np.isfinite(sub) & (sub > self.disp_valid_min)
        if not np.any(mask):
            return imgL, imgR, disp
        max_bg = float(sub[mask].max())
        if max_bg >= d0 - self.zbuffer_margin:
            return imgL, imgR, disp  # 미적용

        # 합성
        L_out = imgL.copy()
        _alpha_blend(L_out, patch_r, alpha_r, y0, x0)

        dL_out = dL.copy()
        subd = dL_out[y0:y0 + rh, x0:x0 + rw]
        subd[mask] = disp_r[mask]
        dL_out[y0:y0 + rh, x0:x0 + rw] = subd

        canvasR, amaskR = _forward_warp_splat_to_right(patch_r, alpha_r, disp_r, H, W, y0, x0)
        a3 = amaskR[..., None]
        R_out = np.clip((1 - a3) * imgR.astype(np.float32) + a3 * canvasR.astype(np.float32), 0, 255).astype(np.uint8)

        # 원래 부호로 복원
        disp_out = dL_out * orig_sign
        return L_out, R_out, disp_out
