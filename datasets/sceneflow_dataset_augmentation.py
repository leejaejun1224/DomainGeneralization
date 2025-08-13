import os
import random
from PIL import Image, ImageFile
from torch.utils.data import Dataset
import numpy as np
import cv2
from datasets.data_io import get_transform, read_all_lines, pfm_imread
import torchvision.transforms as transforms  # 호환 유지
import torch
import math
from datasets.aug.car_patch_modeule import CarPatchAugmenter

ImageFile.LOAD_TRUNCATED_IMAGES = True


# =========================================================
# 보조 유틸 (로버스트 특성 + 마스크)
# =========================================================
def _normalize01(a, eps=1e-6):
    a = a.astype(np.float32)
    mn, mx = a.min(), a.max()
    return (a - mn) / (mx - mn + eps)

def rank_transform(gray, win=7):
    """윈도우 내 현재 픽셀보다 작은 픽셀 비율(0~1). 밝기/감마 변화에 강함."""
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_RGB2GRAY)
    gray = gray.astype(np.float32)
    k = win // 2
    pad = cv2.copyMakeBorder(gray, k, k, k, k, cv2.BORDER_REFLECT)
    h, w = gray.shape
    out = np.zeros_like(gray, dtype=np.float32)
    for dy in range(win):
        for dx in range(win):
            patch = pad[dy:dy+h, dx:dx+w]
            out += (patch < gray).astype(np.float32)
    out /= (win * win)
    return out

def log_magnitude(img, sigma=1.0):
    """LoG 응답 절댓값 정규화."""
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_blur = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma)
    log = cv2.Laplacian(img_blur, cv2.CV_32F, ksize=3)
    return _normalize01(np.abs(log))

def structure_tensor_coherence(img, sigma=2.0, blur=3):
    """coherence=(λ1-λ2)/(λ1+λ2+eps): 0~1, 평탄/에지 구분."""
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    Ix = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    Iy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    Ixx = cv2.GaussianBlur(Ix * Ix, (blur | 1, blur | 1), sigma)
    Iyy = cv2.GaussianBlur(Iy * Iy, (blur | 1, blur | 1), sigma)
    Ixy = cv2.GaussianBlur(Ix * Iy, (blur | 1, blur | 1), sigma)
    tmp = np.sqrt((Ixx - Iyy) ** 2 + 4.0 * (Ixy ** 2))
    l1 = 0.5 * (Ixx + Iyy + tmp)
    l2 = 0.5 * (Ixx + Iyy - tmp)
    coh = (l1 - l2) / (l1 + l2 + 1e-6)
    return _normalize01(coh)

def gradient_magnitude_rgb(img):
    dx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(np.sum(dx * dx, axis=-1) + np.sum(dy * dy, axis=-1))
    return _normalize01(mag)

def _left_valid_mask_signed(disp, negate=False):
    """
    Left 기준 유효 마스크. negate=True면 내부적으로 부호를 뒤집어(+ disparity) 좌표 체크.
    """
    dd = (-disp if negate else disp).astype(np.float32)
    h, w = dd.shape
    xs = np.arange(w, dtype=np.float32)[None, :].repeat(h, axis=0)
    valid = (dd > 0.0) & (xs - dd >= 0.0) & (xs - dd < w)
    return valid.astype(np.uint8)


# =========================================================
# StereoAugmentor (강증강)
# =========================================================
class StereoAugmentor:
    """
    KITTI/자체 도메인(바닥/유리/야간/날씨/센서 차이)에 강한 스테레오 증강.
    - 기하 유지: 좌우 동일 크롭/리사이즈/수평 스케일만 사용(수평 스케일은 disp도 동일 배율 적용)
    - 좌/우 비대칭 포토메트릭: 감마/노출/화이트밸런스/노이즈/블러/JPEG 등
    - 유리(상부) 반사/플레어/편광 근사, 바닥(하부) 젖은 도로/거울, 그림자, 비/안개
    """

    def __init__(self,
                 p_hflip=0.0,
                 p_hscale=0.0,
                 hscale_range=(0.9, 1.1),
                 p_basic=1.0,
                 p_noise=0.7,
                 p_blur=0.5,
                 p_jpeg=0.5,
                 p_vignette=0.5,
                 p_color_shade=0.5,
                 p_shadow=0.6,
                 p_fog=0.2,
                 p_rain=0.2,
                 p_glass=0.4,
                 p_wet=0.2,
                 p_lowtex=0.4,
                 seed=None):
        self.p_hflip = p_hflip
        self.p_hscale = p_hscale
        self.hscale_range = hscale_range
        self.p_basic = p_basic
        self.p_noise = p_noise
        self.p_blur = p_blur
        self.p_jpeg = p_jpeg
        self.p_vignette = p_vignette
        self.p_color_shade = p_color_shade
        self.p_shadow = p_shadow
        self.p_fog = p_fog
        self.p_rain = p_rain
        self.p_glass = p_glass
        self.p_wet = p_wet
        self.p_lowtex = p_lowtex
        self.rng = np.random.RandomState(seed if seed is not None else np.random.randint(0, 2**31 - 1))

    # ---------- 유틸 ----------
    @staticmethod
    def _to_float(img):
        return img.astype(np.float32) / 255.0

    @staticmethod
    def _to_uint8(imgf):
        return np.clip(imgf * 255.0 + 0.5, 0, 255).astype(np.uint8)

    def _gamma(self, img, gamma):
        imgf = self._to_float(img)
        imgf = np.power(np.clip(imgf, 0.0, 1.0), gamma)
        return self._to_uint8(imgf)

    def _brightness(self, img, scale):
        imgf = self._to_float(img)
        imgf = imgf * scale
        return self._to_uint8(imgf)

    def _contrast(self, img, scale):
        imgf = self._to_float(img)
        mean = imgf.mean(axis=(0, 1), keepdims=True)
        imgf = (imgf - mean) * scale + mean
        return self._to_uint8(imgf)

    def _saturation(self, img, scale):
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * scale, 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    def _white_balance(self, img, r_gain, b_gain):
        imgf = self._to_float(img)
        imgf[..., 0] *= r_gain   # R
        imgf[..., 2] *= b_gain   # B
        return self._to_uint8(imgf)

    def _gaussian_noise(self, img, sigma):
        imgf = self._to_float(img)
        noise = self.rng.normal(0.0, sigma, imgf.shape).astype(np.float32)
        return self._to_uint8(imgf + noise)

    def _defocus_blur(self, img, k):
        if k <= 1:
            return img
        return cv2.GaussianBlur(img, (k | 1, k | 1), 0)

    def _motion_blur(self, img, k, angle):
        if k <= 1:
            return img
        kernel = np.zeros((k, k), dtype=np.float32)
        kernel[k // 2, :] = 1.0
        M = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), angle, 1.0)
        kernel = cv2.warpAffine(kernel, M, (k, k))
        kernel = kernel / (kernel.sum() + 1e-6)
        return cv2.filter2D(img, -1, kernel)

    def _jpeg(self, img, q):
        ok, enc = cv2.imencode('.jpg', cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), q])
        if not ok:
            return img
        dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
        return cv2.cvtColor(dec, cv2.COLOR_BGR2RGB)

    def _vignette(self, img, strength):
        h, w = img.shape[:2]
        yy, xx = np.indices((h, w), dtype=np.float32)
        cx, cy = w * 0.5, h * 0.5
        r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        r = r / (r.max() + 1e-6)
        mask = 1.0 - strength * (r ** 2)
        imgf = self._to_float(img)
        imgf *= mask[..., None]
        return self._to_uint8(imgf)

    def _color_shading(self, img, gains):
        imgf = self._to_float(img)
        imgf[..., 0] *= gains[0]
        imgf[..., 1] *= gains[1]
        imgf[..., 2] *= gains[2]
        return self._to_uint8(imgf)

    def _shadow(self, img, n_poly=1, min_v=0.4, max_v=0.85):
        h, w = img.shape[:2]
        overlay = img.copy()
        for _ in range(n_poly):
            n = self.rng.randint(3, 8)
            pts = np.stack([self.rng.randint(0, w, n), self.rng.randint(0, h, n)], axis=1).astype(np.int32)
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [pts], 255)
            v = self.rng.uniform(min_v, max_v)
            overlay[mask == 255] = (overlay[mask == 255].astype(np.float32) * v).astype(np.uint8)
        return overlay

    def _fog(self, img, beta, A, vertical=True):
        """
        대기 산란 근사: I = I*t + A*(1-t), t = exp(-beta * d)
        """
        h, w = img.shape[:2]
        imgf = self._to_float(img).astype(np.float32)

        if vertical:
            # (H,1,1)로 확장 → (H,W,3)에 브로드캐스트
            y = np.linspace(1.0, 0.0, h, dtype=np.float32)
            t = np.exp(-beta * y).astype(np.float32)[:, None, None]
        else:
            # (1,W,1)로 확장
            x = np.linspace(0.0, 1.0, w, dtype=np.float32)
            t = np.exp(-beta * x).astype(np.float32)[None, :, None]

        A = np.float32(A)
        out = imgf * t + A * (1.0 - t)
        return self._to_uint8(out)

    def _rain(self, img, density, length, angle, intensity):
        h, w = img.shape[:2]
        num = int(density * h * w / 4096.0)
        layer = np.zeros((h, w), dtype=np.float32)
        dx = np.cos(np.deg2rad(angle))
        dy = np.sin(np.deg2rad(angle))
        for _ in range(num):
            x0 = self.rng.randint(0, w)
            y0 = self.rng.randint(0, h)
            x1 = int(x0 + dx * length)
            y1 = int(y0 + dy * length)
            cv2.line(layer, (x0, y0), (x1, y1), 1.0, 1)
        layer = cv2.GaussianBlur(layer, (3, 3), 0)
        imgf = self._to_float(img)
        out = 1.0 - (1.0 - imgf) * (1.0 - intensity * layer[..., None])  # screen blend
        return self._to_uint8(out)

    def _glass_mask(self, h, w):
        cy = int(h * self.rng.uniform(0.15, 0.45))
        cx = int(w * 0.5)
        ax = int(w * self.rng.uniform(0.25, 0.45))
        ay = int(h * self.rng.uniform(0.12, 0.25))
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(mask, (cx, cy), (ax, ay), 0, 0, 360, 255, -1)
        return mask

    def _road_mask(self, h, w):
        top_y = int(h * self.rng.uniform(0.55, 0.7))
        x_margin = int(w * self.rng.uniform(0.1, 0.3))
        pts = np.array([[x_margin, h-1],
                        [w - x_margin, h-1],
                        [w - int(x_margin*0.5), top_y],
                        [int(x_margin*0.5), top_y]], dtype=np.int32)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 255)
        return mask

    def _apply_mask_blend(self, img, overlay, mask, alpha):
        a = (mask.astype(np.float32) / 255.0) * alpha
        imgf = self._to_float(img)
        overf = self._to_float(overlay)
        out = imgf * (1.0 - a[..., None]) + overf * (a[..., None])
        return self._to_uint8(out)

    def _glass_reflection(self, img, mask, alpha, tint=(0.95, 1.0, 1.05)):
        ref = cv2.flip(img, 1)
        ref = cv2.GaussianBlur(ref, (11, 11), 0)
        ref = (self._to_float(ref) * np.array(tint)[None, None, :]).clip(0, 1)
        ref = self._to_uint8(ref)
        return self._apply_mask_blend(img, ref, mask, alpha)

    def _glass_flare(self, img, mask, strength):
        h, w = img.shape[:2]
        overlay = img.copy()
        num = self.rng.randint(1, 3)
        for _ in range(num):
            cx = self.rng.randint(int(w*0.2), int(w*0.8))
            cy = self.rng.randint(int(h*0.05), int(h*0.5))
            rx = self.rng.randint(int(w*0.05), int(w*0.15))
            ry = self.rng.randint(int(h*0.02), int(h*0.08))
            tmp = np.zeros((h, w), dtype=np.uint8)
            cv2.ellipse(tmp, (cx, cy), (rx, ry), self.rng.uniform(-30, 30), 0, 360, 255, -1)
            blur = cv2.GaussianBlur(tmp, (0, 0), sigmaX=rx*0.6, sigmaY=ry*0.6)
            for c in range(3):
                overlay[..., c] = np.clip(overlay[..., c].astype(np.float32) + strength*255*blur/255.0, 0, 255)
        return np.where(mask[..., None] > 0, overlay, img).astype(np.uint8)

    def _glass_polarize(self, img, mask, sat_drop=0.2, val_drop=0.1):
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
        m = (mask > 0).astype(np.float32)
        hsv[..., 1] = hsv[..., 1] * (1.0 - sat_drop * m)
        hsv[..., 2] = hsv[..., 2] * (1.0 - val_drop * m)
        out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        return out

    def _wet_road_reflection(self, img, mask, alpha, blur_ks=9):
        h, w = img.shape[:2]
        ref_src_top = img[:h//2, :, :]
        ref = cv2.flip(ref_src_top, 0)
        ref = cv2.resize(ref, (w, h), interpolation=cv2.INTER_LINEAR)
        ref = cv2.GaussianBlur(ref, (blur_ks, blur_ks), 0)
        return self._apply_mask_blend(img, ref, mask, alpha)

    def _low_texture_patch(self, img, mask, smooth_k=7, contrast_scale=0.7):
        blur = cv2.bilateralFilter(img, d=5, sigmaColor=50, sigmaSpace=7)
        blur = self._contrast(blur, contrast_scale)
        return np.where(mask[..., None] > 0, blur, img).astype(np.uint8)

    def _basic_photometric_lr(self, imgL, imgR):
        gL, gR = self.rng.uniform(0.8, 1.2), self.rng.uniform(0.8, 1.2)
        bL, bR = self.rng.uniform(0.7, 1.3), self.rng.uniform(0.7, 1.3)
        cL, cR = self.rng.uniform(0.8, 1.2), self.rng.uniform(0.8, 1.2)
        sL, sR = self.rng.uniform(0.6, 1.4), self.rng.uniform(0.6, 1.4)
        wlL, wlR = self.rng.uniform(0.9, 1.1), self.rng.uniform(0.9, 1.1)   # R 게인
        wbL, wbR = self.rng.uniform(0.9, 1.1), self.rng.uniform(0.9, 1.1)   # B 게인

        def apply(img, g, b, c, s, rg, bg):
            out = self._gamma(img, g)
            out = self._brightness(out, b)
            out = self._contrast(out, c)
            out = self._saturation(out, s)
            out = self._white_balance(out, rg, bg)
            return out

        return apply(imgL, gL, bL, cL, sL, wlL, wbL), apply(imgR, gR, bR, cR, sR, wlR, wbR)

    def _noise_blur_jpeg_vignette_color(self, img):
        out = img
        if self.rng.rand() < self.p_noise:
            sigma = self.rng.uniform(0.0, 0.02)  # 0~0.02
            out = self._gaussian_noise(out, sigma)
        if self.rng.rand() < self.p_blur:
            if self.rng.rand() < 0.5:
                k = self.rng.randint(1, 4) * 2 + 1  # 3,5,7
                out = self._defocus_blur(out, k)
            else:
                k = self.rng.randint(3, 10)
                angle = self.rng.uniform(-20, 20)
                out = self._motion_blur(out, k, angle)
        if self.rng.rand() < self.p_jpeg:
            q = int(self.rng.uniform(30, 90))
            out = self._jpeg(out, q)
        if self.rng.rand() < self.p_vignette:
            strength = self.rng.uniform(0.1, 0.4)
            out = self._vignette(out, strength)
        if self.rng.rand() < self.p_color_shade:
            gains = 1.0 + self.rng.uniform(-0.08, 0.08, size=3)  # ±8%
            out = self._color_shading(out, gains)
        return out

    def _pair_weather_shadows(self, imgL, imgR):
        if self.rng.rand() < self.p_shadow:
            n_poly = self.rng.randint(1, 3)
            imgL = self._shadow(imgL, n_poly=n_poly)
            imgR = self._shadow(imgR, n_poly=n_poly)

        if self.rng.rand() < self.p_fog:
            beta = self.rng.uniform(0.02, 0.08)
            A = self.rng.uniform(0.8, 1.0)
            imgL = self._fog(imgL, beta, A, vertical=True)
            imgR = self._fog(imgR, beta, A, vertical=True)

        if self.rng.rand() < self.p_rain:
            density = self.rng.uniform(0.3, 0.9)
            length = self.rng.randint(8, 20)
            angle = self.rng.uniform(-20, 20)
            intensity = self.rng.uniform(0.3, 0.7)
            imgL = self._rain(imgL, density, length, angle, intensity)
            imgR = self._rain(imgR, density, length, angle, intensity)

        return imgL, imgR

    def _pair_glass_road(self, imgL, imgR):
        h, w = imgL.shape[:2]

        # 유리: 반사 + 플레어 + 편광
        if self.rng.rand() < self.p_glass:
            gmask = self._glass_mask(h, w)
            aL = self.rng.uniform(0.10, 0.35)
            aR = max(0.0, aL + self.rng.uniform(-0.05, 0.05))
            imgL = self._glass_reflection(imgL, gmask, aL, tint=(0.95, 1.0, 1.05))
            imgR = self._glass_reflection(imgR, gmask, aR, tint=(0.95, 1.0, 1.05))
            imgL = self._glass_flare(imgL, gmask, strength=self.rng.uniform(0.05, 0.2))
            imgR = self._glass_flare(imgR, gmask, strength=self.rng.uniform(0.05, 0.2))
            imgL = self._glass_polarize(imgL, gmask, sat_drop=self.rng.uniform(0.1, 0.25),
                                        val_drop=self.rng.uniform(0.05, 0.2))
            imgR = self._glass_polarize(imgR, gmask, sat_drop=self.rng.uniform(0.1, 0.25),
                                        val_drop=self.rng.uniform(0.05, 0.2))

        # 바닥: 젖은 반사 + 저질감 패치
        rmask = self._road_mask(h, w)
        if self.rng.rand() < self.p_wet:
            a = self.rng.uniform(0.10, 0.35)
            imgL = self._wet_road_reflection(imgL, rmask, a, blur_ks=self.rng.choice([7, 9, 11]))
            imgR = self._wet_road_reflection(imgR, rmask,
                                             max(0.0, a + self.rng.uniform(-0.05, 0.05)),
                                             blur_ks=self.rng.choice([7, 9, 11]))
        if self.rng.rand() < self.p_lowtex:
            imgL = self._low_texture_patch(imgL, rmask, smooth_k=7, contrast_scale=self.rng.uniform(0.6, 0.85))
            imgR = self._low_texture_patch(imgR, rmask, smooth_k=7, contrast_scale=self.rng.uniform(0.6, 0.85))

        return imgL, imgR

    def _horizontal_scale_pair(self, imgL, imgR, disp, sx):
        """수평 스케일만 적용 → disparity도 동일 배율로 스케일"""
        h, w = imgL.shape[:2]
        new_w = max(32, int(round(w * sx)))
        imgL = cv2.resize(imgL, (new_w, h), interpolation=cv2.INTER_LINEAR)
        imgR = cv2.resize(imgR, (new_w, h), interpolation=cv2.INTER_LINEAR)
        disp = cv2.resize(disp.astype(np.float32), (new_w, h), interpolation=cv2.INTER_NEAREST) * sx
        return imgL, imgR, disp

    def _hflip_pair(self, imgL, imgR, disp):
        imgL = cv2.flip(imgL, 1)
        imgR = cv2.flip(imgR, 1)
        disp = cv2.flip(disp, 1) * -1.0  # 좌우가 바뀌므로 부호 반전
        return imgL, imgR, disp

    def __call__(self, imgL, imgR, disp):
        """
        imgL, imgR: np.uint8 (H,W,3) RGB
        disp: np.float32 (H,W)
        """
        # (0) H-flip
        # if self.rng.rand() < self.p_hflip:
        #     imgL, imgR, disp = self._hflip_pair(imgL, imgR, disp)

        # # (1) 수평 스케일
        # if self.rng.rand() < self.p_hscale:
        #     sx = self.rng.uniform(self.hscale_range[0], self.hscale_range[1])
        #     imgL, imgR, disp = self._horizontal_scale_pair(imgL, imgR, disp, sx)

        # # (2) 좌/우 비대칭 포토메트릭
        # if self.rng.rand() < self.p_basic:
        #     imgL, imgR = self._basic_photometric_lr(imgL, imgR)

        # # (3) 센서 잡음/블러/JPEG/비네팅/컬러셰이딩
        imgL = self._noise_blur_jpeg_vignette_color(imgL)
        imgR = self._noise_blur_jpeg_vignette_color(imgR)

        # (4) 날씨/그림자
        imgL, imgR = self._pair_weather_shadows(imgL, imgR)

        # (5) 유리/바닥 특화
        imgL, imgR = self._pair_glass_road(imgL, imgR)

        return imgL, imgR, disp


# =========================================================
# FlyingThingDataset
# =========================================================
class FlyingThingDataset(Dataset):
    def __init__(self, datapath, list_filename, training, max_len=None, aug=True, prior=None,
                 use_aux_feats=True,             # 보조 특징 채널 사용
                 invalid_disp_nonpos=True,       # disparity <= 0 invalid 처리
                 negate_disp=True,               # 반환 시 disparity 부호 반전(기존 파이프라인 호환)
                 erase_low=False,                # 구코드 호환 옵션(권장: False)
                 min_valid_ratio=0.01,           # 크롭 내 유효 disparity 최소 비율
                 max_crop_tries=30):             # 크롭 재시도 횟수
        self.datapath = datapath
        self.left_filenames, self.right_filenames, self.disp_filenames = self.load_path(list_filename)
        self.training = training
        self.data_len = len(self.left_filenames)
        self.max_len = max_len
        self.aug = aug
        self.erase_low = erase_low
        self.prior_path = prior

        self.use_aux_feats = use_aux_feats
        self.invalid_disp_nonpos = invalid_disp_nonpos
        self.negate_disp = negate_disp
        self.min_valid_ratio = float(min_valid_ratio)
        self.max_crop_tries = int(max_crop_tries)

        if self.training:
            assert self.disp_filenames is not None

        # 증강기
        self.augmentor = StereoAugmentor() if (self.training and self.aug) else None
        self.car_patch_aug = None  # CarPatchAugmenter는 별도 구현 필요
        # self.car_patch_aug = CarPatchAugmenter(
        #     aug_prob=0.10,                 # 이미지 10%에 적용 (원하시는 비율로 조정)
        #     ymin_ratio=0.70, ymax_ratio=1.00,
        #     height_base=150, width_base=300, size_jitter=0.20,
        #     disp_mean=70.0, disp_jitter=10.0,
        #     zbuffer_margin=0.5, disp_valid_min=0.1,
        #     base_gray_range=(30, 250),
        #     shape='random',
        #     rotate_deg_range=(-18.0, 18.0),
        #     corner='random',
        #     noise_level=0.0, noise_cells=0,
        #     seed=42
        # ) if (self.training and self.aug) else None

    def load_path(self, list_filename):
        lines = read_all_lines(list_filename)
        splits = [line.split() for line in lines]
        left_images = [x[0] for x in splits]
        right_images = [x[1] for x in splits]
        if len(splits[0]) == 2:  # ground truth not available
            return left_images, right_images, None
        else:
            disp_images = [x[2] for x in splits]
            return left_images, right_images, disp_images

    def load_image(self, filename):
        filename = os.path.expanduser(filename)
        return Image.open(filename).convert('RGB')

    def load_prior(self):
        if self.prior_path is not None:
            prior_path = os.path.expanduser(self.prior_path)
            if not os.path.exists(os.path.expanduser(prior_path)):
                raise FileNotFoundError(f"Prior file {prior_path} does not exist.")
            prior_data = np.load(prior_path)
        else:
            prior_data = 0.0
        return prior_data

    def load_disp(self, filename):
        filename = os.path.expanduser(filename)
        data, scale = pfm_imread(filename)
        data = np.ascontiguousarray(data, dtype=np.float32)
        # NaN/Inf 정리
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        return data

    def __len__(self):
        return self.max_len if self.max_len is not None else self.data_len

    # ---------- 메인 ----------
    def __getitem__(self, index):
        index = index if index < self.data_len else random.randint(0, self.data_len - 1)
        left_img = self.load_image(os.path.join(self.datapath, self.left_filenames[index]))
        right_img = self.load_image(os.path.join(self.datapath, self.right_filenames[index]))
        disparity = self.load_disp(os.path.join(self.datapath, self.disp_filenames[index]))

        # === disparity 기본 정리 ===
        # if self.erase_low:
        #     # 구코드 호환(권장 X): 대부분 0이 될 수 있어 주의
        #     disparity[disparity > -2] = 0
        # if self.invalid_disp_nonpos:
        #     disparity[disparity <= 0] = 0  # 0 이하 invalid
        # disparity = np.nan_to_num(disparity, nan=0.0, posinf=0.0, neginf=0.0)

        prior_data = self.load_prior()

        if self.training:
            
            if self.aug and self.car_patch_aug is not None:
                L_np = np.array(left_img)   # PIL -> np.uint8 (RGB)
                R_np = np.array(right_img)
                disparity = np.asarray(disparity, dtype=np.float32)
                L_np, R_np, disparity = self.car_patch_aug(L_np, R_np, disparity)
                left_img  = Image.fromarray(L_np)
                right_img = Image.fromarray(R_np)

            # --------- 증강 ----------
            if self.aug and self.augmentor is not None:
                L = np.array(left_img)
                R = np.array(right_img)
                L, R, disparity = self.augmentor(L, R, disparity)
                left_img = Image.fromarray(L)
                right_img = Image.fromarray(R)
                disparity = np.nan_to_num(disparity, nan=0.0, posinf=0.0, neginf=0.0)

            # ---------- 크롭(유효 픽셀 비율 보장) ----------
            w, h = left_img.size
            crop_list = [[512,128],[512,256],[768,256],[768,512]]
            num = random.randint(0, 3)
            crop_w, crop_h = crop_list[num][0], crop_list[num][1]

            if w < crop_w or h < crop_h:
                pad_w = max(0, crop_w - w)
                pad_h = max(0, crop_h - h)
                left_img_np = np.pad(np.array(left_img), ((pad_h, 0), (0, pad_w), (0, 0)), mode='edge')
                right_img_np = np.pad(np.array(right_img), ((pad_h, 0), (0, pad_w), (0, 0)), mode='edge')
                disparity = np.pad(disparity, ((pad_h, 0), (0, pad_w)), mode='edge')
                left_img = Image.fromarray(left_img_np)
                right_img = Image.fromarray(right_img_np)
                w, h = left_img.size

            # 크롭 재시도 루프
            # tries = 0
            while True:
                x1 = random.randint(0, w - crop_w)
                y1 = random.randint(0, h - crop_h)
                disp_patch = disparity[y1:y1 + crop_h, x1:x1 + crop_w]

                # negate_disp 기준으로 양의 disparity를 valid로 판단
                dd = (-disp_patch if self.negate_disp else disp_patch)
                valid_ratio = float((dd > 0).mean()) if dd.size > 0 else 0.0

                if valid_ratio >= self.min_valid_ratio or tries >= self.max_crop_tries:
                    break
                tries += 1
            x1 = random.randint(0, w - crop_w)
            y1 = random.randint(0, h - crop_h)
            left_img  = left_img.crop((x1, y1, x1 + crop_w, y1 + crop_h))
            right_img = right_img.crop((x1, y1, x1 + crop_w, y1 + crop_h))
            disparity = disparity[y1:y1 + crop_h, x1:x1 + crop_w]
            # disparity = np.nan_to_num(disparity, nan=0.0, posinf=0.0, neginf=0.0)

            # ---------- 저해상도 GT ----------
            disparity_low    = cv2.resize(disparity, (crop_w//4, crop_h//4), interpolation=cv2.INTER_NEAREST)
            disparity_low_r8 = cv2.resize(disparity, (crop_w//8, crop_h//8), interpolation=cv2.INTER_NEAREST)

            # ---------- 보조 특징/마스크 ----------
            left_np  = np.array(left_img)
            right_np = np.array(right_img)

            gradL = gradient_magnitude_rgb(left_np)
            gradR = gradient_magnitude_rgb(right_np)
            logL  = log_magnitude(left_np, sigma=1.0)
            logR  = log_magnitude(right_np, sigma=1.0)
            rankL = rank_transform(left_np, win=7)
            rankR = rank_transform(right_np, win=7)
            cohL  = structure_tensor_coherence(left_np, sigma=2.0, blur=3)
            cohR  = structure_tensor_coherence(right_np, sigma=2.0, blur=3)

            robust_left  = torch.from_numpy(np.stack([gradL, logL, rankL, cohL], axis=0)).float()
            robust_right = torch.from_numpy(np.stack([gradR, logR, rankR, cohR], axis=0)).float()

            # negate 기준에 맞춘 valid mask
            valid_mask = torch.from_numpy(_left_valid_mask_signed(disparity, self.negate_disp)).float()
            gradient_map = torch.from_numpy(gradL).float()

            # ---------- 텐서 변환 ----------
            processed = get_transform()
            left_img_t  = processed(left_img)   # [3,H,W], float
            right_img_t = processed(right_img)

            # prior 텐서화
            if isinstance(prior_data, np.ndarray):
                prior_t = torch.from_numpy(np.nan_to_num(prior_data, nan=0.0, posinf=0.0, neginf=0.0)).float()
            else:
                prior_t = torch.tensor(float(prior_data), dtype=torch.float32)

            sample = {
                "left": left_img_t,
                "right": right_img_t,
                "disparity": torch.from_numpy(disparity).float(),
                "valid_mask": valid_mask,
                "gradient_map": gradient_map,
                "disparity_low": torch.from_numpy(disparity_low).float(),
                "disparity_low_r8": torch.from_numpy(disparity_low_r8).float(),
                "left_filename": self.left_filenames[index],
                "right_filename": self.right_filenames[index],
                "prior": prior_t,
            }
            if self.use_aux_feats:
                sample["robust_left"]  = robust_left
                sample["robust_right"] = robust_right

            # 부호 일관성(최종 반환)
            if self.negate_disp:
                sample["disparity"]        = -sample["disparity"]
                sample["disparity_low"]    = -sample["disparity_low"]
                sample["disparity_low_r8"] = -sample["disparity_low_r8"]

            return sample

        else:
            # --------- 평가 경로(증강 없음) ----------
            w, h = left_img.size
            crop_w, crop_h = 960, 512
            left_img  = left_img.crop((w - crop_w, h - crop_h, w, h))
            right_img = right_img.crop((w - crop_w, h - crop_h, w, h))
            disparity = disparity[h - crop_h:h, w - crop_w: w]
            disparity = np.nan_to_num(disparity, nan=0.0, posinf=0.0, neginf=0.0)

            left_np = np.array(left_img)
            gradL = gradient_magnitude_rgb(left_np)
            gradient_map = torch.from_numpy(gradL).float()

            disparity_low = cv2.resize(disparity, (crop_w//4, crop_h//4), interpolation=cv2.INTER_NEAREST)
            valid_mask = torch.from_numpy(_left_valid_mask_signed(disparity, self.negate_disp)).float()

            # 보조특징(옵션)
            if self.use_aux_feats:
                right_np = np.array(right_img)
                robust_left  = torch.from_numpy(
                    np.stack([gradL, log_magnitude(left_np), rank_transform(left_np),
                              structure_tensor_coherence(left_np)], axis=0)).float()
                robust_right = torch.from_numpy(
                    np.stack([gradient_magnitude_rgb(right_np), log_magnitude(right_np), rank_transform(right_np),
                              structure_tensor_coherence(right_np)], axis=0)).float()

            processed = get_transform()
            left_img_t  = processed(left_img)
            right_img_t = processed(right_img)

            sample = {
                "left": left_img_t,
                "right": right_img_t,
                "disparity": torch.from_numpy(disparity).float(),
                "top_pad": 0,
                "right_pad": 0,
                "gradient_map": gradient_map,
                "disparity_low": torch.from_numpy(disparity_low).float(),
                "valid_mask": valid_mask,
                "left_filename": self.left_filenames[index],
                "right_filename": self.right_filenames[index],
            }
            if self.use_aux_feats:
                sample["robust_left"]  = robust_left
                sample["robust_right"] = robust_right

            if self.negate_disp:
                sample["disparity"]     = -sample["disparity"]
                sample["disparity_low"] = -sample["disparity_low"]
            return sample
