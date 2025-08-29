# =========================================
# Kitti → FT3D 스타일 테스트-타임 전처리
# =========================================
from torchvision.transforms import functional as F
from torchvision.transforms.functional import InterpolationMode
import numpy as np
import cv2
from PIL import Image, ImageFilter
import torchvision.transforms as transforms
from datasets.kitti2015 import *

# ---- [A] 유틸 함수들 ----
def _pil_to_gray_np(img_pil):
    arr = np.array(img_pil.convert("RGB"), dtype=np.uint8)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    return gray

def _apply_vertical_shift_pil(img_pil, dy):
    if dy == 0:
        return img_pil
    return F.affine(img_pil, angle=0.0, translate=(0, int(dy)), scale=1.0,
                    shear=[0.0, 0.0], interpolation=InterpolationMode.BILINEAR, fill=0)

def _estimate_vertical_shift(imgL_pil, imgR_pil, dy_range=(-2, 2)):
    """간단한 스캔으로 좌우의 수직 오프셋 추정(중앙 80% 영역 MSE 최소)"""
    L = _pil_to_gray_np(imgL_pil)
    R = _pil_to_gray_np(imgR_pil)
    H, W = L.shape
    y0, y1 = int(0.1*H), int(0.9*H)

    best_dy, best_mse = 0, 1e9
    for dy in range(dy_range[0], dy_range[1]+1):
        if dy >= 0:
            R2 = np.pad(R, ((dy, 0), (0, 0)), mode='edge')[:H, :]
        else:
            R2 = np.pad(R, ((0, -dy), (0, 0)), mode='edge')[-dy:H, :]
        mse = np.mean((L[y0:y1, :] - R2[y0:y1, :])**2)
        if mse < best_mse:
            best_mse, best_dy = mse, dy
    return int(best_dy)

def _color_match_right_to_left(imgL_pil, imgR_pil, method="meanvar"):
    """R을 L에 맞춤(좌우 색 통계 일치). method='meanvar' 권장."""
    L = np.array(imgL_pil.convert("RGB")).astype(np.float32)
    R = np.array(imgR_pil.convert("RGB")).astype(np.float32)

    if method == "meanvar":
        for c in range(3):
            mL, sL = L[..., c].mean(), L[..., c].std() + 1e-6
            mR, sR = R[..., c].mean(), R[..., c].std() + 1e-6
            R[..., c] = ((R[..., c] - mR) / sR) * sL + mL
        R = np.clip(R, 0, 255).astype(np.uint8)
        return imgL_pil, Image.fromarray(R)
    else:
        # (선택) 히스토그램 매칭 구현 가능. 기본은 mean/var 정합.
        return imgL_pil, imgR_pil

def _progressive_tone_pair(imgL_pil, imgR_pil, gamma=1.15, sat=1.15, contrast=1.05):
    """FT3D의 진하고 선명한 톤을 모사(좌우 동일 파라미터 적용)"""
    def _apply(img):
        out = F.adjust_gamma(img, gamma=gamma, gain=1.0)
        out = F.adjust_saturation(out, sat)
        out = F.adjust_contrast(out, contrast)
        return out
    return _apply(imgL_pil), _apply(imgR_pil)

def _horizontal_stretch_fix_width(img_pil, sx):
    """가로 스케일 sx 후 원폭 복원(중앙 크롭/패드). 메타 반환."""
    w, h = img_pil.size
    new_w = max(1, int(round(w * float(sx))))
    img2 = img_pil.resize((new_w, h), Image.BICUBIC)

    meta = {"w": w, "h": h, "sx": float(sx), "new_w": new_w,
            "crop_left": 0, "pad_left": 0, "pad_right": 0}
    if new_w > w:
        left = (new_w - w) // 2
        img2 = img2.crop((left, 0, left + w, h))
        meta["crop_left"] = left
    elif new_w < w:
        pad_total = w - new_w
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        img2 = F.pad(img2, [pad_left, 0, pad_right, 0], fill=0)
        meta["pad_left"] = pad_left
        meta["pad_right"] = pad_right
    return img2, meta

def _unsharp_mask_np(rgb_np, blur_sigma=0.5, amount=1.0):
    """간단한 언샤프 마스크 (OpenCV)"""
    blur = cv2.GaussianBlur(rgb_np, (0, 0), blur_sigma)
    sharp = cv2.addWeighted(rgb_np, 1.0 + amount, blur, -amount, 0)
    return np.clip(sharp, 0, 255).astype(np.uint8)

def _light_denoise_np(rgb_np, strength=3):
    """약한 디노이즈(양방향 가능). 여기서는 fast NLM 대신 Gaussian으로 가볍게."""
    if strength <= 0:
        return rgb_np
    return cv2.GaussianBlur(rgb_np, (3, 3), 0)

def _suppress_specular_np(rgb_np, v_thresh=0.85, s_thresh=0.35, reduce=0.85):
    """
    HSV에서 (밝음↑ & 채도↓) 픽셀은 스펙큘러 후보 → V를 줄임.
    """
    hsv = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2HSV).astype(np.float32)
    H, S, V = hsv[..., 0], hsv[..., 1] / 255.0, hsv[..., 2] / 255.0
    mask = (V > v_thresh) & (S < s_thresh)
    V[mask] = V[mask] * reduce
    hsv[..., 1] = np.clip(S * 255.0, 0, 255)
    hsv[..., 2] = np.clip(V * 255.0, 0, 255)
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    return out

def _tensorize_and_normalize(pil_img, target_size, normalize):
    t = F.to_tensor(pil_img) * 255.0
    t = ImageProcessor.pad_to_size(t, target_size)
    return normalize(t)

# ---- [B] sx 선택(정책) ----
def _choose_sx_ratio(fxB_src=None, fxB_tgt=None, default=1.0):
    if fxB_src is None or fxB_tgt is None or fxB_tgt == 0:
        return float(default)
    return float(fxB_src) / float(fxB_tgt)

def _choose_sx_by_stereoBM(imgL_pil, imgR_pil, sx_candidates=(0.8, 1.0, 1.2, 1.4),
                           down_w=640, num_disp=128, block_size=15):
    """
    후보 sx에 대해 간단한 StereoBM으로 포토메트릭 워핑 오차를 비교해 최적 sx 선택.
    """
    def _score(L_pil, R_pil):
        L = np.array(L_pil.convert("RGB"))
        R = np.array(R_pil.convert("RGB"))
        # 다운샘플
        scale = down_w / L.shape[1]
        new_size = (down_w, max(1, int(round(L.shape[0] * scale))))
        Ls = cv2.resize(L, new_size, interpolation=cv2.INTER_AREA)
        Rs = cv2.resize(R, new_size, interpolation=cv2.INTER_AREA)

        Lg = cv2.cvtColor(Ls, cv2.COLOR_RGB2GRAY)
        Rg = cv2.cvtColor(Rs, cv2.COLOR_RGB2GRAY)
        # StereoBM
        num_disp16 = int(np.ceil(num_disp/16.0))*16
        stereo = cv2.StereoBM_create(numDisparities=num_disp16, blockSize=block_size)
        disp = stereo.compute(Lg, Rg).astype(np.float32) / 16.0  # [H,W]
        disp[disp < 0] = 0.0

        # 오른쪽을 왼쪽으로 워핑
        Hh, Ww = Lg.shape
        grid_x, grid_y = np.meshgrid(np.arange(Ww), np.arange(Hh))
        map_x = (grid_x - disp).astype(np.float32)
        map_y = grid_y.astype(np.float32)
        Rw = cv2.remap(Rs, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        # 중앙 영역 포토메트릭 MSE
        y0, y1 = int(0.1*Hh), int(0.9*Hh)
        x0, x1 = int(0.1*Ww), int(0.9*Ww)
        mse = np.mean((Ls[y0:y1, x0:x1, :].astype(np.float32) - Rw[y0:y1, x0:x1, :].astype(np.float32))**2)
        return mse

    best_sx, best_score = 1.0, 1e18
    for sx in sx_candidates:
        Ls, _ = _horizontal_stretch_fix_width(imgL_pil, sx)
        Rs, _ = _horizontal_stretch_fix_width(imgR_pil, sx)
        score = _score(Ls, Rs)
        if score < best_score:
            best_score, best_sx = score, sx
    return float(best_sx)

# ---- [C] 메인 Transform 클래스 ----
class Kitti2FT3DPreprocess:
    """
    6단계 전처리 파이프라인 (테스트-타임):
    (1) 정렬 보정(수직 오프셋 추정&보정)
    (2) 좌우 색 정합(mean/var)
    (3) 전역 톤 보정(감마/채도/대비)
    (4) 수평 스케일 보정(sx)
    (5) 디노이즈 + 샤픈
    (6) 스펙큘러 억제

    sx_policy: {"none", "fixed", "ratio", "search"}
      - fixed: sx_fixed 사용
      - ratio: fxB_src/fxB_tgt
      - search: StereoBM 기반 후보 탐색
    """
    def __init__(self,
                 target_size=(1248, 384),
                 normalize=transforms.Normalize(ImageProcessor.IMAGENET_MEAN, ImageProcessor.IMAGENET_STD),
                 rect_dy_range=(-2, 2),
                 color_match="meanvar",
                 tone_gamma=1.15, tone_sat=1.15, tone_contrast=1.05,
                 sx_policy="ratio", sx_fixed=1.0, fxB_src=None, fxB_tgt=None,
                 sx_candidates=(0.8, 1.0, 1.2, 1.4),
                 denoise_strength=2, unsharp_amount=0.7, unsharp_sigma=0.7,
                 specular_v_thresh=0.85, specular_s_thresh=0.35, specular_reduce=0.85,
                 return_stages=False):
        self.target_size = target_size
        self.normalize = normalize

        self.rect_dy_range = rect_dy_range
        self.color_match = color_match
        self.tone_gamma = tone_gamma
        self.tone_sat = tone_sat
        self.tone_contrast = tone_contrast

        self.sx_policy = sx_policy
        self.sx_fixed = sx_fixed
        self.fxB_src = fxB_src
        self.fxB_tgt = fxB_tgt
        self.sx_candidates = sx_candidates

        self.denoise_strength = denoise_strength
        self.unsharp_amount = unsharp_amount
        self.unsharp_sigma = unsharp_sigma

        self.specular_v_thresh = specular_v_thresh
        self.specular_s_thresh = specular_s_thresh
        self.specular_reduce = specular_reduce

        self.return_stages = return_stages

    def __call__(self, left_pil: Image.Image, right_pil: Image.Image):
        stages = {}  # 시각화를 위해 PIL/ndarray 단계별 보관

        # Stage 0: 원본
        L0, R0 = left_pil.convert("RGB"), right_pil.convert("RGB")
        if self.return_stages:
            stages["0_orig_L"] = L0.copy()
            stages["0_orig_R"] = R0.copy()

        # (1) 수직 오프셋 추정 및 보정(우영상에만 적용)
        dy = _estimate_vertical_shift(L0, R0, dy_range=self.rect_dy_range)
        L1, R1 = L0, _apply_vertical_shift_pil(R0, dy)
        if self.return_stages:
            stages["1_rect_L"] = L1.copy(); stages["1_rect_R"] = R1.copy()

        # (2) 좌우 색 정합(오른쪽을 왼쪽에 맞춤)
        L2, R2 = _color_match_right_to_left(L1, R1, method=self.color_match)
        if self.return_stages:
            stages["2_color_L"] = L2.copy(); stages["2_color_R"] = R2.copy()

        # (3) 전역 톤 보정(좌우 동일)
        L3, R3 = _progressive_tone_pair(L2, R2, gamma=self.tone_gamma,
                                        sat=self.tone_sat, contrast=self.tone_contrast)
        if self.return_stages:
            stages["3_tone_L"] = L3.copy(); stages["3_tone_R"] = R3.copy()

        # (4) 수평 스케일 sx 결정 및 적용(좌우 동기)
        if self.sx_policy == "fixed":
            sx = float(self.sx_fixed)
        elif self.sx_policy == "ratio":
            sx = _choose_sx_ratio(self.fxB_src, self.fxB_tgt, default=1.0)
        elif self.sx_policy == "search":
            sx = _choose_sx_by_stereoBM(L3, R3, sx_candidates=self.sx_candidates)
        else:  # "none"
            sx = 1.0

        L4, metaL = _horizontal_stretch_fix_width(L3, sx)
        R4, metaR = _horizontal_stretch_fix_width(R3, sx)
        T_meta = {"sx": sx, "dy": dy, **metaL}
        if self.return_stages:
            stages["4_hscale_L"] = L4.copy(); stages["4_hscale_R"] = R4.copy()

        # (5) 디노이즈 + 샤픈
        L5 = np.array(L4, dtype=np.uint8); R5 = np.array(R4, dtype=np.uint8)
        L5 = _light_denoise_np(L5, strength=self.denoise_strength)
        R5 = _light_denoise_np(R5, strength=self.denoise_strength)
        L5 = _unsharp_mask_np(L5, blur_sigma=self.unsharp_sigma, amount=self.unsharp_amount)
        R5 = _unsharp_mask_np(R5, blur_sigma=self.unsharp_sigma, amount=self.unsharp_amount)
        L5_pil = Image.fromarray(L5); R5_pil = Image.fromarray(R5)
        if self.return_stages:
            stages["5_denoiseLsharp_L"] = L5_pil.copy(); stages["5_denoiseLsharp_R"] = R5_pil.copy()

        # (6) 스펙큘러 억제(선택)
        L6 = _suppress_specular_np(L5, v_thresh=self.specular_v_thresh,
                                   s_thresh=self.specular_s_thresh, reduce=self.specular_reduce)
        R6 = _suppress_specular_np(R5, v_thresh=self.specular_v_thresh,
                                   s_thresh=self.specular_s_thresh, reduce=self.specular_reduce)
        L6_pil, R6_pil = Image.fromarray(L6), Image.fromarray(R6)
        if self.return_stages:
            stages["6_spec_suppress_L"] = L6_pil.copy(); stages["6_spec_suppress_R"] = R6_pil.copy()

        # 텐서화 + 패딩 + 정규화(기존 스타일)
        tL = _tensorize_and_normalize(L6_pil, self.target_size, self.normalize)
        tR = _tensorize_and_normalize(R6_pil, self.target_size, self.normalize)

        out = {"left_pre": tL, "right_pre": tR, "meta": T_meta}
        if self.return_stages:
            out["stages_pil"] = stages
        return out


# =============================
# 전처리 6단계 시각화 스크립트
# =============================
import matplotlib.pyplot as plt

def visualize_kitti2ft3d_pipeline(left_pil, right_pil):
    preproc = Kitti2FT3DPreprocess(
        target_size=ImageProcessor.TARGET_SIZE,
        sx_policy="search", sx_candidates=(0.8, 1.0, 1.2, 1.4),
        return_stages=True
    )
    out = preproc(left_pil, right_pil)
    stages = out["stages_pil"]  # dict: "<idx>_<name>_{L|R}" -> PIL

    # 단계명 정렬
    step_names = sorted({k[:-2] for k in stages.keys()})  # "0_orig", "1_rect", ...
    n = len(step_names)

    plt.figure(figsize=(12, 2.6*n))
    for i, step in enumerate(step_names, 1):
        L = stages[f"{step}_L"]; R = stages[f"{step}_R"]
        ax1 = plt.subplot(n, 2, 2*i - 1); ax2 = plt.subplot(n, 2, 2*i)
        ax1.imshow(L); ax1.set_title(f"{step} (Left)"); ax1.axis("off")
        ax2.imshow(R); ax2.set_title(f"{step} (Right)"); ax2.axis("off")
    plt.tight_layout()
    plt.show()

    # 최종 텐서 이미지 확인(정규화 전용이라 직접 보기엔 적합치 않음)
    print("Chosen sx:", out["meta"]["sx"], "  dy:", out["meta"]["dy"])


if __name__=="__main__":
    
    # 사용 예시
    left_img = Image.open("/home/jaejun/dataset/kitti_2015/training/image_2/000020_10.png").convert("RGB")
    right_img = Image.open("/home/jaejun/dataset/kitti_2015/training/image_3/000020_10.png").convert("RGB")
    visualize_kitti2ft3d_pipeline(left_img, right_img)
