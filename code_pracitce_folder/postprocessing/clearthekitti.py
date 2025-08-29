# =========================================
# Kitti → FT3D 스타일 전처리(6단계) + 저장
# =========================================
import os, json, random
import numpy as np
import cv2
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
from torchvision.transforms.functional import InterpolationMode
from scipy.ndimage import gaussian_filter

# --- ImageProcessor는 질문자가 이미 가진 클래스의 pad_to_size/정규화 mean-std를 사용합니다. ---
class ImageProcessor:
    """Handles image processing operations"""
    
    # Constants
    TARGET_SIZE = (1248, 384)  # (width, height)
    CROP_SIZE = (512, 320)    # (width, height)
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    
    @staticmethod
    def pad_to_size(tensor, target_size):
        """Pad tensor to target size"""
        if len(tensor.shape) == 3:
            _, h, w = tensor.shape
        else:
            h, w = tensor.shape
            
        target_w, target_h = target_size
        top_pad = max(0, target_h - h)
        right_pad = max(0, target_w - w)
        
        if top_pad > 0 or right_pad > 0:
            if len(tensor.shape) == 3:
                if isinstance(tensor, torch.Tensor):
                    tensor = torch.nn.functional.pad(tensor, (0, right_pad, top_pad, 0), mode='constant', value=0)
                else:
                    tensor = np.lib.pad(tensor, ((0, 0), (max(0, top_pad), 0), (0, max(0, right_pad))), 
                                      mode='constant', constant_values=0)
            else:
                tensor = np.lib.pad(tensor, ((max(0, top_pad), 0), (0, max(0, right_pad))), 
                                  mode='constant', constant_values=0)
        return tensor

    @staticmethod
    def create_depth_map(disparity):
        """Create depth map from disparity"""
        valid_mask = disparity > 0
        if not np.any(valid_mask):
            return disparity.copy()
            
        disp_min = np.min(disparity[valid_mask])
        disp_max = np.max(disparity[valid_mask])
        
        depth_map = np.zeros_like(disparity)
        if disp_max != disp_min:
            depth_map[valid_mask] = 1e-4 + (disparity[valid_mask] - disp_min) / (disp_max - disp_min)
        else:
            depth_map[valid_mask] = 1.0
            
        return depth_map

    @staticmethod
    def compute_textureless_score(left_img):
        """Compute textureless score from left image"""
        left_array = np.array(left_img).astype(np.float32) / 255.0
        gray = cv2.cvtColor((left_array * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(gx*gx + gy*gy)
        
        mag_norm = (mag - mag.min()) / (mag.max() - mag.min() + 1e-6)
        textureless = 1.0 - mag_norm
        textureless_sm = gaussian_filter(textureless, sigma=5)
        
        return torch.from_numpy(textureless_sm).unsqueeze(0).float()
# ---------------------- 유틸 ----------------------
def _ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def _pil_to_gray_np(img_pil):
    arr = np.array(img_pil.convert("RGB"), dtype=np.uint8)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    return gray

def _estimate_vertical_shift(imgL_pil, imgR_pil, dy_range=(-2, 2)):
    """좌우 수직 오프셋을 간단 스캔으로 추정"""
    L = _pil_to_gray_np(imgL_pil)
    R = _pil_to_gray_np(imgR_pil)
    H, W = L.shape
    y0, y1 = int(0.1*H), int(0.9*H)
    best_dy, best_mse = 0, 1e9
    for dy in range(dy_range[0], dy_range[1] + 1):
        if dy >= 0:
            R2 = np.pad(R, ((dy, 0), (0, 0)), mode='edge')[:H, :]
        else:
            R2 = np.pad(R, ((0, -dy), (0, 0)), mode='edge')[-dy:H, :]
        mse = np.mean((L[y0:y1, :] - R2[y0:y1, :])**2)
        if mse < best_mse:
            best_mse, best_dy = mse, dy
    return int(best_dy)

def _apply_vertical_shift_pil(img_pil, dy):
    if dy == 0:
        return img_pil
    return F.affine(img_pil, angle=0.0, translate=(0, int(dy)), scale=1.0,
                    shear=[0.0, 0.0], interpolation=InterpolationMode.BILINEAR, fill=0)

def _color_match_right_to_left(imgL_pil, imgR_pil):
    """R의 채널별 mean/std를 L에 맞춤"""
    L = np.array(imgL_pil.convert("RGB")).astype(np.float32)
    R = np.array(imgR_pil.convert("RGB")).astype(np.float32)
    for c in range(3):
        mL, sL = L[..., c].mean(), L[..., c].std() + 1e-6
        mR, sR = R[..., c].mean(), R[..., c].std() + 1e-6
        R[..., c] = ((R[..., c] - mR) / sR) * sL + mL
    R = np.clip(R, 0, 255).astype(np.uint8)
    return imgL_pil, Image.fromarray(R)

def _progressive_tone_pair(imgL_pil, imgR_pil, gamma=1.15, sat=1.15, contrast=1.05):
    """감마/채도/대비를 좌우 동일 파라미터로 보정"""
    def _apply(img):
        out = F.adjust_gamma(img, gamma=gamma, gain=1.0)
        out = F.adjust_saturation(out, sat)
        out = F.adjust_contrast(out, contrast)
        return out
    return _apply(imgL_pil), _apply(imgR_pil)

def _horizontal_stretch_fix_width(img_pil, sx):
    """가로 스케일 sx 후 원폭 복원(중앙 크롭/패드) + meta"""
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

def _light_denoise_np(rgb_np, strength=2):
    if strength <= 0:
        return rgb_np
    return cv2.GaussianBlur(rgb_np, (3, 3), 0)

def _unsharp_mask_np(rgb_np, blur_sigma=0.7, amount=0.7):
    blur = cv2.GaussianBlur(rgb_np, (0, 0), blur_sigma)
    sharp = cv2.addWeighted(rgb_np, 1.0 + amount, blur, -amount, 0)
    return np.clip(sharp, 0, 255).astype(np.uint8)

def _suppress_specular_np(rgb_np, v_thresh=0.85, s_thresh=0.35, reduce=0.85):
    """HSV에서 (밝음↑, 채도↓)을 스펙큘러로 가정하고 V 감소"""
    hsv = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2HSV).astype(np.float32)
    S = hsv[..., 1] / 255.0
    V = hsv[..., 2] / 255.0
    mask = (V > v_thresh) & (S < s_thresh)
    V[mask] = V[mask] * reduce
    hsv[..., 1] = np.clip(S * 255.0, 0, 255)
    hsv[..., 2] = np.clip(V * 255.0, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

def _absdiff_heatmap(a_pil, b_pil):
    a = np.array(a_pil.convert("RGB"), dtype=np.int16)
    b = np.array(b_pil.convert("RGB"), dtype=np.int16)
    d = np.abs(a - b).mean(axis=2).astype(np.uint8)  # 회색조
    return Image.fromarray(d)

def _crop_roi_by_ratio(img_pil, x0_ratio=0.70, y0_ratio=0.55, x1_ratio=0.96, y1_ratio=0.98, scale=3.0):
    """오른쪽 하단(검은 차량 영역) 확대"""
    W, H = img_pil.size
    x0, y0 = int(W * x0_ratio), int(H * y0_ratio)
    x1, y1 = int(W * x1_ratio), int(H * y1_ratio)
    crop = np.array(img_pil)[y0:y1, x0:x1, :]
    crop = cv2.resize(crop, (int((x1 - x0) * scale), int((y1 - y0) * scale)), interpolation=cv2.INTER_NEAREST)
    return Image.fromarray(crop)

def _tensorize_and_normalize(pil_img, target_size, normalize):
    t = F.to_tensor(pil_img) * 255.0
    t = ImageProcessor.pad_to_size(t, target_size)
    return normalize(t)

def _choose_sx_ratio(fxB_src=None, fxB_tgt=None, default=1.0):
    if fxB_src is None or fxB_tgt is None or fxB_tgt == 0:
        return float(default)
    return float(fxB_src) / float(fxB_tgt)

def _choose_sx_by_stereoBM(imgL_pil, imgR_pil, sx_candidates=(0.8, 1.0, 1.2, 1.4),
                           down_w=640, num_disp=128, block_size=15):
    """후보 sx로 간단 StereoBM 매칭 후 포토메트릭 워핑 오차 최소값 선택"""
    def _score(L_pil, R_pil):
        L = np.array(L_pil.convert("RGB")); R = np.array(R_pil.convert("RGB"))
        scale = down_w / L.shape[1]
        new_size = (down_w, max(1, int(round(L.shape[0] * scale))))
        Ls = cv2.resize(L, new_size, interpolation=cv2.INTER_AREA)
        Rs = cv2.resize(R, new_size, interpolation=cv2.INTER_AREA)
        Lg = cv2.cvtColor(Ls, cv2.COLOR_RGB2GRAY)
        Rg = cv2.cvtColor(Rs, cv2.COLOR_RGB2GRAY)
        num_disp16 = int(np.ceil(num_disp / 16.0)) * 16
        stereo = cv2.StereoBM_create(numDisparities=num_disp16, blockSize=block_size)
        disp = stereo.compute(Lg, Rg).astype(np.float32) / 16.0
        disp[disp < 0] = 0.0
        H, W = Lg.shape
        gx, gy = np.meshgrid(np.arange(W), np.arange(H))
        map_x = (gx - disp).astype(np.float32)
        map_y = gy.astype(np.float32)
        Rw = cv2.remap(Rs, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        y0, y1 = int(0.1*H), int(0.9*H); x0, x1 = int(0.1*W), int(0.9*W)
        mse = np.mean((Ls[y0:y1, x0:x1, :].astype(np.float32) - Rw[y0:y1, x0:x1, :].astype(np.float32))**2)
        return mse

    best_sx, best_score = 1.0, 1e18
    for sx in sx_candidates:
        Ls, _ = _horizontal_stretch_fix_width(imgL_pil, sx)
        Rs, _ = _horizontal_stretch_fix_width(imgR_pil, sx)
        sc = _score(Ls, Rs)
        if sc < best_score:
            best_sx, best_score = float(sx), sc
    return best_sx

# ---------------------- 메인 클래스 ----------------------
class Kitti2FT3DPreprocessAndSave:
    """
    6단계 전처리(테스트-타임) + 단계별 저장
    (1) 정렬 보정, (2) 좌우 색 정합, (3) 전역 톤, (4) H-Scale, (5) 디노이즈+샤픈, (6) 스펙큘러 억제
    sx_policy: {"none","fixed","ratio","search"}
    저장 결과: {save_root}/{sample_id}_{step:02d}_{name}_{L|R}.png
               ROI: {save_root}/{sample_id}_{step:02d}_{name}_{L|R}_roi.png
               차이맵(2단계): {sample_id}_02_diff_before.png, {sample_id}_02_diff_after.png
               메타: {sample_id}_meta.json
    """
    def __init__(self,
                 target_size=(1248, 384),
                 normalize=None,
                 # 1) 정렬 보정
                 rect_dy_range=(-2, 2),
                 # 2) 좌우 색 정합
                 color_match_method="meanvar",
                 # 3) 전역 톤
                 tone_gamma=1.15, tone_sat=1.15, tone_contrast=1.05,
                 # 4) H-Scale
                 sx_policy="ratio", sx_fixed=1.0, fxB_src=None, fxB_tgt=None, sx_candidates=(0.8, 1.0, 1.2, 1.4),
                 # 5) 디노이즈+샤픈
                 denoise_strength=2, unsharp_sigma=0.7, unsharp_amount=0.7,
                 # 6) 스펙큘러 억제
                 specular_v_thresh=0.85, specular_s_thresh=0.35, specular_reduce=0.85,
                 # 저장 옵션
                 save_root=None, save_full=True, save_roi=True,
                 roi_box=(0.70, 0.55, 0.96, 0.98), roi_scale=3.0,
                 save_meta=True):
        self.target_size = target_size
        self.normalize = normalize or transforms.Normalize(ImageProcessor.IMAGENET_MEAN, ImageProcessor.IMAGENET_STD)

        self.rect_dy_range = rect_dy_range
        self.color_match_method = color_match_method

        self.tone_gamma = tone_gamma
        self.tone_sat = tone_sat
        self.tone_contrast = tone_contrast

        self.sx_policy = sx_policy
        self.sx_fixed = sx_fixed
        self.fxB_src = fxB_src
        self.fxB_tgt = fxB_tgt
        self.sx_candidates = sx_candidates

        self.denoise_strength = denoise_strength
        self.unsharp_sigma = unsharp_sigma
        self.unsharp_amount = unsharp_amount

        self.specular_v_thresh = specular_v_thresh
        self.specular_s_thresh = specular_s_thresh
        self.specular_reduce = specular_reduce

        self.save_root = save_root
        self.save_full = save_full
        self.save_roi = save_roi
        self.roi_box = roi_box
        self.roi_scale = roi_scale
        self.save_meta = save_meta

        if self.save_root is not None:
            _ensure_dir(self.save_root)

    # 저장 헬퍼
    def _save_step(self, sample_id, step, name, imgL_pil, imgR_pil):
        if self.save_root is None:
            return
        if self.save_full:
            imgL_pil.save(os.path.join(self.save_root, f"{sample_id}_{step:02d}_{name}_L.png"))
            imgR_pil.save(os.path.join(self.save_root, f"{sample_id}_{step:02d}_{name}_R.png"))
        if self.save_roi:
            x0, y0, x1, y1 = self.roi_box
            roiL = _crop_roi_by_ratio(imgL_pil, x0, y0, x1, y1, self.roi_scale)
            roiR = _crop_roi_by_ratio(imgR_pil, x0, y0, x1, y1, self.roi_scale)
            roiL.save(os.path.join(self.save_root, f"{sample_id}_{step:02d}_{name}_L_roi.png"))
            roiR.save(os.path.join(self.save_root, f"{sample_id}_{step:02d}_{name}_R_roi.png"))

    def _save_diff(self, sample_id, before_pair, after_pair):
        if self.save_root is None:
            return
        Lb, Rb = before_pair
        La, Ra = after_pair
        d_before = _absdiff_heatmap(Lb, Rb)
        d_after  = _absdiff_heatmap(La, Ra)
        d_before.save(os.path.join(self.save_root, f"{sample_id}_02_diff_before.png"))
        d_after.save(os.path.join(self.save_root, f"{sample_id}_02_diff_after.png"))

    def __call__(self, left_pil: Image.Image, right_pil: Image.Image, sample_id: str = "sample"):
        # Stage 0: 원본 저장
        L0, R0 = left_pil.convert("RGB"), right_pil.convert("RGB")
        self._save_step(sample_id, 0, "orig", L0, R0)

        # 1) 정렬 보정
        dy = _estimate_vertical_shift(L0, R0, dy_range=self.rect_dy_range)
        L1, R1 = L0, _apply_vertical_shift_pil(R0, dy)
        self._save_step(sample_id, 1, f"rect_dy{dy:+d}", L1, R1)

        # 2) 좌우 색 정합
        L2, R2 = _color_match_right_to_left(L1, R1)
        self._save_step(sample_id, 2, "colormatch", L2, R2)
        self._save_diff(sample_id, before_pair=(L1, R1), after_pair=(L2, R2))

        # 3) 전역 톤
        L3, R3 = _progressive_tone_pair(L2, R2, gamma=self.tone_gamma,
                                        sat=self.tone_sat, contrast=self.tone_contrast)
        self._save_step(sample_id, 3, "tone", L3, R3)

        # 4) H-Scale 결정 및 적용
        if self.sx_policy == "fixed":
            sx = float(self.sx_fixed)
        elif self.sx_policy == "ratio":
            sx = _choose_sx_ratio(self.fxB_src, self.fxB_tgt, default=1.0)
        elif self.sx_policy == "search":
            sx = _choose_sx_by_stereoBM(L3, R3, sx_candidates=self.sx_candidates)
        else:
            sx = 1.0
        L4, metaL = _horizontal_stretch_fix_width(L3, sx)
        R4, metaR = _horizontal_stretch_fix_width(R3, sx)
        self._save_step(sample_id, 4, f"hscale_sx{sx:.2f}", L4, R4)

        # 5) 디노이즈 + 샤픈
        L5 = _light_denoise_np(np.array(L4, dtype=np.uint8), strength=self.denoise_strength)
        R5 = _light_denoise_np(np.array(R4, dtype=np.uint8), strength=self.denoise_strength)
        L5 = _unsharp_mask_np(L5, blur_sigma=self.unsharp_sigma, amount=self.unsharp_amount)
        R5 = _unsharp_mask_np(R5, blur_sigma=self.unsharp_sigma, amount=self.unsharp_amount)
        L5_pil, R5_pil = Image.fromarray(L5), Image.fromarray(R5)
        self._save_step(sample_id, 5, "sharpen", L5_pil, R5_pil)

        # 6) 스펙큘러 억제
        L6 = _suppress_specular_np(L5, v_thresh=self.specular_v_thresh,
                                   s_thresh=self.specular_s_thresh, reduce=self.specular_reduce)
        R6 = _suppress_specular_np(R5, v_thresh=self.specular_v_thresh,
                                   s_thresh=self.specular_s_thresh, reduce=self.specular_reduce)
        L6_pil, R6_pil = Image.fromarray(L6), Image.fromarray(R6)
        self._save_step(sample_id, 6, "specsuppress", L6_pil, R6_pil)

        # 메타 저장
        meta = {
            "dy": dy,
            "sx": float(sx),
            "tone": {"gamma": self.tone_gamma, "sat": self.tone_sat, "contrast": self.tone_contrast},
            "rect_dy_range": list(self.rect_dy_range),
        }
        if self.save_root is not None and self.save_meta:
            with open(os.path.join(self.save_root, f"{sample_id}_meta.json"), "w") as f:
                json.dump(meta, f, indent=2)

        # 텐서 변환 + 패딩 + 정규화
        tL = _tensorize_and_normalize(L6_pil, self.target_size, self.normalize)
        tR = _tensorize_and_normalize(R6_pil, self.target_size, self.normalize)

        return {"left_pre": tL, "right_pre": tR, "meta": meta}
# =========================================
# main(): 입력 경로를 받아 6단계 전처리 결과 저장
#   - 전제: Kitti2FT3DPreprocessAndSave, ImageProcessor 가 위에 정의되어 있음
# =========================================
import argparse
import os
from pathlib import Path
from PIL import Image
import numpy as np
import torchvision.transforms as T

def _save_final_padded_from_step6(step6_L_path, step6_R_path, target_size, out_dir, sample_id):
    """
    step6(스펙큘러 억제) 이미지를 타깃 해상도(TARGET_SIZE)로 패딩한
    '모델 입력용 최종 이미지(정규화 전, uint8)'를 저장합니다.
    """
    step6_L_path = Path(step6_L_path)
    step6_R_path = Path(step6_R_path)
    if not (step6_L_path.exists() and step6_R_path.exists()):
        print("[WARN] step6 이미지가 없어 최종 패딩본 저장을 건너뜁니다.")
        return

    L6 = Image.open(step6_L_path).convert("RGB")
    R6 = Image.open(step6_R_path).convert("RGB")

    to_tensor = T.ToTensor()
    tL = to_tensor(L6) * 255.0  # [C,H,W], 0..255
    tR = to_tensor(R6) * 255.0

    # ImageProcessor.pad_to_size는 질문자 코드의 구현을 사용합니다.
    tL_pad = ImageProcessor.pad_to_size(tL, target_size)
    tR_pad = ImageProcessor.pad_to_size(tR, target_size)

    # tensor -> uint8 PIL
    L_pad = np.clip(tL_pad.permute(1, 2, 0).cpu().numpy(), 0, 255).astype(np.uint8)
    R_pad = np.clip(tR_pad.permute(1, 2, 0).cpu().numpy(), 0, 255).astype(np.uint8)
    L_pad = Image.fromarray(L_pad)
    R_pad = Image.fromarray(R_pad)

    L_pad.save(Path(out_dir) / f"{sample_id}_final_padded_L.png")
    R_pad.save(Path(out_dir) / f"{sample_id}_final_padded_R.png")


def build_argparser():
    p = argparse.ArgumentParser(description="KITTI → FT3D 스타일 6단계 전처리(저장 포함)")
    p.add_argument("--left",  default="/home/jaejun/dataset/kitti_2015/training/image_2/000020_10.png")
    p.add_argument("--right", default="/home/jaejun/dataset/kitti_2015/training/image_3/000020_10.png")
    p.add_argument("--out",   default="./outputs")
    p.add_argument("--id",    default=None,  help="샘플 ID(미지정 시 left 파일명 기반)")

    # 수평 스케일 정책
    p.add_argument("--sx-policy", choices=["none", "fixed", "ratio", "search"],
                   default="ratio", help="H-Scale 결정 정책")
    p.add_argument("--sx-fixed", type=float, default=1.0, help="sx-policy=fixed일 때 수평 스케일 값")
    p.add_argument("--fxB-src",  type=float, default=None, help="sx-policy=ratio일 때 FT3D 측 fx*B")
    p.add_argument("--fxB-tgt",  type=float, default=None, help="sx-policy=ratio일 때 KITTI 측 fx*B")
    p.add_argument("--sx-candidates", type=str, default="0.8,1.0,1.2,1.4",
                   help="sx-policy=search일 때 후보 리스트(콤마 구분)")

    # 톤/정렬/노이즈 파라미터(필요 시 조정)
    p.add_argument("--rect-range", type=str, default="-2,2", help="수직 정렬 탐색범위 예: -2,2")
    p.add_argument("--tone-gamma",   type=float, default=1.15)
    p.add_argument("--tone-sat",     type=float, default=1.15)
    p.add_argument("--tone-contrast",type=float, default=1.05)
    p.add_argument("--denoise",      type=int,   default=2)
    p.add_argument("--unsharp-sigma",type=float, default=0.7)
    p.add_argument("--unsharp-amt",  type=float, default=0.7)
    p.add_argument("--spec-v",       type=float, default=0.85)
    p.add_argument("--spec-s",       type=float, default=0.35)
    p.add_argument("--spec-reduce",  type=float, default=0.85)

    # 저장 옵션
    p.add_argument("--no-roi", action="store_true", help="ROI 저장 비활성화")
    p.add_argument("--no-full", action="store_true", help="단계별 전체 프레임 저장 비활성화")
    return p


def main():
    args = build_argparser().parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 샘플 ID 기본값: 왼쪽 파일명(확장자 제외)
    sample_id = args.id or Path(args.left).stem

    # 입력 로드
    left_pil  = Image.open(args.left).convert("RGB")
    right_pil = Image.open(args.right).convert("RGB")

    # 파라미터 파싱
    sx_candidates = tuple(float(x) for x in args.sx_candidates.split(","))
    rect_range = tuple(int(x) for x in args.rect_range.split(","))  # (-2,2) 등

    # 전처리기 생성 (앞서 정의된 클래스 사용)
    preproc = Kitti2FT3DPreprocessAndSave(
        target_size=ImageProcessor.TARGET_SIZE,
        # 1) 정렬
        rect_dy_range=rect_range,
        # 2) 색 정합
        color_match_method="meanvar",
        # 3) 톤
        tone_gamma=args.tone_gamma, tone_sat=args.tone_sat, tone_contrast=args.tone_contrast,
        # 4) H-Scale
        sx_policy=args.sx_policy, sx_fixed=args.sx_fixed,
        fxB_src=args.fxB_src, fxB_tgt=args.fxB_tgt, sx_candidates=sx_candidates,
        # 5) 디노이즈/샤픈
        denoise_strength=args.denoise, unsharp_sigma=args.unsharp_sigma, unsharp_amount=args.unsharp_amt,
        # 6) 스펙큘러
        specular_v_thresh=args.spec_v, specular_s_thresh=args.spec_s, specular_reduce=args.spec_reduce,
        # 저장
        save_root=str(out_dir),
        save_full=not args.no_full,
        save_roi=not args.no_roi,
        roi_box=(0.70, 0.55, 0.96, 0.98),  # 검은 차량 영역(대략) 확대
        roi_scale=3.0,
        save_meta=True
    )

    # 실행(단계별 PNG + ROI + 메타 저장)
    out = preproc(left_pil, right_pil, sample_id=sample_id)

    # 최종(step6) 이미지를 TARGET_SIZE로 패딩한 '모델입력용' PNG도 저장
    step6_L = out_dir / f"{sample_id}_06_specsuppress_L.png"
    step6_R = out_dir / f"{sample_id}_06_specsuppress_R.png"
    _save_final_padded_from_step6(step6_L, step6_R, ImageProcessor.TARGET_SIZE, out_dir, sample_id)

    print(f"[DONE] 저장 경로: {out_dir}")
    print(f" - 단계별: {sample_id}_00~06_*.png")
    print(f" - 최종 패딩본: {sample_id}_final_padded_L.png, {sample_id}_final_padded_R.png")
    print(f" - 메타: {sample_id}_meta.json")


if __name__ == "__main__":
    main()
