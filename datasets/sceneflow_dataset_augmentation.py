# =========================================
# (1) Imports
# =========================================
import os
import random
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import cv2
from PIL import Image
from torch.utils.data import Dataset
import torch

from datasets.data_io import get_transform, read_all_lines, pfm_imread
import torchvision.transforms as transforms  # 호환 유지
from datasets.aug.multiple_aug import *      # StereoAugmentor, aux feature funcs
# from datasets.aug.car_patch_modeule import CarPatchAugmenter  # 사용 시 주석 해제


# =========================================
# (2) Helpers: disparity 유효 마스크
# =========================================
def _left_valid_mask_signed(disp: np.ndarray, negate: bool = False) -> np.ndarray:
    """
    Left 기준 유효 마스크. negate=True면 내부적으로 부호를 뒤집어(+ disparity) 좌표 체크.
    반환: uint8 [H,W], {0,1}
    """
    dd = (-disp if negate else disp).astype(np.float32)
    h, w = dd.shape
    xs = np.arange(w, dtype=np.float32)[None, :].repeat(h, axis=0)
    valid = (dd > 0.0) & (xs - dd >= 0.0) & (xs - dd < w)
    return valid.astype(np.uint8)


# =========================================
# (3) Helpers: 파일/경로 & 마스크 I/O
# =========================================
def _nan_to_zero_f32(arr: np.ndarray) -> np.ndarray:
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)

def _read_occ_mask_png(path: Optional[str]) -> Optional[np.ndarray]:
    """
    Sceneflow occlusion PNG: 255=occluded, 0=non-occluded
    반환: np.float32 [H,W], {0.0, 1.0} 또는 None
    """
    if not path:
        return None
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None
    return (m > 127).astype(np.float32)

def _expanduser_norm(path: str) -> str:
    return os.path.expanduser(path).replace("\\", "/")

def _img_to_np(img: Image.Image) -> np.ndarray:
    # PIL(Image, RGB) -> np.uint8[H, W, 3]
    return np.array(img, copy=False)

def _np_to_img(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(arr)


# =========================================
# (4) Helpers: 패딩 & 크롭 좌표
# =========================================
def _pad_images_for_crop(
    left_np: np.ndarray,
    right_np: np.ndarray,
    disp_left: np.ndarray,
    target_w: int,
    target_h: int,
    occ_left: Optional[np.ndarray] = None,
    disp_right: Optional[np.ndarray] = None,
    occ_right: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    (left/right RGB 이미지, 좌/우 disparity, 좌/우 오큘루전)을
    요청 크롭 크기를 만족하도록 오른쪽/아래에 패딩.
    - 이미지/디스패리티: 'edge' 패딩
    - 오큘루전: 0 상수 패딩
    """
    h, w = left_np.shape[:2]
    pad_w = max(0, target_w - w)
    pad_h = max(0, target_h - h)
    if pad_w == 0 and pad_h == 0:
        return left_np, right_np, disp_left, occ_left, disp_right, occ_right

    pad_img = ((pad_h, 0), (0, pad_w), (0, 0))
    pad_arr = ((pad_h, 0), (0, pad_w))

    left_np  = np.pad(left_np,  pad_img, mode='edge')
    right_np = np.pad(right_np, pad_img, mode='edge')
    disp_left = np.pad(disp_left, pad_arr, mode='edge')

    if occ_left is not None:
        occ_left = np.pad(occ_left, pad_arr, mode='constant', constant_values=0.0)
    if disp_right is not None:
        disp_right = np.pad(disp_right, pad_arr, mode='edge')
    if occ_right is not None:
        occ_right = np.pad(occ_right, pad_arr, mode='constant', constant_values=0.0)

    return left_np, right_np, disp_left, occ_left, disp_right, occ_right


def _random_crop_with_min_valid(
    disp_left: np.ndarray,
    crop_w: int,
    crop_h: int,
    min_valid_ratio: float,
    negate_disp: bool,
    max_tries: int,
) -> Tuple[int, int]:
    """
    좌 disparity에서 유효 픽셀 비율(min_valid_ratio)을 만족하는
    크롭 좌표 (x1, y1)를 랜덤 시도 내에서 선택.
    실패 시 마지막 시도로 무조건 랜덤 좌표 반환.
    """
    H, W = disp_left.shape
    tries = 0
    last_xy = (0, 0)

    while tries < max_tries:
        x1 = random.randint(0, W - crop_w)
        y1 = random.randint(0, H - crop_h)
        last_xy = (x1, y1)

        patch = disp_left[y1:y1 + crop_h, x1:x1 + crop_w]
        dd = (-patch if negate_disp else patch)
        valid_ratio = float((dd > 0).mean()) if dd.size > 0 else 0.0
        if valid_ratio >= min_valid_ratio:
            return x1, y1
        tries += 1

    # 실패: 마지막 후보 반환
    return last_xy


def _resize_quarter(arr: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    return cv2.resize(arr, (out_w, out_h), interpolation=cv2.INTER_NEAREST)


# =========================================
# (5) Dataset 본체
# =========================================
class FlyingThingDataset(Dataset):
    """
    FlyingThings3D 계열 스테레오 데이터셋.
    - 좌/우 RGB, 좌/우 disparity(.pfm), 좌/우 occlusion(.png) 지원
    - 학습 시 랜덤 크롭 + (선택)증강, 평가 시 고정 크롭
    """

    def __init__(
        self,
        datapath: str,
        list_filename: str,
        training: bool,
        max_len: Optional[int] = None,
        aug: bool = True,
        prior: Optional[str] = None,

        use_aux_feats: bool = True,         # 보조 특징 채널 사용
        invalid_disp_nonpos: bool = False,  # True면 disparity <= 0 무효화(원본 실동작과 일치 위해 False가 기본)
        negate_disp: bool = True,           # 반환 시 disparity 부호 반전(기존 파이프라인 호환)
        erase_low: bool = False,            # (비권장) 구코드 호환 옵션
        min_valid_ratio: float = 0.01,      # 크롭 내 유효 disparity 최소 비율
        max_crop_tries: int = 30,

        occ_root: Optional[str] = "~/dataset/flyingthing/FlyingThings3D_subset_disparity_occlusions/FlyingThings3D_subset/train/disparity_occlusions",
        right_disp_root: Optional[str] = "/home/jaejun/dataset/flyingthing/FlyingThings3D_subset_disparity/FlyingThings3D_subset/train/disparity/right",
        use_occ_left: bool = True,
        use_occ_right: bool = True,
    ):
        # --- 기본 경로/설정 ---
        self.datapath = _expanduser_norm(datapath)
        self.training = bool(training)
        self.max_len = max_len
        self.aug = bool(aug)
        self.prior_path = _expanduser_norm(prior) if prior is not None else None

        self.use_aux_feats = bool(use_aux_feats)
        self.invalid_disp_nonpos = bool(invalid_disp_nonpos)
        self.negate_disp = bool(negate_disp)
        self.erase_low = bool(erase_low)
        self.min_valid_ratio = float(min_valid_ratio)
        self.max_crop_tries = int(max_crop_tries)

        self.occ_root = _expanduser_norm(occ_root) if occ_root is not None else None
        self.right_disp_root = _expanduser_norm(right_disp_root) if right_disp_root is not None else None
        self.use_occ_left = bool(use_occ_left)
        self.use_occ_right = bool(use_occ_right)

        # --- 리스트 파일 로드 ---
        self.left_filenames, self.right_filenames, self.disp_filenames = self._load_path(list_filename)
        self.data_len = len(self.left_filenames)
        if self.training:
            assert self.disp_filenames is not None, "Training=True면 disparity GT가 필요합니다."

        # --- Occlusion 인덱스 사전 구축(옵션) ---
        self._occ_idx: Dict[str, Dict[str, Dict[str, List[str]]]] = {"left": None, "right": None}
        if self.occ_root and os.path.isdir(self.occ_root):
            self._build_occ_index()
        else:
            self.occ_root = None  # 사용 안 함

        # --- 증강기 구성 ---
        self.augmentor = StereoAugmentor() if (self.training and self.aug) else None
        self.car_patch_aug = None
        # 필요 시 활성화 예시:
        # self.car_patch_aug = CarPatchAugmenter(...) if (self.training and self.aug) else None

        self._transform = get_transform()

    # -----------------------------------------
    # (5-1) 경로/파일 로드
    # -----------------------------------------
    def _load_path(self, list_filename: str) -> Tuple[List[str], List[str], Optional[List[str]]]:
        lines = read_all_lines(list_filename)
        splits = [line.split() for line in lines]
        left_images = [x[0] for x in splits]
        right_images = [x[1] for x in splits]
        if len(splits[0]) == 2:
            return left_images, right_images, None
        else:
            disp_images = [x[2] for x in splits]
            return left_images, right_images, disp_images

    def _load_image(self, filename: str) -> Image.Image:
        return Image.open(_expanduser_norm(filename)).convert('RGB')

    def _load_prior(self):
        if self.prior_path is None:
            return 0.0
        path = _expanduser_norm(self.prior_path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Prior file {path} does not exist.")
        return np.load(path)

    def _load_disp(self, filename: str) -> np.ndarray:
        data, _scale = pfm_imread(_expanduser_norm(filename))
        return _nan_to_zero_f32(np.ascontiguousarray(data, dtype=np.float32))

    def _derive_right_disp_path(self, left_disp_full_path: str) -> Optional[str]:
        """
        left disp의 전체 경로에서 '/left/'를 '/right/'로 치환.
        해당 토큰이 없으면 basename만 사용해 self.right_disp_root 아래로 구성.
        """
        p = _expanduser_norm(left_disp_full_path)
        if "/left/" in p:
            right_p = p.replace("/left/", "/right/")
            return right_p
        if self.right_disp_root:
            return os.path.join(self.right_disp_root, os.path.basename(p))
        return None

    # -----------------------------------------
    # (5-2) Occlusion 인덱스/탐색
    # -----------------------------------------
    def _build_occ_index(self):
        def collect(view: str) -> Dict[str, Dict[str, List[str]]]:
            base = os.path.join(self.occ_root, view)
            if not os.path.isdir(base):
                return {"by_rel": {}, "by_base": {}}

            by_rel, by_base = {}, {}
            for dp, _, fnames in os.walk(base):
                for f in fnames:
                    if not f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                        continue
                    full = os.path.join(dp, f)
                    rel = os.path.relpath(full, base).replace("\\", "/")
                    by_rel[rel] = full
                    by_base.setdefault(f, []).append(full)
            return {"by_rel": by_rel, "by_base": by_base}

        self._occ_idx["left"]  = collect("left")
        self._occ_idx["right"] = collect("right")

    def _find_occ_path_for_view(self, rel_path: str, view: str) -> Optional[str]:
        """
        rel_path: 리스트 파일에 들어있던 상대 경로(또는 유사 문자열).
        우선 순위:
          1) '.../view/<suffix>' 의 <suffix>로 매칭
          2) basename으로 매칭(여럿이면 사전순 첫 항목)
        """
        if not self.occ_root:
            return None
        idx = self._occ_idx.get(view)
        if not idx:
            return None

        p = rel_path.replace("\\", "/")
        key = None
        token = f"/{view}/"
        if token in p:
            key = p.split(token, 1)[1]
        elif p.startswith(f"{view}/"):
            key = p[len(f"{view}/"):]
        if key:
            hit = idx["by_rel"].get(key)
            if hit and os.path.exists(hit):
                return hit

        base = os.path.basename(p)
        cands = idx["by_base"].get(base, [])
        if len(cands) == 1:
            return cands[0]
        elif len(cands) > 1:
            return sorted(cands)[0]
        return None

    # -----------------------------------------
    # (5-3) 기본 인터페이스
    # -----------------------------------------
    def __len__(self) -> int:
        return self.max_len if self.max_len is not None else self.data_len

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        # 인덱스 보정(샘플링 확장 허용)
        if index >= self.data_len:
            index = random.randint(0, self.data_len - 1)

        # --- 경로 확보 ---
        left_rel  = self.left_filenames[index]
        right_rel = self.right_filenames[index]

        left_img  = self._load_image(os.path.join(self.datapath, left_rel))
        right_img = self._load_image(os.path.join(self.datapath, right_rel))

        # 좌 disparity
        disp_left_full = os.path.join(self.datapath, self.disp_filenames[index]) if self.disp_filenames else None
        disparity_left = self._load_disp(disp_left_full) if disp_left_full else None

        # 우 disparity(있으면)
        disparity_right = None
        if disp_left_full:
            right_disp_path = self._derive_right_disp_path(disp_left_full)
            if right_disp_path and os.path.exists(_expanduser_norm(right_disp_path)):
                disparity_right = self._load_disp(right_disp_path)

        # Occlusion 마스크(옵션)
        occ_left = None
        occ_right = None
        if self.occ_root:
            if self.use_occ_left:
                occ_left_path = self._find_occ_path_for_view(left_rel, "left")
                occ_left = _read_occ_mask_png(occ_left_path)
            if self.use_occ_right:
                occ_right_path = self._find_occ_path_for_view(right_rel, "right")
                occ_right = _read_occ_mask_png(occ_right_path)

        # Prior
        prior_data = self._load_prior()

        # -------------------------------------
        # 학습 경로
        # -------------------------------------
        if self.training:
            assert disparity_left is not None, "Training=True면 disparity GT가 필요합니다."

            # (선택) Car Patch Aug
            if self.aug and self.car_patch_aug is not None:
                L_np = _img_to_np(left_img)
                R_np = _img_to_np(right_img)
                disparity_left = disparity_left.astype(np.float32)
                L_np, R_np, disparity_left = self.car_patch_aug(L_np, R_np, disparity_left)
                left_img, right_img = _np_to_img(L_np), _np_to_img(R_np)

            # (선택) StereoAug
            if self.aug and self.augmentor is not None:
                L_np = _img_to_np(left_img)
                R_np = _img_to_np(right_img)
                L_np, R_np, disparity_left = self.augmentor(L_np, R_np, disparity_left)
                left_img, right_img = _np_to_img(L_np), _np_to_img(R_np)
                disparity_left = _nan_to_zero_f32(disparity_left)

            # (옵션) invalid_disp_nonpos 처리
            if self.invalid_disp_nonpos:
                disparity_left = disparity_left.copy()
                disparity_left[disparity_left <= 0] = 0.0

            # 크롭 사이즈 선택
            crop_list = [(512, 128), (512, 256), (768, 256), (768, 512)]
            crop_w, crop_h = random.choice(crop_list)

            # 패딩(필요 시)
            L_np = _img_to_np(left_img)
            R_np = _img_to_np(right_img)
            L_np, R_np, disparity_left, occ_left, disparity_right, occ_right = _pad_images_for_crop(
                L_np, R_np, disparity_left, crop_w, crop_h, occ_left, disparity_right, occ_right
            )

            # 크롭 좌표 선택(유효 비율 보장)
            x1, y1 = _random_crop_with_min_valid(
                disparity_left, crop_w, crop_h, self.min_valid_ratio, self.negate_disp, self.max_crop_tries
            )

            # 실제 크롭
            left_img  = _np_to_img(L_np[y1:y1 + crop_h, x1:x1 + crop_w])
            right_img = _np_to_img(R_np[y1:y1 + crop_h, x1:x1 + crop_w])
            disparity = disparity_left[y1:y1 + crop_h, x1:x1 + crop_w]

            disp_right_crop = None
            if disparity_right is not None:
                disp_right_crop = disparity_right[y1:y1 + crop_h, x1:x1 + crop_w]

            occ_left_crop = occ_left[y1:y1 + crop_h, x1:x1 + crop_w] if occ_left is not None else None
            occ_right_crop = occ_right[y1:y1 + crop_h, x1:x1 + crop_w] if occ_right is not None else None

            # 저해상도 GT/마스크
            q_w, q_h = crop_w // 4, crop_h // 4
            disparity_low = _resize_quarter(disparity, q_w, q_h)
            disp_right_low = _resize_quarter(disp_right_crop, q_w, q_h) if disp_right_crop is not None else None
            occ_left_low  = _resize_quarter(occ_left_crop,  q_w, q_h) if occ_left_crop is not None else None
            occ_right_low = _resize_quarter(occ_right_crop, q_w, q_h) if occ_right_crop is not None else None

            # 보조 특징(옵션)
            aux_left = aux_right = None
            if self.use_aux_feats:
                L_np_crop = _img_to_np(left_img)
                R_np_crop = _img_to_np(right_img)
                gradL = gradient_magnitude_rgb(L_np_crop)
                logL  = log_magnitude(L_np_crop, sigma=1.0)
                rankL = rank_transform(L_np_crop, win=7)
                cohL  = structure_tensor_coherence(L_np_crop, sigma=2.0, blur=3)
                aux_left = torch.from_numpy(np.stack([gradL, logL, rankL, cohL], axis=0)).float()

                gradR = gradient_magnitude_rgb(R_np_crop)
                logR  = log_magnitude(R_np_crop, sigma=1.0)
                rankR = rank_transform(R_np_crop, win=7)
                cohR  = structure_tensor_coherence(R_np_crop, sigma=2.0, blur=3)
                aux_right = torch.from_numpy(np.stack([gradR, logR, rankR, cohR], axis=0)).float()

            # 텐서 변환
            left_t  = self._transform(left_img)
            right_t = self._transform(right_img)

            # Prior 텐서화
            if isinstance(prior_data, np.ndarray):
                prior_t = torch.from_numpy(_nan_to_zero_f32(prior_data)).float()
            else:
                prior_t = torch.tensor(float(prior_data), dtype=torch.float32)

            # 샘플 구성
            sample: Dict[str, torch.Tensor] = {
                "left": left_t,
                "right": right_t,
                "disparity": torch.from_numpy(disparity).float(),
                "disparity_low": torch.from_numpy(disparity_low).float(),
                "left_filename": self.left_filenames[index],
                "right_filename": self.right_filenames[index],
                "prior": prior_t,
            }
            if disp_right_crop is not None:
                sample["disparity_right"] = torch.from_numpy(disp_right_crop).float()
            if disp_right_low is not None:
                sample["disparity_right_low"] = torch.from_numpy(disp_right_low).float()

            if occ_left_crop is not None:
                sample["occ_mask"] = torch.from_numpy(occ_left_crop).float()
            if occ_left_low is not None:
                sample["occ_mask_low"] = torch.from_numpy(occ_left_low).float()

            if occ_right_crop is not None:
                sample["occ_mask_right"] = torch.from_numpy(occ_right_crop).float()
            if occ_right_low is not None:
                sample["occ_mask_right_low"] = torch.from_numpy(occ_right_low).float()

            if aux_left is not None and aux_right is not None:
                sample["aux_left"]  = aux_left
                sample["aux_right"] = aux_right

            # 부호 일관성
            if self.negate_disp:
                sample["disparity"]     = -sample["disparity"]
                sample["disparity_low"] = -sample["disparity_low"]
                if "disparity_right" in sample:
                    sample["disparity_right"] = -sample["disparity_right"]
                if "disparity_right_low" in sample:
                    sample["disparity_right_low"] = -sample["disparity_right_low"]

            return sample

        # -------------------------------------
        # 평가 경로 (증강 없음, 하단-우측 고정 크롭)
        # -------------------------------------
        w, h = left_img.size
        crop_w, crop_h = 960, 512
        x1, y1 = max(0, w - crop_w), max(0, h - crop_h)

        left_img  = left_img.crop((x1, y1, x1 + crop_w, y1 + crop_h))
        right_img = right_img.crop((x1, y1, x1 + crop_w, y1 + crop_h))

        disparity = disparity_left[y1:y1 + crop_h, x1:x1 + crop_w]
        disparity = _nan_to_zero_f32(disparity)

        disp_right_crop = None
        if disparity_right is not None:
            disp_right_crop = disparity_right[y1:y1 + crop_h, x1:x1 + crop_w]

        occ_left_crop = None
        occ_right_crop = None
        if self.use_occ_left and occ_left is not None:
            occ_left_crop = occ_left[y1:y1 + crop_h, x1:x1 + crop_w]
        if self.use_occ_right and occ_right is not None:
            occ_right_crop = occ_right[y1:y1 + crop_h, x1:x1 + crop_w]

        q_w, q_h = crop_w // 4, crop_h // 4
        disparity_low = _resize_quarter(disparity, q_w, q_h)

        left_t  = self._transform(left_img)
        right_t = self._transform(right_img)

        sample: Dict[str, torch.Tensor] = {
            "left": left_t,
            "right": right_t,
            "disparity": torch.from_numpy(disparity).float(),
            "top_pad": torch.tensor(0, dtype=torch.int32),
            "right_pad": torch.tensor(0, dtype=torch.int32),
            "disparity_low": torch.from_numpy(disparity_low).float(),
            "left_filename": self.left_filenames[index],
            "right_filename": self.right_filenames[index],
        }

        if disp_right_crop is not None:
            disp_right_low = _resize_quarter(disp_right_crop, q_w, q_h)
            sample["disparity_right"] = torch.from_numpy(disp_right_crop).float()
            sample["disparity_right_low"] = torch.from_numpy(disp_right_low).float()

        if occ_left_crop is not None:
            occ_left_low = _resize_quarter(occ_left_crop, q_w, q_h)
            sample["occ_mask"] = torch.from_numpy(occ_left_crop).float()
            sample["occ_mask_low"] = torch.from_numpy(occ_left_low).float()

        if occ_right_crop is not None:
            occ_right_low = _resize_quarter(occ_right_crop, q_w, q_h)
            sample["occ_mask_right"] = torch.from_numpy(occ_right_crop).float()
            sample["occ_mask_right_low"] = torch.from_numpy(occ_right_low).float()

        if self.negate_disp:
            sample["disparity"]     = -sample["disparity"]
            sample["disparity_low"] = -sample["disparity_low"]
            if "disparity_right" in sample:
                sample["disparity_right"] = -sample["disparity_right"]
            if "disparity_right_low" in sample:
                sample["disparity_right_low"] = -sample["disparity_right_low"]

        return sample
