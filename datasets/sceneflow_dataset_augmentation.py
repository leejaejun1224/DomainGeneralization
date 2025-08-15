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
from datasets.aug.multiple_aug import *

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
# FlyingThingDataset
# =========================================================
class FlyingThingDataset(Dataset):
    def __init__(self, datapath, list_filename, training, max_len=None, aug=True, prior=None,
                 use_aux_feats=True,             # 보조 특징 채널 사용
                 invalid_disp_nonpos=True,       # disparity <= 0 invalid 처리
                 negate_disp=True,               # 반환 시 disparity 부호 반전(기존 파이프라인 호환)
                 erase_low=False,                # 구코드 호환 옵션(권장: False)
                 min_valid_ratio=0.01,           # 크롭 내 유효 disparity 최소 비율
                 max_crop_tries=30,
                 occ_root="~/dataset/flyingthing/FlyingThings3D_subset_disparity_occlusions/FlyingThings3D_subset/train/disparity_occlusions",
                 use_occ_left=True   # 왼쪽 뷰 기준 마스크 사용
                 ):             # 크롭 재시도 횟수
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
        
        self.occ_root = os.path.expanduser(occ_root) if occ_root is not None else None
        self.use_occ_left = bool(use_occ_left)
        self._occ_idx = {"left": None, "right": None}  # suffix/basename 매핑

        if self.occ_root and os.path.isdir(self.occ_root):
            self._build_occ_index()   # 한 번만 인덱싱
        else:
            self.occ_root = None  # 경로 없으면 사용 안 함
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
    # ---- 추가: occlusion 파일 인덱싱(좌/우 각각) ----
    def _build_occ_index(self):
        def collect(view):
            base = os.path.join(self.occ_root, view)
            if not os.path.isdir(base):
                return {"by_rel": {}, "by_base": {}}

            by_rel, by_base = {}, {}
            for dp, _, fnames in os.walk(base):
                for f in fnames:
                    if not f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                        continue
                    full = os.path.join(dp, f)
                    rel = os.path.relpath(full, base).replace("\\", "/")  # view/ 이후 경로
                    by_rel[rel] = full
                    by_base.setdefault(f, []).append(full)
            return {"by_rel": by_rel, "by_base": by_base}

        self._occ_idx["left"]  = collect("left")
        self._occ_idx["right"] = collect("right")

    # ---- 추가: 원본 left 파일 경로로부터 occlusion 파일 경로 찾기 ----
    def _find_occ_path_for_left(self, left_rel_path):
        """
        left_rel_path: 리스트 파일에 들어있던 left 상대경로(또는 그에 준하는 문자열).
        우선 순위:
          1) '.../left/<suffix>' 의 <suffix>로 매칭  (권장 구조)
          2) basename(예: '000000.png')으로 매칭
        """
        if not self.occ_root:
            return None

        p = left_rel_path.replace("\\", "/")
        # left/ 이후 suffix 추출
        key = None
        if "/left/" in p:
            key = p.split("/left/", 1)[1]
        elif p.startswith("left/"):
            key = p[len("left/"):]
        # 1) suffix로 시도
        if key:
            hit = self._occ_idx["left"]["by_rel"].get(key)
            if hit and os.path.exists(hit):
                return hit
        # 2) basename으로 fallback
        base = os.path.basename(p)
        cands = self._occ_idx["left"]["by_base"].get(base, [])
        if len(cands) == 1:
            return cands[0]
        elif len(cands) > 1:
            # 동일 이름이 여럿이면 첫 번째 선택(필요 시 더 정교한 규칙 추가)
            return sorted(cands)[0]
        return None

    # ---- 추가: occlusion PNG를 0/1 float32로 읽기 ----
    @staticmethod
    def _read_occ_mask_png(path):
        """
        Sceneflow occlusion PNG: 255=occluded, 0=non-occluded
        반환: np.float32 [H,W], {0.0, 1.0}
        """
        if path is None or (not os.path.exists(path)):
            return None
        m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if m is None:
            return None
        return (m > 127).astype(np.float32)

    def __len__(self):
        return self.max_len if self.max_len is not None else self.data_len

    # ---------- 메인 ----------
    def __getitem__(self, index):
        index = index if index < self.data_len else random.randint(0, self.data_len - 1)
        left_path_rel  = self.left_filenames[index]   # 리스트 파일에 기록된 상대경로
        right_path_rel = self.right_filenames[index]

        left_img = self.load_image(os.path.join(self.datapath, left_path_rel))
        right_img = self.load_image(os.path.join(self.datapath, right_path_rel))
        disparity = self.load_disp(os.path.join(self.datapath, self.disp_filenames[index]))
        
        occ_full = None
        if self.use_occ_left and self.occ_root:
            occ_path = self._find_occ_path_for_left(left_path_rel)
            occ_full = self._read_occ_mask_png(occ_path)
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
                if occ_full is not None:
                    occ_full = np.pad(occ_full, ((pad_h, 0), (0, pad_w)), mode="constant", constant_values=0.0)


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
            if occ_full is not None:
                occ_full = occ_full[y1:y1 + crop_h, x1:x1 + crop_w]
            # disparity = np.nan_to_num(disparity, nan=0.0, posinf=0.0, neginf=0.0)

            # ---------- 저해상도 GT ----------
            disparity_low    = cv2.resize(disparity, (crop_w//4, crop_h//4), interpolation=cv2.INTER_NEAREST)
            disparity_low_r8 = cv2.resize(disparity, (crop_w//8, crop_h//8), interpolation=cv2.INTER_NEAREST)
            
            if occ_full is not None:
                occ_low = cv2.resize(occ_full, (crop_w//4, crop_h//4), interpolation=cv2.INTER_NEAREST)
            else:
                occ_low = None

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
                
                
            if occ_full is not None:
                sample["occ_mask"] = torch.from_numpy(occ_full).float()          # [H,W], 1=occluded
                if occ_low is not None:
                    sample["occ_mask_low"] = torch.from_numpy(occ_low).float()    # [H/4,W/4]
                
                
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
            
            
            occ_full = None
            if self.use_occ_left and self.occ_root:
                occ_path = self._find_occ_path_for_left(left_path_rel)
                occ_full = self._read_occ_mask_png(occ_path)
                if occ_full is not None:
                    occ_full = occ_full[h - crop_h:h, w - crop_w:w]
                    occ_low  = cv2.resize(occ_full, (crop_w//4, crop_h//4), interpolation=cv2.INTER_NEAREST)

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
            if occ_full is not None:
                
                sample["occ_mask"] = torch.from_numpy(occ_full).float()
                sample["occ_mask_low"] = torch.from_numpy(occ_low).float()
                
            if self.negate_disp:
                sample["disparity"]     = -sample["disparity"]
                sample["disparity_low"] = -sample["disparity_low"]
            return sample
