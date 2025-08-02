import os
import random
from PIL import Image, ImageFile
from torch.utils.data import Dataset
import numpy as np
import cv2
from datasets.data_io import get_transform, read_all_lines, pfm_imread
import torchvision.transforms as transforms
import torch
import torchvision
ImageFile.LOAD_TRUNCATED_IMAGES = True


def _random_rect_shadow(img,
                        h_ratio=(0.2, 0.5),    # 직사각형 높이 비율
                        w_ratio=(0.1, 0.9),    # 너비 비율
                        y_min_ratio=0.35,        # 그림자 y 시작 하한(전체비율)
                        alpha_range=(0.25, 0.5),# 어둡게 정도(1=원본)
                        blur_sigma=1):
    """
    이미지 하단부에 부드러운 네모 그림자 적용 후 새 이미지 반환.
    """
    h, w = img.shape[:2]

    rh = int(h * random.uniform(*h_ratio))
    rw = int(w * random.uniform(*w_ratio))
    y0 = random.randint(int(h * y_min_ratio), max(h - rh - 1, int(h * y_min_ratio)))
    x0 = random.randint(0, w - rw - 1)

    mask = np.ones((h, w), np.float32)
    mask[y0:y0 + rh, x0:x0 + rw] = random.uniform(*alpha_range)
    mask = cv2.GaussianBlur(mask, (0, 0), blur_sigma)

    out = (img.astype(np.float32) * mask[..., None]).clip(0, 255).astype(img.dtype)
    return out


def add_rect_shadow_pair(left_img, right_img, prob=0.5, **shadow_kwargs):
    """
    좌·우 이미지에 동일 파라미터로 네모 그림자를 적용.
    """
    if random.random() > prob:
        return left_img, right_img

    left_arr  = _random_rect_shadow(np.array(left_img),  **shadow_kwargs)
    right_arr = _random_rect_shadow(np.array(right_img), **shadow_kwargs)

    return Image.fromarray(left_arr), Image.fromarray(right_arr)

class FlyingThingDataset(Dataset):
    def __init__(self, datapath, list_filename, training,
                 max_len=None, aug=False, prior=None,
                 shadow_prob=0.5):
        self.datapath = datapath
        self.left_filenames, self.right_filenames, self.disp_filenames = self.load_path(list_filename)
        self.training = training
        self.data_len = len(self.left_filenames)
        self.max_len = max_len
        self.aug = aug
        self.erase_low = True
        self.prior_path = prior
        self.shadow_prob = shadow_prob        # ← 그림자 증강 확률
        if self.training:
            assert self.disp_filenames is not None

    # ---------- Loader helpers ---------- #
    def load_path(self, list_filename):
        lines = read_all_lines(list_filename)
        splits = [ln.split() for ln in lines]
        left = [x[0] for x in splits]
        right = [x[1] for x in splits]
        disp = [x[2] for x in splits] if len(splits[0]) == 3 else None
        return left, right, disp

    def load_image(self, filename):
        return Image.open(os.path.expanduser(filename)).convert('RGB')

    def load_prior(self):
        if self.prior_path is None:
            return 0.0
        path = os.path.expanduser(self.prior_path)
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        return np.load(path)

    def load_disp(self, filename):
        data, _ = pfm_imread(os.path.expanduser(filename))
        return np.ascontiguousarray(data, np.float32)

    def __len__(self):
        return self.max_len if self.max_len else self.data_len

    # ---------- Main getter ---------- #
    def __getitem__(self, idx):
        if idx >= self.data_len:
            idx = random.randint(0, self.data_len - 1)

        L = self.load_image(os.path.join(self.datapath, self.left_filenames[idx]))
        R = self.load_image(os.path.join(self.datapath, self.right_filenames[idx]))
        disp = self.load_disp(os.path.join(self.datapath, self.disp_filenames[idx]))

        # gradient map (전체 해상도)
        grad = cv2.Sobel(np.array(L), cv2.CV_32F, 1, 0, 3)**2 + \
               cv2.Sobel(np.array(L), cv2.CV_32F, 0, 1, 3)**2
        grad = np.sqrt(grad.sum(-1))
        grad = grad / (grad.max() + 1e-5)

        if self.erase_low:
            disp[disp > -2] = 0

        prior = self.load_prior()

        # ---------- TRAIN ---------- #
        if self.training:
            if self.aug:
                # 색감 기본 증강
                for img, br, gm, ct, st in ((L, *np.random.uniform((0.5, 0.8, 0.8, 0.0),
                                                                   (2.0, 1.2, 1.2, 1.4)))
                                             ,):
                    pass  # placeholder (kept minimal to focus on shadow)

                # 그림자 증강
                L, R = add_rect_shadow_pair(L, R,
                                            prob=self.shadow_prob,
                                            h_ratio=(0.12, 0.30),
                                            w_ratio=(0.25, 0.70),
                                            y_min_ratio=0.55,
                                            alpha_range=(0.35, 0.55),
                                            blur_sigma=15)

            # 랜덤 크롭
            w, h = L.size
            crop_w, crop_h = random.choice([[512, 128], [512, 256], [768, 256], [768, 512]])
            x1, y1 = random.randint(0, w - crop_w), random.randint(0, h - crop_h)

            L = L.crop((x1, y1, x1 + crop_w, y1 + crop_h))
            R = R.crop((x1, y1, x1 + crop_w, y1 + crop_h))
            disp_c = disp[y1:y1 + crop_h, x1:x1 + crop_w]
            disp_low = cv2.resize(disp_c, (crop_w//4, crop_h//4), cv2.INTER_NEAREST)
            disp_low8 = cv2.resize(disp_c, (crop_w//8, crop_h//8), cv2.INTER_NEAREST)
            grad_c = grad[y1:y1 + crop_h, x1:x1 + crop_w]

            to_tensor = get_transform()
            return {
                "left": to_tensor(L),
                "right": to_tensor(R),
                "disparity": -disp_c,
                "gradient_map": torch.from_numpy(grad_c).float(),
                "disparity_low": -disp_low,
                "disparity_low_r8": -disp_low8,
                "left_filename": self.left_filenames[idx],
                "right_filename": self.right_filenames[idx],
                "prior": prior
            }

        # ---------- EVAL ---------- #
        w, h = L.size
        crop_w, crop_h = 960, 512
        L = L.crop((w-crop_w, h-crop_h, w, h))
        R = R.crop((w-crop_w, h-crop_h, w, h))
        disp_c = disp[h-crop_h:h, w-crop_w:w]
        grad_c = grad[h-crop_h:h, w-crop_w:w]
        disp_low = cv2.resize(disp_c, (crop_w//4, crop_h//4), cv2.INTER_NEAREST)

        to_tensor = get_transform()
        return {
            "left": to_tensor(L),
            "right": to_tensor(R),
            "disparity": -disp_c,
            "top_pad": 0,
            "right_pad": 0,
            "gradient_map": grad_c,
            "disparity_low": -disp_low,
            "left_filename": self.left_filenames[idx],
            "right_filename": self.right_filenames[idx]
        }
