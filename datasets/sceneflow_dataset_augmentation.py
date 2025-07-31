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


# --------------------------- #
#   Random shadow  utilities  #
# --------------------------- #
def _poly_shadow_mask(h, w,
                      vertices=None,
                      feather=0.6,
                      strength_range=(0.4, 0.8)):
    """
    랜덤 폴리곤 그림자 마스크 생성 (0~1, 0=어둡게)
    """
    if vertices is None:
        # 3~6각형 임의 생성
        num_pts = random.randint(3, 6)
        xs = np.random.randint(int(0.0*w), int(1.0*w), num_pts)
        ys = np.random.randint(int(0.0*h), int(1.0*h), num_pts)
        vertices = np.vstack([xs, ys]).T.astype(np.int32)

    mask = np.ones((h, w), np.float32)
    cv2.fillConvexPoly(mask, vertices, 0.0, lineType=cv2.LINE_AA)

    # feather
    k = int(max(h, w) * 0.05)
    k = k if k % 2 == 1 else k + 1
    mask = cv2.GaussianBlur(mask, (k, k), 0)
    mask = mask ** (1.0 - feather)

    strength = random.uniform(*strength_range)   # 0.4~0.8
    mask = 1.0 - mask * strength                 # 1 → 그대로, 0.2~0.6 → 어둡게
    mask = mask[..., None]                       # (H,W,1)
    return mask


def add_shadow_pair(left_img, right_img,
                    prob=0.5,
                    feather=0.6,
                    strength_range=(0.4, 0.8)):
    """
    좌/우 이미지에 동일 그림자 마스크 곱하기.
    """
    if random.random() > prob:
        return left_img, right_img

    img_np = np.array(left_img).astype(np.float32) / 255.0
    h, w = img_np.shape[:2]
    mask = _poly_shadow_mask(h, w, feather=feather, strength_range=strength_range)

    def apply(img):
        arr = np.array(img).astype(np.float32) / 255.0
        arr = (arr * mask).clip(0, 1)
        return Image.fromarray((arr * 255).astype(np.uint8))

    return apply(left_img), apply(right_img)



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
                L, R = add_shadow_pair(L, R,
                                       prob=self.shadow_prob,
                                       feather=0.6,
                                       strength_range=(0.2, 0.6))

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
