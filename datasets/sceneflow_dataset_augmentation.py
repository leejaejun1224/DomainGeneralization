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
#   Specular highlight utils  #
# --------------------------- #
def add_specular_highlight_pil(img,
                               center=None,          # (x,y)
                               radius=80,
                               strength=0.7,         # 0~1
                               feather=0.6,          # 0~1
                               mode="screen",        # "screen" or "add"
                               blobs=1,
                               seed=None):
    """
    PIL Image -> PIL Image
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    img_np = np.array(img).astype(np.float32) / 255.0  # (H,W,3)
    h, w = img_np.shape[:2]
    out = img_np.copy()

    for _ in range(blobs):
        if center is None:
            cx = random.randint(int(0.2*w), int(0.8*w))
            cy = random.randint(int(0.2*h), int(0.8*h))
        else:
            cx, cy = center

        r = random.randint(int(0.5*radius), int(1.5*radius))
        s = random.uniform(strength*0.7, strength*1.2)

        yy, xx = np.mgrid[0:h, 0:w]
        r2 = ((xx - cx) ** 2 + (yy - cy) ** 2).astype(np.float32)
        sigma = (r ** 2)
        mask = np.exp(-r2 / (2 * sigma))
        mask = mask ** (1.0 - feather)
        mask = (mask * s).clip(0, 1)[..., None]  # (H,W,1)

        if mode == "screen":
            out = 1 - (1 - out) * (1 - mask)
        elif mode == "add":
            out = np.clip(out + mask, 0, 1)
        else:
            raise ValueError("mode should be 'screen' or 'add'")

    out_uint8 = (out * 255).astype(np.uint8)
    return Image.fromarray(out_uint8)


def add_specular_pair(left_img, right_img,
                      prob=0.5,
                      blobs_range=(1, 4),
                      radius_range=(20, 80),
                      strength_range=(0.4, 0.9),
                      feather=0.6):
    """
    좌/우 이미지에 비슷하지만 살짝 다른 하이라이트를 추가.
    """
    if random.random() > prob:
        return left_img, right_img  # 증강 X

    blobs = random.randint(*blobs_range)
    radius = random.randint(*radius_range)
    strength = random.uniform(*strength_range)

    seed = random.randint(0, 10_000)
    left_aug = add_specular_highlight_pil(left_img,
                                          radius=radius,
                                          strength=strength,
                                          feather=feather,
                                          blobs=blobs,
                                          seed=seed)
    # 오른쪽은 약간 다른 위치/세기로
    right_aug = add_specular_highlight_pil(right_img,
                                           radius=radius + random.randint(-10, 10),
                                           strength=strength * random.uniform(0.8, 1.1),
                                           feather=feather,
                                           blobs=blobs,
                                           seed=seed + 1)

    return left_aug, right_aug


# --------------------------- #
#        Dataset class        #
# --------------------------- #
class FlyingThingDataset(Dataset):
    def __init__(self, datapath, list_filename, training, max_len=None, aug=False, prior=None,
                 specular_prob=0.35):
        self.datapath = datapath
        self.left_filenames, self.right_filenames, self.disp_filenames = self.load_path(list_filename)
        self.training = training
        self.data_len = len(self.left_filenames)
        self.max_len = max_len
        self.aug = aug
        self.erase_low = True  # Erase low disparity values
        self.prior_path = prior
        self.specular_prob = specular_prob  # <--- 추가

        if self.training:
            assert self.disp_filenames is not None

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
        return data

    def __len__(self):
        return self.max_len if self.max_len is not None else self.data_len

    def __getitem__(self, index):
        index = index if index < self.data_len else random.randint(0, self.data_len - 1)
        left_img = self.load_image(os.path.join(self.datapath, self.left_filenames[index]))
        right_img = self.load_image(os.path.join(self.datapath, self.right_filenames[index]))
        disparity = self.load_disp(os.path.join(self.datapath, self.disp_filenames[index]))

        # Sobel for gradient map (left only, original res)
        left_img_np = np.array(left_img)
        dx_imgL = cv2.Sobel(left_img_np, cv2.CV_32F, 1, 0, ksize=3)
        dy_imgL = cv2.Sobel(left_img_np, cv2.CV_32F, 0, 1, ksize=3)
        dxy_imgL = np.sqrt(np.sum(dx_imgL**2, axis=-1) + np.sum(dy_imgL**2, axis=-1))
        dxy_imgL = dxy_imgL / (np.max(dxy_imgL) + 1e-5)

        if self.erase_low:
            disparity[disparity > -2] = 0  # Erase low disparity values

        prior_data = self.load_prior()

        if self.training:
            # -------- 기본 색감 증강 --------
            if self.aug:
                random_brightness = np.random.uniform(0.5, 2.0, 2)
                random_gamma = np.random.uniform(0.8, 1.2, 2)
                random_contrast = np.random.uniform(0.8, 1.2, 2)
                random_satur = np.random.uniform(.0, 1.4, 2)

                left_img = torchvision.transforms.functional.adjust_brightness(left_img, random_brightness[0])
                left_img = torchvision.transforms.functional.adjust_gamma(left_img, random_gamma[0])
                left_img = torchvision.transforms.functional.adjust_contrast(left_img, random_contrast[0])
                left_img = torchvision.transforms.functional.adjust_saturation(left_img, random_satur[0])

                right_img = torchvision.transforms.functional.adjust_brightness(right_img, random_brightness[1])
                right_img = torchvision.transforms.functional.adjust_gamma(right_img, random_gamma[1])
                right_img = torchvision.transforms.functional.adjust_contrast(right_img, random_contrast[1])
                right_img = torchvision.transforms.functional.adjust_saturation(right_img, random_satur[1])

                # -------- Specular 증강 --------
                left_img, right_img = add_specular_pair(left_img, right_img,
                                                        prob=self.specular_prob,
                                                        blobs_range=(1, 4),
                                                        radius_range=(20, 80),
                                                        strength_range=(0.2, 0.5),
                                                        feather=0.6)

            # -------- Random crop --------
            w, h = left_img.size
            crop_list = [[512, 128], [512, 256], [768, 256], [768, 512]]
            num = random.randint(0, 3)

            crop_w, crop_h = crop_list[num][0], crop_list[num][1]
            x1 = random.randint(0, w - crop_w)
            y1 = random.randint(0, h - crop_h)

            left_img = left_img.crop((x1, y1, x1 + crop_w, y1 + crop_h))
            right_img = right_img.crop((x1, y1, x1 + crop_w, y1 + crop_h))
            disparity_c = disparity[y1:y1 + crop_h, x1:x1 + crop_w]
            disparity_low = cv2.resize(disparity_c, (crop_w // 4, crop_h // 4), interpolation=cv2.INTER_NEAREST)
            disparity_low_r8 = cv2.resize(disparity_c, (crop_w // 8, crop_h // 8), interpolation=cv2.INTER_NEAREST)
            gradient_map = dxy_imgL[y1:y1 + crop_h, x1:x1 + crop_w]

            # to tensor, normalize
            processed = get_transform()
            left_img = processed(left_img)
            right_img = processed(right_img)
            gradient_map = torch.from_numpy(gradient_map).float()

            return {"left": left_img,
                    "right": right_img,
                    "disparity": disparity_c * -1,
                    "gradient_map": gradient_map,
                    "disparity_low": disparity_low * -1,
                    "disparity_low_r8": disparity_low_r8 * -1,
                    "left_filename": self.left_filenames[index],
                    "right_filename": self.right_filenames[index],
                    "prior": prior_data}
        else:
            # -------- Eval crop --------
            w, h = left_img.size
            crop_w, crop_h = 960, 512

            left_img = left_img.crop((w - crop_w, h - crop_h, w, h))
            right_img = right_img.crop((w - crop_w, h - crop_h, w, h))
            disparity_c = disparity[h - crop_h:h, w - crop_w: w]
            gradient_map = dxy_imgL[h - crop_h:h, w - crop_w: w]
            disparity_low = cv2.resize(disparity_c, (crop_w // 4, crop_h // 4), interpolation=cv2.INTER_NEAREST)

            processed = get_transform()
            left_img = processed(left_img)
            right_img = processed(right_img)

            if disparity is not None:
                return {"left": left_img,
                        "right": right_img,
                        "disparity": disparity_c * -1,
                        "top_pad": 0,
                        "right_pad": 0,
                        "gradient_map": gradient_map,
                        "disparity_low": disparity_low * -1,
                        "left_filename": self.left_filenames[index],
                        "right_filename": self.right_filenames[index],}
            else:
                return {"left": left_img,
                        "right": right_img,
                        "left_filename": self.left_filenames[index],
                        "right_filename": self.right_filenames[index]}
