import os
import random
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import cv2
from datasets.data_io import get_transform, read_all_lines, pfm_imread
import torchvision.transforms as transforms
import torch
import torchvision
import matplotlib.pyplot as plt


class KITTI2015Dataset(Dataset):
    def __init__(self, datapath, list_filename, training, max_len=None, aug=False):
        self.datapath = datapath
        self.left_filenames, self.right_filenames, self.disp_filenames = self.load_path(list_filename)
        self.training = training
        self.data_len = len(self.left_filenames)
        self.max_len = max_len
        self.aug = aug
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

    def load_disp(self, filename):
        filename = os.path.expanduser(filename)
        data = Image.open(filename)
        data = np.array(data, dtype=np.float32) / 256.
        return data

    def __len__(self):
        return self.max_len if self.max_len is not None else self.data_len

    def __getitem__(self, index):
        idx = index if index < self.data_len else random.randint(0, self.data_len - 1)


        left_img = self.load_image(os.path.join(self.datapath, self.left_filenames[idx]))
        right_img = self.load_image(os.path.join(self.datapath, self.right_filenames[idx]))

        if self.disp_filenames:  # has disparity ground truth
            disparity = self.load_disp(os.path.join(self.datapath, self.disp_filenames[idx]))
        else:
            disparity = None

        if self.training:
            
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



            w, h = left_img.size
            crop_w, crop_h = 512, 256

            x1 = random.randint(0, w - crop_w)
            # y1 = random.randint(0, h - crop_h)
            if  random.randint(0, 10) >= int(8):
                y1 = random.randint(0, h - crop_h)
            else:
                y1 = random.randint(int(0.3 * h), h - crop_h)

            # random crop
            left_img = left_img.crop((x1, y1, x1 + crop_w, y1 + crop_h))
            right_img = right_img.crop((x1, y1, x1 + crop_w, y1 + crop_h))
            left_img_half = left_img.resize((crop_w//2, crop_h//2), Image.BICUBIC)
            right_img_half = right_img.resize((crop_w//2, crop_h//2), Image.BICUBIC)
            left_img_low = left_img.resize((crop_w//4, crop_h//4), Image.BICUBIC)
            right_img_low = right_img.resize((crop_w//4, crop_h//4), Image.BICUBIC)
            
            disparity = disparity[y1:y1 + crop_h, x1:x1 + crop_w]
            disparity_low = cv2.resize(disparity, (crop_w//4, crop_h//4), interpolation=cv2.INTER_NEAREST)
            disparity_half = cv2.resize(disparity, (crop_w//2, crop_h//2), interpolation=cv2.INTER_NEAREST)

            valid_mask = disparity_low > 0
            if np.any(valid_mask):
                disp_min = np.min(disparity_low[valid_mask])
                disp_max = np.max(disparity_low[valid_mask])
                # 유효한 값이 모두 동일하지 않을 때
                if disp_max != disp_min:
                    depth_low = np.zeros_like(disparity_low)
                    depth_low[valid_mask] = 1e-4 + (disparity_low[valid_mask] - disp_min) / (disp_max - disp_min) 
                else:
                    # 모든 유효 값이 동일하다면, 논리적으로 scaling이 어려우므로 기본값 1로 설정 (0이 아닌 부분만)
                    depth_low = np.ones_like(disparity_low)
                    depth_low[~valid_mask] = 0
            else:
                depth_low = disparity_low.copy()


            # to tensor, normalize
            processed = get_transform()
            left_img = processed(left_img)
            right_img = processed(right_img)
            left_img_half = processed(left_img_half)
            right_img_half = processed(right_img_half)
            left_img_low = processed(left_img_low)
            right_img_low = processed(right_img_low)

            return {"left": left_img,
                    "right": right_img,
                    "left_low": left_img_low,
                    "right_low": right_img_low,
                    "left_half": left_img_half,
                    "right_half": right_img_half,
                    "disparity": disparity,
                    "disparity_low": disparity_low,
                    "disparity_half": disparity_half,
                    "depth_low": depth_low,
                    "left_filename" : self.left_filenames[idx],
                    "right_filename" : self.right_filenames[idx]}

        else:
            w, h = left_img.size

            # normalize
            processed = get_transform()
            left_img_half = left_img.resize((1248//2, 384//2), Image.BICUBIC)
            right_img_half = right_img.resize((1248//2, 384//2), Image.BICUBIC)
            left_img_low = left_img.resize((1248//4, 384//4), Image.BICUBIC)
            right_img_low = right_img.resize((1248//4, 384//4), Image.BICUBIC)
            left_img = processed(left_img).numpy()
            right_img = processed(right_img).numpy()
            left_img_low = processed(left_img_low).numpy()
            right_img_low = processed(right_img_low).numpy()
            left_img_half = processed(left_img_half).numpy()
            right_img_half = processed(right_img_half).numpy()

            # pad to size 1248x384
            top_pad = 384 - h
            right_pad = 1248 - w
            assert top_pad > 0 and right_pad > 0
            # pad images
            left_img = np.lib.pad(left_img, ((0, 0), (top_pad, 0), (0, right_pad)), mode='constant', constant_values=0)
            right_img = np.lib.pad(right_img, ((0, 0), (top_pad, 0), (0, right_pad)), mode='constant',
                                   constant_values=0)
            # pad disparity gt
            if disparity is not None:
                assert len(disparity.shape) == 2
                disparity = np.lib.pad(disparity, ((top_pad, 0), (0, right_pad)), mode='constant', constant_values=0)
                disparity_half = cv2.resize(disparity, (1248//2, 384//2), interpolation=cv2.INTER_NEAREST)
                disparity_low = cv2.resize(disparity, (1248//4, 384//4), interpolation=cv2.INTER_NEAREST)
                valid_mask = disparity_low > 0
                if np.any(valid_mask):
                    disp_min = np.min(disparity_low[valid_mask])
                    disp_max = np.max(disparity_low[valid_mask])
                    # 유효한 값이 모두 동일하지 않을 때
                    if disp_max != disp_min:
                        depth_low = np.zeros_like(disparity_low)
                        depth_low[valid_mask] = 1e-4 + (disparity_low[valid_mask] - disp_min) / (disp_max - disp_min) * (10 - 1e-4)
                    else:
                        # 모든 유효 값이 동일하다면, 논리적으로 scaling이 어려우므로 기본값 1로 설정 (0이 아닌 부분만)
                        depth_low = np.ones_like(disparity_low)
                        depth_low[~valid_mask] = 0
                else:
                    depth_low = disparity_low.copy()

            if disparity is not None:
                return {"left": left_img,
                        "right": right_img,
                        "left_half": left_img_half,
                        "right_half": right_img_half,
                        "left_low": left_img_low,
                        "right_low": right_img_low,
                        "disparity": disparity,
                        "disparity_low": disparity_low,
                        "disparity_half": disparity_half,
                        "depth_low": depth_low,
                        "left_filename": self.left_filenames[idx],
                        "right_filename": self.right_filenames[idx]}
            else:
                return {"left": left_img,
                        "right": right_img,
                        "left_half": left_img_half,
                        "right_half": right_img_half,
                        "left_low": left_img_low,
                        "right_low": right_img_low, 
                        "left_filename": self.left_filenames[idx],
                        "right_filename": self.right_filenames[idx]}