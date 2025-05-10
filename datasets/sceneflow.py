import os
import random
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import cv2
from datasets.data_io import get_transform, read_all_lines, pfm_imread
import torchvision.transforms as transforms
import torch
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import matplotlib.pyplot as plt
import torchvision

class FlyingThingDataset(Dataset):
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
        left_img_np = np.array(left_img)
        dx_imgL = cv2.Sobel(left_img_np,cv2.CV_32F,1,0,ksize=3)
        dy_imgL = cv2.Sobel(left_img_np,cv2.CV_32F,0,1,ksize=3)
        dxy_imgL=np.sqrt(np.sum(np.square(dx_imgL),axis=-1)+np.sum(np.square(dy_imgL),axis=-1))
        dxy_imgL = dxy_imgL/(np.max(dxy_imgL)+1e-5)

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
            y1 = random.randint(0, h - crop_h)

            # random crop
            left_img = left_img.crop((x1, y1, x1 + crop_w, y1 + crop_h))
            right_img = right_img.crop((x1, y1, x1 + crop_w, y1 + crop_h))
            disparity = disparity[y1:y1 + crop_h, x1:x1 + crop_w]
            disparity_low = cv2.resize(disparity, (crop_w//4, crop_h//4), interpolation=cv2.INTER_NEAREST)
            disparity_low_r8 = cv2.resize(disparity, (crop_w//8, crop_h//8), interpolation=cv2.INTER_NEAREST)
            gradient_map = dxy_imgL[y1:y1 + crop_h, x1:x1 + crop_w]

            # to tensor, normalize
            processed = get_transform()
            left_img = processed(left_img)
            right_img = processed(right_img)
            gradient_map = torch.from_numpy(gradient_map)


            
            return {"left": left_img,
                    "right": right_img,
                    "disparity": disparity*-1,
                    "gradient_map":gradient_map,
                    "disparity_low":disparity_low*-1,
                    "disparity_low_r8":disparity_low_r8*-1,
                    "left_filename": self.left_filenames[index],
                    "right_filename": self.right_filenames[index]}
        else:
            w, h = left_img.size
            crop_w, crop_h = 960, 512

            left_img = left_img.crop((w - crop_w, h - crop_h, w, h))
            right_img = right_img.crop((w - crop_w, h - crop_h, w, h))
            disparity = disparity[h - crop_h:h, w - crop_w: w]
            gradient_map = dxy_imgL[h - crop_h:h, w - crop_w: w]
            disparity_low = cv2.resize(disparity, (crop_w//4, crop_h//4), interpolation=cv2.INTER_NEAREST)

            processed = get_transform()
            left_img = processed(left_img)
            right_img = processed(right_img)

            if disparity is not None:
                return {"left": left_img,
                        "right": right_img,
                        "disparity": disparity*-1,
                        "top_pad": 0,
                        "right_pad": 0,
                        "gradient_map":gradient_map,
                        "disparity_low":disparity_low*-1,
                        "left_filename": self.left_filenames[index],
                        "right_filename": self.right_filenames[index]}
            
            else:
                return {"left": left_img,
                        "right": right_img,
                        "left_filename": self.left_filenames[index],
                        "right_filename": self.right_filenames[index]}
