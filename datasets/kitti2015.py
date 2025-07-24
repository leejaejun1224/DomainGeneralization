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
from scipy.ndimage import gaussian_filter
from datasets.transform import HFStereoV2


class KITTI2015Dataset(Dataset):
    def __init__(self, datapath, list_filename, training, max_len=None, aug=False, prior=None):
        self.datapath = datapath
        self.left_filenames, self.right_filenames, self.disp_filenames = self.load_path(list_filename)
        self.training = training
        self.data_len = len(self.left_filenames)
        self.max_len = max_len
        self.aug = aug
        self.hf_transform = HFStereoV2(use_edge_enhancement=True)
        self.prior_path = prior
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
    
    def load_prior(self):
        if self.prior_path is not None:
            prior_path = os.path.expanduser(self.prior_path)
            if not os.path.exists(os.path.expanduser(prior_path)):
                raise FileNotFoundError(f"Prior file {prior_path} does not exist.")
            prior_data = np.load(prior_path)
        else:
            prior_data = 0.0
        
        return prior_data
    
    def calculate_overlap(self, x1_1, y1_1, x1_2, y1_2, crop_w, crop_h):
        """Calculate overlap region between two crops"""
        x_overlap_start = max(x1_1, x1_2)
        y_overlap_start = max(y1_1, y1_2)
        x_overlap_end = min(x1_1 + crop_w, x1_2 + crop_w)
        y_overlap_end = min(y1_1 + crop_h, y1_2 + crop_h)
        
        overlap_w = max(0, x_overlap_end - x_overlap_start)
        overlap_h = max(0, y_overlap_end - y_overlap_start)
        
        return {
            'x': x_overlap_start,
            'y': y_overlap_start,
            'width': overlap_w,
            'height': overlap_h
        }
    
    def __len__(self):
        return self.max_len if self.max_len is not None else self.data_len

    def __getitem__(self, index):
        idx = index if index < self.data_len else random.randint(0, self.data_len - 1)

        left_img = self.load_image(os.path.join(self.datapath, self.left_filenames[idx]))
        right_img = self.load_image(os.path.join(self.datapath, self.right_filenames[idx]))

        if self.disp_filenames:
            disparity = self.load_disp(os.path.join(self.datapath, self.disp_filenames[idx]))
        else:
            disparity = None

        # textureless score 계산 (기존과 동일)
        left_low_np = np.array(left_img).astype(np.float32) / 255.0
        gray = cv2.cvtColor((left_low_np*255).astype(np.uint8),
                            cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(gx*gx + gy*gy)
        mag_norm = (mag - mag.min()) / (mag.max() - mag.min() + 1e-6)
        textureless = 1.0 - mag_norm
        textureless_sm = gaussian_filter(textureless, sigma=5)
        textureless_score = torch.from_numpy(textureless_sm).unsqueeze(0).float()
        self.aug = False
        
        if self.training:
            w, h = left_img.size
            crop_w, crop_h = 512, 256
            x1 = random.randint(0, w - crop_w)
            y1 = random.randint(int(0.3 * h), h - crop_h)
            left_img = left_img.crop((x1, y1, x1 + crop_w, y1 + crop_h))
            right_img = right_img.crop((x1, y1, x1 + crop_w, y1 + crop_h))
            
            # 전체 이미지를 1248×384로 리사이즈 또는 패딩
            w, h = left_img.size
            
            # Multi-scale images
            left_img_half = left_img.resize((w//2, h//2), Image.BICUBIC)
            right_img_half = right_img.resize((w//2, h//2), Image.BICUBIC)
            left_img_low = left_img.resize((w//4, h//4), Image.BICUBIC)
            right_img_low = right_img.resize((w//4, h//4), Image.BICUBIC)
            
            processed = get_transform()
            left_img = processed(left_img)
            right_img = processed(right_img)
            left_img_half = processed(left_img_half)
            right_img_half = processed(right_img_half)
            left_img_low = processed(left_img_low)
            right_img_low = processed(right_img_low)

            prior_data = self.load_prior()

            disparity = disparity[y1:y1 + crop_h, x1:x1 + crop_w]
            disparity_low = cv2.resize(disparity, (crop_w//4, crop_h//4), interpolation=cv2.INTER_NEAREST)
            disparity_half = cv2.resize(disparity, (crop_w//2, crop_h//2), interpolation=cv2.INTER_NEAREST)

            # Depth map 계산 (기존과 동일)
            valid_mask = disparity > 0
            if np.any(valid_mask):
                disp_min = np.min(disparity[valid_mask])
                disp_max = np.max(disparity[valid_mask])
                if disp_max != disp_min:
                    depth_map = np.zeros_like(disparity)
                    depth_map[valid_mask] = 1e-4 + (disparity[valid_mask] - disp_min) / (disp_max - disp_min) 
                else:
                    depth_map = np.ones_like(disparity)
                    depth_map[~valid_mask] = 0
            else:
                depth_map = disparity.copy()

            return {
                "left": left_img,
                "right": right_img,
                "left_strong_aug": left_img,
                "right_strong_aug": right_img,
                "left_low": left_img_low,
                "right_low": right_img_low,
                "left_half": left_img_half,
                "right_half": right_img_half,
                "disparity": disparity,
                "disparity_low": disparity_low,
                "disparity_half": disparity_half,
                "depth_map": depth_map,
                "textureless_score": textureless_score,
                "left_filename": self.left_filenames[idx],
                "right_filename": self.right_filenames[idx],
                "prior": prior_data
            }

        else:
            w, h = left_img.size
            # crop_w, crop_h = 512, 256
            crop_w, crop_h = 1152, 320
            
            # Generate two random crop coordinates
            x1_1 = random.randint(0, w - crop_w)
            y1_1 = random.randint(0, h - crop_h)
            x1_2 = random.randint(0, w - crop_w)
            y1_2 = random.randint(0, h - crop_h)
            
            # Create random crops
            left_random_1 = left_img.crop((x1_1, y1_1, x1_1 + crop_w, y1_1 + crop_h))
            right_random_1 = right_img.crop((x1_1, y1_1, x1_1 + crop_w, y1_1 + crop_h))
            left_random_2 = left_img.crop((x1_2, y1_2, x1_2 + crop_w, y1_2 + crop_h))
            right_random_2 = right_img.crop((x1_2, y1_2, x1_2 + crop_w, y1_2 + crop_h))
            
            # Calculate overlap region
            overlap_coords = self.calculate_overlap(x1_1, y1_1, x1_2, y1_2, crop_w, crop_h)

            # normalize
            processed = get_transform()
            left_img_half = left_img.resize((1248//2, 384//2), Image.BICUBIC)
            right_img_half = right_img.resize((1248//2, 384//2), Image.BICUBIC)
            left_img_low = left_img.resize((1248//4, 384//4), Image.BICUBIC)
            right_img_low = right_img.resize((1248//4, 384//4), Image.BICUBIC)
            
            # Process random crops
            left_random_1 = processed(left_random_1).numpy()
            right_random_1 = processed(right_random_1).numpy()
            left_random_2 = processed(left_random_2).numpy()
            right_random_2 = processed(right_random_2).numpy()
            
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
                valid_mask = disparity > 0
                if np.any(valid_mask):
                    disp_min = np.min(disparity[valid_mask])
                    disp_max = np.max(disparity[valid_mask])
                    if disp_max != disp_min:
                        depth_map = np.zeros_like(disparity)
                        depth_map[valid_mask] = 1e-4 + (disparity[valid_mask] - disp_min) / (disp_max - disp_min)
                    else:
                        depth_map = np.ones_like(disparity)
                        depth_map[~valid_mask] = 0
                else:
                    depth_map = disparity.copy()

            if disparity is not None:
                return {
                    "left": left_img,
                    "right": right_img,
                    "left_half": left_img_half,
                    "right_half": right_img_half,
                    "left_low": left_img_low,
                    "right_low": right_img_low,
                    "textureless_score": textureless_score,
                    "disparity": disparity,
                    "disparity_low": disparity_low,
                    "disparity_half": disparity_half,
                    "depth_map": depth_map,
                    "left_filename": self.left_filenames[idx],
                    "right_filename": self.right_filenames[idx],
                    # Random crops
                    "random_coord_1" : (x1_1, y1_1),
                    "left_random_1": left_random_1,
                    "right_random_1": right_random_1,
                    
                    "random_coord_2" : (x1_2, y1_2),
                    "left_random_2": left_random_2,
                    "right_random_2": right_random_2,
                    # Overlap coordinates
                    "overlap_coords": overlap_coords
                }
            else:
                return {
                    "left": left_img,
                    "right": right_img,
                    "left_half": left_img_half,
                    "right_half": right_img_half,
                    "left_low": left_img_low,
                    "right_low": right_img_low, 
                    "textureless_score": textureless_score,
                    "left_filename": self.left_filenames[idx],
                    "right_filename": self.right_filenames[idx],
                    # Random crops
                    "random_coord_1" : (x1_1, y1_1),
                    "left_random_1": left_random_1,
                    "right_random_1": right_random_1,
                    
                    "random_coord_2" : (x1_2, y1_2),
                    "left_random_2": left_random_2,
                    "right_random_2": right_random_2,
                    # Overlap coordinates
                    "overlap_coords": overlap_coords
                }
