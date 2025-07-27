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


def pad_to_384_1248(tensor):
    """Pad tensor to 384x1248 size"""
    _, h, w = tensor.shape
    target_h, target_w = 384, 1248
    
    top_pad = max(0, target_h - h)
    right_pad = max(0, target_w - w)
    
    if top_pad > 0 or right_pad > 0:
        tensor = torch.nn.functional.pad(tensor, (0, right_pad, top_pad, 0), mode='constant', value=0)
    
    return tensor


class JinoTransform:
    def __init__(self):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        self.weak_photo = transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)

        self.strong_photo = transforms.Compose([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
            transforms.RandomGrayscale(0.2),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: t * 255.0),
            transforms.Lambda(pad_to_384_1248),
            # transforms.GaussianBlur(7, sigma=(0.1,2.0)),
            transforms.Normalize(mean, std),
        ])
        self.normalize = transforms.Normalize(mean, std)

    @staticmethod
    def _to_tensor_and_norm(pil_img, normalize):
        t = transforms.functional.to_tensor(pil_img) * 255.0
        t = pad_to_384_1248(t)
        return normalize(t)

    def __call__(self, pil_img):
        # weak = self._to_tensor_and_norm(self.weak_photo(pil_img), self.normalize)
        weak = self._to_tensor_and_norm(pil_img, self.normalize)
        strong = self.strong_photo(pil_img)
        return weak, strong


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
        
        # JinoTransform 초기화
        self.jino_transform = JinoTransform()
        
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

    def compute_textureless_score(self, left_img):
        """Compute textureless score from left image"""
        left_low_np = np.array(left_img).astype(np.float32) / 255.0
        gray = cv2.cvtColor((left_low_np*255).astype(np.uint8),
                            cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(gx*gx + gy*gy)
        mag_norm = (mag - mag.min()) / (mag.max() - mag.min() + 1e-6)
        textureless = 1.0 - mag_norm
        textureless_sm = gaussian_filter(textureless, sigma=5)
        return torch.from_numpy(textureless_sm).unsqueeze(0).float()

    def create_multi_scale_images(self, left_img, right_img, target_w=1248, target_h=384):
        """Create multi-scale versions of images"""
        processed = get_transform()
        
        # Multi-scale images
        left_img_half = left_img.resize((target_w//2, target_h//2), Image.BICUBIC)
        right_img_half = right_img.resize((target_w//2, target_h//2), Image.BICUBIC)
        left_img_low = left_img.resize((target_w//4, target_h//4), Image.BICUBIC)
        right_img_low = right_img.resize((target_w//4, target_h//4), Image.BICUBIC)
        
        # Apply transforms
        left_img_processed = processed(left_img)
        right_img_processed = processed(right_img)
        left_img_half = processed(left_img_half)
        right_img_half = processed(right_img_half)
        left_img_low = processed(left_img_low)
        right_img_low = processed(right_img_low)
        
        return (left_img_processed, right_img_processed, 
                left_img_half, right_img_half, 
                left_img_low, right_img_low)

    def pad_to_target_size(self, img_array, target_w=1248, target_h=384):
        """Pad image array to target size"""
        if len(img_array.shape) == 3:  # [C, H, W]
            _, h, w = img_array.shape
            top_pad = target_h - h
            right_pad = target_w - w
            if top_pad > 0 or right_pad > 0:
                img_array = np.lib.pad(img_array, ((0, 0), (max(0, top_pad), 0), (0, max(0, right_pad))), 
                                     mode='constant', constant_values=0)
        else:  # [H, W]
            h, w = img_array.shape
            top_pad = target_h - h
            right_pad = target_w - w
            if top_pad > 0 or right_pad > 0:
                img_array = np.lib.pad(img_array, ((max(0, top_pad), 0), (0, max(0, right_pad))), 
                                     mode='constant', constant_values=0)
        return img_array

    def create_depth_map(self, disparity):
        """Create depth map from disparity"""
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
        return depth_map

    def generate_random_coords(self, w, h, crop_w, crop_h, num_crops=4):
        """Generate random crop coordinates"""
        coords = []
        for _ in range(num_crops):
            x = random.randint(0, w - crop_w)
            y = random.randint(int(0.1 * h), h - crop_h)
            coords.append((x, y))
        return coords

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

        # Compute textureless score
        textureless_score = self.compute_textureless_score(left_img)
        
        w, h = left_img.size
        crop_w, crop_h = 1152, 320
        target_w, target_h = 1248, 384

        # Generate random crop coordinates
        random_coords = self.generate_random_coords(w, h, crop_w, crop_h, num_crops=9)
        x1_1, y1_1 = random_coords[0]
        x1_2, y1_2 = random_coords[1]
        x1_3, y1_3 = random_coords[2]
        x1_4, y1_4 = random_coords[3]
        x1_5, y1_5 = random_coords[4]
        x1_6, y1_6 = random_coords[5]
        x1_7, y1_7 = random_coords[6]
        x1_8, y1_8 = random_coords[7]
        x1_9, y1_9 = random_coords[8]


        if self.training:
            # For training, use traditional crop approach for main images
            x1 = random.randint(0, w - crop_w)
            y1 = random.randint(0, h - crop_h)
            left_img_main = left_img
            right_img_main = right_img
            
            # Create multi-scale images for main crop
            (left_img_processed, right_img_processed, 
             left_img_half, right_img_half, 
             left_img_low, right_img_low) = self.create_multi_scale_images(left_img_main, right_img_main, crop_w, crop_h)
            
            # Strong augmentation 적용
            if self.aug:
                left_weak, left_strong = self.jino_transform(left_img_main)
                right_weak, right_strong = self.jino_transform(right_img_main)
            else:
                # Augmentation이 비활성화된 경우 weak만 사용
                left_weak, _ = self.jino_transform(left_img_main)
                right_weak, _ = self.jino_transform(right_img_main)
                left_strong = left_weak.clone()
                right_strong = right_weak.clone()
            
            prior_data = self.load_prior()
            
            # Process disparity for main crop
            disparity_main = disparity
            disparity_low = cv2.resize(disparity_main, (crop_w//4, crop_h//4), interpolation=cv2.INTER_NEAREST)
            disparity_half = cv2.resize(disparity_main, (crop_w//2, crop_h//2), interpolation=cv2.INTER_NEAREST)
            depth_map = self.create_depth_map(disparity_main)
            
        else:
            # For validation/test, resize to target size
            left_img_resized = left_img.resize((target_w, target_h), Image.BICUBIC)
            right_img_resized = right_img.resize((target_w, target_h), Image.BICUBIC)
            
            # Create multi-scale images
            (left_img_processed, right_img_processed, 
             left_img_half, right_img_half, 
             left_img_low, right_img_low) = self.create_multi_scale_images(left_img_resized, right_img_resized, target_w, target_h)

        # Common processing for both training and validation
        # Convert processed images to numpy for cropping
        processed = get_transform()
        left_img_full = processed(left_img).numpy()
        right_img_full = processed(right_img).numpy()
        
        # Pad full images to target size
        left_img_full = self.pad_to_target_size(left_img_full, target_w, target_h)
        right_img_full = self.pad_to_target_size(right_img_full, target_w, target_h)
        
        # Create random crops from padded full images
        left_random_1 = left_img_full[:, y1_1:y1_1 + crop_h, x1_1:x1_1 + crop_w]
        right_random_1 = right_img_full[:, y1_1:y1_1 + crop_h, x1_1:x1_1 + crop_w]
        left_random_2 = left_img_full[:, y1_2:y1_2 + crop_h, x1_2:x1_2 + crop_w]
        right_random_2 = right_img_full[:, y1_2:y1_2 + crop_h, x1_2:x1_2 + crop_w]
        left_random_3 = left_img_full[:, y1_3:y1_3 + crop_h, x1_3:x1_3 + crop_w]
        right_random_3 = right_img_full[:, y1_3:y1_3 + crop_h, x1_3:x1_3 + crop_w]
        left_random_4 = left_img_full[:, y1_4:y1_4 + crop_h, x1_4:x1_4 + crop_w]
        right_random_4 = right_img_full[:, y1_4:y1_4 + crop_h, x1_4:x1_4 + crop_w]
        left_random_5 = left_img_full[:, y1_5:y1_5 + crop_h, x1_5:x1_5 + crop_w]
        right_random_5 = right_img_full[:, y1_5:y1_5 + crop_h, x1_5:x1_5 + crop_w]
        left_random_6 = left_img_full[:, y1_6:y1_6 + crop_h, x1_6:x1_6 + crop_w]
        right_random_6 = right_img_full[:, y1_6:y1_6 + crop_h, x1_6:x1_6 + crop_w]
        left_random_7 = left_img_full[:, y1_7:y1_7 + crop_h, x1_7:x1_7 + crop_w]
        right_random_7 = right_img_full[:, y1_7:y1_7 + crop_h, x1_7:x1_7 + crop_w]
        left_random_8 = left_img_full[:, y1_8:y1_8 + crop_h, x1_8:x1_8 + crop_w]
        right_random_8 = right_img_full[:, y1_8:y1_8 + crop_h, x1_8:x1_8 + crop_w]
        left_random_9 = left_img_full[:, y1_9:y1_9 + crop_h, x1_9:x1_9 + crop_w]
        right_random_9 = right_img_full[:, y1_9:y1_9 + crop_h, x1_9:x1_9 + crop_w]

        # Calculate overlap coordinates
        overlap_coords = self.calculate_overlap(x1_1, y1_1, x1_2, y1_2, crop_w, crop_h)

        # Process disparity if available
        if disparity is not None:
            if not self.training:
                # Pad disparity for validation/test
                disparity = self.pad_to_target_size(disparity, target_w, target_h)
                disparity_half = cv2.resize(disparity, (target_w//2, target_h//2), interpolation=cv2.INTER_NEAREST)
                disparity_low = cv2.resize(disparity, (target_w//4, target_h//4), interpolation=cv2.INTER_NEAREST)
                depth_map = self.create_depth_map(disparity)

        # Prepare common return data
        common_data = {
            "left_filename": self.left_filenames[idx],
            "right_filename": self.right_filenames[idx],
            "textureless_score": textureless_score,
            # Random crops (common for both training and validation)
            "random_coord_1": (x1_1, y1_1),
            "left_random_1": left_random_1,
            "right_random_1": right_random_1,
            "random_coord_2": (x1_2, y1_2),
            "left_random_2": left_random_2,
            "right_random_2": right_random_2,
            "random_coord_3": (x1_3, y1_3),
            "left_random_3": left_random_3,
            "right_random_3": right_random_3,
            "random_coord_4": (x1_4, y1_4),
            "left_random_4": left_random_4,
            "right_random_4": right_random_4,
            "random_coord_5": (x1_5, y1_5),
            "left_random_5": left_random_5,
            "right_random_5": right_random_5,
            "random_coord_6": (x1_6, y1_6),
            "left_random_6": left_random_6,
            "right_random_6": right_random_6,
            "random_coord_7": (x1_7, y1_7),
            "left_random_7": left_random_7,
            "right_random_7": right_random_7,
            "random_coord_8": (x1_8, y1_8),
            "left_random_8": left_random_8,
            "right_random_8": right_random_8,
            "random_coord_9": (x1_9, y1_9),
            "left_random_9": left_random_9,
            "right_random_9": right_random_9,
            "overlap_coords": overlap_coords
        }

        if self.training:
            # Training specific data
            training_data = {
                "left": left_img_full,
                "right": right_img_full,
                "left_strong_aug": left_strong,  # JinoTransform의 strong augmentation 결과
                "right_strong_aug": right_strong,  # JinoTransform의 strong augmentation 결과
                "left_low": left_img_low,
                "right_low": right_img_low,
                "left_half": left_img_half,
                "right_half": right_img_half,
                "disparity": disparity_main,
                "disparity_low": disparity_low,
                "disparity_half": disparity_half,
                "depth_map": depth_map,
                "prior": prior_data
            }
            training_data.update(common_data)
            return training_data
        else:
            # Validation/test specific data
            validation_data = {
                "left": left_img_full,
                "right": right_img_full,
                "left_half": left_img_half.numpy(),
                "right_half": right_img_half.numpy(),
                "left_low": left_img_low.numpy(),
                "right_low": right_img_low.numpy()
            }
            
            if disparity is not None:
                validation_data.update({
                    "disparity": disparity,
                    "disparity_low": disparity_low,
                    "disparity_half": disparity_half,
                    "depth_map": depth_map
                })
            
            validation_data.update(common_data)
            return validation_data
