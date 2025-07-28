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


class ImageProcessor:
    """Handles image processing operations"""
    
    # Constants
    TARGET_SIZE = (1248, 384)  # (width, height)
    CROP_SIZE = (1152, 320)    # (width, height)
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


class JinoTransform:
    """Handles image augmentation transforms"""
    
    def __init__(self):
        self.weak_photo = transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        self.strong_photo = transforms.Compose([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
            transforms.RandomGrayscale(0.2),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: t * 255.0),
            transforms.Lambda(lambda t: ImageProcessor.pad_to_size(t, ImageProcessor.TARGET_SIZE)),
            transforms.Normalize(ImageProcessor.IMAGENET_MEAN, ImageProcessor.IMAGENET_STD),
        ])
        self.normalize = transforms.Normalize(ImageProcessor.IMAGENET_MEAN, ImageProcessor.IMAGENET_STD)

    def _to_tensor_and_norm(self, pil_img, normalize):
        """Convert PIL image to tensor and normalize"""
        tensor = transforms.functional.to_tensor(pil_img) * 255.0
        tensor = ImageProcessor.pad_to_size(tensor, ImageProcessor.TARGET_SIZE)
        return normalize(tensor)

    def __call__(self, pil_img):
        weak = self._to_tensor_and_norm(pil_img, self.normalize)
        strong = self.strong_photo(pil_img)
        return weak, strong


class KITTI2015Dataset(Dataset):
    """KITTI 2015 stereo dataset"""
    
    NUM_RANDOM_CROPS = 9
    
    def __init__(self, datapath, list_filename, training, max_len=None, aug=False, prior=None):
        self.datapath = datapath
        self.left_filenames, self.right_filenames, self.disp_filenames = self._load_paths(list_filename)
        self.training = training
        self.data_len = len(self.left_filenames)
        self.max_len = max_len
        self.aug = aug
        self.prior_path = prior
        
        # Initialize transforms
        self.hf_transform = HFStereoV2(use_edge_enhancement=True)
        self.jino_transform = JinoTransform()
        self.processor = ImageProcessor()
        
        if self.training:
            assert self.disp_filenames is not None

    def _load_paths(self, list_filename):
        """Load image and disparity file paths"""
        lines = read_all_lines(list_filename)
        splits = [line.split() for line in lines]
        
        left_images = [x[0] for x in splits]
        right_images = [x[1] for x in splits]
        
        if len(splits[0]) == 2:  # No ground truth available
            return left_images, right_images, None
        else:
            disp_images = [x[2] for x in splits]
            return left_images, right_images, disp_images

    def _load_image(self, filename):
        """Load and convert image to RGB"""
        return Image.open(os.path.expanduser(filename)).convert('RGB')

    def _load_disparity(self, filename):
        """Load disparity image"""
        data = Image.open(os.path.expanduser(filename))
        return np.array(data, dtype=np.float32) / 256.0
    
    def _load_prior(self):
        """Load prior data if available"""
        if self.prior_path is None:
            return 0.0
            
        prior_path = os.path.expanduser(self.prior_path)
        if not os.path.exists(prior_path):
            raise FileNotFoundError(f"Prior file {prior_path} does not exist.")
        return np.load(prior_path)
    
    def _generate_random_coords(self, w, h, num_crops=None):
        """Generate random crop coordinates"""
        if num_crops is None:
            num_crops = self.NUM_RANDOM_CROPS
            
        crop_w, crop_h = self.processor.CROP_SIZE
        coords = []
        
        for _ in range(num_crops):
            x = random.randint(0, w - crop_w)
            y = random.randint(int(0.1 * h), h - crop_h)
            coords.append((x, y))
            
        return coords

    def _calculate_overlap(self, coord1, coord2):
        """Calculate overlap region between two crops"""
        x1_1, y1_1 = coord1
        x1_2, y1_2 = coord2
        crop_w, crop_h = self.processor.CROP_SIZE
        
        x_overlap_start = max(x1_1, x1_2)
        y_overlap_start = max(y1_1, y1_2)
        x_overlap_end = min(x1_1 + crop_w, x1_2 + crop_w)
        y_overlap_end = min(y1_1 + crop_h, y1_2 + crop_h)
        
        return {
            'x': x_overlap_start,
            'y': y_overlap_start,
            'width': max(0, x_overlap_end - x_overlap_start),
            'height': max(0, y_overlap_end - y_overlap_start)
        }

    def _create_multiscale_images(self, left_img, right_img):
        """Create multi-scale versions of images"""
        target_w, target_h = self.processor.TARGET_SIZE
        processed = get_transform()
        
        # Create different scales
        scales = {
            'full': (target_w, target_h),
            'half': (target_w//2, target_h//2),
            'low': (target_w//4, target_h//4)
        }
        
        results = {}
        for scale_name, (w, h) in scales.items():
            left_scaled = left_img.resize((w, h), Image.BICUBIC)
            right_scaled = right_img.resize((w, h), Image.BICUBIC)
            
            results[f'left_{scale_name}'] = processed(left_scaled)
            results[f'right_{scale_name}'] = processed(right_scaled)
            
        return results

    def _create_random_crops(self, img_array, coords):
        """Create random crops from image array"""
        crop_w, crop_h = self.processor.CROP_SIZE
        crops = {}
        
        for i, (x, y) in enumerate(coords, 1):
            crops[f'crop_{i}'] = img_array[:, y:y + crop_h, x:x + crop_w]
            
        return crops

    def _process_disparity(self, disparity):
        """Process disparity at multiple scales"""
        target_w, target_h = self.processor.TARGET_SIZE
        
        # Pad original disparity
        disparity_padded = self.processor.pad_to_size(disparity, self.processor.TARGET_SIZE)
        
        # Create different scales
        disparity_half = cv2.resize(disparity_padded, (target_w//2, target_h//2), interpolation=cv2.INTER_NEAREST)
        disparity_low = cv2.resize(disparity_padded, (target_w//4, target_h//4), interpolation=cv2.INTER_NEAREST)
        depth_map = self.processor.create_depth_map(disparity_padded)
        
        return {
            'disparity': disparity_padded,
            'disparity_half': disparity_half,
            'disparity_low': disparity_low,
            'depth_map': depth_map
        }

    def __len__(self):
        return self.max_len if self.max_len is not None else self.data_len

    def __getitem__(self, index):
        # Handle index overflow
        idx = index if index < self.data_len else random.randint(0, self.data_len - 1)
        
        # Load images
        left_img = self._load_image(os.path.join(self.datapath, self.left_filenames[idx]))
        right_img = self._load_image(os.path.join(self.datapath, self.right_filenames[idx]))
        
        # Load disparity if available
        disparity = None
        if self.disp_filenames:
            disparity = self._load_disparity(os.path.join(self.datapath, self.disp_filenames[idx]))
        
        # Compute textureless score
        textureless_score = self.processor.compute_textureless_score(left_img)
        
        # Generate random crop coordinates
        w, h = left_img.size
        random_coords = self._generate_random_coords(w, h)
        
        # Prepare images based on training/validation mode
        if self.training:
            # Use original images for training
            main_left, main_right = left_img, right_img
        else:
            # Resize for validation/test
            target_w, target_h = self.processor.TARGET_SIZE
            main_left = left_img.resize((target_w, target_h), Image.BICUBIC)
            main_right = right_img.resize((target_w, target_h), Image.BICUBIC)
        
        # Create multi-scale images
        multiscale_data = self._create_multiscale_images(main_left, main_right)
        
        # Process full images
        processed = get_transform()
        left_full = processed(left_img).numpy()
        right_full = processed(right_img).numpy()
        
        # Pad to target size
        left_full = self.processor.pad_to_size(left_full, self.processor.TARGET_SIZE)
        right_full = self.processor.pad_to_size(right_full, self.processor.TARGET_SIZE)
        
        # Create random crops
        left_crops = self._create_random_crops(left_full, random_coords)
        right_crops = self._create_random_crops(right_full, random_coords)
        
        # Prepare base return data
        return_data = {
            "left_filename": self.left_filenames[idx],
            "right_filename": self.right_filenames[idx],
            "textureless_score": textureless_score,
            "left": left_full,
            "right": right_full,
            "overlap_coords": self._calculate_overlap(random_coords[0], random_coords[1])
        }
        
        # Add multi-scale data
        return_data.update(multiscale_data)
        
        # Add random crops and coordinates
        for i, coord in enumerate(random_coords, 1):
            return_data[f"random_coord_{i}"] = coord
            return_data[f"left_random_{i}"] = left_crops[f'crop_{i}']
            return_data[f"right_random_{i}"] = right_crops[f'crop_{i}']
        
        # Handle training-specific data
        if self.training:
            # Apply augmentation if enabled
            if self.aug:
                left_weak, left_strong = self.jino_transform(main_left)
                right_weak, right_strong = self.jino_transform(main_right)
            else:
                left_weak, _ = self.jino_transform(main_left)
                right_weak, _ = self.jino_transform(main_right)
                left_strong = left_weak.clone()
                right_strong = right_weak.clone()
            
            return_data.update({
                "left_strong_aug": left_strong,
                "right_strong_aug": right_strong,
                "prior": self._load_prior()
            })
        else:
            # Convert tensors to numpy for validation
            for key in ['left_half', 'right_half', 'left_low', 'right_low']:
                if key in return_data:
                    return_data[key] = return_data[key].numpy()
        
        # Add disparity data if available
        if disparity is not None:
            disparity_data = self._process_disparity(disparity)
            return_data.update(disparity_data)
        
        return return_data
