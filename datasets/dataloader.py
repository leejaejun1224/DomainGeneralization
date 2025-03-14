import os
import torch
import random
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import cv2
from transformers import SegformerImageProcessor
from datasets.data_io import read_all_lines, reshape_image, reshape_disparity, get_transform


class PrepareDataset(Dataset):
    def __init__(self, source_datapath, target_datapath, sourcefile_list, targetfile_list, training):
        self.source_datapath = source_datapath
        self.target_datapath = target_datapath
        self.training = training
        
        # Load source and target paths
        self.source_paths = self._load_dataset_paths(sourcefile_list)
        self.target_paths = self._load_dataset_paths(targetfile_list)

        # Validate training requirements
        if self.training and self.source_paths['disp_filenames'] is None:
            raise AssertionError("Training requires source disparity data")

    def _load_dataset_paths(self, filelist):
        """Load and parse dataset file paths"""
        left, right, disp = self.load_path(filelist)
        return {
            'left_filenames': left,
            'right_filenames': right,
            'disp_filenames': disp
        }

    def load_path(self, list_filename):
        lines = read_all_lines(list_filename)
        splits = [line.split() for line in lines]
        left_images = [x[0] for x in splits]
        right_images = [x[1] for x in splits]
        disp_images = [x[2] for x in splits] if len(splits[0]) > 2 else None
        return left_images, right_images, disp_images

    def load_image(self, filename):
        filename = os.path.expanduser(filename)
        return Image.open(filename).convert('RGB')

    def load_disp(self, filename):
        filename = os.path.expanduser(filename)
        data = Image.open(filename)
        return np.array(data, dtype=np.float32) / 256.0

    def __len__(self):
        return min(len(self.source_paths['left_filenames']), 
                  len(self.target_paths['left_filenames']))

    def _load_stereo_pair(self, datapath, left_file, right_file, disp_file=None):
        left_img = self.load_image(os.path.join(datapath, left_file))
        right_img = self.load_image(os.path.join(datapath, right_file))
        disparity = None
        if disp_file:
            disparity = self.load_disp(os.path.join(datapath, disp_file))
        return left_img, right_img, disparity

    def _random_crop(self, left_img, right_img, disparity=None, crop_size=(512, 256)):
        w, h = left_img.size
        crop_w, crop_h = crop_size
        
        x1 = random.randint(0, w - crop_w)
        y1 = random.randint(int(0.3 * h), h - crop_h) if random.randint(0, 10) < 8 else random.randint(0, h - crop_h)

        # Crop images
        left_img = left_img.crop((x1, y1, x1 + crop_w, y1 + crop_h))
        right_img = right_img.crop((x1, y1, x1 + crop_w, y1 + crop_h))

        # Crop and downsample disparity if available
        disp_crop = None
        disp_low = None
        if disparity is not None:
            disp_crop = disparity[y1:y1 + crop_h, x1:x1 + crop_w]
            disp_low = cv2.resize(disp_crop, (crop_w // 4, crop_h // 4), interpolation=cv2.INTER_NEAREST)

        return left_img, right_img, disp_crop, disp_low

    ### 차라리 training이란 testing을 따로 분리를 하자.
    def _prepare_training_sample(self, index):
        
        src_left, src_right, src_disp = self._load_stereo_pair(
            self.source_datapath,
            self.source_paths['left_filenames'][index],
            self.source_paths['right_filenames'][index],
            self.source_paths['disp_filenames'][index]
        )

        tgt_left, tgt_right, tgt_disp = self._load_stereo_pair(
            self.target_datapath,
            self.target_paths['left_filenames'][index],
            self.target_paths['right_filenames'][index],
            self.target_paths['disp_filenames'][index] if self.target_paths['disp_filenames'] else None
        )

        src_left, src_right, src_disp, src_disp_low = self._random_crop(src_left, src_right, src_disp)
        tgt_left, tgt_right, tgt_disp, tgt_disp_low = self._random_crop(tgt_left, tgt_right, tgt_disp)

        transform = get_transform()
        return {
            "src_left": transform(src_left),
            "src_right": transform(src_right),
            "src_disparity": src_disp,
            "src_disparity_low": src_disp_low,
            "tgt_left": transform(tgt_left),
            "tgt_right": transform(tgt_right),
            "tgt_disparity": tgt_disp,
            "tgt_disparity_low": tgt_disp_low,
            "source_left_filename": self.source_paths['left_filenames'][index],
            "source_right_filename": self.source_paths['right_filenames'][index],
            "target_left_filename": self.target_paths['left_filenames'][index],
            "target_right_filename": self.target_paths['right_filenames'][index]
        }

    def _prepare_test_sample(self, index):

        ### kitti의 경우 test set에 대해서는 disparity 데이터가 없다.
        src_left, src_right, src_disp = self._load_stereo_pair(
            self.source_datapath,
            self.source_paths['left_filenames'][index],
            self.source_paths['right_filenames'][index],
            self.source_paths['disp_filenames'][index] if self.source_paths['disp_filenames'] else None
        )

        tgt_left, tgt_right, tgt_disp = self._load_stereo_pair(
            self.target_datapath,
            self.target_paths['left_filenames'][index],
            self.target_paths['right_filenames'][index],
            self.target_paths['disp_filenames'][index] if self.target_paths['disp_filenames'] else None
        )

        src_left = torch.from_numpy(reshape_image(src_left)).float()
        src_right = torch.from_numpy(reshape_image(src_right)).float()
        src_disp = torch.from_numpy(reshape_disparity(src_disp)).float() if src_disp is not None else None

        tgt_left = torch.from_numpy(reshape_image(tgt_left)).float()
        tgt_right = torch.from_numpy(reshape_image(tgt_right)).float()
        tgt_disp = torch.from_numpy(reshape_disparity(tgt_disp)).float() if tgt_disp is not None else None

        result = {
            "src_left": src_left,
            "src_right": src_right,
            "tgt_left": tgt_left,
            "tgt_right": tgt_right,
            "source_left_filename": self.source_paths['left_filenames'][index],
            "source_right_filename": self.source_paths['right_filenames'][index],
            "target_left_filename": self.target_paths['left_filenames'][index],
            "target_right_filename": self.target_paths['right_filenames'][index]
        }

        if src_disp is not None:
            result["src_disparity"] = src_disp
        if tgt_disp is not None:
            result["tgt_disparity"] = tgt_disp

        return result

    def __getitem__(self, index):
        if self.training:
            return self._prepare_training_sample(index)
        return self._prepare_test_sample(index)