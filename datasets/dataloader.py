import os
import random
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import cv2
from datasets.data_io import get_transform, read_all_lines, pfm_imread
import torchvision.transforms as transforms
import torch
import matplotlib.pyplot as plt


class PrepareDataset(Dataset):
    def __init__(self, source_datapath, target_datapath, sourcefile_list, targetfile_list, training):
        self.source_datapath = source_datapath
        self.target_datapath = target_datapath
        self.source_left_filenames, self.source_right_filenames, self.source_disp_filenames = self.load_path(sourcefile_list)
        self.target_left_filenames, self.target_right_filenames, self.target_disp_filenames = self.load_path(targetfile_list)
        self.training = training
        if self.training:
            assert self.source_disp_filenames is not None

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
        return Image.open(filename).convert('RGB')

    def load_disp(self, filename):
        data = Image.open(filename)
        data = np.array(data, dtype=np.float32) / 256.
        return data

    def __len__(self):
        return len(self.source_left_filenames)

    def __getitem__(self, index):
        
        print(self.source_datapath)
        src_left_img = self.load_image(os.path.join(self.source_datapath, self.source_left_filenames[index]))
        src_right_img = self.load_image(os.path.join(self.source_datapath, self.source_right_filenames[index]))
        src_disparity = self.load_disp(os.path.join(self.source_datapath, self.source_disp_filenames[index]))
        
        tgt_left_img = self.load_image(os.path.join(self.target_datapath, self.target_left_filenames[index]))
        tgt_right_img = self.load_image(os.path.join(self.target_datapath, self.target_right_filenames[index]))

        if self.target_disp_filenames:  # 만약에 target 이미지에 대해서 disparity 참값을 가지고 있다면
            tgt_disparity = self.load_disp(os.path.join(self.target_datapath, self.target_disp_filenames[index]))
        else:
            tgt_disparity = None

        if self.training:
            w, h = src_left_img.size
            crop_w, crop_h = 512, 256

            x1 = random.randint(0, w - crop_w)
            # y1 = random.randint(0, h - crop_h)
            if  random.randint(0, 10) >= int(8):
                y1 = random.randint(0, h - crop_h)
            else:
                y1 = random.randint(int(0.3 * h), h - crop_h)

            # random crop source / target
            src_left_img = src_left_img.crop((x1, y1, x1 + crop_w, y1 + crop_h))
            src_right_img = src_right_img.crop((x1, y1, x1 + crop_w, y1 + crop_h))
            src_disparity = src_disparity[y1:y1 + crop_h, x1:x1 + crop_w]
            src_disparity_low = cv2.resize(src_disparity, (crop_w//4, crop_h//4), interpolation=cv2.INTER_NEAREST)

            tgt_left_img = tgt_left_img.crop((x1, y1, x1 + crop_w, y1 + crop_h))
            tgt_right_img = tgt_right_img.crop((x1, y1, x1 + crop_w, y1 + crop_h))
            if tgt_disparity is not None:
                tgt_disparity = tgt_disparity[y1:y1 + crop_h, x1:x1 + crop_w]
                tgt_disparity_low = cv2.resize(tgt_disparity, (crop_w//4, crop_h//4), interpolation=cv2.INTER_NEAREST)


            # to tensor, normalize
            processed = get_transform()
            src_left_img = processed(src_left_img)
            src_right_img = processed(src_right_img)
            
            tgt_left_img = processed(tgt_left_img)
            tgt_right_img = processed(tgt_right_img)
            

            return {"src_left": src_left_img,
                    "src_right": src_right_img,
                    "src_disparity": src_disparity,
                    "src_disparity_low": src_disparity_low,
                    "tgt_left": tgt_left_img,
                    "tgt_right": tgt_right_img,
                    "tgt_disparity": tgt_disparity,
                    "tgt_disparity_low": tgt_disparity_low}

        else:
            w, h = src_left_img.size

            # normalize
            processed = get_transform()
            src_left_img = processed(src_left_img).numpy()
            src_right_img = processed(src_right_img).numpy()
            tgt_left_img = processed(tgt_left_img).numpy()
            tgt_right_img = processed(tgt_right_img).numpy()
            
            
            # pad to size 1248x384
            top_pad = 384 - h
            right_pad = 1248 - w
            assert top_pad > 0 and right_pad > 0
            # pad images
            src_left_img = np.lib.pad(src_left_img, ((0, 0), (top_pad, 0), (0, right_pad)), mode='constant', constant_values=0)
            src_right_img = np.lib.pad(src_right_img, ((0, 0), (top_pad, 0), (0, right_pad)), mode='constant',
                                   constant_values=0)
            src_disparity = np.lib.pad(src_disparity, ((top_pad, 0), (0, right_pad)), mode='constant', constant_values=0)

            
            tgt_left_img = np.lib.pad(tgt_left_img, ((0, 0), (top_pad, 0), (0, right_pad)), mode='constant', constant_values=0)
            tgt_right_img = np.lib.pad(tgt_right_img, ((0, 0), (top_pad, 0), (0, right_pad)), mode='constant',
                                   constant_values=0)
            
            if tgt_disparity is not None:
                assert len(tgt_disparity.shape) == 2
                tgt_disparity = np.lib.pad(tgt_disparity, ((top_pad, 0), (0, right_pad)), mode='constant', constant_values=0)


            if tgt_disparity is not None:
                return {"src_left": src_left_img,
                        "src_right": src_right_img,
                        "src_disparity": src_disparity,
                        "tgt_left": tgt_left_img,
                        "tgt_right": tgt_right_img,
                        "tgt_disparity": tgt_disparity,
                        "top_pad": top_pad,
                        "right_pad": right_pad}
            else:
                return {"src_left": src_left_img,
                        "src_right": src_right_img,
                        "tgt_left": tgt_left_img,
                        "tgt_right": tgt_right_img,
                        "top_pad": top_pad,
                        "right_pad": right_pad,
                        "source_left_filename": self.source_left_filenames[index],
                        "source_right_filename": self.source_right_filenames[index],
                        "target_left_filename": self.target_left_filenames[index],
                        "target_right_filename": self.target_right_filenames[index]}
