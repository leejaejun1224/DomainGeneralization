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

        self.source_left_filenames, self.source_right_filenames, self.source_disp_filenames = self.load_path(sourcefile_list)
        self.target_left_filenames, self.target_right_filenames, self.target_disp_filenames = self.load_path(targetfile_list)

        self.training = training
        if self.training:
            # source에 대한 disparity는 반드시 있어야 함
            assert self.source_disp_filenames is not None, "Training 시에는 source의 disparity가 필요합니다."

    def load_path(self, list_filename):
        lines = read_all_lines(list_filename)
        splits = [line.split() for line in lines]
        left_images = [x[0] for x in splits]
        right_images = [x[1] for x in splits]
        # ground truth가 없는 경우 (파일에 좌/우 경로만 존재)
        if len(splits[0]) == 2:
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
        data = np.array(data, dtype=np.float32) / 256.0
        return data

    def __len__(self):
        # source와 target 둘 중 작은 길이를 Dataset 크기로 설정
        return min(len(self.source_left_filenames), len(self.target_left_filenames))

    def __getitem__(self, index):
        src_index = index
        tgt_index = index

        # (1) 이미지 / disparity 불러오기
        src_left_img = self.load_image(os.path.join(self.source_datapath, self.source_left_filenames[src_index]))
        src_right_img = self.load_image(os.path.join(self.source_datapath, self.source_right_filenames[src_index]))
        
        if self.source_disp_filenames:
            src_disparity = self.load_disp(os.path.join(self.source_datapath, self.source_disp_filenames[src_index]))
        else:
            src_disparity = None


        tgt_left_img = self.load_image(os.path.join(self.target_datapath, self.target_left_filenames[tgt_index]))
        tgt_right_img = self.load_image(os.path.join(self.target_datapath, self.target_right_filenames[tgt_index]))

        if self.target_disp_filenames:
            tgt_disparity = self.load_disp(os.path.join(self.target_datapath, self.target_disp_filenames[tgt_index]))
        else:
            tgt_disparity = None

        # (2) 학습일 때
        if self.training:
            # random crop (512x256)
            w, h = src_left_img.size
            crop_w, crop_h = 512, 256

            x1 = random.randint(0, w - crop_w)
            # 30% 지점부터 crop or 0부터 crop 중 랜덤
            if random.randint(0, 10) >= 8:
                y1 = random.randint(0, h - crop_h)
            else:
                y1 = random.randint(int(0.3 * h), h - crop_h)

            # target도 동일하게 random crop
            target_w, target_h = tgt_left_img.size
            target_x1 = random.randint(0, target_w - crop_w)
            if random.randint(0, 10) >= 8:
                target_y1 = random.randint(0, target_h - crop_h)
            else:
                target_y1 = random.randint(int(0.3 * target_h), target_h - crop_h)

            # source crop
            src_left_img = src_left_img.crop((x1, y1, x1 + crop_w, y1 + crop_h))
            src_right_img = src_right_img.crop((x1, y1, x1 + crop_w, y1 + crop_h))
            src_disparity = src_disparity[y1:y1 + crop_h, x1:x1 + crop_w]
            src_disparity_low = cv2.resize(src_disparity, (crop_w // 4, crop_h // 4), interpolation=cv2.INTER_NEAREST)

            # target crop
            tgt_left_img = tgt_left_img.crop((target_x1, target_y1, target_x1 + crop_w, target_y1 + crop_h))
            tgt_right_img = tgt_right_img.crop((target_x1, target_y1, target_x1 + crop_w, target_y1 + crop_h))

            if tgt_disparity is not None:
                tgt_disparity = tgt_disparity[target_y1:target_y1 + crop_h, target_x1:target_x1 + crop_w]
                tgt_disparity_low = cv2.resize(tgt_disparity, (crop_w // 4, crop_h // 4), interpolation=cv2.INTER_NEAREST)
            else:
                tgt_disparity_low = None

            # ToTensor + Normalize
            transform_func = get_transform()
            src_left_img = transform_func(src_left_img)   # (C,H,W) tensor
            src_right_img = transform_func(src_right_img)
            tgt_left_img = transform_func(tgt_left_img)
            tgt_right_img = transform_func(tgt_right_img)

            return {
                "src_left": src_left_img,   # torch.Tensor
                "src_right": src_right_img,
                "src_disparity": src_disparity,  # 여기는 아직 numpy
                "src_disparity_low": src_disparity_low,  # numpy
                "tgt_left": tgt_left_img,
                "tgt_right": tgt_right_img,
                "tgt_disparity": tgt_disparity,
                "tgt_disparity_low": tgt_disparity_low
            }

        # (3) 검증/테스트일 때
        else:
            # reshape_image / reshape_disparity를 적용 (1248×384 맞춤)
            # 함수 내부에서 get_transform() 호출 -> ToTensor & Normalize -> numpy array로 반환
            src_left_img_np = reshape_image(src_left_img)    # shape: (C, 384, 1248)
            src_right_img_np = reshape_image(src_right_img)

            if src_disparity is not None:
                src_disparity_np = reshape_disparity(src_disparity)  # shape: (384, 1248)
            else:
                src_disparity_np = None

            tgt_left_img_np = reshape_image(tgt_left_img)
            tgt_right_img_np = reshape_image(tgt_right_img)

            if tgt_disparity is not None:
                tgt_disparity_np = reshape_disparity(tgt_disparity)
            else:
                tgt_disparity_np = None

            # numpy -> torch
            src_left_img_t = torch.from_numpy(src_left_img_np).float()   # (C,384,1248)
            src_right_img_t = torch.from_numpy(src_right_img_np).float()

            if src_disparity_np is not None:
                src_disparity_t = torch.from_numpy(src_disparity_np).float() # (384,1248)
            else:
                src_disparity_t = None

            tgt_left_img_t = torch.from_numpy(tgt_left_img_np).float()
            tgt_right_img_t = torch.from_numpy(tgt_right_img_np).float()

            if tgt_disparity_np is not None:
                tgt_disparity_t = torch.from_numpy(tgt_disparity_np).float()
            else:
                tgt_disparity_t = None

            # 결과 딕셔너리 반환
            if tgt_disparity_t is not None:
                return {
                    "src_left": src_left_img_t,
                    "src_right": src_right_img_t,
                    "src_disparity": src_disparity_t,
                    "tgt_left": tgt_left_img_t,
                    "tgt_right": tgt_right_img_t,
                    "tgt_disparity": tgt_disparity_t,
                    "source_left_filename": self.source_left_filenames[src_index],
                    "source_right_filename": self.source_right_filenames[src_index],
                    "target_left_filename": self.target_left_filenames[tgt_index],
                    "target_right_filename": self.target_right_filenames[tgt_index]
                }
            else:
                return {
                    "src_left": src_left_img_t,
                    "src_right": src_right_img_t,
                    "src_disparity": src_disparity_t,
                    "tgt_left": tgt_left_img_t,
                    "tgt_right": tgt_right_img_t,
                    "source_left_filename": self.source_left_filenames[src_index],
                    "source_right_filename": self.source_right_filenames[src_index],
                    "target_left_filename": self.target_left_filenames[tgt_index],
                    "target_right_filename": self.target_right_filenames[tgt_index]
                }