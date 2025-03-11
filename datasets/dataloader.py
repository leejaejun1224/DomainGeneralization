import os
import random
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import cv2
from datasets.data_io import get_transform, read_all_lines, pfm_imread, reshape_image, reshape_disparity


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
        filename = os.path.expanduser(filename)
        return Image.open(filename).convert('RGB')

    def load_image1(self, filename):
        filename = os.path.expanduser(filename)
        img = Image.open(filename).convert('RGB')
        
        # 파일 이름을 이미지에 추가하기 위해 OpenCV 형식으로 변환
        img_cv = np.array(img)  # PIL -> NumPy 변환 (H, W, C)
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)  # RGB -> BGR 변환 (OpenCV는 BGR 사용)
        
        # 텍스트 위치 및 스타일 설정
        text = os.path.basename(filename)  # 파일 이름만 추출
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_color = (0, 255, 0)  # 초록색
        thickness = 2
        position = (50, 50)  # 좌측 상단
        
        # 이미지에 텍스트 추가
        img_cv = cv2.putText(img_cv, text, position, font, font_scale, font_color, thickness, cv2.LINE_AA)

        # 저장할 경로 설정 (예제: "debug_images/" 디렉토리에 저장)
        save_dir = "debug_images"
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, os.path.basename(filename))
        cv2.imwrite(save_path, img_cv)  # 이미지 저장

        print(f"Saved image with filename text: {save_path}")  # 디버깅용 출력
        
        return img  # 원본 PIL 이미지 반환

    def load_disp(self, filename):
        filename = os.path.expanduser(filename)
        data = Image.open(filename)
        data = np.array(data, dtype=np.float32) / 256.
        return data

    def __len__(self):
        # 이 함수가 필요한 이유 : dataloader 클래스에서 이 길이의 안쪽에 있는 dataset의 인덱스를 가져옴
        # 고로 하나가 더 커버리면 반대쪽에서 가져올 인덱스가 없어서 에러가 남. 
        # 이러면 작은 놈을 따라갈 수 밖에 없음,
        return min(len(self.source_left_filenames), len(self.target_left_filenames))

    def __getitem__(self, index):
        src_index = index
        tgt_index = index
        src_left_img = self.load_image(os.path.join(self.source_datapath, self.source_left_filenames[src_index]))        
        src_right_img = self.load_image(os.path.join(self.source_datapath, self.source_right_filenames[src_index]))
        src_disparity = self.load_disp(os.path.join(self.source_datapath, self.source_disp_filenames[src_index]))

        tgt_left_img = self.load_image(os.path.join(self.target_datapath, self.target_left_filenames[tgt_index]))
        tgt_right_img = self.load_image(os.path.join(self.target_datapath, self.target_right_filenames[tgt_index]))

        if self.target_disp_filenames:   # target 이미지에 대해 disparity 참값이 있는 경우
            tgt_disparity = self.load_disp(os.path.join(self.target_datapath, self.target_disp_filenames[tgt_index]))
        else:
            tgt_disparity = None

        # 이미지 및 disparity 크기가 (width > 2000, height > 1000) 인 경우, 1242x375로 리사이즈
        # if src_left_img.size[0] > 2000 and src_left_img.size[1] > 1000:
        #     src_left_img = src_left_img.resize((1242, 375), resample=Image.LANCZOS)
        #     src_right_img = src_right_img.resize((1242, 375), resample=Image.LANCZOS)
        #     src_disparity = cv2.resize(src_disparity, (1242, 375), interpolation=cv2.INTER_LANCZOS4)
        # if tgt_left_img.size[0] > 2000 and tgt_left_img.size[1] > 1000:
        #     tgt_left_img = tgt_left_img.resize((1242, 375), resample=Image.LANCZOS)
        #     tgt_right_img = tgt_right_img.resize((1242, 375), resample=Image.LANCZOS)
        #     if tgt_disparity is not None:
        #         tgt_disparity = cv2.resize(tgt_disparity, (1242, 375), interpolation=cv2.INTER_LANCZOS4)

        if self.training:

            w, h = src_left_img.size
            crop_w, crop_h = 512, 256

            x1 = random.randint(0, w - crop_w)
            if random.randint(0, 10) >= 8:
                y1 = random.randint(0, h - crop_h)
            else:
                y1 = random.randint(int(0.3 * h), h - crop_h)

            target_w, target_h = tgt_left_img.size
            target_x1 = random.randint(0, target_w - crop_w)
            if random.randint(0, 10) >= 8:
                target_y1 = random.randint(0, target_h - crop_h)
            else:
                target_y1 = random.randint(int(0.3 * target_h), target_h - crop_h)

            # source 이미지 및 disparity random crop
            src_left_img = src_left_img.crop((x1, y1, x1 + crop_w, y1 + crop_h))
            src_right_img = src_right_img.crop((x1, y1, x1 + crop_w, y1 + crop_h))
            src_disparity = src_disparity[y1:y1 + crop_h, x1:x1 + crop_w]
            src_disparity_low = cv2.resize(src_disparity, (crop_w // 4, crop_h // 4), interpolation=cv2.INTER_NEAREST)

            # target 이미지 및 disparity random crop
            tgt_left_img = tgt_left_img.crop((target_x1, target_y1, target_x1 + crop_w, target_y1 + crop_h))
            tgt_right_img = tgt_right_img.crop((target_x1, target_y1, target_x1 + crop_w, target_y1 + crop_h))
            if tgt_disparity is not None:
                tgt_disparity = tgt_disparity[target_y1:target_y1 + crop_h, target_x1:target_x1 + crop_w]
                tgt_disparity_low = cv2.resize(tgt_disparity, (crop_w // 4, crop_h // 4), interpolation=cv2.INTER_NEAREST)

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
            src_left_img = reshape_image(src_left_img)
            src_right_img = reshape_image(src_right_img)
            src_disparity = reshape_disparity(src_disparity)
            tgt_left_img = reshape_image(tgt_left_img)
            tgt_right_img = reshape_image(tgt_right_img)

            if tgt_disparity is not None:
                tgt_disparity = reshape_disparity(tgt_disparity)

            if tgt_disparity is not None:
                return {"src_left": src_left_img,
                        "src_right": src_right_img,
                        "src_disparity": src_disparity,
                        "tgt_left": tgt_left_img,
                        "tgt_right": tgt_right_img,
                        "tgt_disparity": tgt_disparity,
                        "source_left_filename": self.source_left_filenames[src_index],
                        "source_right_filename": self.source_right_filenames[src_index],
                        "target_left_filename": self.target_left_filenames[tgt_index],
                        "target_right_filename": self.target_right_filenames[tgt_index]}
            else:
                return {"src_left": src_left_img,
                        "src_right": src_right_img,
                        "tgt_left": tgt_left_img,
                        "tgt_right": tgt_right_img,
                        "source_left_filename": self.source_left_filenames[src_index],
                        "source_right_filename": self.source_right_filenames[src_index],
                        "target_left_filename": self.target_left_filenames[tgt_index],
                        "target_right_filename": self.target_right_filenames[tgt_index]}


# if __name__ == "__main__":
#     source_datapath = '/home/jaejun/dataset/kitti'
#     target_datapath = '/home/jaejun/dataset/cityscapes'
#     sourcefile_list = './filenames/source/kitti_2015_train.txt'
#     targetfile_list = './filenames/target/cityscapes_train.txt'
#     dataset = PrepareDataset(source_datapath, target_datapath, sourcefile_list, targetfile_list, training=True)
#     dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1)
#     for i, data in enumerate(dataloader):
#         print(i, " :", data['src_left'].shape)
#         print(i, " :", data['tgt_left'].shape)