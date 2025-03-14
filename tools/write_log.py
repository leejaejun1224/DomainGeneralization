import os
import cv2
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from skimage import exposure

def save_att(data_batch, dir_name):
    os.makedirs(dir_name + '/att', exist_ok=True)
    # print("data_batch['src_pred_disp']", len(data_batch['src_pred_disp']))

    # this for loop is for batch size
    for i in range(data_batch['src_pred_disp'][0].shape[0]):
        att_prob, _ = data_batch['src_pred_disp'][2][i].max(dim=0, keepdim=True)
        att_prob = att_prob.squeeze(0).cpu().numpy()
        source_left_filename = data_batch['source_left_filename'][i].split('/')[-1]
        plt.imsave(os.path.join(dir_name, 'att', source_left_filename), att_prob, cmap='gray')

def save_gt(data_batch, dir_name):
    os.makedirs(dir_name + '/save_gt', exist_ok=True)
    for i in range(data_batch['src_pred_disp'][0].shape[0]):
        source_left_filename = data_batch['source_left_filename'][i].split('/')[-1]
        plt.imsave(os.path.join(dir_name, 'save_gt', source_left_filename), data_batch['src_disparity'][i].cpu().numpy(), cmap='jet')


def save_disparity(data_batch, dir_name):
    os.makedirs(dir_name + '/disp/src', exist_ok=True)
    os.makedirs(dir_name + '/disp/tgt', exist_ok=True)
    pred_src_dir = os.path.join(dir_name, 'disp', 'src')
    os.makedirs(pred_src_dir, exist_ok=True)
    for i in range(data_batch['src_pred_disp'][0].shape[0]):

        pred_src = data_batch['src_pred_disp'][0][i].cpu().numpy()
        source_left_filename = data_batch['source_left_filename'][i].split('/')[-1]
        plt.imsave(os.path.join(pred_src_dir, source_left_filename), pred_src, cmap='jet')

        pred_tgt_dir = os.path.join(dir_name, 'disp', 'tgt')
        os.makedirs(pred_tgt_dir, exist_ok=True)
        pred_tgt = data_batch['tgt_pred_disp'][0][i].cpu().numpy()
        target_left_filename = data_batch['target_left_filename'][i].split('/')[-1] 
        plt.imsave(os.path.join(pred_tgt_dir, target_left_filename), pred_tgt, cmap='jet')


def compare(data_batch, dir_name):
    cost_volume_compare_dir = os.path.join(dir_name, 'cost_volume_compare')
    os.makedirs(cost_volume_compare_dir, exist_ok=True)
    if 'src_disparity' in data_batch.keys():
        for i in range(data_batch['src_pred_disp'][0].shape[0]):

            source_left_filename = data_batch['source_left_filename'][i].split('/')[-1]
            shape_map_resized = F.interpolate(data_batch['src_shape_map'][i].float(), scale_factor=4, mode="nearest")
            shape_map_resized = shape_map_resized.squeeze(0).squeeze(0).cpu().numpy()
            disparity = data_batch['src_disparity'][i].cpu().numpy()


            if shape_map_resized.shape != disparity.shape:
                print("shape_map_resized shape : ", shape_map_resized.shape)
                print("disparity shape : ", disparity.shape)

            diff = np.abs(shape_map_resized - disparity)
            mask = np.where(diff <= 1, disparity, 0).astype(disparity.dtype)
            plt.imsave(os.path.join(cost_volume_compare_dir, source_left_filename), mask, cmap='jet')

def save_entropy(data_batch, dir_name):
    # 저장할 디렉터리 생성
    entropy_dir = os.path.join(dir_name, 'entropy')
    os.makedirs(entropy_dir, exist_ok=True)

    for i in range(data_batch['src_pred_disp'][0].shape[0]):
        shape_map = data_batch['src_shape_map'][i]

        # 그룹 평균 계산
        # shape_map_avg = shape_map.mean(dim=0, keepdim=True)

        # 해상도 확대 (scale_factor=4) 및 차원 축소
        shape_map_resized = F.interpolate(shape_map.float(), scale_factor=4, mode="nearest")
        shape_map_resized = shape_map_resized.squeeze(0).squeeze(0).cpu().numpy()
        
        # map_min = shape_map_resized.min()
        # map_max = shape_map_resized.max()
        # shape_map_norm = (shape_map_resized - map_min) / (map_max - map_min + 1e-8)

        # # 차이를 더 강조하기 위해 추가 스케일링
        # alpha = 2.0  # 예: 2배 확대
        # shape_map_scaled = shape_map_norm * alpha
        # 스케일링 이후 1을 초과한 값은 1로 클리핑
        # shape_map_scaled = torch.clamp(shape_map_scaled, 0, 1)
        # 저장할 파일 이름 설정
        source_left_filename = data_batch['source_left_filename'][i].split('/')[-1]
        save_path = os.path.join(entropy_dir, source_left_filename)

        # 컬러바 포함 이미지 저장 (컬러바 크기 조절)
        plt.figure(figsize=(12, 8))
        img = plt.imshow(shape_map_resized, cmap='jet')
        

        
        # 컬러바 추가 (크기 조절)
        cbar = plt.colorbar(img, fraction=0.015, pad=0.04)
        cbar.ax.tick_params(labelsize=8)  # 컬러바 숫자 크기 조절

        plt.axis('off')  # 축을 없애서 깔끔하게 저장
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
        plt.close()



        # plt.imsave(os.path.join(entropy_dir, source_left_filename), shape_map_resized.cpu().numpy(), cmap='jet')

def save_metrics(metrics, dir_name):
    total_epe = 0
    total_d1 = 0
    total_thres1 = 0
    total_thres2 = 0
    total_thres3 = 0
    count = 0

    for disp_metrics in metrics.values():
        if all(key in disp_metrics for key in ['EPE', 'D1', 'Thres1', 'Thres2', 'Thres3']):
            total_epe += disp_metrics['EPE'][0]
            total_d1 += disp_metrics['D1'][0]
            total_thres1 += disp_metrics['Thres1'][0]
            total_thres2 += disp_metrics['Thres2'][0]
            total_thres3 += disp_metrics['Thres3'][0]
            count += 1

    averages = {
        'average_metric' : {
            'EPE' : total_epe / count if count > 0 else 0,
            'D1' : total_d1 / count if count > 0 else 0,
            'Thres1' : total_thres1 / count if count > 0 else 0,
            'Thres2' : total_thres2 / count if count > 0 else 0,
            'Thres3' : total_thres3 / count if count > 0 else 0
        }
    }

    metrics.update(averages)
    metrics_dir = os.path.join(dir_name, 'metrics')
    os.makedirs(metrics_dir, exist_ok=True)
    with open(os.path.join(metrics_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)