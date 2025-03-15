import os
import cv2
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from skimage import exposure

class Logger:
    
    def __init__(self, save_dir):
        self.save_dir = save_dir
        self._setup_directories()


    def _setup_directories(self):
        self.att_dir = os.path.join(self.save_dir, 'att')
        self.gt_dir = os.path.join(self.save_dir, 'save_gt') 
        self.disp_dir_src = os.path.join(self.save_dir, 'disp', 'src')
        self.disp_dir_tgt = os.path.join(self.save_dir, 'disp', 'tgt')
        self.entropy_dir = os.path.join(self.save_dir, 'entropy')
        self.metrics_dir = os.path.join(self.save_dir, 'metrics')

        os.makedirs(self.att_dir, exist_ok=True)
        os.makedirs(self.gt_dir, exist_ok=True)
        os.makedirs(self.disp_dir_src, exist_ok=True)
        os.makedirs(self.disp_dir_tgt, exist_ok=True)
        os.makedirs(self.entropy_dir, exist_ok=True)
        os.makedirs(self.metrics_dir, exist_ok=True)



    def _save_image(self, data, filename, directory, cmap='gray'):
        plt.imsave(os.path.join(directory, filename), data, cmap=cmap)


    def save_att(self, data_batch):
        print(len(data_batch['src_pred_disp']))
        att_prob, _ = data_batch['src_pred_disp'][2].max(dim=0, keepdim=True)
        att_prob = att_prob.squeeze(0).cpu().numpy()
        filename = data_batch['source_left_filename'].split('/')[-1]
        self._save_image(att_prob, filename, self.att_dir)


    def save_gt(self, data_batch):
        ### just for debug and eye check
        if 'src_disparity' in data_batch.keys():
            filename = data_batch['source_left_filename'].split('/')[-1]
            gt_disp = data_batch['src_disparity'].cpu().numpy()
            self._save_image(gt_disp, filename, self.gt_dir, cmap='jet')


    def save_disparity(self, data_batch):
        pred_src = data_batch['src_pred_disp'][0].cpu().numpy()
        src_filename = data_batch['source_left_filename'].split('/')[-1]
        self._save_image(pred_src, src_filename, self.disp_dir_src, cmap='jet')

        pred_tgt = data_batch['tgt_pred_disp'][0].cpu().numpy()
        tgt_filename = data_batch['target_left_filename'].split('/')[-1]
        self._save_image(pred_tgt, tgt_filename, self.disp_dir_tgt, cmap='jet')


    def save_entropy(self, data_batch):
        shape_map = data_batch['tgt_shape_map']
        shape_map_resized = F.interpolate(shape_map.float(), scale_factor=4, mode="nearest")
        shape_map_resized = shape_map_resized.squeeze(0).squeeze(0).cpu().numpy()
        
        filename = data_batch['target_left_filename'].split('/')[-1]
        save_path = os.path.join(self.entropy_dir, filename)

        # Create figure with colorbar
        plt.figure(figsize=(12, 8))
        img = plt.imshow(shape_map_resized, cmap='jet')
        cbar = plt.colorbar(img, fraction=0.015, pad=0.04)
        cbar.ax.tick_params(labelsize=8)
        plt.axis('off')
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
        plt.close()

    def save_metrics(self, metrics):
        totals = {metric: 0 for metric in ['EPE', 'D1', 'Thres1', 'Thres2', 'Thres3']}
        count = 0

        # 모든 이미지에 대해서 다 
        for disp_metrics in metrics.values():
            if all(key in disp_metrics for key in totals.keys()):
                for metric in totals:
                    totals[metric] += disp_metrics[metric][0]
                count += 1

        # 평균
        averages = {
            'average_metric': {
                metric: totals[metric] / count if count > 0 else 0 
                for metric in totals
            }
        }

        # metrics
        metrics.update(averages)
        os.makedirs(self.metrics_dir, exist_ok=True)
        with open(os.path.join(self.metrics_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=4)


    def log(self, data_batch, metrics):
        self.save_entropy(data_batch)
        self.save_gt(data_batch)
        self.save_att(data_batch)
        self.save_disparity(data_batch)
        # self.save_metrics(metrics)