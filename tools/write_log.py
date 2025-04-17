import os
import cv2
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from skimage import exposure
from .metrics import EPE_metric, D1_metric, Thres_metric, tensor2float




class Logger:
    def __init__(self, save_dir, max_disp=256):
        self.save_dir = save_dir
        self.metrics_dict = {
            'source': {},
            'target': {}
        }
        self.max_disp = max_disp
        self._setup_directories()


    def _setup_directories(self):
        self.att_dir = os.path.join(self.save_dir, 'att')
        self.gt_dir_src = os.path.join(self.save_dir, 'save_gt', 'src') 
        self.gt_dir_tgt = os.path.join(self.save_dir, 'save_gt', 'tgt') 
        self.disp_dir_src = os.path.join(self.save_dir, 'disp', 'src')
        self.disp_dir_tgt = os.path.join(self.save_dir, 'disp', 'tgt')
        self.entropy_dir = os.path.join(self.save_dir, 'entropy')
        self.top_one_dir = os.path.join(self.save_dir, 'top_one')
        self.metrics_dir = os.path.join(self.save_dir, 'metrics')
        self.depth_dir = os.path.join(self.save_dir, 'depth')
        os.makedirs(self.att_dir, exist_ok=True)
        os.makedirs(self.gt_dir_src, exist_ok=True)
        os.makedirs(self.gt_dir_tgt, exist_ok=True)
        os.makedirs(self.disp_dir_src, exist_ok=True)
        os.makedirs(self.disp_dir_tgt, exist_ok=True)
        os.makedirs(self.entropy_dir, exist_ok=True)
        os.makedirs(self.metrics_dir, exist_ok=True)
        os.makedirs(self.top_one_dir, exist_ok=True)
        os.makedirs(self.depth_dir, exist_ok=True)

    def _save_image(self, data, filename, directory, cmap='gray'):
        plt.imsave(os.path.join(directory, filename), data, cmap=cmap)


    def save_att(self, data_batch):
        att_prob = data_batch['src_confidence_map_s']
        att_prob = F.interpolate(att_prob.unsqueeze(0), 
                        scale_factor=4, 
                        mode='bilinear', 
                        align_corners=False).squeeze(0)
        att_prob = att_prob.squeeze().cpu().numpy()
        filename = data_batch['src_left_filename'].split('/')[-1]
        
        plt.figure(figsize=(12, 8))
        img = plt.imshow(att_prob, cmap='gray', vmin=0.0, vmax=1.0)
        cbar = plt.colorbar(img, fraction=0.015, pad=0.04)
        cbar.ax.tick_params(labelsize=8)
        plt.axis('off')
        plt.savefig(os.path.join(self.att_dir, filename), bbox_inches='tight', pad_inches=0.1)
        plt.close()

    def save_depth_map(self, data_batch):
        depth_map = data_batch['depth_map_s'].squeeze().cpu().numpy()
        filename = data_batch['src_left_filename'].split('/')[-1]
        plt.figure(figsize=(12, 8))
        img = plt.imshow(depth_map, cmap='jet', vmin=0, vmax=1)
        cbar = plt.colorbar(img, fraction=0.015, pad=0.04)
        cbar.ax.tick_params(labelsize=8)
        plt.axis('off')
        plt.savefig(os.path.join(self.depth_dir, filename), bbox_inches='tight', pad_inches=0.1)
        plt.close()

    def save_gt(self, data_batch):
        # Add colorbar with min=0, max=255 for source disparity
        if 'src_disparity' in data_batch.keys():
            filename = data_batch['src_left_filename'].split('/')[-1]
            gt_disp = data_batch['src_disparity'].squeeze().cpu().numpy()
            
            plt.figure(figsize=(12, 8))
            img = plt.imshow(gt_disp, cmap='jet', vmin=0, vmax=192)
            cbar = plt.colorbar(img, fraction=0.015, pad=0.04)
            cbar.ax.tick_params(labelsize=8)
            plt.axis('off')
            plt.savefig(os.path.join(self.gt_dir_src, filename), bbox_inches='tight', pad_inches=0.1)
            plt.close()

        if 'tgt_disparity' in data_batch.keys():
            filename = data_batch['tgt_left_filename'].split('/')[-1]
            gt_disp = data_batch['tgt_disparity'].squeeze().cpu().numpy()
            plt.figure(figsize=(12, 8))
            img = plt.imshow(gt_disp, cmap='jet', vmin=0, vmax=192)
            cbar = plt.colorbar(img, fraction=0.015, pad=0.04)
            cbar.ax.tick_params(labelsize=8)
            plt.axis('off')
            plt.savefig(os.path.join(self.gt_dir_tgt, filename), bbox_inches='tight', pad_inches=0.1)
            plt.close()



    def save_disparity(self, data_batch, log_vars):
        pred_src = data_batch['src_pred_disp_s'][0].squeeze().cpu().numpy()
        src_filename = data_batch['src_left_filename'].split('/')[-1]
        # Create figure with colorbar for source disparity
        plt.figure(figsize=(12, 8))
        img = plt.imshow(pred_src, cmap='jet', vmin=0, vmax=192)
        cbar = plt.colorbar(img, fraction=0.015, pad=0.04)
        cbar.ax.tick_params(labelsize=8)
        plt.axis('off')
        plt.savefig(os.path.join(self.disp_dir_src, src_filename), bbox_inches='tight', pad_inches=0.1)
        plt.close()

        pred_tgt = data_batch['pseudo_disp'][0].squeeze().cpu().numpy()
        tgt_filename = data_batch['tgt_left_filename'].split('/')[-1]
        plt.figure(figsize=(12, 8))
        img = plt.imshow(pred_tgt, cmap='jet', vmin=0, vmax=192)
        cbar = plt.colorbar(img, fraction=0.015, pad=0.04)
        cbar.ax.tick_params(labelsize=8)
        plt.axis('off')
        
        # Add reconstruction loss text in upper right
        recon_loss = log_vars['reconstruction_loss']
        plt.text(0.98, 0.98, f'Recon Loss: {recon_loss:.4f}', 
                horizontalalignment='right',
                verticalalignment='top',
                transform=plt.gca().transAxes,
                color='white',
                fontsize=10,
                bbox=dict(facecolor='black', alpha=0.5))
                
        plt.savefig(os.path.join(self.disp_dir_tgt, tgt_filename), bbox_inches='tight', pad_inches=0.1)
        plt.close()



    def save_entropy(self, data_batch):
        top_one_map = data_batch['tgt_entropy_map_idx_t'] * 4
        top_one_map_resized = F.interpolate(top_one_map.float(), scale_factor=4, mode="nearest")
        top_one_map_resized = top_one_map_resized.squeeze(0).squeeze(0).cpu().numpy()
        filename = data_batch['tgt_left_filename'].split('/')[-1]
        save_path = os.path.join(self.top_one_dir, filename)
        plt.figure(figsize=(12, 8))
        img = plt.imshow(top_one_map_resized, cmap='jet', vmin=0, vmax=192)
        cbar = plt.colorbar(img, fraction=0.015, pad=0.04)
        cbar.ax.tick_params(labelsize=8)
        plt.axis('off')
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
        plt.close()


        entropy_map = data_batch['tgt_entropy_map_t']
        entropy_map_resized = F.interpolate(entropy_map.float(), scale_factor=4, mode="nearest")
        entropy_map_resized = entropy_map_resized.squeeze(0).squeeze(0).cpu().numpy()       
        filename = data_batch['tgt_left_filename'].split('/')[-1]
        save_path = os.path.join(self.entropy_dir, filename)
        plt.figure(figsize=(12, 8))
        img = plt.imshow(entropy_map_resized, cmap='jet')
        cbar = plt.colorbar(img, fraction=0.015, pad=0.04)
        cbar.ax.tick_params(labelsize=8)
        plt.axis('off')
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
        plt.close()

    
    


    def compute_metrics(self, data_batch):
        if 'src_disparity' in data_batch.keys():
            scalar_outputs = {}
            scalar_outputs["EPE"] = [EPE_metric(data_batch['src_pred_disp_s'][0], data_batch['src_disparity'], self.max_disp)]
            scalar_outputs["D1"] = [D1_metric(data_batch['src_pred_disp_s'][0], data_batch['src_disparity'], self.max_disp)]
            scalar_outputs["Thres1"] = [Thres_metric(data_batch['src_pred_disp_s'][0], data_batch['src_disparity'], self.max_disp, 1.0)]
            scalar_outputs["Thres2"] = [Thres_metric(data_batch['src_pred_disp_s'][0], data_batch['src_disparity'], self.max_disp, 2.0)]
            scalar_outputs["Thres3"] = [Thres_metric(data_batch['src_pred_disp_s'][0], data_batch['src_disparity'], self.max_disp, 3.0)]
            self.metrics_dict['source'][data_batch['src_left_filename']] = tensor2float(scalar_outputs)

        if 'tgt_disparity' in data_batch.keys():
            scalar_outputs = {}
            scalar_outputs["EPE"] = [EPE_metric(data_batch['pseudo_disp'][0], data_batch['tgt_disparity'], self.max_disp)]
            scalar_outputs["D1"] = [D1_metric(data_batch['pseudo_disp'][0], data_batch['tgt_disparity'], self.max_disp)]
            scalar_outputs["Thres1"] = [Thres_metric(data_batch['pseudo_disp'][0], data_batch['tgt_disparity'], self.max_disp, 1.0)]
            scalar_outputs["Thres2"] = [Thres_metric(data_batch['pseudo_disp'][0], data_batch['tgt_disparity'], self.max_disp, 2.0)]
            scalar_outputs["Thres3"] = [Thres_metric(data_batch['pseudo_disp'][0], data_batch['tgt_disparity'], self.max_disp, 3.0)]
            self.metrics_dict['target'][data_batch['tgt_left_filename']] = tensor2float(scalar_outputs)


    def save_metrics(self):
        # Calculate averages for both domains
        averages = {}
        for domain in ['source', 'target']:
            if self.metrics_dict[domain]:
                totals = {metric: 0 for metric in ['EPE', 'D1', 'Thres1', 'Thres2', 'Thres3']}
                count = 0

                for disp_metrics in self.metrics_dict[domain].values():
                    if all(key in disp_metrics for key in totals.keys()):
                        for metric in totals:
                            totals[metric] += disp_metrics[metric][0]
                        count += 1

                if count > 0:
                    averages[f'{domain}_average_metric'] = {
                        metric: totals[metric] / count
                        for metric in totals
                    }

        # Create new dict with averages at top
        metrics_with_averages = {}
        metrics_with_averages.update(averages)
        metrics_with_averages.update(self.metrics_dict)

        os.makedirs(self.metrics_dir, exist_ok=True)
        with open(os.path.join(self.metrics_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics_with_averages, f, indent=4)


    def log(self, data_batch, log_vars):
        self.save_entropy(data_batch)
        self.save_gt(data_batch)
        self.save_att(data_batch)
        self.save_disparity(data_batch, log_vars)
        self.compute_metrics(data_batch)
        # self.save_depth_map(data_batch)