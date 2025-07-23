import os
import cv2
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from skimage import exposure
from .metrics import EPE_metric, D1_metric, Thres_metric, tensor2float
from PIL import Image



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
        self.error_dir = os.path.join(self.save_dir, 'error_map')
        self.cost_dir = os.path.join(self.save_dir, 'cost')
        self.occlusion_dir = os.path.join(self.save_dir, 'occlusion_mask')
        os.makedirs(self.att_dir, exist_ok=True)
        os.makedirs(self.gt_dir_src, exist_ok=True)
        os.makedirs(self.gt_dir_tgt, exist_ok=True)
        os.makedirs(self.disp_dir_src, exist_ok=True)
        os.makedirs(self.disp_dir_tgt, exist_ok=True)
        os.makedirs(self.entropy_dir, exist_ok=True)
        os.makedirs(self.metrics_dir, exist_ok=True)
        os.makedirs(self.top_one_dir, exist_ok=True)
        os.makedirs(self.depth_dir, exist_ok=True)
        os.makedirs(self.error_dir, exist_ok=True)
        os.makedirs(self.cost_dir, exist_ok=True)
        os.makedirs(self.occlusion_dir, exist_ok=True)

    def _save_image(self, data, filename, directory, cmap='gray'):
        plt.imsave(os.path.join(directory, filename), data, cmap=cmap)
        
        
        
    def save_cost_volume_max_prob(self, cost, filename, save_dir):
        """
        Cost volume을 softmax로 확률 변환 후 최대 확률 값 시각화
        
        Args:
            cost: torch.Tensor [1,1,24,96,312] - cost volume
            filename: str - 저장할 파일명
            save_dir: str - 저장할 디렉토리
        """
        import torch.nn.functional as F
        
        # Cost를 확률로 변환
        cost_squeezed = cost.squeeze(1)  # [1,24,96,312]
        prob = F.softmax(cost_squeezed, dim=1)  # disparity 차원에서 softmax
        
        # GPU 텐서를 CPU로 이동
        prob_np = prob.squeeze().detach().cpu().numpy()  # [24,96,312]
        os.makedirs(save_dir, exist_ok=True)
        
        # 1) Disparity별 최대 확률 값
        prob_max = prob_np.max(axis=0)  # [96,312] - 각 픽셀의 최대 확률
        
        plt.figure(figsize=(12, 8))
        img = plt.imshow(prob_max, cmap='jet', vmin=0, vmax=1)
        cbar = plt.colorbar(img, fraction=0.015, pad=0.04)
        cbar.ax.tick_params(labelsize=8)
        plt.title(f'Maximum Probability Values\nRange: [{prob_max.min():.4f}, {prob_max.max():.4f}]')
        plt.xlabel('Width')
        plt.ylabel('Height')
        plt.axis('off')
        
        max_prob_path = os.path.join(save_dir, f"{filename}_max_prob.png")
        plt.savefig(max_prob_path, bbox_inches='tight', pad_inches=0.1, dpi=150)
        plt.close()
        
        
        # 2) 최대 확률을 가지는 disparity index
        max_prob_disp = np.argmax(prob_np, axis=0)  # [96,312]
        
        plt.figure(figsize=(12, 8))
        plt.imshow(max_prob_disp, cmap='viridis')
        plt.colorbar(fraction=0.015, pad=0.04)
        plt.title('Disparity Index with Maximum Probability')
        plt.axis('off')
        
        max_disp_path = os.path.join(save_dir, f"{filename}_max_prob_disp.png")
        plt.savefig(max_disp_path, bbox_inches='tight', pad_inches=0.1)
        plt.close()
        

    def save_att(self, data_batch):
        # att_prob = data_batch['src_confidence_map_s']
        
        att_prob = data_batch['tgt_confidence_map_s'].unsqueeze(1)
        mask = data_batch['valid_disp'] > 0
        # att_prob = data_batch['src_confidence_map_s']
        # att_prob = F.interpolate(att_prob, 
        #                 scale_factor=4, 
        #                 mode='bilinear', 
        #                 align_corners=False)
        att_prob = att_prob.squeeze().cpu().numpy()
        filename = data_batch['tgt_left_filename'].split('/')[-1]
        
        plt.figure(figsize=(12, 8))
        img = plt.imshow(att_prob, cmap='jet')
        cbar = plt.colorbar(img, fraction=0.015, pad=0.04)
        cbar.ax.tick_params(labelsize=8)
        plt.axis('off')
        plt.savefig(os.path.join(self.att_dir, filename), bbox_inches='tight', pad_inches=0.1)
        plt.close()

    def save_depth_map(self, data_batch):
        depth_map = data_batch['valid_disp'].squeeze().cpu().numpy()
        filename = data_batch['src_left_filename'].split('/')[-1]
        plt.figure(figsize=(12, 8))
        img = plt.imshow(depth_map, cmap='jet', vmin=0, vmax=192)
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
        # recon_loss = log_vars['reconstruction_loss']
        # plt.text(0.98, 0.98, f'Recon Loss: {recon_loss:.4f}', 
        #         horizontalalignment='right',
        #         verticalalignment='top',
        #         transform=plt.gca().transAxes,
        #         color='white',
        #         fontsize=10,
        #         bbox=dict(facecolor='black', alpha=0.5))
                
        plt.savefig(os.path.join(self.disp_dir_tgt, tgt_filename), bbox_inches='tight', pad_inches=0.1)
        plt.close()
        
    def save_occlusion_mask_simple(self, data_batch):
        # occlusion mask 가져오기
        occlusion_mask = data_batch['occlusion_mask_l']
        
        # 텐서를 numpy로 변환
        occlusion_mask_np = occlusion_mask.squeeze(0).cpu().numpy()
        
        # 파일명 생성
        filename = data_batch['tgt_left_filename'].split('/')[-1]
        filename_base = filename.split('.')[0]
        filename_occlusion = f"{filename_base}_occlusion.png"
        
        # 저장 경로 설정
        save_path = os.path.join(self.occlusion_dir, filename_occlusion)
        
        # 순수 흑백 이미지로 저장
        plt.figure(figsize=(12, 8))
        plt.imshow(occlusion_mask_np, cmap='gray', vmin=0, vmax=1)
        plt.axis('off')
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()



    def save_entropy(self, data_batch):
        # top_one_map = data_batch['tgt_entropy_map_idx_t_1'] * 4
        # top_one_map_resized = F.interpolate(top_one_map.float(), scale_factor=4, mode="nearest")
        
        top_one_map_resized = data_batch['tgt_refined_pred_disp_t']
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


        entropy_map = data_batch['confidence_entropy_map_s']
        # entropy_map = data_batch['tgt_entropy_map_idx_t_2']
        # entropy_map_resized = F.interpolate(entropy_map.float(), scale_factor=4, mode="bilinear")
        entropy_map_resized = entropy_map.squeeze(0).squeeze(0).cpu().numpy()       
        filename = data_batch['tgt_left_filename'].split('/')[-1]
        save_path = os.path.join(self.entropy_dir, filename)
        plt.figure(figsize=(12, 8))
        img = plt.imshow(entropy_map_resized, cmap='jet')
        cbar = plt.colorbar(img, fraction=0.015, pad=0.04)
        cbar.ax.tick_params(labelsize=8)
        plt.axis('off')
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
        plt.close()



    """
    빨간색 : d1오차 해당 범위
    초록색 : d1오차 이내 범위
    회색 : d1오차 조건에 해당이 안되는 부분
    검정 : gt 없는 부분
    """ 
    def save_error_map(self,
                    data_batch,
                    x1_1: int = None,
                    y1_1: int = None, 
                    x1_2: int = None,
                    y1_2: int = None,
                    abs_thresh: float = 1.1,
                    rel_thresh: float = 0.05,
                    dilation: int = 1,
                    bins: int = 50):
        x1_1 = data_batch['tgt_random_coord_1'][0]
        y1_1 = data_batch['tgt_random_coord_1'][1]
        x1_2 = data_batch['tgt_random_coord_2'][0]
        y1_2 = data_batch['tgt_random_coord_2'][1]
        # Helper function to compute error map for any prediction
        def compute_error_data(gt_crop, pred_crop):
            valid = (gt_crop > 0) & (pred_crop > 0)
            abs_err = (gt_crop - pred_crop).abs()
            rel_err = abs_err / gt_crop.clamp(min=1e-6)
            
            bad = valid & (abs_err >= abs_thresh) & (rel_err >= rel_thresh)
            good = valid & (abs_err <= abs_thresh) & (rel_err <= rel_thresh)
            
            bad_np = bad.numpy().astype(np.uint8)
            good_np = good.numpy().astype(np.uint8)
            
            if dilation > 1:
                kernel = np.ones((dilation, dilation), np.uint8)
                bad_np = cv2.dilate(bad_np, kernel)
                
            return valid, bad_np, good_np, bad
        
        # Helper function to create error map image
        def create_error_image(valid, bad_np, good_np, H, W):
            img = np.zeros((H, W, 3), dtype=np.uint8)
            img[:, :, :] = (255, 0, 0)  # Default red
            img[~valid.numpy()] = (0, 0, 0)  # Invalid pixels black
            img[bad_np > 0] = (255, 0, 0)    # Bad pixels red
            img[good_np > 0] = (0, 255, 0)   # Good pixels green
            return img

        # Load main data
        gt = data_batch['tgt_disparity'].squeeze().detach().cpu()
        pred = data_batch['pseudo_disp'][0].squeeze().detach().cpu()
        
        # Process main prediction
        valid_main, bad_np_main, good_np_main, bad_main = compute_error_data(gt, pred)
        H, W = bad_np_main.shape
        img_main = create_error_image(valid_main, bad_np_main, good_np_main, H, W)
        
        # Compute histogram for main prediction
        bin_centers_main, counts_total_main, counts_bad_main = self.compute_gt_and_bad_counts(
            gt.numpy(), bad_main.numpy(), bins
        )
        
        # Check if random crops exist and coordinates are provided
        has_random_crops = (
            'pseudo_disp_random_1' in data_batch and 
            'pseudo_disp_random_2' in data_batch and
            all(coord is not None for coord in [x1_1, y1_1, x1_2, y1_2])
        )
        
        filename = os.path.basename(data_batch['tgt_left_filename'].split('/')[-1])
        save_path = os.path.join(self.error_dir, filename)
        
        if has_random_crops:
            # Process random crop 1
            pred_r1 = data_batch['pseudo_disp_random_1'][0].squeeze().detach().cpu()
            crop_h, crop_w = pred_r1.shape
            gt_crop1 = gt[y1_1:y1_1+crop_h, x1_1:x1_1+crop_w]
            valid_r1, bad_np_r1, good_np_r1, bad_r1 = compute_error_data(gt_crop1, pred_r1)
            img_r1 = create_error_image(valid_r1, bad_np_r1, good_np_r1, crop_h, crop_w)
            
            bin_centers_r1, counts_total_r1, counts_bad_r1 = self.compute_gt_and_bad_counts(
                gt_crop1.numpy(), bad_r1.numpy(), bins
            )
            
            # Process random crop 2
            pred_r2 = data_batch['pseudo_disp_random_2'][0].squeeze().detach().cpu()
            gt_crop2 = gt[y1_2:y1_2+crop_h, x1_2:x1_2+crop_w]
            
            valid_r2, bad_np_r2, good_np_r2, bad_r2 = compute_error_data(gt_crop2, pred_r2)
            img_r2 = create_error_image(valid_r2, bad_np_r2, good_np_r2, crop_h, crop_w)
            
            bin_centers_r2, counts_total_r2, counts_bad_r2 = self.compute_gt_and_bad_counts(
                gt_crop2.numpy(), bad_r2.numpy(), bins
            )
            
            # Plot all three together
            self.plot_error_map_with_count_histogram_multi(
                [img_main, img_r1, img_r2],
                [bin_centers_main, bin_centers_r1, bin_centers_r2],
                [counts_total_main, counts_total_r1, counts_total_r2],
                [counts_bad_main, counts_bad_r1, counts_bad_r2],
                ['Full Prediction', f'Random Crop 1 ({x1_1},{y1_1})', f'Random Crop 2 ({x1_2},{y1_2})'],
                save_path
            )
        else:
            # Plot only main prediction
            self.plot_error_map_with_count_histogram(img_main, bin_centers_main, counts_total_main, counts_bad_main, save_path)


    def compute_gt_and_bad_counts(self, gt_np, bad_np, bins=10):
        """GT disparity 값별 총 카운트와 에러 카운트를 계산"""
        # 유효한 GT 값만 필터링 (gt > 0)
        valid_mask = gt_np > 0
        gt_valid = gt_np[valid_mask]
        bad_valid = bad_np[valid_mask]

        if len(gt_valid) == 0:
            return np.array([]), np.array([]), np.array([])

        # bins 경계 생성
        bin_edges = np.linspace(0.0, 80.0, bins + 1)

        # 각 bin별 총 픽셀 수
        counts_total, _ = np.histogram(gt_valid, bins=bin_edges)

        # 각 bin별 에러 픽셀 수
        counts_bad, _ = np.histogram(gt_valid[bad_valid.astype(bool)], bins=bin_edges)

        # bin 중앙값
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        return bin_centers, counts_total, counts_bad


    def plot_error_map_with_count_histogram(self, img, bin_centers, counts_total, counts_bad, save_path):
        """에러맵과 카운트 히스토그램을 함께 플롯"""
        plt.figure(figsize=(16, 6), dpi=100)
        
        # 에러맵 서브플롯
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.axis('off')
        plt.title('Error Map', fontsize=14)
        
        # 히스토그램 서브플롯 (그룹 막대 그래프)
        plt.subplot(1, 2, 2)
        if len(bin_centers) > 0:
            # 막대 너비와 위치 설정
            bar_width = (bin_centers[1] - bin_centers[0]) * 0.35 if len(bin_centers) > 1 else 0.35
            x_pos = np.arange(len(bin_centers))
            
            # 그룹 막대 그래프 생성
            plt.bar(x_pos - bar_width/2, counts_total, bar_width, 
                label='Total GT Count', color='blue', alpha=0.7)
            plt.bar(x_pos + bar_width/2, counts_bad, bar_width, 
                label='Bad Count', color='red', alpha=0.7)
            
            # X축 라벨 설정
            plt.xticks(x_pos, [f'{center:.1f}' for center in bin_centers], rotation=45)
            plt.xlabel('GT Disparity Value', fontsize=12)
            plt.ylabel('Count', fontsize=12)
            plt.title('GT Count and Bad Count by Disparity Bin', fontsize=14)
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # 막대 위에 수치 표시
            for i, (total, bad) in enumerate(zip(counts_total, counts_bad)):
                if total > 0:
                    plt.text(i - bar_width/2, total + max(counts_total) * 0.01, 
                            str(total), ha='center', va='bottom', fontsize=9)
                if bad > 0:
                    plt.text(i + bar_width/2, bad + max(counts_total) * 0.01, 
                            str(bad), ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
        plt.close()


    def plot_error_map_with_count_histogram_multi(self, imgs, bin_centers_list, counts_total_list, counts_bad_list, titles, save_path):
        """Multiple error maps and histograms side by side"""
        n_plots = len(imgs)
        plt.figure(figsize=(8 * n_plots, 12), dpi=100)
        
        for i in range(n_plots):
            # Error map
            plt.subplot(2, n_plots, i + 1)
            plt.imshow(imgs[i])
            plt.axis('off')
            plt.title(f'{titles[i]} - Error Map', fontsize=14)
            
            # Histogram
            plt.subplot(2, n_plots, i + 1 + n_plots)
            bin_centers = bin_centers_list[i]
            counts_total = counts_total_list[i]
            counts_bad = counts_bad_list[i]
            
            if len(bin_centers) > 0:
                bar_width = (bin_centers[1] - bin_centers[0]) * 0.35 if len(bin_centers) > 1 else 0.35
                x_pos = np.arange(len(bin_centers))
                
                plt.bar(x_pos - bar_width/2, counts_total, bar_width, 
                    label='Total GT Count', color='blue', alpha=0.7)
                plt.bar(x_pos + bar_width/2, counts_bad, bar_width, 
                    label='Bad Count', color='red', alpha=0.7)
                
                plt.xticks(x_pos, [f'{center:.1f}' for center in bin_centers], rotation=45)
                plt.xlabel('GT Disparity Value', fontsize=12)
                plt.ylabel('Count', fontsize=12)
                plt.title(f'{titles[i]} - Histogram', fontsize=14)
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                # Add count labels on bars
                for j, (total, bad) in enumerate(zip(counts_total, counts_bad)):
                    if total > 0:
                        plt.text(j - bar_width/2, total + max(counts_total) * 0.01, 
                                str(total), ha='center', va='bottom', fontsize=9)
                    if bad > 0:
                        plt.text(j + bar_width/2, bad + max(counts_total) * 0.01, 
                                str(bad), ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
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
        self.save_depth_map(data_batch)
        self.save_error_map(data_batch)
        self.save_cost_volume_max_prob(data_batch['cost_t'], data_batch['tgt_left_filename'].split('/')[-1], self.cost_dir)
        self.save_occlusion_mask_simple(data_batch)