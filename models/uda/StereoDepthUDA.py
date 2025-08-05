import math
import numpy as np
import torch
import random
import torch.nn as nn
from copy import deepcopy
import torch.nn.functional as F
from models.losses.loss import get_loss
from models.estimator import __models__
from models.decoder.mono import MonoDepthDecoder
from models.uda.decorator import StereoDepthUDAInference
from models.uda.label_generator import RobustDisparityGenerator

from models.uda.utils import *
from models.losses.loss import *
# from models.losses.photometric import photometric_loss, photometric_loss_low, photometric_loss_half
import time
from models.losses.monoloss import MonoDepthLoss
from models.losses.photometric import *
from models.tools.prior_setting import *
### if student => model = 's'
### if teacher => model = 't'


def multi_scale_consistency_filter(data_batch, scales=[1, 2, 4]):
    """
    여러 스케일에서 consistency 검증
    """
    disp_pred = data_batch['tgt_pred_disp_s'][0]
    img_left = data_batch['tgt_left'] 
    img_right = data_batch['tgt_right']
    consistent_masks = []
    
    for scale in scales:
        # 다운샘플링
        h, w = img_left.shape[-2:]
        new_h, new_w = h // scale, w // scale
        
        img_l_down = F.interpolate(img_left, (new_h, new_w), mode='bilinear')
        img_r_down = F.interpolate(img_right, (new_h, new_w), mode='bilinear')
        disp_down = F.interpolate(disp_pred.unsqueeze(1), (new_h, new_w), mode='bilinear') / scale
        
        # Photometric error 계산
        warped = warp_image(img_r_down, disp_down.squeeze(1), 'R->L')
        photo_error = (img_l_down - warped).abs().mean(1, keepdim=True)
        
        # 다시 원본 크기로 업샘플링
        photo_error_up = F.interpolate(photo_error, (h, w), mode='bilinear')
        consistent_masks.append(photo_error_up < 0.1)
    
    # 모든 스케일에서 consistent한 영역만 선택
    final_mask = torch.stack(consistent_masks).all(dim=0)
    return final_mask


class StereoDepthUDA(StereoDepthUDAInference):
    def __init__(self, cfg, student_optimizer=0.0, teacher_optimizer=0.0):
        super().__init__(cfg)
        self.cfg = cfg
        self.student_optimizer = student_optimizer
        self.teacher_optimizer = teacher_optimizer
        self.depth_loss = MonoDepthLoss()
        self.jino_loss = JINOLoss()
        self.entropy_threshold = 2.5
        self.generator = RobustDisparityGenerator()


    def update_ema(self, iter, alpha=0.99):
        alpha_teacher = min(1 - 1 / (iter + 1), alpha)
        for ema_param, param in zip(self.teacher_model.parameters(), self.student_model.parameters()):
            if not param.data.shape:  # scalar tensor
                ema_param.data = \
                    alpha_teacher * ema_param.data + \
                    (1 - alpha_teacher) * param.data
            else:
                ema_param.data[:] = \
                    alpha_teacher * ema_param[:].data[:] + \
                    (1 - alpha_teacher) * param[:].data[:]


    def init_ema(self):
        for ema_param, param in zip(self.teacher_model.parameters(), self.student_model.parameters()):
            ema_param.data.copy_(param.data)
        self.ema_initialized = True


    def student_state_dict(self):
        return self.student_model.state_dict()
    

    def teacher_state_dict(self):
        return self.teacher_model.state_dict()
    
    
    def val_step(self, data_batch):
        pass
    
    def freeze_specific_modules(self):
        modules_to_freeze = [
            self.student_model.feature,
            # self.student_model.feature_up,
            # self.student_model.stem_2,
            # self.student_model.stem_4,
            # self.student_model.conv,  # self.conf가 실제로는 self.conv인 것 같습니다
            # self.student_model.desc
            
            self.teacher_model.feature,
        ]
        
        for module in modules_to_freeze:
            for param in module.parameters():
                param.requires_grad = False

    def unfreeze_specific_modules(self):
        modules_to_unfreeze = [
            self.student_model.feature,
            # self.student_model.feature_up,
            # self.student_model.stem_2,
            # self.student_model.stem_4,
            # self.student_model.conv,
            # self.student_model.desc
            
            self.teacher_model.feature,
        ]
        
        for module in modules_to_unfreeze:
            for param in module.parameters():
                param.requires_grad = True


    "args : optimizer, data_batchhhhh"
    def train_step(self, data_batch, optimizer, iter, epoch, temperature=1.0):
        
        optimizer.zero_grad()
        log_vars = self.forward_train(data_batch, epoch, temperature)
        optimizer.step()
        self.update_ema(iter, alpha=0.99)
        
        return log_vars
    
    
    def get_loss_weights(self, epoch, total_epochs=200):
        """
        Epoch에 따른 동적 loss weight 조정
        """
        # Phase 1: Warm-up (0-50 epochs)
        if epoch < 50:
            return {
                'supervised': 1.0,      # Source supervision 강화
                'pseudo': 0.1,          # Pseudo-label 점진적 증가
                'jino': 0.05,           # Feature alignment 시작
                'photometric': 0.0,     # Self-supervision 비활성화
                'confidence': 0.0       # Confidence loss 비활성화
            }
        
        # Phase 2: Adaptation (50-150 epochs)
        elif epoch < 150:
            progress = (epoch - 50) / 100.0
            return {
                'supervised': 0.8, #max(0.3, 1.0 - 0.7 * progress),  # 점진적 감소
                'pseudo': 0.2,#min(0.8, 0.1 + 0.7 * progress),      # 점진적 증가
                'jino': 0.2,#min(0.2, 0.05 + 0.15 * progress),      # Feature alignment 강화
                'photometric': min(0.2, 0.3 * progress),       # Self-supervision 추가
                'confidence': min(0.1, 0.1 * progress)         # Confidence regularization
            }
        
        # Phase 3: Fine-tuning (150+ epochs)
        else:
            return {
                'supervised': 0.2,      # 최소한의 source supervision
                'pseudo': 0.6,          # 주요 supervision
                'jino': 0.35,           # Feature alignment 유지
                'photometric': 0.4,     # Self-supervision 강화
                'confidence': 0.05      # Confidence regularization
            }


    "back propagation"
    "forward propagation"
    def forward_train(self, data_batch, epoch, temperature=0.5):
        
        self.freeze_specific_modules()
        self.student_model.freeze_original_network()
        

        src_pred, map, features = self.student_forward(data_batch['src_left'], data_batch['src_right'])
        data_batch['src_pred_disp_s'] = src_pred
        data_batch['src_confidence_map_s'] = map[0]
        data_batch['src_corr_volume_s_1'] = map[1]
        data_batch['src_mask_pred_s'] = map[2]
        data_batch['src_corr_volume_s_2'] = map[3]
        data_batch['features_s'] = features[0]
        data_batch['attn_weights_s'] = features[1]
        data_batch['cost_s'] = features[2]
        
        
        
        # data_batch['src`_attn_loss_s'] = features[2]
        # data_batch['pos_encodings_s'] = features[2]

        tgt_pred, map, features = self.student_forward(data_batch['tgt_left_strong_aug'], data_batch['tgt_right_strong_aug'])  
        data_batch['tgt_pred_disp_s'] = tgt_pred
        data_batch['tgt_disp_diff'] = map[0]
        data_batch['tgt_corr_volume_s_1'] = map[1]
        data_batch['tgt_mask_pred_s'] = map[2]
        data_batch['tgt_corr_volume_s_2'] = map[3]
        data_batch['features_s'] = features[0]
        data_batch['tgt_attn_weights_s'] = features[1]
        data_batch['tgt_left_feature_s_aug'] = features[3]
        data_batch['tgt_right_feature_s_aug'] = features[4]
        

        tgt_pred, _, _ = self.student_forward(data_batch['tgt_left'], data_batch['tgt_right'])  
        data_batch['tgt_pred_disp_s_for_loss'] = tgt_pred[0]

        
        tgt_pred, map, features = self.student_forward(data_batch['tgt_right'], data_batch['tgt_left'], mode='right')  
        data_batch['tgt_pred_disp_s_reverse'] = tgt_pred

        with torch.no_grad():
            pseudo_disp, map, features = self.teacher_forward(
                data_batch['tgt_left'], data_batch['tgt_right'])
            data_batch['pseudo_disp'] = pseudo_disp
            data_batch['pseudo_disp_for_loss'] = pseudo_disp[0]
            data_batch['confidence_map'] = map[0]
            data_batch['tgt_corr_volume_t_1'] = map[1]
            data_batch['tgt_mask_pred_t'] = map[2]
            data_batch['tgt_corr_volume_t_2'] = map[3]
            data_batch['features_t'] = features[0]
            data_batch['tgt_attn_weights_t'] = features[1]
            data_batch['tgt_left_feature_t'] = features[3]
            data_batch['tgt_right_feature_t'] = features[4]
            
            for i in range(0, 9):
                pseudo_disp, map, features = self.teacher_forward(
                    data_batch[f'tgt_left_random_{i+1}'], data_batch[f'tgt_right_random_{i+1}'])
                data_batch[f'pseudo_disp_random_{i+1}'] = pseudo_disp

        supervised_loss = calc_supervised_train_loss(data_batch, model='s', epoch=epoch)
        calc_entropy(data_batch, temperature=temperature, threshold=self.entropy_threshold)
        calc_confidence_entropy(data_batch,threshold=10, k=12, temperature=0.5)
        compute_photometric_error(data_batch, threshold=0.010)

        # directional_loss = calc_directional_loss(data_batch)


        diff_mask = multi_scale_consistency_filter(data_batch)
        data_batch['tgt_refined_pred_disp_t'], diff_mask = refine_disparity(data_batch, diff_mask, threshold=3.0)
        
        avg_disparity, _, _ = self.generator.generate_robust_disparity(data_batch)
        data_batch['avg_pseudo_disp'] = avg_disparity.unsqueeze(0)
        
        pseudo_loss, true_ratio = calc_pseudo_loss(data_batch, diff_mask, threshold=0.2, model='s')
        if torch.isnan(pseudo_loss).any():
            pseudo_loss = torch.tensor(0.0, device=data_batch['tgt_pred_disp_s'][0].device)

        consist_photo_loss = consistency_photometric_loss(data_batch)
        # confidence_loss = calc_entropy_loss(data_batch)

        lora_loss = calc_adaptor_loss(data_batch, T=2.0)
        band_kl_loss = calc_band_kl_loss(data_batch)


        patch_size = random.randint(1, 8)

        jino_loss = self.jino_loss(data_batch["tgt_left_feature_s_aug"], data_batch["tgt_left_feature_t"], patch_size) \
         + self.jino_loss(data_batch["tgt_right_feature_s_aug"], data_batch["tgt_right_feature_t"], patch_size)

        
        weights = self.get_loss_weights(epoch, total_epochs=200)
        

        # total_loss = (weights['supervised'] * supervised_loss + 
        #           weights['pseudo'] * pseudo_loss + 
        #           weights['jino'] * jino_loss + 
        #           weights['photometric'] * consist_photo_loss["loss_total"] + 
        #           weights['confidence'] * confidence_loss) 
        # total_loss = 0.5 * supervised_loss + 1.0 * directional_loss # + 0.0 * pseudo_loss + 0.5 * jino_loss 
        # total_loss = consist_photo_loss['loss_total'] 
        # total_loss = 0.2 * supervised_loss + 1.0 * pseudo_loss + 1.0 * lora_loss
        total_loss = 0.2*supervised_loss + 1.0 * pseudo_loss + 0.5*lora_loss #+ 0.2*band_kl_loss
        
        
        ## pred, gt, mask, weights
        weight = [1.0]
        gt_tgt_disp = data_batch['tgt_disparity']
        mask = (gt_tgt_disp > 0) & (gt_tgt_disp < 256)
        student_loss = get_loss(data_batch['tgt_pred_disp_s_for_loss'], gt_tgt_disp, mask, weight)
        teacher_loss = get_loss(data_batch['pseudo_disp_for_loss'], gt_tgt_disp, mask, weight)

        log_vars = {
            'loss': total_loss.item(),
            'supervised_loss': supervised_loss.item(),
            'unsupervised_loss': student_loss.item(),
            'true_ratio': 0.0,
            'reconstruction_loss': 0.0,
            'depth_loss': 0.0,
            'entropy_loss': lora_loss.item(),
            'consist_loss' : 0.0, #consist_photo_loss["loss_total"].item(),
            'target_valid_loss_student' : student_loss.item(),
            'target_valid_loss_teacher' : teacher_loss.item(),
        }

        total_loss.backward()

        return log_vars

    # automatically make model's training member variable False
    @torch.no_grad()
    def forward_test(self, data_batch, epoch):
        src_pred, map, features = self.student_forward(data_batch['src_left'], data_batch['src_right'])
        data_batch['src_pred_disp_s'] = src_pred
        data_batch['src_confidence_map_s'] = map[0]
        data_batch['src_corr_volume_s_1'] = map[1]
        data_batch['src_mask_pred_s'] = map[2]
        data_batch['src_corr_volume_s_2'] = map[3]
        data_batch['features_s'] = features[0]
        data_batch['attn_weights_s'] = features[1]
        # data_batch['src_attn_loss_s'] = features[2]
        # data_batch['pos_encodings_s'] = features[2]

        tgt_pred, map, features = self.student_forward(data_batch['tgt_left'], data_batch['tgt_right'])  
        data_batch['tgt_pred_disp_s'] = tgt_pred
        data_batch['tgt_disp_diff'] = map[0]
        data_batch['tgt_corr_volume_s_1'] = map[1]
        data_batch['tgt_mask_pred_s'] = map[2]
        data_batch['tgt_corr_volume_s_2'] = map[3]
        data_batch['features_s'] = features[0]
        data_batch['attn_weights_s'] = features[1]
        data_batch['cost_s'] = features[2]

        tgt_pred, _, _ = self.student_forward(data_batch['tgt_left'], data_batch['tgt_right'])  
        data_batch['tgt_pred_disp_s_for_loss'] = tgt_pred[0]
        
        tgt_pred, map, features = self.student_forward(data_batch['tgt_right'], data_batch['tgt_left'], mode='right')  
        data_batch['tgt_pred_disp_s_reverse'] = tgt_pred


        with torch.no_grad():
            pseudo_disp, map, features = self.teacher_forward(
                data_batch['tgt_left'], data_batch['tgt_right'])
            data_batch['pseudo_disp'] = pseudo_disp
            data_batch['confidence_map'] = map[0]
            data_batch['tgt_corr_volume_t_1'] = map[1]
            data_batch['tgt_mask_pred_t'] = map[2]
            data_batch['tgt_corr_volume_t_2'] = map[3]
            data_batch['features_t'] = features[0]
            data_batch['attn_weights_t'] = features[1]
            data_batch['cost_t'] = features[2]
            
            for i in range(0, 9):
                pseudo_disp, map, features = self.teacher_forward(
                    data_batch[f'tgt_left_random_{i+1}'], data_batch[f'tgt_right_random_{i+1}'])
                data_batch[f'pseudo_disp_random_{i+1}'] = pseudo_disp
                
                
            # data_batch['tgt_attn_loss_t'] = features[2]
            # data_batch['pos_encodings_t'] = features[2]

        # data_batch['depth_map_s'] = self.decode_forward(data_batch['features_s'])
        
        supervised_loss = calc_supervised_train_loss(data_batch, model='s', epoch=epoch)
        # calc_entropy(data_batch, threshold=self.entropy_threshold)
        calc_entropy(data_batch, threshold=2.487)
        data_batch['tgt_refined_pred_disp_t'], diff_mask = refine_disparity(data_batch, threshold=20.0)
        calc_confidence_entropy(data_batch,threshold=1.3, k=12, temperature=0.2)
        compute_photometric_error(data_batch, threshold=0.03)
        
        avg_disparity, _, _ = self.generator.generate_robust_disparity(data_batch)
        data_batch['avg_pseudo_disp'] = avg_disparity.unsqueeze(0)
    
        # lora_loss = calc_adaptor_loss(data_batch, T=2.0)

        # confidence_loss = calc_entropy_loss(data_batch)
        consist_photo_loss = consistency_photometric_loss(data_batch)
        pseudo_loss, true_ratio = calc_pseudo_loss(data_batch, diff_mask, threshold=0.2, model='s')
        
        
        weight = [1.0]
        gt_tgt_disp = data_batch['tgt_disparity']
        mask = (gt_tgt_disp > 0) & (gt_tgt_disp < 256)
        student_loss = get_loss(data_batch['tgt_pred_disp_s_for_loss'], gt_tgt_disp, mask, weight)
        
        # total_loss = 0.1 * supervised_loss #+ 0.5 * pseudo_loss + 0.1 * entropy_loss
        total_loss = 1.0 * supervised_loss 
        log_vars = {
            'loss': total_loss.item(),
            'supervised_loss': supervised_loss.item(),
            'unsupervised_loss': student_loss.item(),
            'true_ratio': 0.0,
            'reconstruction_loss': 0.0,
            'depth_loss': 0.0,
            'entropy_loss': 0.0
        }
        return log_vars