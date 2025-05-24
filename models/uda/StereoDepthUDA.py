import math
import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
import torch.nn.functional as F
from models.losses.loss import get_loss
from models.estimator import __models__
from models.decoder.mono import MonoDepthDecoder
from models.uda.decorator import StereoDepthUDAInference

from models.uda.utils import *
from models.losses.loss import *
# from models.losses.photometric import photometric_loss, photometric_loss_low, photometric_loss_half
import time
from models.losses.monoloss import MonoDepthLoss
from models.losses.photometric import *
### if student => model = 's'
### if teacher => model = 't'
class StereoDepthUDA(StereoDepthUDAInference):
    def __init__(self, cfg, student_optimizer=0.0, teacher_optimizer=0.0):
        super().__init__(cfg)
        self.cfg = cfg
        self.student_optimizer = student_optimizer
        self.teacher_optimizer = teacher_optimizer
        self.depth_loss = MonoDepthLoss()

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
    

    "args : optimizer, data_batchhhhh"
    def train_step(self, data_batch, optimizer, iter, epoch, temperature=1.0):
        
        optimizer.zero_grad()
        log_vars = self.forward_train(data_batch, epoch, temperature)
        optimizer.step()
        self.update_ema(iter, alpha=0.99)
        
        return log_vars
    

    "back propagation"
    "forward propagation"
    def forward_train(self, data_batch, epoch, temperature=0.5):
        
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

        tgt_pred, map, features = self.student_forward(data_batch['tgt_left'], data_batch['tgt_right'])  
        data_batch['tgt_pred_disp_s'] = tgt_pred
        data_batch['tgt_confidence_map_s'] = map[0]
        data_batch['tgt_corr_volume_s_1'] = map[1]
        data_batch['tgt_mask_pred_s'] = map[2]
        data_batch['tgt_corr_volume_s_2'] = map[3]
        data_batch['features_s'] = features[0]
        data_batch['attn_weights_s'] = features[1]


        tgt_pred, map, features = self.student_forward(data_batch['src_right'], data_batch['src_left'], mode='right')  
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
            # data_batch['tgt_attn_loss_t'] = features[2]
            # data_batch['pos_encodings_t'] = features[2]
        supervised_loss = calc_supervised_train_loss(data_batch, model='s')
        calc_entropy(data_batch, temperature=temperature, threshold=0.00090)
        calc_confidence_entropy(data_batch, k=12, temperature=0.5)
        compute_photometric_error(data_batch, threshold=0.010)

        data_batch['tgt_refined_pred_disp_t'], diff_mask = refine_disparity(data_batch, threshold=1.0)
        pseudo_loss, true_ratio = calc_pseudo_loss(data_batch, diff_mask, threshold=0.2, model='s')

        consist_photo_loss = consistency_photometric_loss(data_batch)
        entropy_loss = calc_entropy_loss(data_batch)

        if epoch < 150:
            mask_loss = 0.0
        total_loss = 1.0 * supervised_loss #+ 1.0 * pseudo_loss + 0.2 * entropy_loss #+ 0.1 * consist_photo_loss['loss_total'] 
        
        # total_loss = consist_photo_loss['loss_total']

        log_vars = {
            'loss': total_loss.item(),
            'supervised_loss': supervised_loss.item(),
            'unsupervised_loss': pseudo_loss.item(),
            'true_ratio': 0.0,
            'reconstruction_loss': 0.0,
            'depth_loss': 0.0,
            'entropy_loss': entropy_loss.item(),
            'consist_loss' : consist_photo_loss['loss_total'].item()
        }

        total_loss.backward()

        return log_vars

    # automatically make model's training member variable False
    @torch.no_grad()
    def forward_test(self, data_batch):
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
        data_batch['tgt_confidence_map_s'] = map[0]
        data_batch['tgt_corr_volume_s_1'] = map[1]
        data_batch['tgt_mask_pred_s'] = map[2]
        data_batch['tgt_corr_volume_s_2'] = map[3]
        data_batch['features_s'] = features[0]
        data_batch['attn_weights_s'] = features[1]
        data_batch['cost_s'] = features[2]

    
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
            # data_batch['tgt_attn_loss_t'] = features[2]
            # data_batch['pos_encodings_t'] = features[2]

        # data_batch['depth_map_s'] = self.decode_forward(data_batch['features_s'])
        

        supervised_loss = calc_supervised_train_loss(data_batch, model='s')
        calc_entropy(data_batch, threshold=0.00089)
        calc_confidence_entropy(data_batch, k=12, temperature=0.5)
        compute_photometric_error(data_batch, threshold=0.011)
        data_batch['tgt_refined_pred_disp_t'], diff_mask = refine_disparity(data_batch, threshold=1.0)

        entropy_loss = calc_entropy_loss(data_batch)
        consist_photo_loss = consistency_photometric_loss(data_batch)
        pseudo_loss, true_ratio = calc_pseudo_loss(data_batch, diff_mask, threshold=0.2, model='s')

        # total_loss = 0.0 * supervised_loss + 1.0 * pseudo_loss + 0.1 * entropy_loss
        total_loss = 1.0 * supervised_loss 
        log_vars = {
            'loss': total_loss.item(),
            'supervised_loss': supervised_loss.item(),
            'unsupervised_loss': pseudo_loss.item(),
            'true_ratio': 0.0,
            'reconstruction_loss': 0.0,
            'depth_loss': 0.0,
            'entropy_loss': entropy_loss.item()
        }
        return log_vars



    def meta_pseudo_training(self, data_batch, optimizer, iter, threshold):
        
        self.student_optimizer.zero_grad()
        self.teacher_optimizer.zero_grad()
        

        teacher_supervised_loss = calc_supervised_train_loss(data_batch, model='t')
        # teacher_unsupervised_loss = calc_pseudo_loss(data_batch, threshold)

        student_loss_previous_labeled = calc_supervised_train_loss(data_batch, model='s')

        ### 그러면 여기서의 loss는 teacher와 student의 soft label이다. 
        student_loss_previous_unlabeled = calc_pseudo_soft_loss(data_batch, threshold, model='t')
        ### 여기 사이에서 backpropagation이 student에게 발생을 해야한다. 
        ### 여기서 lambda는 일단 0.5로 고정을 한 번 해보자.
        student_total_loss = student_loss_previous_labeled + 0.5 * student_loss_previous_unlabeled
        student_total_loss.backward()


        self.student_optimizer.step()
        self.student_optimizer.zero_grad()


        student_loss_updated_labeled = calc_supervised_train_loss(data_batch, model='s')

        ## 이렇게도 쓰고 결국에는 수렴도 한다고는 하는데 일단 나는 아래의 방식을 따를 예정
        # student_update_signal = student_loss_updated_labeled - student_loss_previous_labeled
        student_update_signal = student_loss_previous_labeled - student_loss_updated_labeled

        with torch.no_grad():
            # Student가 새로 업데이트된 상태로 unlabeled predict
            student_tgt_pred, _ = self.student_forward(data_batch['tgt_left'], data_batch['tgt_right'])
            # stereo -> shape (B, H, W)
        teacher_disp, _ = self.teacher_forward(data_batch['tgt_left'], data_batch['tgt_right'])
        
        teacher_loss_mpl = F.l1_loss(teacher_disp, student_tgt_pred, reduction='mean')

        t_loss_mpl = student_update_signal * teacher_loss_mpl
        teacher_total_loss = teacher_supervised_loss + t_loss_mpl
        teacher_total_loss.backward()

        self.teacher_optimizer.step()

        total_loss = student_total_loss + teacher_total_loss
        log_vars = {
            'loss': total_loss.item(),
            'supervised_loss': student_total_loss.item(),
            'unsupervised_loss': teacher_total_loss.item(),
            'true_ratio': 0.0,
            'reconstruction_loss': 0.0
        }

        return log_vars