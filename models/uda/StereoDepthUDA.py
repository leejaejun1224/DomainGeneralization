import math
import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
from models.estimator.Fast_ACV import Fast_ACVNet
from models.losses.loss import get_loss
from models.estimator import __models__
from models.uda.decorator import StereoDepthUDAInference

from models.losses.loss import calc_supervised_train_loss, calc_supervised_val_loss, calc_pseudo_loss



class StereoDepthUDA(StereoDepthUDAInference):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = cfg

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
    
    
    def val_step(self):
        pass
    
    "args : optimizer, data_batchhhhh"
    def train_step(self, data_batch, optimizer, iter):
        
        optimizer.zero_grad()
        log_vars = self.forward_train(data_batch)
        optimizer.step()
        self.update_ema(iter, alpha=0.99)
        
        return log_vars
    

    # automatically make model's training member variable False
    @torch.no_grad()
    def forward_test(self, data_batch):
        
        data_batch['src_pred_disp'] = self.forward(data_batch['src_left'], data_batch['src_right'])
        data_batch['tgt_pred_disp'] = self.forward(data_batch['tgt_left'], data_batch['tgt_right'])
        
        with torch.no_grad():
            pseudo_disp, confidence_map = self.ema_forward(
                data_batch['tgt_left'], data_batch['tgt_right'])
            data_batch['pseudo_disp'] = pseudo_disp
            data_batch['confidence_map'] = confidence_map
    
        supervised_loss = calc_supervised_val_loss(data_batch)
        pseudo_loss = calc_pseudo_loss(data_batch, self.cfg)
        total_loss = supervised_loss + pseudo_loss

        log_vars = {
            'loss': total_loss.item(),
            'supervised_loss': supervised_loss.item(),
            'unsupervised_loss': pseudo_loss.item()
        }
    
        return log_vars
    
    
    "back propagation"
    "forward propagation"
    
    
    def forward_train(self, data_batch):
        
        src_pred = self.forward(data_batch['src_left'], data_batch['src_right'])
        data_batch['src_pred_disp'] = src_pred
        
        tgt_pred = self.forward(data_batch['tgt_left'], data_batch['tgt_right'])  
        data_batch['tgt_pred_disp'] = tgt_pred
        
        
        with torch.no_grad():
            pseudo_disp, confidence_map = self.ema_forward(
                data_batch['tgt_left'], data_batch['tgt_right'])
            data_batch['pseudo_disp'] = pseudo_disp
            data_batch['confidence_map'] = confidence_map

        supervised_loss = calc_supervised_train_loss(data_batch)
        pseudo_loss = calc_pseudo_loss(data_batch, self.cfg)
        total_loss = supervised_loss + pseudo_loss

        log_vars = {
            'loss': total_loss.item(),
            'supervised_loss': supervised_loss.item(),
            'unsupervised_loss': pseudo_loss.item()
        }
        total_loss.backward()
    
        return log_vars