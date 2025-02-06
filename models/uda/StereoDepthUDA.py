import math
import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
from models.estimator.Fast_ACV import Fast_ACVNet
from models.losses.loss import get_loss
from models.estimator import __models__

class StereoDepthUDA(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.alpha = cfg['uda']['alpha']
        
        # student model
        self.student_model = __models__[cfg['model']](maxdisp=cfg['maxdisp'], 
                                att_weights_only=cfg['att_weights_only'])

        # ema teacher model
        self.teacher_model = __models__[cfg['model']](maxdisp=cfg['maxdisp'],
                                    att_weights_only=cfg['att_weights_only'])
        
        # flag for initializing EMA weights
        self.ema_initialized = False


    def forward(self, left, right):
        output, _ = self.student_model(left, right)
        return output
    

    @torch.no_grad()
    def ema_forward(self, left, right, return_confidence=True):
        output, confidence_map = self.teacher_model(left, right)
        if return_confidence:
            return output[1], confidence_map
        else:
            return output[1]


    def update_ema(self, iter, alpha=0.99):
        alpha_teacher = min(1 - 1 / (iter + 1), self.alpha)
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