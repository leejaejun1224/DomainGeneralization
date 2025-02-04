import math
import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
from model.Fast_ACV import Fast_ACVNet
from model.loss import get_loss


class StereoDepthUDA(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.alpha = cfg['uda']['alpha']
        
        model = cfg['model']['name']

        # student model
        self.model = model(maxdisp=cfg['model']['maxdisp'], 
                                          att_weights_only=cfg['model']['att_weights_only'])

        # ema teacher model
        self.ema_model = model(maxdisp=cfg['model']['maxdisp'],
                                        att_weights_only=cfg['model']['att_weights_only'])
        
        # flag for initializing EMA weights
        self.ema_initialized = False

    def forward(self, left, right):
        output, _ = self.model(left, right)
        return output
    
    @torch.no_grad()
    def ema_forward(self, left, right, return_confidence=True):
        output, confidence_map = self.ema_model(left, right)
        if return_confidence:
            return output[1], confidence_map
        else:
            return output[1]

    def update_ema(self, iter, alpha=0.99):
        alpha_teacher = min(1 - 1 / (iter + 1), self.alpha)
        for ema_param, param in zip(self.ema_model.parameters(), self.model.parameters()):
            if not param.data.shape:  # scalar tensor
                ema_param.data = \
                    alpha_teacher * ema_param.data + \
                    (1 - alpha_teacher) * param.data
            else:
                ema_param.data[:] = \
                    alpha_teacher * ema_param[:].data[:] + \
                    (1 - alpha_teacher) * param[:].data[:]

    def init_ema(self):
        for ema_param, param in zip(self.ema_model.parameters(), self.model.parameters()):
            ema_param.data.copy_(param.data)    
        self.ema_initialized = True


    def ema_state_dict(self):
        return self.ema_model.state_dict()