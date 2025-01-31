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
        
        # student model
        self.model = Fast_ACVNet(maxdisp=192, att_weights_only=False)
        
        # ema teacher model
        # ema_cfg = deepcopy(cfg['model'])
        self.ema_model = Fast_ACVNet(maxdisp=192, att_weights_only=False)
        
        # flag for initializing EMA weights
        self.ema_initialized = False

    def forward(self, data_batch):
        
        return self.model(data_batch['left'], data_batch['right'])
    

    @torch.no_grad()
    def ema_forward(self, data_batch):
        return self.ema_model(data_batch['left'], data_batch['right'])

    def update_ema(self, alpha=0.99):
        for ema_param, param in zip(self.ema_model.parameters(), self.model.parameters()):
            ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)

    def init_ema(self):
        for ema_param, param in zip(self.ema_model.parameters(), self.model.parameters()):
            ema_param.data.copy_(param.data)    
        self.ema_initialized = True


