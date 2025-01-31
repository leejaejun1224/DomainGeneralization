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

    def update_ema(self, alpha=0.99):
        for ema_param, param in zip(self.ema_model.parameters(), self.model.parameters()):
            ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)

    def init_ema(self):
        for ema_param, param in zip(self.ema_model.parameters(), self.model.parameters()):
            ema_param.data.copy_(param.data)    
        self.ema_initialized = True


