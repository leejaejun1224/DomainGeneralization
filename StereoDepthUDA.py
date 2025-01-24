import math
import numpy as np
import torch
import torch.nn as nn
from tools import get_model
from copy import deepcopy
from .model import build_stereo_model


class StereoDepthUDA(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        ema_cfg = deepcopy(cfg['model'])
        self.ema_model = build_stereo_model(ema_cfg)

    # for train
    # 여기에서 model은 그대로 있고 data_batch만 받아서 student 모델을 학습을 하고 loss를 돌려줌.
    def forward_train(self, data_batch):
        
        if self.init_weight == True:
            self._init_ema_weight()
        
        if self.ema_upate == True:
            self._init_ema_update()


        outputs = dict()
        outputs['loss'] = 0
        outputs['log_vars'] = 0
        return outputs

    # aka teacher network
    def get_model(self):
        return get_model(self.ema_model)



    def ema_forward():
        return
    
    def _update_ema_weight(alpha):
        return 
    
    def _init_ema_weight():
        return 
    
    def _calc_supervised_loss():
        return 
    
    def _calc_pseudo_loss():
        return 
    