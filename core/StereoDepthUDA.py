import math
import numpy as np
import torch
import torch.nn as nn



class StereoDepthUDA(nn.Module0):
    def __init__(self, cfg):
        super().__init__()
        
        
    # just for inference or eval
    def forward(self):
        
        
        return 

    # for train
    def forward_train(self, data_batch):
        
        
        
        
        outputs = dict()
        outputs['loss'] = 0
        outputs['log_vars'] = 0
        return outputs
    
    def ema_forward():
        return
    
    def update_ema():
        return 
    
    def init_ema():
        return 
    
    def _calc_supervised_loss():
        return 
    
    def _calc_pseudo_loss():
        return 
    