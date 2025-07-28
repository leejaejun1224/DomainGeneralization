import math
import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
from models.estimator.Fast_ACV import Fast_ACVNet
from models.losses.loss import get_loss
from models.decoder.mono import MonoDepthDecoder
from models.decoder.monodpt import DPTDepthDecoder
from models.estimator import __models__
from models.head.head import GeometricHead


class StereoDepthUDAInference(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        # student model
        self.student_model = __models__[cfg['model']](maxdisp=cfg['maxdisp'], 
                                att_weights_only=cfg['att_weights_only'], enable_lora=cfg['student_lora'])

        # ema teacher model
        self.teacher_model = __models__[cfg['model']](maxdisp=cfg['maxdisp'],
                                    att_weights_only=cfg['att_weights_only'], enable_lora=cfg['teacher_lora'])
        
        # self.decoder = MonoDepthDecoder(max_disp=60)
        # flag for initializing EMA weights
        self.ema_initialized = False
        self.cfg = cfg
        # self.train = True
        self.set_model()
    
    def set_model(self):
        if self.cfg['uda']['train_source_only']:
            for param in self.teacher_model.parameters():
                param.requires_grad = False
            self.teacher_model.eval()

    def student_forward(self, left, right, mode='left'):
        output, map, features = self.student_model(left, right, mode)
        return output, map, features
    

    def teacher_forward(self, left, right, mode='left'):
        if self.cfg['uda']['train_source_only']:
            with torch.no_grad():
                output, map, features = self.teacher_model(left, right, mode)
        else:
            output, map, features = self.teacher_model(left, right, mode)
        
        return output, map, features
    
        
    def decode_forward(self, features):
        depth_map = self.decoder(features)
        return depth_map



