import math
import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
from models.estimator.Fast_ACV import Fast_ACVNet
from models.losses.loss import get_loss
from models.decoder.mono import MonoDepthDecoder
from models.estimator import __models__
from models.head.head import GeometricHead


class StereoDepthUDAInference(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        # student model
        self.student_model = __models__[cfg['model']](maxdisp=cfg['maxdisp'], 
                                att_weights_only=cfg['att_weights_only'])

        # ema teacher model
        self.teacher_model = __models__[cfg['model']](maxdisp=cfg['maxdisp'],
                                    att_weights_only=cfg['att_weights_only'])
        self.decoder = MonoDepthDecoder()
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

    def student_forward(self, left, right):
        output, map, features = self.student_model(left, right)
        return output, map, features
    

    def teacher_forward(self, left, right):
        if self.cfg['uda']['train_source_only']:
            with torch.no_grad():
                output, map, features = self.teacher_model(left, right)
        else:
            output, map, features = self.teacher_model(left, right)
        
        return output, map, features
        
    def decode_forward(self, pos_encodings):
        depth_map = self.decoder(pos_encodings)
        return depth_map



