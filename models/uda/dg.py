import torch
import torch.nn as nn
from copy import deepcopy

from models.estimator import __models__

class DG(nn.Module):
    def __init__(self, cfg):
        super(DG, self).__init__()
        self.estimator = __models__[cfg['uda']['estimator']](cfg)
