import torch
import numpy as np
import torch.nn.functional as F


def calc_prior(data_batch):
    src_prior = data_batch['src_prior']
    tgt_prior = data_batch['tgt_prior']
    
    common_len = min(len(src_prior), len(tgt_prior))
    src_prior = src_prior[:common_len]
    tgt_prior = tgt_prior[:common_len]
    
    eps = 1e-8
    prior_ratio = torch.where(src_prior > eps, 
                             tgt_prior / src_prior, 
                             torch.ones_like(tgt_prior))
    prior_ratio = torch.log(prior_ratio + 1)
    
    return prior_ratio