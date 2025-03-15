import os
import json
import math
import datetime
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
from models.uda import __models__
from torch.utils.data import DataLoader
from datasets import __datasets__
from datasets.dataloader import PrepareDataset
from experiment import prepare_cfg
from tools.metrics import EPE_metric, D1_metric, Thres_metric, tensor2float
from tools.write_log import Logger
from collections import OrderedDict
from itertools import starmap

cudnn.benchmark = True
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'

def setup_args():
    parser = argparse.ArgumentParser(description="StereoDepth Unsupervised Domain Adaptation")
    parser.add_argument('--dataset_config', default='./config/datasets/cityscapes_to_kitti2015.py', help='source domain and target domain name')
    parser.add_argument('--uda_config', default='./config/uda/kit15_cityscapes.py', help='UDA model preparation')
    parser.add_argument('--seed', default=1, metavar='S', help='random seed(default = 1)')
    parser.add_argument('--log_dir', default='./log', help='log directory')
    parser.add_argument('--ckpt', default='', help='checkpoint')
    parser.add_argument('--compute_metrics', default=True, help='compute error')
    parser.add_argument('--save_disp', default=True, help='save disparity')
    parser.add_argument('--save_att', default=True, help='save attention')
    parser.add_argument('--save_heatmap', default=False, help='save heatmap')
    parser.add_argument('--save_entropy', default=True, help='save entropy')
    parser.add_argument('--save_gt', default=True, help='save gt')
    parser.add_argument('--compare_costvolume', default=True, help='compare costvolume')
    return parser.parse_args()

def setup_model(cfg, ckpt_path):
    model = __models__['StereoDepthUDA'](cfg)
    checkpoint = torch.load(ckpt_path)
    model.student_model.load_state_dict(checkpoint['student_state_dict'])
    model.teacher_model.load_state_dict(checkpoint['teacher_state_dict'])
    model.to('cuda:0')
    return model

def process_batch(data_batch):
    for key in data_batch:
        if isinstance(data_batch[key], torch.Tensor):
            data_batch[key] = data_batch[key].cuda()
    return data_batch

def split_batch(data_batch, batch_idx):
    batch = {}
    for key, value in data_batch.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value[batch_idx].detach().unsqueeze(0)
        elif isinstance(value, list):
            if isinstance(value[0], torch.Tensor):
                batch[key] = [v[batch_idx].detach().unsqueeze(0) for v in value]
            else:
                batch[key] = value[batch_idx]
    return batch


def main():
    args = setup_args()
    assert args.ckpt != '', 'checkpoint is required !!'

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    
    log_dir = '/'.join(args.ckpt.split('/')[:-1])

    cfg = prepare_cfg(args, mode='test')
    
    test_dataset = PrepareDataset(
        source_datapath=cfg['dataset']['src_root'],
        target_datapath=cfg['dataset']['tgt_root'],
        sourcefile_list=cfg['dataset']['src_filelist'],
        targetfile_list=cfg['dataset']['tgt_filelist'],
        training=False
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=cfg['test_batch_size'],
        shuffle=False,
        num_workers=cfg['test_num_workers'],
        drop_last=False
    )

    model = setup_model(cfg, args.ckpt)
    model.eval()
    
    metrics_dict = {}
    logger = Logger(log_dir)
    logger._setup_directories()

    with torch.no_grad():
        for _, data_batch in enumerate(test_loader):
            src = True if 'src_disparity' in data_batch.keys() else False
            tgt = True if 'tgt_disparity' in data_batch.keys() else False

            
            data_batch = process_batch(data_batch)
            log_vars = model.forward_test(data_batch)
            for i in range(cfg['test_batch_size']):
                batch = split_batch(data_batch, i)
                logger.log(batch)

    logger.save_metrics()
    return 0

if __name__=="__main__":
    main()