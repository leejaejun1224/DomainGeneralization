import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import StereoDepthUDA


from torch.utils.data import DataLoader
from datasets import __datasets__
from datasets.dataloader import PrepareDataset
from experiment import prepare_cfg

cudnn.benchmark = True
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'

parser = argparse.ArgumentParser(description="StereoDepth Unsupervised Domain Adaptation")
parser.add_argument('--dataset_config', default='./config/datasets/kitti2015_to_kitti2012.py', help='source domain and target domain name')
parser.add_argument('--model_config', default='', help='UDA model preparation')
parser.add_argument('--seed', default=1, metavar='S', help='random seed(default = 1)')
parser.add_argument('--log_dir', default='./log', help='log directory')

args = parser.parse_args()
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
os.makedirs(args.log_dir, exist_ok=True)



cfg = prepare_cfg(args)
train_dataset = PrepareDataset(, training=True)
test_dataset = PrepareDataset(, training=False)



def train_step(model, data_batch, optimizer):
    model.train()
    optimizer.zero_grad()
    
    outputs = model.forward_train(data_batch) 
    loss = outputs['loss']
    loss.backward()
    optimizer.step()
    
    # ema update here?
    model.update_ema()
    
    return outputs['log_vars'] 
    
    
    

def main(args):
    
    
    for epoch in range(args.num_epochs):
        for data_batch in train_loader:
            log_vars = train_step(args.model, data_batch, optimizer)
    
    
    
    
    return






if __name__=="__main__":
    # argparser
    main()