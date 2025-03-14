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
from datasets.dataloader import PrepareDataset
from experiment import prepare_cfg, adjust_learning_rate
from tools.plot_loss import plot_loss_graph, plot_true_ratio
from tools.metrics import EPE_metric, D1_metric, Thres_metric
from models.tools.threshold_manager import ThresholdManager

cudnn.benchmark = True
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'

def parse_args():
    parser = argparse.ArgumentParser(description="StereoDepth Unsupervised Domain Adaptation")
    parser.add_argument('--dataset_config', default='./config/datasets/cityscapes_to_kitti2015.py', help='source domain and target domain name')
    parser.add_argument('--uda_config', default='./config/uda/kit15_cityscapes.py', help='UDA model preparation')
    parser.add_argument('--seed', default=1, metavar='S', help='random seed(default = 1)')
    parser.add_argument('--log_dir', default='./log', help='log directory')
    parser.add_argument('--compute_metrics', default=True, help='compute metrics')
    return parser.parse_args()

def setup_environment(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    dir_name = datetime.datetime.now().strftime("%Y-%m-%d_%H_%M")
    save_dir = os.path.join(args.log_dir, dir_name)
    os.makedirs(save_dir, exist_ok=True)
    return save_dir

def setup_data_loaders(cfg):
    train_dataset = PrepareDataset(
        source_datapath=cfg['dataset']['src_root'],
        target_datapath=cfg['dataset']['tgt_root'], 
        sourcefile_list=cfg['dataset']['src_filelist'],
        targetfile_list=cfg['dataset']['tgt_filelist'],
        training=True
    )
    
    test_dataset = PrepareDataset(
        source_datapath=cfg['dataset']['src_root'],
        target_datapath=cfg['dataset']['tgt_root'],
        sourcefile_list=cfg['dataset']['src_filelist'], 
        targetfile_list=cfg['dataset']['tgt_filelist'],
        training=False
    )
    
    train_loader = DataLoader(train_dataset, batch_size=cfg['batch_size'], shuffle=True, num_workers=cfg['num_workers'], drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=cfg['batch_size'], shuffle=False, num_workers=cfg['num_workers'], drop_last=False)
    
    return train_loader, test_loader

def compute_metrics_dict(data_batch):
    return {
        "EPE": [EPE_metric(data_batch['src_pred_disp'][0], data_batch['src_disparity'], data_batch['mask'])],
        "D1": [D1_metric(data_batch['src_pred_disp'][0], data_batch['src_disparity'], data_batch['mask'])],
        "Thres1": [Thres_metric(data_batch['src_pred_disp'][0], data_batch['src_disparity'], data_batch['mask'], 1.0)],
        "Thres2": [Thres_metric(data_batch['src_pred_disp'][0], data_batch['src_disparity'], data_batch['mask'], 2.0)],
        "Thres3": [Thres_metric(data_batch['src_pred_disp'][0], data_batch['src_disparity'], data_batch['mask'], 3.0)]
    }

def process_batch(data_batch):
    for key in data_batch:
        if isinstance(data_batch[key], torch.Tensor):
            data_batch[key] = data_batch[key].cuda()
    return data_batch

def train_epoch(model, train_loader, optimizer, threshold_manager, epoch, cfg, args):
    model.train()
    adjust_learning_rate(optimizer, epoch, cfg['lr'], cfg['adjust_lr'])
    
    true_ratios, train_losses, train_pseudo_losses = [], [], []
    
    for batch_idx, data_batch in enumerate(train_loader):
        data_batch = process_batch(data_batch)
        image_ids = data_batch['target_left_filename']
        threshold_manager.initialize_log(image_ids)
        
        log_vars = model.train_step(data_batch, optimizer, batch_idx)
        
        if not math.isnan(log_vars['loss']):
            train_losses.append(log_vars['loss'])
            train_pseudo_losses.append(log_vars['unsupervised_loss'])
            true_ratios.append(log_vars['true_ratio'])
            
            threshold_manager.update_log(image_ids, log_vars['true_ratio'], log_vars['unsupervised_loss'], epoch)
            
            if args.compute_metrics:
                scalar_outputs = compute_metrics_dict(data_batch)
    
    if train_losses:
        return {
            'train_loss': sum(train_losses) / len(train_losses),
            'true_ratio_train': sum(true_ratios) / len(true_ratios),
            'train_pseudo_loss': sum(train_pseudo_losses) / len(train_pseudo_losses)
        }
    return {'train_loss': 0, 'true_ratio_train': 0, 'train_pseudo_loss': 0}

def validate(model, test_loader):
    model.eval()
    val_losses, val_pseudo_losses, true_ratios = [], [], []
    
    with torch.no_grad():
        for data_batch in test_loader:
            data_batch = process_batch(data_batch)
            log_vars = model.forward_test(data_batch)
            
            if not math.isnan(log_vars['loss']):
                val_losses.append(log_vars['loss'])
                true_ratios.append(log_vars['true_ratio'])
                val_pseudo_losses.append(log_vars['unsupervised_loss'])
    
    if val_losses:
        return {
            'val_loss': sum(val_losses) / len(val_losses),
            'true_ratio_val': sum(true_ratios) / len(true_ratios),
            'val_pseudo_loss': sum(val_pseudo_losses) / len(val_pseudo_losses)
        }
    return {'val_loss': 0, 'true_ratio_val': 0, 'val_pseudo_loss': 0}

def save_checkpoint(model, optimizer, epoch, save_dir):
    checkpoint = {
        'epoch': epoch,
        'student_state_dict': model.student_state_dict(),
        'teacher_state_dict': model.teacher_state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    torch.save(checkpoint, os.path.join(save_dir, f'checkpoint_epoch{epoch+1}.pth'))

def main():
    args = parse_args()
    save_dir = setup_environment(args)
    
    cfg = prepare_cfg(args)
    log_dict = {'parameters': cfg}
    
    train_loader, test_loader = setup_data_loaders(cfg)
    
    model = __models__['StereoDepthUDA'](cfg)
    model.to('cuda:0')
    model.init_ema()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
    log_dict['student_params'] = sum(p.numel() for p in model.student_model.parameters())
    log_dict['teacher_params'] = sum(p.numel() for p in model.teacher_model.parameters())
    
    threshold_manager = ThresholdManager(save_dir=save_dir)
    
    for epoch in range(cfg['epoch']):
        train_metrics = train_epoch(model, train_loader, optimizer, threshold_manager, epoch, cfg, args)
        print(f'Epoch [{epoch + 1}/{cfg["epoch"]}] Average Loss: {train_metrics["train_loss"]:.4f}')
        
        if (epoch + 1) % cfg['val_interval'] == 0:
            val_metrics = validate(model, test_loader)
            print(f'Validation Loss: {val_metrics["val_loss"]:.4f}')
            
            if (epoch + 1) % cfg['save_interval'] == 0:
                save_checkpoint(model, optimizer, epoch, save_dir)
            
            log_dict[f'epoch_{epoch+1}'] = {**train_metrics, **val_metrics}
    
    with open(f'{save_dir}/training_log.json', 'w') as f:
        json.dump(log_dict, f, indent=4)
        
    threshold_manager.save_log()
    plot_loss_graph(log_dict, f'{save_dir}/loss_graph.png')
    plot_true_ratio(log_dict, f'{save_dir}/true_ratio_graph.png')
    
    return 0

if __name__=="__main__":
    main()