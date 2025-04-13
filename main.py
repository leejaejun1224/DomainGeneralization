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
from datasets import __datasets__
from torch.utils.data import DataLoader
from datasets.dataloader import PrepareDataset
from experiment import prepare_cfg, adjust_learning_rate
from tools.plot_loss import plot_loss_graph, plot_true_ratio, plot_threshold, plot_reconstruction_loss
from tools.metrics import EPE_metric, D1_metric, Thres_metric
from models.tools.threshold_manager import EntropyThresholdManager, ThresholdManager

cudnn.benchmark = True
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'

def parse_args():
    parser = argparse.ArgumentParser(description="StereoDepth Unsupervised Domain Adaptation")
    parser.add_argument('--dataset_config', default='./config/datasets/cityscapes_to_kitti2015.py', help='source domain and target domain name')
    parser.add_argument('--uda_config', default='./config/uda/kit15_cityscapes.py', help='UDA model preparation')
    parser.add_argument('--seed', default=1, metavar='S', help='random seed(default = 1)')
    parser.add_argument('--log_dir', default='./log', help='log directory')
    parser.add_argument('--compute_metrics', default=True, help='compute metrics')
    parser.add_argument('--checkpoint', default=None, help='load checkpoint')
    return parser.parse_args()

def setup_environment(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    dir_name = datetime.datetime.now().strftime("%Y-%m-%d_%H_%M")
    save_dir = os.path.join(args.log_dir, dir_name)
    os.makedirs(save_dir, exist_ok=True)
    return save_dir

def setup_train_loaders(cfg):
    source_dataset = __datasets__[cfg['dataset']['src_type']](
        datapath=cfg['dataset']['src_root'],
        list_filename=cfg['dataset']['src_filelist'],
        training=True, 
        aug=True
    )
    
    target_dataset = __datasets__[cfg['dataset']['tgt_type']](
        datapath=cfg['dataset']['tgt_root'],
        list_filename=cfg['dataset']['tgt_filelist'],
        training=True,
        aug=False
    )

    max_len = max(len(source_dataset), len(target_dataset))
    source_dataset.max_len = max_len
    target_dataset.max_len = max_len


    source_loader = DataLoader(
        source_dataset, 
        batch_size=cfg['test_batch_size'],
        shuffle=False,
        num_workers=cfg['test_num_workers'],
        drop_last=False
    )
    
    target_loader = DataLoader(
        target_dataset, 
        batch_size=cfg['test_batch_size'],
        shuffle=False,
        num_workers=cfg['test_num_workers'],
        drop_last=False
    )
    return source_loader, target_loader


def setup_test_loaders(cfg):
    source_dataset = __datasets__[cfg['dataset']['src_type']](
        datapath=cfg['dataset']['src_root'],
        list_filename=cfg['dataset']['src_filelist'],
        training=False
    )
    target_dataset = __datasets__[cfg['dataset']['tgt_type']](
        datapath=cfg['dataset']['tgt_root'],
        list_filename=cfg['dataset']['tgt_filelist'],
        training=False
    )

    max_len = max(len(source_dataset), len(target_dataset))
    source_dataset.max_len = max_len
    target_dataset.max_len = max_len

    source_loader = DataLoader(
        source_dataset, 
        batch_size=cfg['test_batch_size'],
        shuffle=False,
        num_workers=cfg['test_num_workers'],
        drop_last=False
    )
    
    target_loader = DataLoader(
        target_dataset, 
        batch_size=cfg['test_batch_size'],
        shuffle=False,
        num_workers=cfg['test_num_workers'],
        drop_last=False
    )
    return source_loader, target_loader


def compute_metrics_dict(data_batch):
    return {
        "EPE": [EPE_metric(data_batch['src_pred_disp_s'][0], data_batch['src_disparity'], data_batch['mask'])],
        "D1": [D1_metric(data_batch['src_pred_disp_s'][0], data_batch['src_disparity'], data_batch['mask'])],
        "Thres1": [Thres_metric(data_batch['src_pred_disp_s'][0], data_batch['src_disparity'], data_batch['mask'], 1.0)],
        "Thres2": [Thres_metric(data_batch['src_pred_disp_s'][0], data_batch['src_disparity'], data_batch['mask'], 2.0)],
        "Thres3": [Thres_metric(data_batch['src_pred_disp_s'][0], data_batch['src_disparity'], data_batch['mask'], 3.0)]
    }

def process_batch(data_batch, source_batch, target_batch):
    for key in source_batch:
        if isinstance(source_batch[key], torch.Tensor):
            data_batch['src_' + key] = source_batch[key].cuda()
        else:
            data_batch['src_' + key] = source_batch[key]
    for key in target_batch:
        if isinstance(target_batch[key], torch.Tensor):
            data_batch['tgt_' + key] = target_batch[key].cuda()
        else:
            data_batch['tgt_' + key] = target_batch[key]
    return data_batch

def train_epoch(model, source_loader, target_loader, optimizer, threshold_manager, epoch, cfg, args):
    model.train()
    current_lr = adjust_learning_rate(optimizer, epoch, cfg['lr'], cfg['adjust_lr'])
    
    true_ratios, train_losses, train_supervised_losses, train_pseudo_losses, reconstruction_losses = [], [], [], [], []
    average_threshold = []
    for batch_idx, (source_batch, target_batch) in enumerate(zip(source_loader, target_loader)):
        data_batch = {}
        data_batch = process_batch(data_batch, source_batch, target_batch)
        image_ids = data_batch['tgt_left_filename']
        threshold_manager.initialize_log(image_ids)
        threshold = threshold_manager.get_threshold(image_ids).float()
        average_threshold.append(threshold.mean().item())
        temperature = max(0.5, 1.0 - 0.5 * (epoch / cfg['epoch']))
        log_vars = model.train_step(data_batch, optimizer, batch_idx, threshold, temperature=temperature)
        
        if not math.isnan(log_vars['loss']):
            train_losses.append(log_vars['loss'])
            train_supervised_losses.append(log_vars['supervised_loss'])
            train_pseudo_losses.append(log_vars['unsupervised_loss'])
            true_ratios.append(log_vars['true_ratio'])
            reconstruction_losses.append(log_vars['reconstruction_loss'])
            threshold_manager.update_log(image_ids, log_vars['true_ratio'], log_vars['unsupervised_loss'], epoch)
            
            if args.compute_metrics:
                scalar_outputs = compute_metrics_dict(data_batch)

    print("average_threshold", sum(average_threshold)/len(average_threshold))

    if train_losses:
        return {
            'train_loss': sum(train_losses) / len(train_losses),
            'train_supervised_loss': sum(train_supervised_losses) / len(train_supervised_losses),
            'true_ratio_train': sum(true_ratios) / len(true_ratios),
            'train_pseudo_loss': sum(train_pseudo_losses) / len(train_pseudo_losses),
            'average_threshold': sum(average_threshold)/len(average_threshold),
            'reconstruction_loss': sum(reconstruction_losses)/len(reconstruction_losses),
            'learning_rate': current_lr
        }
    return {'train_loss': 0, 'true_ratio_train': 0, 'train_pseudo_loss': 0, 'reconstruction_loss': 0, 'learning_rate': current_lr}

def validate(model, source_loader, target_loader):
    model.eval()
    val_losses, val_supervised_losses, val_pseudo_losses, true_ratios, reconstruction_losses = [], [], [], [], []
    
    with torch.no_grad():
        for source_batch, target_batch in zip(source_loader, target_loader):
            data_batch = {}
            data_batch = process_batch(data_batch, source_batch, target_batch)
            log_vars = model.forward_test(data_batch)
            
            if not math.isnan(log_vars['loss']):
                val_losses.append(log_vars['loss'])
                true_ratios.append(log_vars['true_ratio'])
                val_supervised_losses.append(log_vars['supervised_loss'])
                val_pseudo_losses.append(log_vars['unsupervised_loss'])
                reconstruction_losses.append(log_vars['reconstruction_loss'])
    if val_losses:
        return {
            'val_loss': sum(val_losses) / len(val_losses),
            'true_ratio_val': sum(true_ratios) / len(true_ratios),
            'val_supervised_loss': sum(val_supervised_losses) / len(val_supervised_losses),
            'val_pseudo_loss': sum(val_pseudo_losses) / len(val_pseudo_losses),
            'reconstruction_loss': sum(reconstruction_losses)/len(reconstruction_losses)
        }
    return {'val_loss': 0, 'true_ratio_val': 0, 'val_pseudo_loss': 0, 'reconstruction_loss': 0}

def save_checkpoint(model, optimizer, epoch, save_dir, current_lr):
    checkpoint = {
        'epoch': epoch,
        'student_state_dict': model.student_state_dict(),
        'teacher_state_dict': model.teacher_state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'learning_rate': current_lr
    }
    torch.save(checkpoint, os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth'))

def main():
    args = parse_args()
    save_dir = setup_environment(args)
    
    cfg = prepare_cfg(args)
    log_dict = {'parameters': cfg}
    
    train_source_loader, train_target_loader = setup_train_loaders(cfg)
    test_source_loader, test_target_loader = setup_test_loaders(cfg)
    
    model = __models__['StereoDepthUDA'](cfg)
    start_epoch = 0
    if args.checkpoint is not None:
        print("checkpoint", args.checkpoint)
        checkpoint = torch.load(args.checkpoint)
        model.student_model.load_state_dict(checkpoint['student_state_dict'])
        model.teacher_model.load_state_dict(checkpoint['teacher_state_dict'])
        start_epoch = int(args.checkpoint.split('_')[-1].split('.')[0])
        print("start_epoch", start_epoch)
        log_dict['ckpt'] = args.checkpoint
        log_dict['ckpt_epoch'] = start_epoch

    model.to('cuda:0')
    model.init_ema()

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
    if args.checkpoint is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
    log_dict['student_params'] = sum(p.numel() for p in model.student_model.parameters())
    log_dict['teacher_params'] = sum(p.numel() for p in model.teacher_model.parameters())
    
    threshold_manager = EntropyThresholdManager(save_dir=save_dir)
    
    for epoch in range(start_epoch, start_epoch + cfg['epoch']):
        train_metrics = train_epoch(model, train_source_loader, train_target_loader, optimizer, threshold_manager, epoch, cfg, args)
        print(f'Epoch [{epoch + 1}/{start_epoch + cfg["epoch"]}] Average Loss: {train_metrics["train_loss"]:.4f}')
        
        if (epoch + 1) % cfg['val_interval'] == 0:
            val_metrics = validate(model, test_source_loader, test_target_loader)
            print(f'Validation Loss: {val_metrics["val_loss"]:.4f}')
            
            if (epoch + 1) % cfg['save_interval'] == 0:
                save_checkpoint(model, optimizer, epoch, save_dir, train_metrics['learning_rate'])
            
            log_dict[f'epoch_{epoch+1}'] = {**train_metrics, **val_metrics}
    
    with open(f'{save_dir}/training_log.json', 'w') as f:
        json.dump(log_dict, f, indent=4)
        
    # threshold_manager.save_log()
    plot_threshold(log_dict, f'{save_dir}/threshold_graph.png')
    plot_loss_graph(log_dict, f'{save_dir}/loss_graph.png')
    plot_true_ratio(log_dict, f'{save_dir}/true_ratio_graph.png')
    plot_reconstruction_loss(log_dict, f'{save_dir}/reconstruction_loss_graph.png')
    
    return 0

if __name__=="__main__":
    main()