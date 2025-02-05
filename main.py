import os
import json
import math
import datetime
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from StereoDepthUDA import StereoDepthUDA


from torch.utils.data import DataLoader
from datasets import __datasets__
from datasets.dataloader import PrepareDataset
from experiment import prepare_cfg
from train import compute_uda_loss
from tools.plot_loss import plot_loss_graph

cudnn.benchmark = True
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'


def val_step(model, data_batch, cfg, train=False):
    model.eval()
    total_loss, log_var = compute_uda_loss(model, data_batch, cfg, train=False)
    return log_var

# train sample one by one
def train_step(model, iter, data_batch, optimizer, cfg):
    model.train()
    optimizer.zero_grad()
    total_loss, log_var = compute_uda_loss(model, data_batch, cfg, train=True)
    total_loss.backward()
    optimizer.step()
    model.update_ema(iter, alpha=0.99)
    return log_var
    
    
def main():
    parser = argparse.ArgumentParser(description="StereoDepth Unsupervised Domain Adaptation")
    parser.add_argument('--dataset_config', default='./config/datasets/kitti2015_to_kitti2012.py', help='source domain and target domain name')
    parser.add_argument('--uda_config', default='./config/uda/kit15_kit12.py', help='UDA model preparation')
    parser.add_argument('--seed', default=1, metavar='S', help='random seed(default = 1)')
    parser.add_argument('--log_dir', default='./log', help='log directory')

    args = parser.parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    dir_name = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M")
    save_dir = args.log_dir + '/' + dir_name
    os.makedirs(save_dir, exist_ok=True)

    cfg = prepare_cfg(args)
    log_dict = {'parameters': cfg}

    train_dataset = PrepareDataset(source_datapath=cfg['dataset']['src_root'],
                                target_datapath=cfg['dataset']['tgt_root'], 
                                sourcefile_list=cfg['dataset']['src_filelist'],
                                targetfile_list=cfg['dataset']['tgt_filelist'],
                                training=True)
    
    test_dataset = PrepareDataset(source_datapath=cfg['dataset']['src_root'],
                                target_datapath=cfg['dataset']['tgt_root'],
                                sourcefile_list=cfg['dataset']['src_filelist'], 
                                targetfile_list=cfg['dataset']['tgt_filelist'],
                                training=False)
    
    train_loader = DataLoader(train_dataset, batch_size=cfg['batch_size'], shuffle=True, num_workers=cfg['num_workers'], drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=cfg['batch_size'], shuffle=False, num_workers=cfg['num_workers'], drop_last=False)
    # print(cfg)

    model = StereoDepthUDA(cfg)
    model.to('cuda:0')
    
    # 이거 init하는 조건은 좀 더 생각을 해봐야겠는데
    model.init_ema()

    # optimizer 좀 더 고민해보자.
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
    # 시작하자잉
    for epoch in range(cfg['epoch']): 
        model.train()
        train_losses = []
        step_loss = {}
        for batch_idx, data_batch in enumerate(train_loader):

            # print(data_batch)
            for key in data_batch:
                if isinstance(data_batch[key], torch.Tensor):
                    data_batch[key] = data_batch[key].cuda()
                    # print(data_batch[key])
            log_vars = train_step(model, epoch, data_batch, optimizer, cfg)
            if not math.isnan(log_vars['loss']):
                train_losses.append(log_vars['loss'])
        
        avg_loss = sum(train_losses) / len(train_losses)
        
        print(f'Epoch [{epoch + 1}/{cfg["epoch"]}] Average Loss: {avg_loss:.4f}')
        step_loss = {'train_loss' : avg_loss}


        if (epoch + 1) % cfg['val_interval'] == 0:
            val_losses = []
            
            with torch.no_grad():
                for data_batch in test_loader:
                    # gpu로 옮기기
                    for key in data_batch:
                        if isinstance(data_batch[key], torch.Tensor):
                            data_batch[key] = data_batch[key].cuda()
                            
                    # EMA model로 검증
                    log_vars = val_step(model, data_batch, cfg, train=False)
                    if not math.isnan(log_vars['loss']):
                        val_losses.append(log_vars['loss'])
            
            avg_val_loss = sum(val_losses) / len(val_losses)
            print(f'Validation Loss: {avg_val_loss:.4f}')
            step_loss['val_loss'] = avg_val_loss 

            # Save checkpoint
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'ema_state_dict': model.ema_state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
                # 'loss': avg_val_loss,
            }
            torch.save(checkpoint, os.path.join(save_dir, f'checkpoint_epoch{epoch+1}.pth'))

        log_dict[f'epoch_{epoch+1}'] = step_loss

    with open(f'{save_dir}/training_log.json', 'w') as f:
        json.dump(log_dict, f, indent=4)
    plot_loss_graph(log_dict, f'{save_dir}/loss_graph.png')

    return 0





if __name__=="__main__":
    # argparser
    main()