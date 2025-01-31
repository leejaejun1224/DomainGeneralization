import os
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

cudnn.benchmark = True
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'



def train_step(model, data_batch, optimizer, cfg):
    model.train()
    optimizer.zero_grad()
    
    total_loss, log_var = compute_uda_loss(model, data_batch, cfg)

    total_loss.backward()
    optimizer.step()
    
    # ema update here?
    model.update_ema(alpha=0.99)
    
    return log_var
    
    
    
def main():

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

    train_dataset = PrepareDataset(source_datapath=cfg['data']['src_root'],
                                target_datapath=cfg['data']['tgt_root'], 
                                sourcefile_list=cfg['data']['src_filelist'],
                                targetfile_list=cfg['data']['tgt_filelist'],
                                training=True)
    test_dataset = PrepareDataset(source_datapath=cfg['data']['src_root'],
                                target_datapath=cfg['data']['tgt_root'],
                                sourcefile_list=cfg['data']['src_filelist'], 
                                targetfile_list=cfg['data']['tgt_filelist'],
                                training=False)
    train_loader = DataLoader(train_dataset, batch_size=cfg['batch_size'], shuffle=True, num_workers=cfg['num_workers'], drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=cfg['batch_size'], shuffle=False, num_workers=cfg['num_workers'], drop_last=False)


    # print(cfg)
    model = StereoDepthUDA(cfg)
    model.to('cuda:0')
    
    
    # 이거 init하는 조건은 좀 더 생각을 해봐야겠는데
    model.init_ema() 

    # optimizer 좀 더 고민해보자.
    # optimizer = torch.optim.Adam(model.parameters(), lr=cfg.optimizer.lr)
    
    # 시작하자잉
    # for epoch in range(cfg.train.num_epochs):
    for epoch in range(100):
        model.train()
        train_losses = []
        
        for batch_idx, data_batch in enumerate(train_loader):

            # print(data_batch)
            for key in data_batch:
                if isinstance(data_batch[key], torch.Tensor):
                    data_batch[key] = data_batch[key].cuda()
                    # print(data_batch[key])
        print(epoch)
        #     log_vars = train_step(model, data_batch, optimizer, cfg)
        #     train_losses.append(log_vars['loss'])
            
        #     if batch_idx % cfg.train.log_interval == 0:
        #         print(f'Epoch [{epoch}/{cfg.train.num_epochs}] Batch [{batch_idx}/{len(train_loader)}] '
        #               f'Loss: {log_vars["loss"]:.4f}')
        
        # avg_loss = sum(train_losses) / len(train_losses)
        # print(f'Epoch [{epoch}/{cfg.train.num_epochs}] Average Loss: {avg_loss:.4f}')
        


        # if epoch % cfg.val.freq == 0:
        #     model.eval()
        #     val_losses = []
            
        #     with torch.no_grad():
        #         for data_batch in test_loader:
        #             # Move data to GPU
        #             for key in data_batch:
        #                 if isinstance(data_batch[key], torch.Tensor):
        #                     data_batch[key] = data_batch[key].cuda()
                            
        #             # Use EMA model for validation
        #             outputs = model.ema_forward(data_batch)
        #             val_losses.append(outputs['loss'].item())
            
        #     avg_val_loss = sum(val_losses) / len(val_losses)
        #     print(f'Validation Loss: {avg_val_loss:.4f}')
            
        #     # Save checkpoint
        #     checkpoint = {
        #         'epoch': epoch,
        #         'model_state_dict': model.state_dict(),
        #         'ema_state_dict': model.ema_state_dict(),
        #         'optimizer_state_dict': optimizer.state_dict(),
        #         'loss': avg_val_loss,
        #     }
        #     torch.save(checkpoint, os.path.join(args.log_dir, f'checkpoint_epoch{epoch}.pth'))
    
    return 0





if __name__=="__main__":
    # argparser
    main()