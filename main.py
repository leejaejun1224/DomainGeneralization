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
from datasets.cityscapes import CityscapesDataset
from experiment import prepare_cfg
# from models.losses.loss import compute_uda_loss
from tools.plot_loss import plot_loss_graph
from tools.compute_metrics import EPE_metric, D1_metric, Thres_metric

cudnn.benchmark = True
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'


# def val_step(model, data_batch, cfg, train=False):
#     model.eval()
#     total_loss, log_var = compute_uda_loss(model, data_batch, cfg, train=False)
#     return log_var

# # train sample one by one
# def train_step(model, iter, data_batch, optimizer, cfg):
#     model.train()
#     optimizer.zero_grad()

#     # inference the model here and add results in data_batch
#     # after that compute uda loss
#     # that can make me change loss function next time.
#     # 왜 가능하냐고? 참조로 전달되니까
#     total_loss, log_var = compute_uda_loss(model, data_batch, cfg, train=True)
#     total_loss.backward()
#     optimizer.step()
#     model.update_ema(iter, alpha=0.99)
#     return log_var
    
    
def main():
    parser = argparse.ArgumentParser(description="StereoDepth Unsupervised Domain Adaptation")
    parser.add_argument('--dataset_config', default='./config/datasets/cityscapes_to_kitti2015.py', help='source domain and target domain name')
    parser.add_argument('--uda_config', default='./config/uda/kit15_cityscapes.py', help='UDA model preparation')
    parser.add_argument('--seed', default=1, metavar='S', help='random seed(default = 1)')
    parser.add_argument('--log_dir', default='./log', help='log directory')
    parser.add_argument('--compute_metrics', default=True, help='compute metrics')

    args = parser.parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    dir_name = datetime.datetime.now().strftime("%Y-%m-%d_%H_%M")
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

    model = __models__['StereoDepthUDA'](cfg)
    model.to('cuda:0')
    
    # 이거 init하는 조건은 좀 더 생각을 해봐야겠는데
    model.init_ema()

    # optimizer 좀 더 고민해보자.
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
    log_dict['student_params'] = sum(p.numel() for p in model.student_model.parameters())
    log_dict['teacher_params'] = sum(p.numel() for p in model.teacher_model.parameters())
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
            log_vars = model.train_step(data_batch, optimizer, batch_idx)
            if not math.isnan(log_vars['loss']):
                train_losses.append(log_vars['loss'])

                # metric을 뭘 계산할건데?
                # target이 teacher이 얼마나 잘 계산이 되었는지는 test.py에서 계산을 하도록 하고
                # source가 student이 얼마나 잘 계산이 되었는지는 여기서 계산을 하도록 하자.
                if args.compute_metrics:
                    scalar_outputs = {}
                    scalar_outputs["EPE"] = [EPE_metric(data_batch['src_pred_disp'][0], data_batch['src_disparity'], data_batch['mask'])]
                    scalar_outputs["D1"] = [D1_metric(data_batch['src_pred_disp'][0], data_batch['src_disparity'], data_batch['mask'])]
                    scalar_outputs["Thres1"] = [Thres_metric(data_batch['src_pred_disp'][0], data_batch['src_disparity'], data_batch['mask'], 1.0)]
                    scalar_outputs["Thres2"] = [Thres_metric(data_batch['src_pred_disp'][0], data_batch['src_disparity'], data_batch['mask'], 2.0)]
                    scalar_outputs["Thres3"] = [Thres_metric(data_batch['src_pred_disp'][0], data_batch['src_disparity'], data_batch['mask'], 3.0)]
        
        if len(train_losses) > 0:
            avg_loss = sum(train_losses) / len(train_losses)
            print(f'Epoch [{epoch + 1}/{cfg["epoch"]}] Average Loss: {avg_loss:.4f}')
            step_loss = {'train_loss' : avg_loss}
        else:
            print(f'Epoch [{epoch + 1}/{cfg["epoch"]}] Average Loss: {0:.4f}')
            step_loss = {'train_loss' : 0}


        if (epoch + 1) % cfg['val_interval'] == 0:
            val_losses = []
            model.eval()
            with torch.no_grad():
                for data_batch in test_loader:
                    # gpu로 옮기기
                    for key in data_batch:
                        if isinstance(data_batch[key], torch.Tensor):
                            data_batch[key] = data_batch[key].cuda()
                    log_vars = model.forward_test(data_batch)
                            
                    # EMA model로 검증
                    # log_vars = val_step(model, data_batch, cfg, train=False)
                    if not math.isnan(log_vars['loss']):
                        val_losses.append(log_vars['loss'])
                    
                        # if args.compute_metrics:
                        #     scalar_outputs = {}
                        #     scalar_outputs["EPE"] = [EPE_metric(data_batch['src_pred_disp'][0], data_batch['src_disparity'], data_batch['mask'])]
                        #     scalar_outputs["D1"] = [D1_metric(data_batch['src_pred_disp'][0], data_batch['src_disparity'], data_batch['mask'])]
                        #     scalar_outputs["Thres1"] = [Thres_metric(data_batch['src_pred_disp'][0], data_batch['src_disparity'], data_batch['mask'], 1.0)]
                        #     scalar_outputs["Thres2"] = [Thres_metric(data_batch['src_pred_disp'][0], data_batch['src_disparity'], data_batch['mask'], 2.0)]
                        #     scalar_outputs["Thres3"] = [Thres_metric(data_batch['src_pred_disp'][0], data_batch['src_disparity'], data_batch['mask'], 3.0)]
            if len(val_losses) > 0:
                avg_val_loss = sum(val_losses) / len(val_losses)
                print(f'Validation Loss: {avg_val_loss:.4f}')
                step_loss['val_loss'] = avg_val_loss 
            else:
                print(f'Validation Loss: {0:.4f}')
                step_loss['val_loss'] = 0

            # Save checkpoint
            if (epoch + 1) % cfg['save_interval'] == 0:
                checkpoint = {
                    'epoch': epoch,
                    'student_state_dict': model.student_state_dict(),
                    'teacher_state_dict': model.teacher_state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                }
                torch.save(checkpoint, os.path.join(save_dir, f'checkpoint_epoch{epoch+1}.pth'))
            
            if 'confidence_map' in data_batch:
                confidence_map_dir = os.path.join(save_dir, 'confidence_maps')
                os.makedirs(confidence_map_dir, exist_ok=True)
                confidence_map = data_batch['confidence_map'].cpu().numpy()
                target_left_filename = data_batch['target_left_filename']
                for idx, conf_map in enumerate(confidence_map):
                    plt.figure(figsize=(10, 8))
                    plt.imshow(conf_map, cmap='viridis')  # viridis is good for confidence visualization
                    plt.colorbar(label='Confidence')
                    plt.title(f'Confidence Map - Epoch {epoch+1} Batch {idx}')
                    plt.savefig(os.path.join(confidence_map_dir, target_left_filename[idx].split('/')[-1]))
                    plt.close() 

            # 이거 좀 더 고민해보자.
            log_dict[f'epoch_{epoch+1}'] = step_loss

    with open(f'{save_dir}/training_log.json', 'w') as f:
        json.dump(log_dict, f, indent=4)
    plot_loss_graph(log_dict, f'{save_dir}/loss_graph.png')

    return 0


if __name__=="__main__":
    main()