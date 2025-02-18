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
from tools.plot_loss import plot_loss_graph


cudnn.benchmark = True
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'


# def test_step(model, data_batch, cfg, train=False):
#     model.eval()
#     total_loss, log_var = compute_uda_loss(model, data_batch, cfg, train=False)
#     pred_disp = log_var['pseudo_disp']
#     return pred_disp


    
def main():
    parser = argparse.ArgumentParser(description="StereoDepth Unsupervised Domain Adaptation")
    parser.add_argument('--dataset_config', default='./config/datasets/cityscapes_to_kitti2015.py', help='source domain and target domain name')
    parser.add_argument('--uda_config', default='./config/uda/kit15_cityscapes.py', help='UDA model preparation')
    parser.add_argument('--seed', default=1, metavar='S', help='random seed(default = 1)')
    parser.add_argument('--log_dir', default='./log', help='log directory')
    parser.add_argument('--ckpt', default='', help='checkpoint', )

    args = parser.parse_args()
    assert args.ckpt != '', 'checkpoint is required !!'

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    dir_name = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M")
    save_dir = '/'.join(args.ckpt.split('/')[:-1]) + '/disp'
    os.makedirs(save_dir, exist_ok=True)

    cfg = prepare_cfg(args)
    log_dict = {'parameters': cfg}

    test_dataset = PrepareDataset(source_datapath=cfg['dataset']['src_root'],
                                target_datapath=cfg['dataset']['tgt_root'],
                                sourcefile_list=cfg['dataset']['src_filelist'], 
                                targetfile_list=cfg['dataset']['tgt_filelist'],
                                training=False)
    
    test_loader = DataLoader(test_dataset, batch_size=cfg['batch_size'], shuffle=False, num_workers=cfg['num_workers'], drop_last=False)
    # print(cfg)

    model = __models__['StereoDepthUDA'](cfg)
    model.to('cuda:0')
    
    model.student_model.load_state_dict(torch.load(args.ckpt)['student_state_dict'])
    model.teacher_model.load_state_dict(torch.load(args.ckpt)['teacher_state_dict'])
    # 이거 init하는 조건은 좀 더 생각을 해봐야겠는데

    # optimizer 좀 더 고민해보자.
    # 시작하자잉
    train_losses = []
    step_loss = {}
    for batch_idx, data_batch in enumerate(test_loader):

        # print(data_batch)
        for key in data_batch:
            if isinstance(data_batch[key], torch.Tensor):
                data_batch[key] = data_batch[key].cuda()
                # print(data_batch[key])
        log_vars = model.forward_test(data_batch)

        pred_src_dir = os.path.join(save_dir, 'src')
        os.makedirs(pred_src_dir, exist_ok=True)
        pred_src = data_batch['src_pred_disp'][0].cpu().numpy()
        plt.imsave(f'{pred_src_dir}/pred_disp_{batch_idx}.png', pred_src[0], cmap='jet')


        pred_tgt_dir = os.path.join(save_dir, 'tgt')
        os.makedirs(pred_tgt_dir, exist_ok=True)
        pred_tgt = data_batch['tgt_pred_disp'][0].cpu().numpy()
        plt.imsave(f'{pred_tgt_dir}/pred_disp_{batch_idx}.png', pred_tgt[0], cmap='jet')

    return 0





if __name__=="__main__":
    # argparser
    main()