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
from tools.write_log import save_disparity, save_metrics, save_att, compare, save_entropy, save_gt
from tools.save_heatmap import save_heatmap
from models.estimator.Fast_ACV_plus import Feature
from collections import OrderedDict

cudnn.benchmark = True
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'



def main():
    parser = argparse.ArgumentParser(description="StereoDepth Unsupervised Domain Adaptation")
    parser.add_argument('--dataset_config', default='./config/datasets/cityscapes_to_kitti2015.py', help='source domain and target domain name')
    parser.add_argument('--uda_config', default='./config/uda/kit15_cityscapes.py', help='UDA model preparation')
    parser.add_argument('--seed', default=1, metavar='S', help='random seed(default = 1)')
    parser.add_argument('--log_dir', default='./log', help='log directory')
    parser.add_argument('--ckpt', default='', help='checkpoint', )
    parser.add_argument('--compute_metrics', default=True, help='compute error')
    parser.add_argument('--save_disp', default=True, help='save disparity')
    parser.add_argument('--save_att', default=True, help='save attention')
    parser.add_argument('--save_heatmap', default=False, help='save heatmap')
    parser.add_argument('--save_entropy', default=True, help='save entropy')
    parser.add_argument('--save_gt', default=True, help='save gt')
    parser.add_argument('--compare_costvolume', default=True, help='compare costvolume')
    args = parser.parse_args()
    assert args.ckpt != '', 'checkpoint is required !!'

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    dir_name = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M")
    log_dir  = '/'.join(args.ckpt.split('/')[:-1])
    save_dir = log_dir + '/disp'
    
    os.makedirs(save_dir, exist_ok=True)

    cfg = prepare_cfg(args, mode='test')
    log_dict = {'parameters': cfg}

    test_dataset = PrepareDataset(source_datapath=cfg['dataset']['src_root'],
                                target_datapath=cfg['dataset']['tgt_root'],
                                sourcefile_list=cfg['dataset']['src_filelist'], 
                                targetfile_list=cfg['dataset']['tgt_filelist'],
                                training=False)
    test_loader = DataLoader(test_dataset, batch_size=cfg['test_batch_size'], shuffle=False, num_workers=cfg['test_num_workers'], drop_last=False)
    # print(cfg)

    model = __models__['StereoDepthUDA'](cfg)
    
    model.student_model.load_state_dict(torch.load(args.ckpt)['student_state_dict'])
    model.teacher_model.load_state_dict(torch.load(args.ckpt)['teacher_state_dict'])
    
    # checkpoint = torch.load(args.ckpt, map_location="cuda:0")
    # new_state_dict = OrderedDict()
    # for k, v in checkpoint['model'].items():
    #     name = k.replace("module.", "")  # DataParallel로 저장된 모델 처리
    #     new_state_dict[name] = v
    # model.student_model.load_state_dict(new_state_dict, strict=True)
    # model.teacher_model.load_state_dict(new_state_dict, strict=True)



    model.to('cuda:0')
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_params}")
    
    # 이거 init하는 조건은 좀 더 생각을 해봐야겠는데

    # optimizer 좀 더 고민해보자.
    # 시작하자잉
    # target에 대해서 metric를 구할 줄 알아야 한다.
    train_losses = []
    step_loss = {}
    metrics_dict = {}
    model.eval()
    for batch_idx, data_batch in enumerate(test_loader):
        # print(data_batch)
        source_filename = data_batch['source_left_filename'][0].split('/')[-1]

        for key in data_batch:
            if isinstance(data_batch[key], torch.Tensor):
                data_batch[key] = data_batch[key].cuda()
                # print(data_batch[key])
        log_vars = model.forward_test(data_batch)

        if args.save_entropy:
            save_entropy(data_batch, log_dir)

        if args.compare_costvolume:
            compare(data_batch, log_dir)
            # [batch, 12, 1, 48, 156]
            # print("entropy_map shape: ", data_batch['src_shape_map'].shape)
        if args.save_gt and 'src_disparity' in data_batch.keys():
            save_gt(data_batch, log_dir)

        if args.save_att:
            save_att(data_batch, log_dir)

        if args.save_disp:
            save_disparity(data_batch, log_dir)

        if args.save_heatmap:   
            image_tensor = data_batch['src_left']
            feature_map, attn_weights = model.student_model.feature(image_tensor)
            save_heatmap(image_tensor, feature_map, attn_weights)

        if args.compute_metrics and 'src_disparity' in data_batch.keys():
            scalar_outputs = {}
            scalar_outputs["EPE"] = [EPE_metric(data_batch['src_pred_disp'][0], data_batch['src_disparity'], data_batch['mask'])]
            scalar_outputs["D1"] = [D1_metric(data_batch['src_pred_disp'][0], data_batch['src_disparity'], data_batch['mask'])]
            scalar_outputs["Thres1"] = [Thres_metric(data_batch['src_pred_disp'][0], data_batch['src_disparity'], data_batch['mask'], 1.0)]
            scalar_outputs["Thres2"] = [Thres_metric(data_batch['src_pred_disp'][0], data_batch['src_disparity'], data_batch['mask'], 2.0)]
            scalar_outputs["Thres3"] = [Thres_metric(data_batch['src_pred_disp'][0], data_batch['src_disparity'], data_batch['mask'], 3.0)]
            metrics_dict[source_filename] = tensor2float(scalar_outputs)

    save_metrics(metrics_dict, log_dir)
            

    return 0





if __name__=="__main__":
    # argparser
    main()