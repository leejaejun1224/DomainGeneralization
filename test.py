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


def compute_error(pred, gt):
    mask = (gt > 0) & (gt < 192)
    error = torch.abs(pred[mask] - gt[mask])
    error = error.mean()
    return error


def test_sample(model, left, right):
    model.eval()
    with torch.no_grad():
        left = left.cuda()
        right = right.cuda()
        output, confidence_map = model(left, right)
    return output[1], confidence_map
    

    
def main():

    parser = argparse.ArgumentParser(description="StereoDepth Unsupervised Domain Adaptation")
    parser.add_argument('--dataset_config', default='./config/datasets/kitti2015_to_kitti2012.py', help='source domain and target domain name')
    parser.add_argument('--model_config', default='', help='UDA model preparation')
    parser.add_argument('--seed', default=1, metavar='S', help='random seed(default = 1)')
    parser.add_argument('--log_dir', default='./log', help='log directory')
    parser.add_argument('--source_only', default=True, type=int, help='batch size')
    parser.add_argument('--checkpoint', default='', help='path to checkpoint')

    args = parser.parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

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


    model = StereoDepthUDA(cfg)
    model.to('cuda:0')

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from {args.checkpoint}")
    

    model.eval()
    total_error = 0
    num_samples = 0

    for batch_idx, data_batch in enumerate(test_loader):
        if args.source_only:
            left, right, disp_gt = data_batch['src_left'], data_batch['src_right'], data_batch['src_disparity']
            output, confidence_map = test_sample(model, left, right)
            error = compute_error(output, disp_gt.cuda())
            total_error += error.item()
            num_samples += 1
            
            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx}: Error = {error.item():.4f}")

        else:
            left, right = data_batch['tgt_left'], data_batch['tgt_right']
            output, confidence_map = test_sample(model, left, right)
            


    if args.source_only:
        avg_error = total_error / num_samples
        print(f"\nAverage Error on Source Domain: {avg_error:.4f}")


if __name__=="__main__":
    main()