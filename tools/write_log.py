import os
import json
import matplotlib.pyplot as plt



def save_disparity(data_batch, dir_name):
    os.makedirs(dir_name, exist_ok=True)
    pred_src_dir = os.path.join(dir_name, 'src')
    os.makedirs(pred_src_dir, exist_ok=True)
    pred_src = data_batch['src_pred_disp'][0].cpu().numpy()
    source_left_filename = data_batch['source_left_filename'][0].split('/')[-1]
    plt.imsave(os.path.join(pred_src_dir, source_left_filename), pred_src[0], cmap='jet')

    pred_tgt_dir = os.path.join(dir_name, 'tgt')
    os.makedirs(pred_tgt_dir, exist_ok=True)
    pred_tgt = data_batch['tgt_pred_disp'][0].cpu().numpy()
    target_left_filename = data_batch['target_left_filename'][0].split('/')[-1] 
    plt.imsave(os.path.join(pred_tgt_dir, target_left_filename), pred_tgt[0], cmap='jet')



def save_metrics(metrics, dir_name):
    os.makedirs(dir_name + '/metrics', exist_ok=True)
    with open(os.path.join(dir_name, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
