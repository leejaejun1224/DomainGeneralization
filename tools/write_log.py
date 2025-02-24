import os
import json
import matplotlib.pyplot as plt

def save_att(data_batch, dir_name):
    os.makedirs(dir_name + '/att', exist_ok=True)
    # print("data_batch['src_pred_disp']", len(data_batch['src_pred_disp']))
    for i in range(data_batch['src_pred_disp'][0].shape[0]):
        att_prob, _ = data_batch['src_pred_disp'][2][i].max(dim=0, keepdim=True)
        att_prob = att_prob.squeeze(0).cpu().numpy()
        source_left_filename = data_batch['source_left_filename'][i].split('/')[-1]
        plt.imsave(os.path.join(dir_name, 'att', source_left_filename), att_prob, cmap='gray')


def save_disparity(data_batch, dir_name):
    os.makedirs(dir_name + '/disp/src', exist_ok=True)
    os.makedirs(dir_name + '/disp/tgt', exist_ok=True)
    pred_src_dir = os.path.join(dir_name, 'disp', 'src')
    os.makedirs(pred_src_dir, exist_ok=True)
    for i in range(data_batch['src_pred_disp'][0].shape[0]):
        pred_src = data_batch['src_pred_disp'][0][i].cpu().numpy()
        source_left_filename = data_batch['source_left_filename'][i].split('/')[-1]
        plt.imsave(os.path.join(pred_src_dir, source_left_filename), pred_src, cmap='jet')

        pred_tgt_dir = os.path.join(dir_name, 'disp', 'tgt')
        os.makedirs(pred_tgt_dir, exist_ok=True)
        pred_tgt = data_batch['tgt_pred_disp'][0][i].cpu().numpy()
        target_left_filename = data_batch['target_left_filename'][i].split('/')[-1] 
        plt.imsave(os.path.join(pred_tgt_dir, target_left_filename), pred_tgt, cmap='jet')



def save_metrics(metrics, dir_name):
    metrics_dir = os.path.join(dir_name, 'metrics')
    os.makedirs(metrics_dir, exist_ok=True)
    with open(os.path.join(metrics_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)