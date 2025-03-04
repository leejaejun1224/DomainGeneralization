import os
import json
import matplotlib.pyplot as plt
import torch.nn.functional as F
def save_att(data_batch, dir_name):
    os.makedirs(dir_name + '/att', exist_ok=True)
    # print("data_batch['src_pred_disp']", len(data_batch['src_pred_disp']))

    # this for loop is for batch size
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

def save_entropy(data_batch, dir_name):
    os.makedirs(os.path.join(dir_name, './entropy'), exist_ok=True)
    entropy_dir = os.path.join(dir_name, 'entropy')

    for i in range(data_batch['src_shape_map'].shape[0]):
        shape_map = data_batch['src_shape_map'][i]
        
        # 여기서는 group의 평균을 계산하려고 했는데 이렇게 하는게 맞나? 이 중에서도 entropy가 높은 놈이 있을텐데 이걸 그냥 갈겨도 되나?
        shape_map_avg = shape_map.mean(dim=0, keepdim=True)  # 그룹 평균 계산

        shape_map_resized = F.interpolate(shape_map_avg, scale_factor=4, mode="bilinear", align_corners=False)
        # shape_map_resized = (shape_map_resized - shape_map_resized.min()) / (shape_map_resized.max() - shape_map_resized.min())
        shape_map_resized = shape_map_resized.squeeze(0).squeeze(0)
        source_left_filename = data_batch['source_left_filename'][i].split('/')[-1]

        plt.imsave(os.path.join(entropy_dir, source_left_filename), shape_map_resized.cpu().numpy(), cmap='gray')


def save_metrics(metrics, dir_name):
    total_epe = 0
    total_d1 = 0
    total_thres1 = 0
    total_thres2 = 0
    total_thres3 = 0
    count = 0

    for disp_metrics in metrics.values():
        if all(key in disp_metrics for key in ['EPE', 'D1', 'Thres1', 'Thres2', 'Thres3']):
            total_epe += disp_metrics['EPE'][0]
            total_d1 += disp_metrics['D1'][0]
            total_thres1 += disp_metrics['Thres1'][0]
            total_thres2 += disp_metrics['Thres2'][0]
            total_thres3 += disp_metrics['Thres3'][0]
            count += 1

    averages = {
        'average_metric' : {
            'EPE' : total_epe / count if count > 0 else 0,
            'D1' : total_d1 / count if count > 0 else 0,
            'Thres1' : total_thres1 / count if count > 0 else 0,
            'Thres2' : total_thres2 / count if count > 0 else 0,
            'Thres3' : total_thres3 / count if count > 0 else 0
        }
    }

    metrics.update(averages)
    metrics_dir = os.path.join(dir_name, 'metrics')
    os.makedirs(metrics_dir, exist_ok=True)
    with open(os.path.join(metrics_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)