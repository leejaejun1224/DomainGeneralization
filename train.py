import torch 
import torch.nn as nn
import torch.nn.functional as F
from models.losses.loss import get_loss, calc_supervised_train_loss, calc_supervised_test_loss, calc_pseudo_loss


def compute_uda_loss(model, data_batch, cfg, train=True):
    src_left = data_batch['src_left']
    src_right = data_batch['src_right']
    src_disp_gt = data_batch['src_disparity']

    tgt_left = data_batch['tgt_left']
    tgt_right = data_batch['tgt_right']

    threshold = cfg['uda']['threshold']

    # print(model.__class__.__name__)
    src_pred = model(src_left, src_right)
    data_batch['src_pred_disp'] = src_pred


    if train:
        src_disp_gt_low = data_batch['src_disparity_low']
        supervised_loss = calc_supervised_train_loss(data_batch)
    else:
        supervised_loss = calc_supervised_test_loss(data_batch)


    with torch.no_grad():
        pseudo_disp, confidence_map = model.ema_forward(
            tgt_left, tgt_right, return_confidence=True
        )
        data_batch['pseudo_disp'] = pseudo_disp
        data_batch['confidence_map'] = confidence_map

    tgt_pred = model(tgt_left, tgt_right)
    data_batch['tgt_pred_disp'] = tgt_pred

    pseudo_loss = calc_pseudo_loss(data_batch, threshold)
    total_loss = supervised_loss + pseudo_loss

    # print("supervised loss : ", supervised_loss)
    # print("pseudo loss : ", pseudo_loss)
    # 로그용
    log_vars = {
        'loss': total_loss.item(),
        'supervised_loss': supervised_loss.item(),
        'unsupervised_loss': pseudo_loss.item()
    }

    
    return total_loss, log_vars