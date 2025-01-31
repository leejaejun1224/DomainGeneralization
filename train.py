import torch
import torch.nn as nn
from model.loss import get_loss



def calc_supervised_loss(self, pred_disp, gt_disp):
    # TODO: multi-scale prediction이라면 pred_disp가 리스트일 수 있음
    #       여기서는 단일 output만 있다고 가정
    mask = (gt_disp > 0) & (gt_disp < 192)
    # scale별 weight 예시
    weights = [1.0]
    return get_loss([pred_disp], [gt_disp], [mask], weights)

def calc_pseudo_loss(self, pred_disp, pseudo_disp, mask):
    # TODO: multi-scale prediction이라면 pred_disp가 리스트일 수 있음
    #       여기서는 단일 output만 있다고 가정
    weights = [1.0]
    return get_loss([pred_disp], [pseudo_disp], [mask], weights)


def compute_uda_loss(model, data_batch, cfg):
    src_left = data_batch['src_left']
    src_right = data_batch['src_right']
    src_disp_gt = data_batch['src_disp']

    tgt_left = data_batch['tgt_left']
    tgt_right = data_batch['tgt_right']

    threshold = cfg['uda']['threshold']

    src_pred = model(src_left, src_right)  
    supervised_loss = calc_supervised_loss(src_pred, src_disp_gt)

    with torch.no_grad():
        pseudo_disp, confidence_map = model.ema_forward(
            tgt_left, tgt_right, return_confidence=True
        )

    tgt_pred = model(tgt_left, tgt_right)

    pseudo_mask = (pseudo_disp > 0) & (pseudo_disp < 192) & (confidence_map > threshold)
    pseudo_loss = calc_pseudo_loss(tgt_pred, pseudo_disp, pseudo_mask)

    total_loss = supervised_loss + pseudo_loss

    # 로그용
    log_vars = {
        'loss': total_loss.item(),
        'supervised_loss': supervised_loss.item(),
        'pseudo_loss': pseudo_loss.item()
    }
    return total_loss, log_vars

    
