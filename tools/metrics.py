import torch
from torch import Tensor
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


def make_nograd_func(func):
    def wrapper(*f_args, **f_kwargs):
        with torch.no_grad():
            ret = func(*f_args, **f_kwargs)
        return ret

    return wrapper


def check_shape_for_metric_computation(*vars):
    assert isinstance(vars, tuple)
    for var in vars:
        assert len(var.size()) == 3
        assert var.size() == vars[0].size()


def compute_metric_for_each_image(metric_func):
    def wrapper(D_ests, D_gts, *nargs):  # masks 제거
        # check_shape_for_metric_computation에서 masks도 제거 필요
        assert isinstance(D_ests, torch.Tensor) and isinstance(D_gts, torch.Tensor)
        assert len(D_ests.size()) == 3 and len(D_gts.size()) == 3
        assert D_ests.size() == D_gts.size()
        
        bn = D_gts.shape[0]  # batch size
        results = []
        for idx in range(bn):
            cur_nargs = [x[idx] if isinstance(x, (Tensor, Variable)) else x for x in nargs]
            ret = metric_func(D_ests[idx], D_gts[idx], *cur_nargs)  # masks 전달 안 함
            results.append(ret)
        if len(results) == 0:
            return torch.tensor(0, dtype=torch.float32, device=D_gts.device)
        else:
            return torch.stack(results).mean()
    return wrapper


@make_nograd_func
@compute_metric_for_each_image
def EPE_metric(pred_disp, gt_disp, max_disp):
    valid_mask = (gt_disp > 0) & (gt_disp < max_disp)
    pred_disp, gt_disp = pred_disp[valid_mask], gt_disp[valid_mask]
    return F.l1_loss(pred_disp, gt_disp, size_average=True)


@make_nograd_func
@compute_metric_for_each_image
def D1_metric(D_est, D_gt, max_disp):
    valid_mask = (D_gt > 0) & (D_gt < max_disp)
    D_est, D_gt = D_est[valid_mask], D_gt[valid_mask]
    E = torch.abs(D_gt - D_est)
    err_mask = (E > 3) & (E / D_gt.abs() > 0.05)
    return torch.mean(err_mask.float())

@make_nograd_func
@compute_metric_for_each_image
def Thres_metric(D_est, D_gt, max_disp, thres):
    assert isinstance(thres, (int, float))
    valid_mask = (D_gt > 0) & (D_gt < max_disp)
    D_est, D_gt = D_est[valid_mask], D_gt[valid_mask]
    E = torch.abs(D_gt - D_est)
    err_mask = E > thres
    return torch.mean(err_mask.float())

def make_iterative_func(func):
    def wrapper(vars):
        if isinstance(vars, list):
            return [wrapper(x) for x in vars]
        elif isinstance(vars, tuple):
            return tuple([wrapper(x) for x in vars])
        elif isinstance(vars, dict):
            return {k: wrapper(v) for k, v in vars.items()}
        else:
            return func(vars)

    return wrapper

@make_iterative_func
def tensor2float(vars):
    if isinstance(vars, float):
        return vars
    elif isinstance(vars, torch.Tensor):
        return vars.data.item()
    else:
        raise NotImplementedError("invalid input type for tensor2float")