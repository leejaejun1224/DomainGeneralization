from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
from torch.autograd.function import Function
import torch.nn.functional as F
import numpy as np
import math


class DomainNorm(nn.Module):
    def __init__(self, channel, l2=True):
        super(DomainNorm, self).__init__()
        self.normalize = nn.InstanceNorm2d(channel, affine=False)
        self.l2 = l2
        self.weight = nn.Parameter(torch.ones(1, channel, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channel, 1, 1))
        self.weight.requires_grad = True
        self.bias.requires_grad = True
    
    def forward(self, x):
        x = self.normalize(x)
        if self.l2:
           x = F.normalize(x, p=2, dim=1)
        x = x * self.weight + self.bias
        return x
        
         
        
class BasicConv(nn.Module):

    def __init__(self, in_channels, out_channels, deconv=False, is_3d=False, bn=True, relu=True, drop_out = 0.0, **kwargs):
        super(BasicConv, self).__init__()

        self.relu = relu
        self.use_bn = bn
        self.use_dropout = drop_out > 0.0
        self.drop_prob = drop_out
        if is_3d:
            if deconv:
                self.conv = nn.ConvTranspose3d(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = nn.Conv3d(in_channels, out_channels, bias=False, **kwargs)
            self.bn = nn.BatchNorm3d(out_channels)
        else:
            if deconv:
                self.conv = nn.ConvTranspose2d(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
            # self.bn = nn.BatchNorm2d(out_channels)
            self.bn = DomainNorm(out_channels, l2=True)

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        if self.relu:
            x = nn.LeakyReLU()(x)#, inplace=True)
        return x



class Conv2x(nn.Module):

    def __init__(self, in_channels, out_channels, deconv=False, is_3d=False, concat=True, keep_concat=True, bn=True, relu=True, keep_dispc=False):
        super(Conv2x, self).__init__()
        self.concat = concat
        self.is_3d = is_3d 
        if deconv and is_3d: 
            kernel = (4, 4, 4)
        elif deconv:
            kernel = 4
        else:
            kernel = 3

        if deconv and is_3d and keep_dispc:
            kernel = (1, 4, 4)
            stride = (1, 2, 2)
            padding = (0, 1, 1)
            self.conv1 = BasicConv(in_channels, out_channels, deconv, is_3d, bn=True, relu=True, kernel_size=kernel, stride=stride, padding=padding)
        else:
            self.conv1 = BasicConv(in_channels, out_channels, deconv, is_3d, bn=True, relu=True, kernel_size=kernel, stride=2, padding=1)

        if self.concat: 
            mul = 2 if keep_concat else 1
            self.conv2 = BasicConv(out_channels*2, out_channels*mul, False, is_3d, bn, relu, kernel_size=3, stride=1, padding=1)
        else:
            self.conv2 = BasicConv(out_channels, out_channels, False, is_3d, bn, relu, kernel_size=3, stride=1, padding=1)

    def forward(self, x, rem):
        x = self.conv1(x)
        if x.shape != rem.shape:
            x = F.interpolate(
                x,
                size=(rem.shape[-2], rem.shape[-1]),
                mode='nearest')
        if self.concat:
            x = torch.cat((x, rem), 1)
        else: 
            x = x + rem
        x = self.conv2(x)
        return x



def disparity_regression(x, maxdisp):
    assert len(x.shape) == 4
    disp_values = torch.arange(0, maxdisp, dtype=x.dtype, device=x.device)
    disp_values = disp_values.view(1, maxdisp, 1, 1)
    return torch.sum(x * disp_values, 1, keepdim=False)


def build_concat_volume(refimg_fea, targetimg_fea, maxdisp):
    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, 2 * C, maxdisp, H, W])
    for i in range(maxdisp):
        if i > 0:
            volume[:, :C, i, :, :] = refimg_fea[:, :, :, :]
            volume[:, C:, i, :, i:] = targetimg_fea[:, :, :, :-i]
        else:
            volume[:, :C, i, :, :] = refimg_fea
            volume[:, C:, i, :, :] = targetimg_fea
    volume = volume.contiguous()
    return volume


def groupwise_correlation(fea1, fea2, num_groups):
    B, C, H, W = fea1.shape
    assert C % num_groups == 0
    channels_per_group = C // num_groups
    cost = (fea1 * fea2).view([B, num_groups, channels_per_group, H, W]).mean(dim=2)
    assert cost.shape == (B, num_groups, H, W)
    return cost

def build_gwc_volume(refimg_fea, targetimg_fea, maxdisp, num_groups):
    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, num_groups, maxdisp, H, W])
    for i in range(maxdisp):
        if i > 0:
            volume[:, :, i, :, i:] = groupwise_correlation(refimg_fea[:, :, :, i:], targetimg_fea[:, :, :, :-i],
                                                           num_groups)
        else:
            volume[:, :, i, :, :] = groupwise_correlation(refimg_fea, targetimg_fea, num_groups)
    volume = volume.contiguous()
    return volume

def groupwise_correlation_norm(fea1, fea2, num_groups):
    B, C, H, W = fea1.shape
    assert C % num_groups == 0
    channels_per_group = C // num_groups
    fea1 = fea1.view([B, num_groups, channels_per_group, H, W])
    fea2 = fea2.view([B, num_groups, channels_per_group, H, W])
    cost = ((fea1/(torch.norm(fea1, 2, 2, True)+1e-05)) * (fea2/(torch.norm(fea2, 2, 2, True)+1e-05))).mean(dim=2)
    assert cost.shape == (B, num_groups, H, W)
    return cost


def build_gwc_volume_norm(refimg_fea, targetimg_fea, maxdisp, num_groups):
    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, num_groups, maxdisp, H, W])
    for i in range(maxdisp):
        if i > 0:
            volume[:, :, i, :, i:] = groupwise_correlation_norm(refimg_fea[:, :, :, i:], targetimg_fea[:, :, :, :-i],
                                                           num_groups)
        else:
            volume[:, :, i, :, :] = groupwise_correlation_norm(refimg_fea, targetimg_fea, num_groups)
    volume = volume.contiguous()
    return volume


def norm_correlation(fea1, fea2):
    cost = torch.mean(((fea1/(torch.norm(fea1, 2, 1, True)+1e-05)) * (fea2/(torch.norm(fea2, 2, 1, True)+1e-05))), dim=1, keepdim=True)
    return cost

def build_norm_correlation_volume(refimg_fea, targetimg_fea, maxdisp, mode='left'):
    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, 1, maxdisp, H, W])
    if mode == 'left':
        for i in range(maxdisp):
            if i > 0:
                volume[:, :, i, :, i:] = norm_correlation(refimg_fea[:, :, :, i:], targetimg_fea[:, :, :, :-i])
            else:
                volume[:, :, i, :, :] = norm_correlation(refimg_fea, targetimg_fea)
        volume = volume.contiguous()
        return volume
    
    else:
        for i in range(maxdisp):
            if i > 0:
                volume[:, :, i, :, i:] = norm_correlation(refimg_fea[:, :, :, :-i], targetimg_fea[:, :, :, i:])
            else:
                volume[:, :, i, :, :] = norm_correlation(refimg_fea, targetimg_fea) 
        return volume


def build_weighted_cost_volume(refimg_fea, targetimg_fea, mask_pred_L, mask_pred_R, maxdisp):
    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, 1, maxdisp, H, W])
    alpha = 0.5
    eps = 0.1
    for i in range(maxdisp):
        if i > 0:
            f1 = refimg_fea[:, :, :, i:]
            f2 = targetimg_fea[:, :, :, :-i]
            corr = norm_correlation(f1, f2)
            left_conf = mask_pred_L[:, :, :, i:]
            right_conf = mask_pred_R[:, :, :, :-i]
            w_raw = left_conf * right_conf
            w = eps + (1.0 - eps)*w_raw.pow(alpha)
            volume[:, :, i, :, i:] = corr * w
            # print((corr*w).abs().mean()/corr.abs().mean())
            # print(w.mean(), w.std(), w.min(), w.max())
        else:
            corr = norm_correlation(refimg_fea, targetimg_fea)
            w_raw = mask_pred_L * mask_pred_R
            w = eps + (1.0 - eps)*w_raw.pow(alpha)
            volume[:, :, i, :, :] = corr * w
            # print((corr*w).abs().mean()/corr.abs().mean())
            # print(w.mean(), w.std(), w.min(), w.max())


    volume = volume.contiguous()
    return volume


def disparity_variance(x, maxdisp, disparity):
    # the shape of disparity should be B,1,H,W, return is the variance of the cost volume [B,1,H,W]
    assert len(x.shape) == 4
    disp_values = torch.arange(0, maxdisp, dtype=x.dtype, device=x.device)
    disp_values = disp_values.view(1, maxdisp, 1, 1)
    disp_values = (disp_values - disparity) ** 2
    return torch.sum(x * disp_values, 1, keepdim=True)

def SpatialTransformer_grid(x, y, disp_range_samples):

    bs, channels, height, width = y.size()
    ndisp = disp_range_samples.size()[1]

    mh, mw = torch.meshgrid([torch.arange(0, height, dtype=x.dtype, device=x.device),
                                 torch.arange(0, width, dtype=x.dtype, device=x.device)])  # (H *W)

    mh = mh.reshape(1, 1, height, width).repeat(bs, ndisp, 1, 1)
    mw = mw.reshape(1, 1, height, width).repeat(bs, ndisp, 1, 1)  # (B, D, H, W)

    cur_disp_coords_y = mh
    cur_disp_coords_x = mw - disp_range_samples

    coords_x = cur_disp_coords_x / ((width - 1.0) / 2.0) - 1.0  # trans to -1 - 1
    coords_y = cur_disp_coords_y / ((height - 1.0) / 2.0) - 1.0
    grid = torch.stack([coords_x, coords_y], dim=4) #(B, D, H, W, 2)

    y_warped = F.grid_sample(y, grid.view(bs, ndisp * height, width, 2), mode='bilinear',
                               padding_mode='zeros', align_corners=True).view(bs, channels, ndisp, height, width)  #(B, C, D, H, W)

    x_warped = x.unsqueeze(2).repeat(1, 1, ndisp, 1, 1) #(B, C, D, H, W)

    return y_warped, x_warped



def volume_entropy_softmax(volume, k=12, temperature=0.5, eps=1e-6):
    vol = volume.squeeze(1)  # [B, D, H, W]
    # 1) Top-K
    topk_vals, _ = torch.topk(vol, k=k, dim=1)  # [B, K, H, W]
    top_one, top_one_idx = torch.topk(vol, k=1, dim=1)  # [B, 1, H, W]
    scaled_topk_vals = topk_vals / temperature
    

    exp_vals = torch.exp(scaled_topk_vals)
    sum_exp = exp_vals.sum(dim=1, keepdim=True) + eps
    
    p = exp_vals / sum_exp  # [B, K, H, W]
    p = torch.clamp(p, eps, 1.0)
    
    # 4) 엔트로피 ㄱㄱ
    H = -(p * p.log()).sum(dim=1) - 2.484  # [B, H, W]
    H = H.unsqueeze(1)
    mask = H < 0.0009
    # H = H * mask
    top_one_idx = top_one_idx*mask
    top_one_idx = top_one_idx.unsqueeze(1)
    # H = H.unsqueeze(1).unsqueeze(1)
    return top_one_idx, H, mask



def peak_confidence_from_volume(volume):
    """
    volume shape: [B, 1, D, H, W]
    return shape: [B, H, W], 각 픽셀에서 'peak - second_peak' 값을 리턴
    """
    # volume: [B, 1, D, H, W] -> [B, D, H, W] 로 squeeze
    vol = volume.squeeze(1)  # shape: [B, D, H, W]
    
    # D 차원(disparity 축)에 대해 최대값 및 argmax 구하기
    max_val, max_idx = vol.max(dim=1)  # shape: [B, H, W]
    
    # 두 번째 최댓값을 찾기 위해, 일단 최대값 위치에 매우 작은 값(-∞나 -1e9) 설정
    # 복사본을 만든 뒤 해당 위치만 -1e9로 세팅
    vol_clone = vol.clone()
    
    b_idxs = torch.arange(vol.shape[0])[:, None, None]     # shape: [B, 1, 1]
    h_idxs = torch.arange(vol.shape[2])[None, :, None]     # shape: [1, H, 1]
    w_idxs = torch.arange(vol.shape[3])[None, None, :]     # shape: [1, 1, W]
    # vol_clone[b, max_idx, h, w] 위치를 아주 작은 값으로 만들어버림
    vol_clone[b_idxs, max_idx, h_idxs, w_idxs] = -1e9
    
    # 이제 두 번째 최댓값을 구한다.
    second_max_val, _ = vol_clone.max(dim=1)  # shape: [B, H, W]
    
    # 최댓값과 두 번째 최댓값의 차이를 confidence로 사용
    peak_confidence = max_val - second_max_val  # shape: [B, H, W]
    mask = peak_confidence < 0.015
    peak_confidence = peak_confidence * mask
    peak_confidence = peak_confidence.unsqueeze(1).unsqueeze(1)
    return peak_confidence


def compute_disparity_from_cost(cost_volume, dim=2, multiplier=4, top_k=1):
    B, C, D, H, W = cost_volume.shape
    
    # Softmax 적용하여 확률 분포 생성
    prob = F.softmax(cost_volume, dim=dim)  # shape: [B, 1, D, H, W]

    # (1) Top-K 확률 및 해당 disparity index 가져오기
    topk_prob, topk_indices = torch.topk(prob, k=top_k, dim=dim)  # (B, 1, K, H, W)
    max_disparity = torch.max(topk_indices)
    print("Max disparity:", max_disparity.item())

    # (2) Top-K disparity index 값 그대로 사용 (이미 정수 값)
    disp_map = torch.sum(topk_prob * topk_indices, dim=dim, keepdim=False)

    # (3) Max disparity 확인
    max_disp = torch.max(disp_map)

    return disp_map







def cost_volume_distribution(cost_volume, dim=2):
    mean = torch.mean(cost_volume, dim=dim, keepdim=True)
    max, _ = torch.max(cost_volume, dim=dim)
    max, _ = torch.max(cost_volume, dim=2)
    # print(max)
    maxmean_distribution = max / (mean + 1e-8)
    return maxmean_distribution


class Propagation(nn.Module):
    def __init__(self):
        super(Propagation, self).__init__()
        self.replicationpad = nn.ReplicationPad2d(1)

    def forward(self, disparity_samples):

        one_hot_filter = torch.zeros(5, 1, 3, 3, device=disparity_samples.device).float()
        one_hot_filter[0, 0, 0, 0] = 1.0
        one_hot_filter[1, 0, 1, 1] = 1.0
        one_hot_filter[2, 0, 2, 2] = 1.0
        one_hot_filter[3, 0, 2, 0] = 1.0
        one_hot_filter[4, 0, 0, 2] = 1.0
        disparity_samples = self.replicationpad(disparity_samples)
        aggregated_disparity_samples = F.conv2d(disparity_samples,
                                                    one_hot_filter,padding=0)
        return aggregated_disparity_samples
        

class Propagation_prob(nn.Module):
    def __init__(self):
        super(Propagation_prob, self).__init__()
        self.replicationpad = nn.ReplicationPad3d((1, 1, 1, 1, 0, 0))

    def forward(self, prob_volume):
        one_hot_filter = torch.zeros(5, 1, 1, 3, 3, device=prob_volume.device).float()
        one_hot_filter[0, 0, 0, 0, 0] = 1.0
        one_hot_filter[1, 0, 0, 1, 1] = 1.0
        one_hot_filter[2, 0, 0, 2, 2] = 1.0
        one_hot_filter[3, 0, 0, 2, 0] = 1.0
        one_hot_filter[4, 0, 0, 0, 2] = 1.0

        prob_volume = self.replicationpad(prob_volume)
        prob_volume_propa = F.conv3d(prob_volume, one_hot_filter,padding=0)


        return prob_volume_propa
        
        
def context_upsample(depth_low, up_weights):


    ###
    # cv (b,1,h,w)
    # sp (b,9,4*h,4*w)
    ###
    b, c, h, w = depth_low.shape
    depth_unfold = F.unfold(depth_low.reshape(b,c,h,w),3,1,1).reshape(b,-1,h,w)
    depth_unfold = F.interpolate(depth_unfold,(h*4,w*4),mode='nearest').reshape(b,9,h*4,w*4)
    depth = (depth_unfold*up_weights).sum(1)
    

    return depth



def regression_topk(cost, disparity_samples, k):

    _, ind = cost.sort(1, True)
    
    ## 2개를 뽑아
    pool_ind = ind[:, :k]
    
    # 그 2개의 index에 해당하는 cost를 추출해.
    cost = torch.gather(cost, 1, pool_ind)

    # 2개의 cost에 대해서 softmax로 prob를 추출해.
    # 그러면 여기에는 나머지는 0이고 어떤 cost가 높은 부분에 해당하는 prob가 담겨 있을 것임.

    prob = F.softmax(cost, 1)

    # 2개의 prob에 대해서 disparity sampling을 하고, 샘플링 된 disparity에 
    # 확률을 곱해줘서 sum을 한다.
    disparity_samples = torch.gather(disparity_samples, 1, pool_ind)    
    pred = torch.sum(disparity_samples * prob, dim=1, keepdim=True)

    return pred, prob


def normalize_channelwise(f, eps=1e-5):
    mean = f.mean(dim=1, keepdim=True)
    std = f.std(dim=1, keepdim=True)
    f = (f-mean)/std
    f = F.normalize(f, p=2, dim=1, eps=eps)
    return f

def cosine_corr(ref, tgt):
    return (ref*tgt).sum(dim=1, keepdim=True)

def build_norm_correlation_volume_dn(ref, tgt, masdisp, as_prob=True, temperature=1.0, eps=1e-5):
    B, C, H, W = ref.shape

    ref = normalize_channelwise(ref, eps=eps)
    tgt = normalize_channelwise(tgt, eps=eps)
    
    volume = ref.new_full([B, 1, masdisp, H, W], fill_value=-1.0)
    mask = ref.new_ones([B, 1, masdisp, H, W], dtype=torch.bool)
    
    for i in range(masdisp):
        if i > 0:
            sim = cosine_corr(ref[:, :, :, i:], tgt[:,:,:,:-i])
            volume[:,:,i, :, i:] = sim
            mask[:,:,i,:,i:] = True
            
        else:
            sim = cosine_corr(ref=ref, tgt=tgt)
            volume[:,:,i] = sim
            mask[:,:,i]=True

    valid = mask.float()
    cnt = valid.sum(dim=2, keepdim=True).clamp_min(1.0)
    mean = (volume*valid).sum(dim=2, keepdim=True) / cnt
    var = ((volume-mean)**2 * valid).sum(dim=2, keepdim=True) / cnt
    std = var.sqrt().clamp_min(eps)
    zvol = (volume - mean) / std
    
    if not as_prob:
        zvol = torch.where(mask, zvol, zvol.new_fill((),-1.0))
        return zvol.contiguous()
    
    logits = zvol / float(temperature)
    logits = logits.masked_fill(~mask, float('-inf'))
    prob = torch.softmax(logits, dim=2)


    return prob.contiguous()