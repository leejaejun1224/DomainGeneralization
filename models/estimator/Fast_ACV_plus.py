from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
from models.estimator.submodules import *
from models.estimator.refiner import *
from models.estimator.adaptor2 import *
from models.estimator.attention_modules.semantic_attn import *
from models.estimator.occlusion import *
from models.estimator.refine.bandwith import *
import math
import gc
import time
import timm
from models.encoder.MiTbackbone import MixVisionTransformer
from transformers import SegformerModel

# =========================
# 추가: 워핑/SSIM/정규화 손실/정제 헤드
# =========================

def _make_base_grid(B, H, W, device, dtype):
    ys, xs = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing='ij'
    )
    xs = xs.unsqueeze(0).expand(B, -1, -1)
    ys = ys.unsqueeze(0).expand(B, -1, -1)
    return xs, ys

def _to_normalized_grid(x_pix, y_pix, H, W):
    gx = 2.0 * x_pix / (W - 1) - 1.0
    gy = 2.0 * y_pix / (H - 1) - 1.0
    return torch.stack([gx, gy], dim=-1)

def warp_right_to_left(img_right, disp_left, padding_mode='border'):
    """
    img_right: [B,3,H,W], disp_left: [B,1,H,W] (px)
    반환: warped_right [B,3,H,W], valid_mask [B,1,H,W]
    """
    B, C, H, W = img_right.shape
    device, dtype = img_right.device, img_right.dtype
    xs, ys = _make_base_grid(B, H, W, device, dtype)
    x_src = xs - disp_left.squeeze(1)
    y_src = ys
    grid = _to_normalized_grid(x_src, y_src, H, W)
    warped = F.grid_sample(img_right, grid, mode='bilinear',
                           padding_mode=padding_mode, align_corners=True)
    valid_x = (grid[..., 0] >= -1.0) & (grid[..., 0] <= 1.0)
    valid_y = (grid[..., 1] >= -1.0) & (grid[..., 1] <= 1.0)
    valid = (valid_x & valid_y).unsqueeze(1).float()
    return warped, valid

def warp_right_disp_to_left(disp_right, disp_left, padding_mode='border'):
    B, C, H, W = disp_right.shape
    device, dtype = disp_right.device, disp_right.dtype
    xs, ys = _make_base_grid(B, H, W, device, dtype)
    x_src = xs - disp_left.squeeze(1)
    y_src = ys
    grid = _to_normalized_grid(x_src, y_src, H, W)
    warped = F.grid_sample(disp_right, grid, mode='bilinear',
                           padding_mode=padding_mode, align_corners=True)
    valid_x = (grid[..., 0] >= -1.0) & (grid[..., 0] <= 1.0)
    valid_y = (grid[..., 1] >= -1.0) & (grid[..., 1] <= 1.0)
    valid = (valid_x & valid_y).unsqueeze(1).float()
    return warped, valid

class SSIM(nn.Module):
    def __init__(self, kernel_size=3, C1=0.01**2, C2=0.03**2):
        super().__init__()
        self.C1 = C1
        self.C2 = C2
        self.pad = kernel_size // 2
        self.pool = nn.AvgPool2d(kernel_size, stride=1, padding=self.pad, count_include_pad=False)

    def forward(self, x, y):
        mu_x = self.pool(x); mu_y = self.pool(y)
        sigma_x  = self.pool(x*x) - mu_x*mu_x
        sigma_y  = self.pool(y*y) - mu_y*mu_y
        sigma_xy = self.pool(x*y) - mu_x*mu_y
        num = (2*mu_x*mu_y + self.C1) * (2*sigma_xy + self.C2)
        den = (mu_x*mu_x + mu_y*mu_y + self.C1) * (sigma_x + sigma_y + self.C2)
        ssim = num / (den + 1e-12)
        return torch.clamp((1 - ssim) / 2, 0, 1)  # dissimilarity map

def charbonnier(x, eps=1e-3):
    return torch.sqrt(x*x + eps*eps)

class StereoRegularizationLoss(nn.Module):
    """
    Photometric(SSIM+Charbonnier) + Left-Right Consistency
    - nonocc_weight: [B,1,H,W], 0~1; in-bounds valid에 곱해 사용
    """
    def __init__(self, alpha_ssim=0.85, w_photo=1.0, w_lr=0.2, occlusion_tau=1.0):
        super().__init__()
        self.alpha = alpha_ssim
        self.w_photo = w_photo
        self.w_lr = w_lr
        self.occlusion_tau = occlusion_tau
        self.ssim = SSIM()

    def forward(self, left, right, d_left, d_right=None, nonocc_weight=None):
        right_warp, valid = warp_right_to_left(right, d_left, padding_mode='border')
        l1_map   = charbonnier(torch.abs(left - right_warp)).mean(1, keepdim=True)
        ssim_map = self.ssim(left, right_warp).mean(1, keepdim=True)
        photo_map = self.alpha * ssim_map + (1 - self.alpha) * l1_map

        if nonocc_weight is not None:
            valid = valid * nonocc_weight

        photo_loss = (photo_map * valid).sum() / (valid.sum() + 1e-6)

        if d_right is not None and self.w_lr > 0:
            d_right_warp, valid_r = warp_right_disp_to_left(d_right, d_left, padding_mode='border')
            lr_map = charbonnier(torch.abs(d_left - d_right_warp))
            valid_lr = valid * valid_r
            lr_loss = (lr_map * valid_lr).sum() / (valid_lr.sum() + 1e-6)
        else:
            lr_loss = torch.tensor(0.0, device=left.device, dtype=left.dtype)

        total = self.w_photo * photo_loss + self.w_lr * lr_loss
        return {'loss': total, 'photo': photo_loss.detach(), 'lr': lr_loss.detach(), 'valid_ratio': valid.mean().detach()}

class ResBlock(nn.Module):
    def __init__(self, ch, dilation=1):
        super().__init__()
        pad = dilation
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=pad, dilation=dilation, bias=False)
        self.bn1   = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=pad, dilation=dilation, bias=False)
        self.bn2   = nn.BatchNorm2d(ch)
        self.relu  = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(x + out)

class DisparityRefinement(nn.Module):
    """
    입력: left,right(0~1), disp_init(px, full-res)
    출력: disp_refined, aux(dict: warped_right, valid_mask, delta)
    """
    def __init__(self, base_ch=64, num_blocks=5, use_error_map=True, max_residual=1.5):
        super().__init__()
        self.use_error_map = use_error_map
        self.max_residual = max_residual
        in_ch = 3 + 3 + 1 + (1 if use_error_map else 0)

        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, 3, padding=1, bias=False),
            nn.GroupNorm(base_ch), nn.ReLU(inplace=True),
            nn.Conv2d(base_ch, base_ch, 3, padding=1, bias=False),
            nn.GroupNorm(base_ch), nn.ReLU(inplace=True),
        )
        dilations = [1, 1, 2, 4, 1][:num_blocks]
        self.blocks = nn.Sequential(*[ResBlock(base_ch, d) for d in dilations])
        self.head = nn.Sequential(
            nn.Conv2d(base_ch, base_ch//2, 3, padding=1, bias=False),
            nn.GroupNorm(base_ch//2), nn.ReLU(inplace=True),
            nn.Conv2d(base_ch//2, 1, 3, padding=1, bias=True)
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, left, right, disp_init):
        right_warp, valid = warp_right_to_left(right, disp_init, padding_mode='border')
        if self.use_error_map:
            err = torch.abs(left - right_warp).mean(1, keepdim=True)
            feat = torch.cat([left, right_warp, disp_init.unsqueeze(1), err], dim=1)
        else:
            feat = torch.cat([left, right_warp, disp_init], dim=1)

        f = self.stem(feat)
        f = self.blocks(f)
        delta = self.head(f)
        delta = self.max_residual * torch.tanh(delta)
        disp_refined = disp_init + delta
        return disp_refined, {'warped_right': right_warp, 'valid_mask': valid, 'delta': delta}


# =========================
# 기존 코드 (Feature/Hourglass 등)
# =========================

class SubModule(nn.Module):
    def __init__(self):
        super(SubModule, self).__init__()

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1); m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1); m.bias.data.zero_()

class FeatureMiTPtr(SubModule):
    def __init__(self):
        super(FeatureMiTPtr, self).__init__()
        self.encoder = SegformerModel.from_pretrained('nvidia/segformer-b0-finetuned-cityscapes-512-1024').encoder

    def forward(self, x):
        outputs = self.encoder(x, output_hidden_states=True, output_attentions=False)
        return outputs.hidden_states, None

class FeatUp(SubModule):
    def __init__(self):
        super(FeatUp, self).__init__()
        chans = [32, 64, 160, 256]
        self.deconv32_16 = Conv2x(chans[3], chans[2], deconv=True, concat=True)
        self.deconv16_8  = Conv2x(chans[2]*2, chans[1], deconv=True, concat=True)
        self.deconv8_4   = Conv2x(chans[1]*2, chans[0], deconv=True, concat=True)
        self.conv4       = BasicConv(chans[0]*2, chans[0], kernel_size=3, stride=1, padding=1)
        self.weight_init()

    def forward(self, featL, featR=None):
        x4, x8, x16, x32 = featL
        y4, y8, y16, y32 = featR
        x16 = self.deconv32_16(x32, x16); y16 = self.deconv32_16(y32, y16)
        x8  = self.deconv16_8(x16, x8);  y8  = self.deconv16_8(y16, y8)
        x4  = self.deconv8_4(x8, x4);    y4  = self.deconv8_4(y8, y4)
        x4  = self.conv4(x4);             y4  = self.conv4(y4)
        return [x4, x8, x16, x32], [y4, y8, y16, y32]

class channelAtt(SubModule):
    def __init__(self, cv_chan, im_chan):
        super(channelAtt, self).__init__()
        self.im_att = nn.Sequential(
            BasicConv(im_chan, im_chan//2, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(im_chan//2, cv_chan, 1)
        )
        self.weight_init()
    def forward(self, cv, im):
        channel_att = self.im_att(im).unsqueeze(2)
        return torch.sigmoid(channel_att) * cv

class hourglass(nn.Module):
    def __init__(self, in_channels):
        super(hourglass, self).__init__()
        self.conv1 = nn.Sequential(BasicConv(in_channels, in_channels*2, is_3d=True, bn=True, relu=True, kernel_size=3, padding=1, stride=2, dilation=1),
                                   BasicConv(in_channels*2, in_channels*2, is_3d=True, bn=True, relu=True, kernel_size=3, padding=1, stride=1, dilation=1))
        self.conv2 = nn.Sequential(BasicConv(in_channels*2, in_channels*4, is_3d=True, bn=True, relu=True, kernel_size=3, padding=1, stride=2, dilation=1),
                                   BasicConv(in_channels*4, in_channels*4, is_3d=True, bn=True, relu=True, kernel_size=3, padding=1, stride=1, dilation=1),
                                   nn.Dropout3d(0.0))
        self.conv2_up = BasicConv(in_channels*4, in_channels*2, deconv=True, is_3d=True, bn=True, relu=True, kernel_size=(4,4,4), padding=(1,1,1), stride=(2,2,2))
        self.conv1_up = BasicConv(in_channels*2, 1, deconv=True, is_3d=True, bn=False, relu=False, kernel_size=(4,4,4), padding=(1,1,1), stride=(2,2,2))
        self.agg = nn.Sequential(BasicConv(in_channels*4, in_channels*2, is_3d=True, kernel_size=1, padding=0, stride=1),
                                 BasicConv(in_channels*2, in_channels*2, is_3d=True, kernel_size=3, padding=1, stride=1),
                                 nn.Dropout3d(0.0),
                                 BasicConv(in_channels*2, in_channels*2, is_3d=True, kernel_size=3, padding=1, stride=1))
        self.feature_att_8    = channelAtt(in_channels*2, 128)
        self.feature_att_16   = channelAtt(in_channels*4, 320)
        self.feature_att_up_8 = channelAtt(in_channels*2, 128)

    def forward(self, x, imgs):
        conv1 = self.feature_att_8(self.conv1(x), imgs[1])
        conv2 = self.feature_att_16(self.conv2(conv1), imgs[2])
        conv1 = torch.cat((self.conv2_up(conv2), conv1), dim=1)
        conv1 = self.feature_att_up_8(self.agg(conv1), imgs[1])
        return self.conv1_up(conv1)

class hourglass_att(nn.Module):
    def __init__(self, in_channels):
        super(hourglass_att, self).__init__()
        self.conv1 = nn.Sequential(BasicConv(in_channels, in_channels*2, is_3d=True, bn=True, relu=True, kernel_size=3, padding=1, stride=2, dilation=1),
                                   BasicConv(in_channels*2, in_channels*2, is_3d=True, bn=True, relu=True, kernel_size=3, padding=1, stride=1, dilation=1))
        self.conv2 = nn.Sequential(BasicConv(in_channels*2, in_channels*4, is_3d=True, bn=True, relu=True, kernel_size=3, padding=1, stride=2, dilation=1),
                                   BasicConv(in_channels*4, in_channels*4, is_3d=True, bn=True, relu=True, kernel_size=3, padding=1, stride=1, dilation=1),
                                   nn.Dropout3d(0.0))
        self.conv3 = nn.Sequential(BasicConv(in_channels*4, in_channels*6, is_3d=True, bn=True, relu=True, kernel_size=3, padding=1, stride=2, dilation=1),
                                   BasicConv(in_channels*6, in_channels*6, is_3d=True, bn=True, relu=True, kernel_size=3, padding=1, stride=1, dilation=1),
                                   nn.Dropout3d(0.0))
        self.conv3_up = BasicConv(in_channels*6, in_channels*4, deconv=True, is_3d=True, bn=True, relu=True, kernel_size=(4,4,4), padding=(1,1,1), stride=(2,2,2))
        self.conv2_up = BasicConv(in_channels*4, in_channels*2, deconv=True, is_3d=True, bn=True, relu=True, kernel_size=(4,4,4), padding=(1,1,1), stride=(2,2,2))
        self.conv1_up = BasicConv(in_channels*2, 1, deconv=True, is_3d=True, bn=False, relu=False, kernel_size=(4,4,4), padding=(1,1,1), stride=(2,2,2))
        self.agg_0 = nn.Sequential(BasicConv(in_channels*8, in_channels*4, is_3d=True, kernel_size=1, padding=0, stride=1),
                                   BasicConv(in_channels*4, in_channels*4, is_3d=True, kernel_size=3, padding=1, stride=1),
                                   nn.Dropout3d(0.0),
                                   BasicConv(in_channels*4, in_channels*4, is_3d=True, kernel_size=3, padding=1, stride=1))
        self.agg_1 = nn.Sequential(BasicConv(in_channels*4, in_channels*2, is_3d=True, kernel_size=1, padding=0, stride=1),
                                   BasicConv(in_channels*2, in_channels*2, is_3d=True, kernel_size=3, padding=1, stride=1),
                                   nn.Dropout3d(0.0),
                                   BasicConv(in_channels*2, in_channels*2, is_3d=True, kernel_size=3, padding=1, stride=1))
        self.feature_att_8       = channelAtt(in_channels*2, 128)
        self.feature_att_16      = channelAtt(in_channels*4, 320)
        self.feature_att_32      = channelAtt(in_channels*6, 256)
        self.feature_att_up_16   = channelAtt(in_channels*4, 320)
        self.feature_att_up_8    = channelAtt(in_channels*2, 128)

    def forward(self, x, imgs):
        conv1 = self.feature_att_8(self.conv1(x), imgs[1])
        conv2 = self.feature_att_16(self.conv2(conv1), imgs[2])
        conv3 = self.feature_att_32(self.conv3(conv2), imgs[3])
        conv2 = torch.cat((self.conv3_up(conv3), conv2), dim=1)
        conv2 = self.feature_att_up_16(self.agg_0(conv2), imgs[2])
        conv1 = torch.cat((self.conv2_up(conv2), conv1), dim=1)
        conv1 = self.feature_att_up_8(self.agg_1(conv1), imgs[1])
        return self.conv1_up(conv1)

# =========================
# 메인 네트워크
# =========================

class Fast_ACVNet_plus(nn.Module):
    def __init__(self, maxdisp, att_weights_only, enable_lora=True,
                 refine_base_ch=64, refine_blocks=5, refine_max_residual=4.0,
                 reg_alpha_ssim=0.85, reg_w_photo=1.0, reg_w_lr=0.2, reg_occ_tau=1.0):
        super(Fast_ACVNet_plus, self).__init__()
        self.maxdisp = maxdisp
        self.att_weights_only = att_weights_only
        self.feature = FeatureMiTPtr()
        self.feature_up = FeatUp()
        self.enable_lora = enable_lora

        self.stem_2 = nn.Sequential(
            BasicConv(3, 32, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            DomainNorm(32), nn.ReLU())
        self.stem_4 = nn.Sequential(
            BasicConv(32, 48, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(48, 48, 3, 1, 1, bias=False),
            DomainNorm(48), nn.ReLU())

        self.spx   = nn.Sequential(nn.ConvTranspose2d(2*32, 9, kernel_size=4, stride=2, padding=1))
        self.spx_2 = Conv2x(64, 32, True)
        self.spx_4 = nn.Sequential(
            BasicConv(80, 64, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(64, 64, 3, 1, 1, bias=False),
            DomainNorm(64), nn.ReLU())

        self.conv = BasicConv(80, 80, kernel_size=3, padding=1, stride=1)
        self.desc = nn.Conv2d(80, 80, kernel_size=1, padding=0, stride=1)
        self.desc_dropout = nn.Dropout2d(0.)

        self.corr_stem = BasicConv(1, 8, is_3d=True, kernel_size=3, stride=1, padding=1)
        self.corr_feature_att_4 = channelAtt(8, 80)
        self.hourglass_att = hourglass_att(8)

        self.concat_feature = nn.Sequential(
            BasicConv(80, 32, kernel_size=3, stride=1, padding=1),
            nn.Dropout2d(0.0),
            nn.Conv2d(32, 16, 3, 1, 1, bias=False))
        self.concat_stem = BasicConv(32, 16, is_3d=True, kernel_size=3, stride=1, padding=1)
        self.concat_feature_att_4 = channelAtt(16, 80)
        self.hourglass = hourglass(16)

        # if enable_lora:
        #     self.adaptor = Adaptor(self.corr_stem, self.corr_feature_att_4, self.hourglass_att,
        #                            adaptor_rank=16, adaptor_alpha=0.3)

        self.occ_head = OcclusionPredictor(feat_ch=80, use_corr=True, use_att=True)

        # === 추가: 정제 헤드 + 정규화 손실 ===
        self.refine_head = DisparityRefinement(base_ch=refine_base_ch,
                                               num_blocks=refine_blocks,
                                               use_error_map=True,
                                               max_residual=refine_max_residual)
        # self.reg_loss = StereoRegularizationLoss(alpha_ssim=reg_alpha_ssim,
        #                                          w_photo=reg_w_photo,
        #                                          w_lr=reg_w_lr,
        #                                          occlusion_tau=reg_occ_tau)
        self.occ_blend_alpha = 1.0  # 기본: GT 없으면 예측 마스크만 사용

        # 오클루전 감독 손실(옵션)
        self.bce_logits = nn.BCEWithLogitsLoss(reduction='mean')

    def set_occ_blend_alpha(self, alpha: float):
        """GT/non-GT 마스크 블렌딩 가중치(0=GT만, 1=예측만)."""
        self.occ_blend_alpha = float(alpha)

    def concat_volume_generator(self, left_input, right_input, disparity_samples):
        right_feature_map, left_feature_map = SpatialTransformer_grid(left_input, right_input, disparity_samples)
        concat_volume = torch.cat((left_feature_map, right_feature_map), dim=1)
        return concat_volume

    def forward(self, left, right, mode,
                gt_occ: torch.Tensor = None,   # [B,1,H,W], 0/1 (1=occ)
                d_right: torch.Tensor = None,  # [B,1,H,W] (있을 때만 LR 사용)
                occ_blend_alpha: float = None, # None이면 self.occ_blend_alpha
                return_reg: bool = False):

        feature_left,  _ = self.feature(left)
        feature_right, _ = self.feature(right)
        features_left, features_right = self.feature_up(feature_left, feature_right)

        stem_2x = self.stem_2(left);  stem_4x = self.stem_4(stem_2x)
        stem_2y = self.stem_2(right); stem_4y = self.stem_4(stem_2y)

        features_left_cat  = torch.cat((features_left[0],  stem_4x), 1)
        features_right_cat = torch.cat((features_right[0], stem_4y), 1)

        match_left  = self.desc_dropout(self.desc(self.conv(features_left_cat)))
        match_right = self.desc_dropout(self.desc(self.conv(features_right_cat)))

        corr_volume_1 = build_norm_correlation_volume_dn(match_left, match_right, self.maxdisp//4)

        if self.enable_lora:
            att_weights = self.adaptor(corr_volume_1, features_left_cat, features_left)
        else:
            corr_volume = self.corr_stem(corr_volume_1)
            cost_att = self.corr_feature_att_4(corr_volume, features_left_cat)
            features_left_for_att = [feat.detach() if self.training else feat for feat in features_left]
            att_weights = self.hourglass_att(cost_att, features_left_for_att)

        T = 1.0
        att_weights_prob = F.softmax(att_weights/T, dim=2)
        prob_flat = att_weights_prob.squeeze(1)
        top2_probs, idx_2 = prob_flat.topk(2, dim=1, largest=True)
        disp_diff = idx_2[:,1].float() - idx_2[:,0].float()

        _, ind = att_weights_prob.sort(2, True)
        k = 16
        ind_k = ind[:, :, :k]
        ind_k = ind_k.sort(2, False)[0]
        att_topk = torch.gather(att_weights_prob, 2, ind_k)
        disparity_sample_topk = ind_k.squeeze(1).float()

        if not self.att_weights_only:
            concat_features_left  = self.concat_feature(features_left_cat)
            concat_features_right = self.concat_feature(features_right_cat)
            concat_volume = self.concat_volume_generator(concat_features_left, concat_features_right, disparity_sample_topk)
            volume = att_topk * concat_volume
            volume = self.concat_stem(volume)
            volume = self.concat_feature_att_4(volume, features_left_cat)

            features_left_for_hg = [feat.detach() if self.training else feat for feat in features_left]
            cost = self.hourglass(volume, features_left_for_hg)

            xspx = self.spx_4(features_left_cat); xspx = self.spx_2(xspx, stem_2x)
            spx_pred = F.softmax(self.spx(xspx), 1)

            # 오클루전 예측
            occ_up, occ_1_4, occ_logit, _ = self.occ_head(
                features_left_cat,
                att_logits=att_weights,
                corr_volume=corr_volume_1,
                spx_pred=spx_pred
            )

        att_prob = torch.gather(att_weights, 2, ind_k).squeeze(1)
        att_prob = F.softmax(att_prob, dim=1)
        pred_att = torch.sum(att_prob * disparity_sample_topk, dim=1)
        pred_att_up = context_upsample(pred_att.unsqueeze(1), spx_pred)

        if self.att_weights_only:
            # 정제 없이 attention 기반 출력만
            return [pred_att_up * 4, pred_att * 4]

        # top-k 회귀
        pred, prob = regression_topk(cost.squeeze(1), disparity_sample_topk, 12)
        pred_up = context_upsample(pred, spx_pred)
        prob_up1 = context_upsample(prob[:,0,:,:].unsqueeze(1), spx_pred)
        prob_up2 = context_upsample(prob[:,1,:,:].unsqueeze(1), spx_pred)
        confidence = prob_up1 + prob_up2

        # ========= 정제(Refinement) =========
        out_full_px = pred_up * 4.0  # 기존 최종값(정제 입력)
        left_det  = left.detach()
        right_det = right.detach()
        out_full_px_det = out_full_px.detach()

        disp_refined, aux_ref = self.refine_head(left_det, right_det, out_full_px_det)

        # # ========= Photometric/LR 정규화 손실(선택) =========
        # reg_dict = None
        # if self.training:
        #     # 오클루전 마스킹: GT와 예측 블렌드
        #     if occ_blend_alpha is None:
        #         alpha = self.occ_blend_alpha if gt_occ is None else 0.0
        #     else:
        #         alpha = float(occ_blend_alpha)

        #     # 예측 마스크(detach) 사용
        #     occ_pred_soft = torch.sigmoid(occ_up).detach()  # [B,1,H,W]
        #     nonocc_pred = 1.0 - occ_pred_soft

        #     if gt_occ is not None:
        #         nonocc_gt = 1.0 - gt_occ.float()
        #         nonocc_weight = (1.0 - alpha) * nonocc_gt + alpha * nonocc_pred
        #     else:
        #         nonocc_weight = nonocc_pred

        #     reg = self.reg_loss(left, right, disp_refined, d_right=d_right, nonocc_weight=nonocc_weight)

        #     # 오클루전 감독(BCE), 가능한 경우 1/4 스케일로 비교
        #     loss_occ = None
        #     if gt_occ is not None:
        #         H4, W4 = occ_logit.shape[-2:]
        #         gt_occ_1_4 = F.interpolate(gt_occ.float(), size=(H4, W4), mode='nearest')
        #         loss_occ = self.bce_logits(occ_logit, gt_occ_1_4)
        #         reg['occ_bce'] = loss_occ.detach()

        #     reg_total = reg['loss'] + (0.0 if loss_occ is None else loss_occ)
        #     reg_dict = {
        #         'reg_total': reg_total,
        #         'photo': reg['photo'],
        #         'lr': reg['lr'],
        #         'valid_ratio': reg['valid_ratio'],
        #         'occ_bce': (torch.tensor(0.0, device=left.device) if loss_occ is None else loss_occ.detach())
        #     }

        # ========= 반환 =========
        # 1) 최종 풀해상도는 정제 결과로 교체(기존: pred_up*4)
        preds_main = [disp_refined.squeeze(1), pred.squeeze(1) * 4, pred_att_up * 4, pred_att * 4]
        pack_a = [disp_diff.detach(), corr_volume_1.detach(), confidence, corr_volume_1.detach()]
        pack_b = [None, att_weights.detach(), cost.detach(), match_left.detach(), match_right.detach()]
        pack_occ = [occ_up, occ_logit]

        # if return_reg and reg_dict is not None:
        #     return preds_main, pack_a, pack_b, pack_occ, reg_dict
        # else:
        return preds_main, pack_a, pack_b, pack_occ

    def freeze_original_network(self):
        for name, param in self.named_parameters():
            if 'adaptor' not in name and 'refine_head' not in name:
                param.requires_grad = False

    def unfreeze_all(self):
        for p in self.parameters():
            p.requires_grad = True
