from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
from models.estimator.submodules import *
import math
import gc
import time
import timm
from models.encoder.MiTbackbone import MixVisionTransformer
from transformers import SegformerModel


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
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class FeatureMiTPtr(SubModule):
    def __init__(self):
        super(FeatureMiTPtr, self).__init__()
        self.model = SegformerModel.from_pretrained('nvidia/segformer-b0-finetuned-ade-512-512')
        self.encoder = self.model.encoder

    def forward(self, x):
        outputs = self.encoder(x, output_hidden_states=True, output_attentions=True)
        return outputs.hidden_states, outputs.attentions

class FeatureMiT(SubModule):
    def __init__(self):
        super(FeatureMiT, self).__init__()
        self.model = MixVisionTransformer(img_size=[256, 512], in_chans=3, embed_dim=[32, 64, 160, 256],
                                      depth=[2, 2, 2, 2], num_heads=[1, 2, 4, 8], qkv_bias=True,
                                      qk_scale=1.0, sr_ratio=[8, 4, 2, 1], proj_drop=[0.0, 0.0, 0.0, 0.0], attn_drop=[0.0, 0.0, 0.0, 0.0],
                                      drop_path_rate=0.1)

    def forward(self, x):
        features, attn_weights, pos_encodings = self.model(x)
        return features, attn_weights, pos_encodings  # [stage1, stage2, stage3, stage4]


class Feature(SubModule):
    def __init__(self):
        super(Feature, self).__init__()
        pretrained =  True
        model = timm.create_model('mobilenetv2_100', pretrained=pretrained, features_only=True)
        layers = [1,2,3,5,6]
        chans = [16, 24, 32, 96, 160]
        self.conv_stem = model.conv_stem
        self.bn1 = model.bn1
        # self.act1 = model.act1
        self.act1 = nn.ReLU6()


        self.block0 = torch.nn.Sequential(*model.blocks[0:layers[0]])
        self.block1 = torch.nn.Sequential(*model.blocks[layers[0]:layers[1]])
        self.block2 = torch.nn.Sequential(*model.blocks[layers[1]:layers[2]])
        self.block3 = torch.nn.Sequential(*model.blocks[layers[2]:layers[3]])
        self.block4 = torch.nn.Sequential(*model.blocks[layers[3]:layers[4]])

    def forward(self, x):
        x = self.act1(self.bn1(self.conv_stem(x)))
        x2 = self.block0(x)
        x4 = self.block1(x2)
        x8 = self.block2(x4)
        x16 = self.block3(x8)
        x32 = self.block4(x16)
        return [x4, x8, x16, x32]

class FeatUp(SubModule):
    def __init__(self):
        super(FeatUp, self).__init__()
        chans = [32, 64, 160, 256]  # Segformer-B0 출력 채널
        self.deconv32_16 = Conv2x(chans[3], chans[2], deconv=True, concat=True)  # 256 -> 160
        self.deconv16_8 = Conv2x(chans[2]*2, chans[1], deconv=True, concat=True)  # 320 -> 64
        self.deconv8_4 = Conv2x(chans[1]*2, chans[0], deconv=True, concat=True)  # 128 -> 32
        self.conv4 = BasicConv(chans[0]*2, chans[0], kernel_size=3, stride=1, padding=1)  # 64 -> 32
        self.weight_init()

    def forward(self, featL, featR=None):
        x4, x8, x16, x32 = featL
        y4, y8, y16, y32 = featR
        x16 = self.deconv32_16(x32, x16)  # [256, H/16, W/16] + [160, H/16, W/16] -> [320, H/8, W/8]
        y16 = self.deconv32_16(y32, y16)
        x8 = self.deconv16_8(x16, x8)     # [320, H/8, W/8] + [64, H/8, W/8] -> [128, H/4, W/4]
        y8 = self.deconv16_8(y16, y8)
        x4 = self.deconv8_4(x8, x4)       # [128, H/4, W/4] + [32, H/4, W/4] -> [64, H/2, W/2]
        y4 = self.deconv8_4(y8, y4)
        x4 = self.conv4(x4)               # [64, H/2, W/2] -> [32, H/2, W/2]
        y4 = self.conv4(y4)
        return [x4, x8, x16, x32], [y4, y8, y16, y32]


class channelAtt(SubModule):
    def __init__(self, cv_chan, im_chan):
        super(channelAtt, self).__init__()

        self.im_att = nn.Sequential(
            BasicConv(im_chan, im_chan//2, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(im_chan//2, cv_chan, 1))

        self.weight_init()

    def forward(self, cv, im):
        channel_att = self.im_att(im).unsqueeze(2)
        cv = torch.sigmoid(channel_att)*cv
        return cv



class hourglass(nn.Module):
    def __init__(self, in_channels):
        super(hourglass, self).__init__()

        self.conv1 = nn.Sequential(BasicConv(in_channels, in_channels*2, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=2, dilation=1),
                                   BasicConv(in_channels*2, in_channels*2, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=1, dilation=1))
                                    
        self.conv2 = nn.Sequential(BasicConv(in_channels*2, in_channels*4, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=2, dilation=1),
                                   BasicConv(in_channels*4, in_channels*4, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=1, dilation=1))                             

        self.conv2_up = BasicConv(in_channels*4, in_channels*2, deconv=True, is_3d=True, bn=True,
                                  relu=True, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))
        self.conv1_up = BasicConv(in_channels*2, 1, deconv=True, is_3d=True, bn=False,
                                  relu=False, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        self.agg = nn.Sequential(BasicConv(in_channels*4, in_channels*2, is_3d=True, kernel_size=1, padding=0, stride=1),
                                 BasicConv(in_channels*2, in_channels*2, is_3d=True, kernel_size=3, padding=1, stride=1),
                                 BasicConv(in_channels*2, in_channels*2, is_3d=True, kernel_size=3, padding=1, stride=1))

        # 채널 수 조정: features_left[1]=128, features_left[2]=320
        self.feature_att_8 = channelAtt(in_channels*2, 128)    # 32 -> 128 (imgs[1])
        self.feature_att_16 = channelAtt(in_channels*4, 320)   # 64 -> 320 (imgs[2])
        self.feature_att_up_8 = channelAtt(in_channels*2, 128)  # 32 -> 128 (imgs[1])

    def forward(self, x, imgs):
        conv1 = self.conv1(x)
        conv1 = self.feature_att_8(conv1, imgs[1])

        conv2 = self.conv2(conv1)
        conv2 = self.feature_att_16(conv2, imgs[2])

        conv2_up = self.conv2_up(conv2)
        conv1 = torch.cat((conv2_up, conv1), dim=1)
        conv1 = self.agg(conv1)
        conv1 = self.feature_att_up_8(conv1, imgs[1])

        conv = self.conv1_up(conv1)
        return conv
    


class hourglass_att(nn.Module):
    def __init__(self, in_channels):
        super(hourglass_att, self).__init__()

        self.conv1 = nn.Sequential(BasicConv(in_channels, in_channels*2, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=2, dilation=1),
                                   BasicConv(in_channels*2, in_channels*2, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=1, dilation=1))
                                    
        self.conv2 = nn.Sequential(BasicConv(in_channels*2, in_channels*4, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=2, dilation=1),
                                   BasicConv(in_channels*4, in_channels*4, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=1, dilation=1))                             

        self.conv3 = nn.Sequential(BasicConv(in_channels*4, in_channels*6, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=2, dilation=1),
                                   BasicConv(in_channels*6, in_channels*6, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=1, dilation=1)) 

        self.conv3_up = BasicConv(in_channels*6, in_channels*4, deconv=True, is_3d=True, bn=True,
                                  relu=True, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))
        self.conv2_up = BasicConv(in_channels*4, in_channels*2, deconv=True, is_3d=True, bn=True,
                                  relu=True, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))
        self.conv1_up = BasicConv(in_channels*2, 1, deconv=True, is_3d=True, bn=False,
                                  relu=False, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        self.agg_0 = nn.Sequential(BasicConv(in_channels*8, in_channels*4, is_3d=True, kernel_size=1, padding=0, stride=1),
                                   BasicConv(in_channels*4, in_channels*4, is_3d=True, kernel_size=3, padding=1, stride=1),
                                   BasicConv(in_channels*4, in_channels*4, is_3d=True, kernel_size=3, padding=1, stride=1))
        self.agg_1 = nn.Sequential(BasicConv(in_channels*4, in_channels*2, is_3d=True, kernel_size=1, padding=0, stride=1),
                                   BasicConv(in_channels*2, in_channels*2, is_3d=True, kernel_size=3, padding=1, stride=1),
                                   BasicConv(in_channels*2, in_channels*2, is_3d=True, kernel_size=3, padding=1, stride=1))

        # 채널 수 조정: features_left[1]=128, features_left[2]=320, features_left[3]=256
        self.feature_att_8 = channelAtt(in_channels*2, 128)    # imgs[1] 채널 128
        self.feature_att_16 = channelAtt(in_channels*4, 320)   # imgs[2] 채널 320
        self.feature_att_32 = channelAtt(in_channels*6, 256)   # imgs[3] 채널 256
        self.feature_att_up_16 = channelAtt(in_channels*4, 320)  # imgs[2] 채널 320
        self.feature_att_up_8 = channelAtt(in_channels*2, 128)   # imgs[1] 채널 128

    def forward(self, x, imgs):
        conv1 = self.conv1(x)
        conv1 = self.feature_att_8(conv1, imgs[1])

        conv2 = self.conv2(conv1)
        conv2 = self.feature_att_16(conv2, imgs[2])

        conv3 = self.conv3(conv2)
        conv3 = self.feature_att_32(conv3, imgs[3])

        conv3_up = self.conv3_up(conv3)
        conv2 = torch.cat((conv3_up, conv2), dim=1)
        conv2 = self.agg_0(conv2)
        conv2 = self.feature_att_up_16(conv2, imgs[2])

        conv2_up = self.conv2_up(conv2)
        conv1 = torch.cat((conv2_up, conv1), dim=1)
        conv1 = self.agg_1(conv1)
        conv1 = self.feature_att_up_8(conv1, imgs[1])

        conv = self.conv1_up(conv1)
        return conv
    

"""
PropagationNetLarge v2
----------------------
* wider ASPP rates   : (3, 6, 12, 24)
* red–dilated stack  : dilations = [1, 2, 4, 8]
* depth‑aware texture–hierarchy attention
    ‑ `depth_prob` (32‑bin soft‑max from EfficientNet depth branch)
    ‑ sigmoid → channel–wise weights → feature × weights
* still takes a *soft* confidence map (entropy‑derived) as 1‑channel input.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------
class _ResDilBlock(nn.Module):
    """3×3 standard + 3×3 dilated conv for larger RF."""
    def __init__(self, ch, dil=2):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(ch, ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ch), nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(ch, ch, 3, 1, padding=dil, dilation=dil, bias=False),
            nn.BatchNorm2d(ch)
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(x + self.conv2(self.conv1(x)))

# ---------------------------------------------------------
class _ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling with custom rates."""
    def __init__(self, ch, out=256, rates=(3, 6, 12, 24)):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, out // len(rates), 3, 1, padding=r, dilation=r,
                          bias=False),
                nn.BatchNorm2d(out // len(rates)), nn.ReLU(inplace=True)
            ) for r in rates
        ])
        self.project = nn.Sequential(
            nn.Conv2d(out, ch, 1, bias=False),
            nn.BatchNorm2d(ch), nn.ReLU(inplace=True)
        )

    def forward(self, x):
        y = torch.cat([b(x) for b in self.branches], dim=1)
        return self.project(y)

# ---------------------------------------------------------
class _NonLocal(nn.Module):
    """Simple Non‑Local block (embedded Gaussian)."""
    def __init__(self, ch, red=2):
        super().__init__()
        self.theta = nn.Conv2d(ch, ch // red, 1, bias=False)
        self.phi   = nn.Conv2d(ch, ch // red, 1, bias=False)
        self.g     = nn.Conv2d(ch, ch // red, 1, bias=False)
        self.out   = nn.Conv2d(ch // red, ch, 1, bias=False)

    def forward(self, x):
        B, C, H, W = x.shape
        th = self.theta(x).reshape(B, -1, H * W)           # [B, C', N]
        ph = self.phi(x).reshape(B, -1, H * W)             # [B, C', N]
        g  = self.g(x).reshape(B, -1, H * W)               # [B, C', N]
        attn = torch.softmax(torch.bmm(th.transpose(1, 2), ph), dim=-1)  # [B, N, N]
        y = torch.bmm(g, attn.transpose(1, 2)).reshape(B, -1, H, W)      # [B, C', H, W]
        return x + self.out(y)

# ---------------------------------------------------------
class DepthTextureAtt(nn.Module):
    """
    Texture‑hierarchy attention (논문 식 7–8)
      depth_prob : [B, 32, H, W]  (channel‑softmax from depth branch)
      feat       : [B, C,  H, W]
    """
    def __init__(self, feat_ch):
        super().__init__()
        self.feat_ch = feat_ch
        self.conv = nn.Sequential(
            nn.Conv2d(32, 32, 1, 1, 0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 1, 1, 0, bias=False),
            nn.ReLU(inplace=True)
            )
    def forward(self, feat, depth_prob):
        depth_prob = self.conv(depth_prob)
        # sigmoid → channel weights A  (0–1)
        A = torch.sigmoid(depth_prob)                  # [B,32,H,W]
        # broadcast to match feat channels (32 bins ≥ feat_ch)
        feat_w = feat * A[:, :self.feat_ch]            # element‑wise
        return feat_w

# ---------------------------------------------------------
class PropagationNetLarge(nn.Module):
    """
    Args
    ----
      feat_ch : input feature channels (32 by default)

    Inputs
    ------
      feat   : [B, feat_ch, H, W]
      conf   : [B, 1,        H, W]   (soft confidence 0‑1)
      depth_prob (optional) : [B,32,H,W]  – depth‑branch soft‑max

    Output
    ------
      feat_ctx : [B, feat_ch, H, W]  (context‑enhanced feature)
    """
    def __init__(self, feat_ch: int = 32):
        super().__init__()
        in_ch = feat_ch + 1              # concatenate confidence map
        base  = 128

        # ---------- Encoder ----------
        self.down1 = nn.Sequential(
            nn.Conv2d(in_ch, base, 7, 1, 3, bias=False),
            nn.BatchNorm2d(base), nn.ReLU(inplace=True)
        )
        self.res1 = _ResDilBlock(base, dil=2)

        self.down2 = nn.Sequential(
            nn.Conv2d(base, base * 2, 3, 2, 1, bias=False),
            nn.BatchNorm2d(base * 2), nn.ReLU(inplace=True)
        )
        self.res2 = _ResDilBlock(base * 2, dil=4)

        # ---------- Bottleneck ----------
        self.aspp = _ASPP(base * 2, rates=(3, 6, 12, 24))
        self.nl   = _NonLocal(base * 2)

        # red‑dilated conv stack (1,2,4,8)
        self.rd   = nn.Sequential(
            _ResDilBlock(base * 2, dil=1),
            _ResDilBlock(base * 2, dil=2),
            _ResDilBlock(base * 2, dil=4),
            _ResDilBlock(base * 2, dil=8),
        )

        # ---------- Decoder ----------
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(base * 2, base, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base), nn.ReLU(inplace=True)
        )
        self.iconv1 = _ResDilBlock(base, dil=2)

        # depth‑aware texture attention
        self.tex_att = DepthTextureAtt(feat_ch)

        # ---------- Output ----------
        self.out = nn.Conv2d(base, feat_ch, 3, 1, 1, bias=False)

    # ---------------------------------
    def forward(self, feat: torch.Tensor,
                conf: torch.Tensor,
                depth_prob = None):
        # concat confidence map
        x = torch.cat([feat, conf], dim=1)     # [B, feat_ch+1, H, W]

        # encoder
        x1 = self.res1(self.down1(x))          # [B,128,H,W]
        x2 = self.res2(self.down2(x1))         # [B,256,H/2,W/2]

        # bottleneck
        x2 = self.rd(self.nl(self.aspp(x2)))   # ASPP → NL → RD

        # decoder
        up = self.up1(x2)                      # [B,128,H,W]
        up = self.iconv1(up + x1)              # skip connection

        # texture‑hierarchy attention
        if depth_prob is not None:
            # depth_prob expected at same (H,W) – resize if necessary
            if depth_prob.shape[-2:] != up.shape[-2:]:
                depth_prob = F.interpolate(depth_prob, size=up.shape[-2:],
                                           mode='bilinear', align_corners=True)
            up = self.tex_att(up, depth_prob)  # apply attention

        return self.out(up)                    # [B, feat_ch, H, W]


            
    
class Fast_ACVNet_plus(nn.Module):
    def __init__(self, maxdisp, att_weights_only):
        super(Fast_ACVNet_plus, self).__init__()
        self.maxdisp = maxdisp
        self.att_weights_only = att_weights_only
        self.feature = FeatureMiTPtr()
        self.feature_up = FeatUp()
        chans = [32, 64, 160, 256]
        self.propagation_net = PropagationNetLarge(feat_ch=chans[0])


        self.stem_2 = nn.Sequential(
            BasicConv(3, 32, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU())
        self.stem_4 = nn.Sequential(
            BasicConv(32, 48, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(48, 48, 3, 1, 1, bias=False),
            nn.BatchNorm2d(48), nn.ReLU())
        self.spx = nn.Sequential(nn.ConvTranspose2d(2*32, 9, kernel_size=4, stride=2, padding=1))
        self.spx_2 = Conv2x(64, 32, True)
        self.spx_4 = nn.Sequential(
            BasicConv(80, 64, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(64, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU())
        self.conv = BasicConv(80, 48, kernel_size=3, padding=1, stride=1)
        self.desc = nn.Conv2d(48, 48, kernel_size=1, padding=0, stride=1)
        self.corr_stem = BasicConv(1, 8, is_3d=True, kernel_size=3, stride=1, padding=1)
        self.corr_feature_att_4 = channelAtt(8, 80)
        self.hourglass_att = hourglass_att(8)
        self.concat_feature = nn.Sequential(
            BasicConv(80, 32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(32, 16, 3, 1, 1, bias=False))
        self.concat_stem = BasicConv(32, 16, is_3d=True, kernel_size=3, stride=1, padding=1)
        self.concat_feature_att_4 = channelAtt(16, 80)
        self.hourglass = hourglass(16)

    def concat_volume_generator(self, left_input, right_input, disparity_samples):
        right_feature_map, left_feature_map = SpatialTransformer_grid(left_input,
                                                                       right_input, disparity_samples)
        concat_volume = torch.cat((left_feature_map, right_feature_map), dim=1)
        return concat_volume
    # forward는 그대로 유지 (디버깅 print 문은 유지)
    def forward(self, left, right):
        feature_left, attn_weights_left  = self.feature(left)
        feature_right, attn_weights_right = self.feature(right)
        features_left, features_right = self.feature_up(feature_left, feature_right)


        stem_2x = self.stem_2(left)
        stem_4x = self.stem_4(stem_2x)
        stem_2y = self.stem_2(right)
        stem_4y = self.stem_4(stem_2y)

        features_left_cat = torch.cat((features_left[0], stem_4x), 1)
        features_right_cat = torch.cat((features_right[0], stem_4y), 1)

        match_left = self.desc(self.conv(features_left_cat))
        match_right = self.desc(self.conv(features_right_cat))

        corr_volume_1 = build_norm_correlation_volume(match_left, match_right, self.maxdisp//4)
        top_one, entropy_map, mask = volume_entropy_softmax(corr_volume_1)
        peak_confidence = peak_confidence_from_volume(corr_volume_1)


        global_feat_L = self.propagation_net(features_left[0], mask, depth_prob=None)
        global_feat_R = self.propagation_net(features_right[0], mask, depth_prob=None)
        match_left_global = self.desc(self.conv(torch.cat((global_feat_L, stem_4x), 1)))
        match_right_global = self.desc(self.conv(torch.cat((global_feat_R, stem_4y), 1)))
        corr_volume_2 = build_norm_correlation_volume(match_left_global, match_right_global, self.maxdisp//4)

        corr_volume = self.corr_stem(corr_volume_2)

        cost_att = self.corr_feature_att_4(corr_volume, features_left_cat)
        att_weights = self.hourglass_att(cost_att, features_left)
        att_weights_prob = F.softmax(att_weights, dim=2)
        _, ind = att_weights_prob.sort(2, True)
        k = 24
        ind_k = ind[:, :, :k]
        ind_k = ind_k.sort(2, False)[0]
        att_topk = torch.gather(att_weights_prob, 2, ind_k)
        disparity_sample_topk = ind_k.squeeze(1).float()

        if not self.att_weights_only:
            concat_features_left = self.concat_feature(features_left_cat)
            concat_features_right = self.concat_feature(features_right_cat)
            concat_volume = self.concat_volume_generator(concat_features_left, concat_features_right, disparity_sample_topk)
            volume = att_topk * concat_volume
            volume = self.concat_stem(volume)
            volume = self.concat_feature_att_4(volume, features_left_cat)
            cost = self.hourglass(volume, features_left)

        xspx = self.spx_4(features_left_cat)
        xspx = self.spx_2(xspx, stem_2x)
        spx_pred = self.spx(xspx)
        spx_pred = F.softmax(spx_pred, 1)

        att_prob = torch.gather(att_weights, 2, ind_k).squeeze(1)
        att_prob = F.softmax(att_prob, dim=1)

        pred_att = torch.sum(att_prob * disparity_sample_topk, dim=1)
        pred_att_up = context_upsample(pred_att.unsqueeze(1), spx_pred)

        if self.att_weights_only:
            return [pred_att_up * 4, pred_att * 4]
        
        pred = regression_topk(cost.squeeze(1), disparity_sample_topk, 2)
        pred_up = context_upsample(pred, spx_pred)
        confidence_map, _ = att_prob.max(dim=1, keepdim=True)
        return [pred_up * 4, pred.squeeze(1) * 4, pred_att_up * 4, pred_att * 4], [confidence_map.squeeze(1), corr_volume_1, att_prob, corr_volume_1],  [feature_left, attn_weights_left]