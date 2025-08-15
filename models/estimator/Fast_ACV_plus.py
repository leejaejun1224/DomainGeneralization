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
        
        ## 이거 이러면 파라미터에 2번 등록이 되네. 전체랑 encoder만이랑 쒯
        self.encoder = SegformerModel.from_pretrained('nvidia/segformer-b0-finetuned-cityscapes-512-1024').encoder


    def forward(self, x):
        outputs = self.encoder(x, output_hidden_states=True, output_attentions=False)

        # return outputs.hidden_states, outputs.attentions
        return outputs.hidden_states, None




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


## 여기서 channel attention을 해서 중요한 부분을 스스로 뽑을텐데 이 부분이 
## 새로운 도메인을 만나면 뭐가 중요한지 학습을 하기 어려울 듯 함.
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
                                             padding=1, stride=1, dilation=1),
                                   nn.Dropout3d(0.0))                             


        self.conv2_up = BasicConv(in_channels*4, in_channels*2, deconv=True, is_3d=True, bn=True,
                                  relu=True, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))
        self.conv1_up = BasicConv(in_channels*2, 1, deconv=True, is_3d=True, bn=False,
                                  relu=False, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))


        self.agg = nn.Sequential(BasicConv(in_channels*4, in_channels*2, is_3d=True, kernel_size=1, padding=0, stride=1),
                                 BasicConv(in_channels*2, in_channels*2, is_3d=True, kernel_size=3, padding=1, stride=1),
                                 nn.Dropout3d(0.0),
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


        self.conv1 = nn.Sequential(BasicConv(in_channels, in_channels*2, is_3d=True, bn=True, relu=True,  kernel_size=3,
                                             padding=1, stride=2, dilation=1),
                                   BasicConv(in_channels*2, in_channels*2, is_3d=True, bn=True, relu=True,  kernel_size=3,
                                             padding=1, stride=1, dilation=1))
                                    
        self.conv2 = nn.Sequential(BasicConv(in_channels*2, in_channels*4, is_3d=True, bn=True, relu=True,  kernel_size=3,
                                             padding=1, stride=2, dilation=1),
                                   BasicConv(in_channels*4, in_channels*4, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=1, dilation=1),
                                   nn.Dropout3d(0.))                             


        self.conv3 = nn.Sequential(BasicConv(in_channels*4, in_channels*6, is_3d=True, bn=True, relu=True,  kernel_size=3,
                                             padding=1, stride=2, dilation=1),
                                   BasicConv(in_channels*6, in_channels*6, is_3d=True, bn=True, relu=True,  kernel_size=3,
                                             padding=1, stride=1, dilation=1),
                                   nn.Dropout3d(0.)) 


        self.conv3_up = BasicConv(in_channels*6, in_channels*4, deconv=True, is_3d=True, bn=True,
                                  relu=True,  kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))
        self.conv2_up = BasicConv(in_channels*4, in_channels*2, deconv=True, is_3d=True, bn=True,
                                  relu=True,  kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))
        self.conv1_up = BasicConv(in_channels*2, 1, deconv=True, is_3d=True, bn=False,
                                  relu=False,  kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))


        self.agg_0 = nn.Sequential(BasicConv(in_channels*8, in_channels*4, is_3d=True, kernel_size=1, padding=0, stride=1),
                                   BasicConv(in_channels*4, in_channels*4, is_3d=True, kernel_size=3, padding=1, stride=1),
                                   nn.Dropout3d(0.),
                                   BasicConv(in_channels*4, in_channels*4, is_3d=True, kernel_size=3, padding=1, stride=1))
        self.agg_1 = nn.Sequential(BasicConv(in_channels*4, in_channels*2, is_3d=True, kernel_size=1, padding=0, stride=1),
                                   BasicConv(in_channels*2, in_channels*2, is_3d=True, kernel_size=3, padding=1, stride=1),
                                   nn.Dropout3d(0.),
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
    


class Fast_ACVNet_plus(nn.Module):
    def __init__(self, maxdisp, att_weights_only, enable_lora=True):
        super(Fast_ACVNet_plus, self).__init__()
        self.maxdisp = maxdisp
        self.att_weights_only = att_weights_only
        # self.feature = FeatureMiT()
        self.feature = FeatureMiTPtr()
        self.feature_up = FeatUp()
        chans = [32, 64, 160, 256]
        self.enable_lora = enable_lora
        lora_rank = 16
        # self.module = RefineCostVolume(feat_ch=32, max_disp=maxdisp)
        # self.propagation_net = PropagationNetLarge(feat_ch=chans[0])



        self.stem_2 = nn.Sequential(
            BasicConv(3, 32, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            # nn.BatchNorm2d(32), nn.ReLU())
            DomainNorm(32), nn.ReLU())
        self.stem_4 = nn.Sequential(
            BasicConv(32, 48, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(48, 48, 3, 1, 1, bias=False),
            # nn.BatchNorm2d(48), nn.ReLU())
            DomainNorm(48), nn.ReLU())
        
        self.spx = nn.Sequential(nn.ConvTranspose2d(2*32, 9, kernel_size=4, stride=2, padding=1))
        self.spx_2 = Conv2x(64, 32, True)
        self.spx_4 = nn.Sequential(
            BasicConv(80, 64, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(64, 64, 3, 1, 1, bias=False),
            # nn.BatchNorm2d(64), nn.ReLU())
            DomainNorm(64), 
            nn.ReLU())
        
        self.conv = BasicConv(80, 80, kernel_size=3, padding=1, stride=1)
        self.desc = nn.Conv2d(80, 80, kernel_size=1, padding=0, stride=1)
        self.desc_dropout = nn.Dropout2d(0.)
        # self.conv = BasicConv(80, 48, kernel_size=3, padding=1, stride=1)
        # self.desc = nn.Conv2d(48, 48, kernel_size=1, padding=0, stride=1)
        
        
        # self.desc1 = nn.Conv2d(48, 48, kernel_size=1, padding=0, stride=1)
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
        
        self.strip_att = RowStripAttention3D(
            cv_chan=8, feat_chan=80,
            heads=1, dim=8,            # 2×16 → 1×8 로 축소
            min_range=16,
            use_feat=True,
            memory_efficient=True,
            block_size=32,             # 64 → 32
            kv_stride=4                # K/V 절반의 절반(가로 1/4)
        )
        
        self.semantic_cross = SemanticCostCrossAttLite(
            cv_chan=8, feat_chan=256,
            heads=1, dim_qk=12, dim_v=12,
            max_tokens=128,      # 여유 없으면 64
            spatial_down=2,      # 쿼리(H/4,W/4) -> (H/8,W/8)에서만 어텐션
            q_chunk=1024,        # 메모리 더 빡세면 512
            dropout=0.0
        )
        self.enable_sematic_attn = False


        # if self.enable_lora:
        #     self.lora_module = AdaptedModel(adaptor_rank=4, adaptor_alpha=0.3)
        if enable_lora:
            self.adaptor = Adaptor(self.corr_stem, self.corr_feature_att_4, self.hourglass_att, 
                 adaptor_rank=16, adaptor_alpha=0.3)

        self.occ_head = OcclusionPredictor(feat_ch=80, use_corr=True, use_att=True)
        
        # === [추가] 픽셀별 밴드폭 제어 스위치/범위/헤드 등록 ===
        self.enable_adaptive_bandwidth = True     # 켜고 끌 수 있는 스위치
        self.tau_range = (0.5, 2.0)               # 온도 범위(클램프)
        self.r_range   = (2.0, 8.0)               # 연속 윈도우 반경 범위(픽셀, 1/4 해상도 기준)
        self.use_radius_mask = True               # r-마스킹 사용할지
        # features_left_cat(80) + entropy(1) + gap(1) = 82ch
        self.bandwidth_head = BandwidthHead(in_ch=82, hidden=64,
                                            tau_range=self.tau_range,
                                            r_range=self.r_range,
                                            predict_r=True)
        # =========================================================

    def concat_volume_generator(self, left_input, right_input, disparity_samples):
        
        ## 여기서 right_feature_map은 오른쪽의 feature map을 disparity만큼 왼쪽 이미지에 맞게 이동을 시킨거임.
        
        right_feature_map, left_feature_map = SpatialTransformer_grid(left_input,
                                                                       right_input, disparity_samples)
        concat_volume = torch.cat((left_feature_map, right_feature_map), dim=1)
        return concat_volume
    
    
    # forward는 그대로 유지 (디버깅 print 문은 유지)
    def forward(self, left, right, mode):
        feature_left, attn_weights_left  = self.feature(left)
        feature_right, attn_weights_right = self.feature(right)
        features_left, features_right = self.feature_up(feature_left, feature_right)



        stem_2x = self.stem_2(left)
        stem_4x = self.stem_4(stem_2x)
        stem_2y = self.stem_2(right)
        stem_4y = self.stem_4(stem_2y)



        features_left_cat = torch.cat((features_left[0], stem_4x), 1)
        features_right_cat = torch.cat((features_right[0], stem_4y), 1)


        ## 애는 local한 영역을 보니까 위에서 넣자
        match_left = self.desc_dropout(self.desc(self.conv(features_left_cat)))
        match_right = self.desc_dropout(self.desc(self.conv(features_right_cat)))


        ## shape [batch, 1, max_disparity//4, h, w]
        corr_volume_1 = build_norm_correlation_volume(match_left, match_right, self.maxdisp//4, mode=mode)


        #### 여기부터
        ## shape [batch, 8, max_disparity//4, h, w]
        ## 이 놈이 위 아래를 연결해주는 핵심부가 됨. 결과는 노션에 정리
        if self.enable_lora:
            att_weights = self.adaptor(corr_volume_1, features_left_cat, features_left)
        
        else:
            corr_volume = self.corr_stem(corr_volume_1)


            cost_att = self.corr_feature_att_4(corr_volume, features_left_cat)
            
            if self.enable_sematic_attn:
                feat_s4_for_att = features_left[3].detach() if self.training else features_left[3]
                cost_att = self.semantic_cross(cost_att, feat_s4_for_att)
            ## left feature는 여기에서는 업데이트가 없도록 함. 
            ## 여기를 잘 맞추기위한 left feature 학습이 없도록 하기 위해
            ## 즉 분리를 하겠다는 거임
            # cost_att, strip_stats = self.strip_att(cost_att, features_left_cat)

            features_left_for_att = [feat.detach() if self.training else feat for feat in features_left]
            att_weights = self.hourglass_att(cost_att, features_left_for_att)
        
        ## for first adaptor
        # if self.enable_lora:
        #     residual_att_weights, _ = self.lora_module(corr_volume_1, features_left_cat, self.att_weights_only)
        #     # Add residual to original attention weights
        #     att_weights = att_weights + residual_att_weights
        
        ## second adaptor
        
        ##### 여기까지 한 뭉탱이
        att_logits = att_weights  # alias

        if self.enable_adaptive_bandwidth:
            # 분포 통계(엔트로피/Top2 gap) - 로짓에서 stop-grad로 추출
            with torch.no_grad():
                p0  = F.softmax(att_logits, dim=2)                 # [B,1,D,H,W]
                p0c = p0.clamp_min(1e-12)
                H0  = -(p0c * p0c.log()).sum(dim=2)               # [B,1,H,W]
                top2 = p0.squeeze(1).topk(2, dim=1).values        # [B,2,H,W]
                gap  = (top2[:,0] - top2[:,1]).unsqueeze(1)       # [B,1,H,W]

            # τ/r 맵 예측
            tau_map, r_map = self.bandwidth_head(features_left_cat, H0, gap)  # [B,1,H,W] each

            # 온도 스케일링
            att_logits_scaled = att_logits / tau_map.unsqueeze(2)             # [B,1,D,H,W]

            # 연속 윈도우 마스킹 (Top-1 주변 |d-d0|<=r)
            if self.use_radius_mask:
                B_, C_, D_, H4_, W4_ = att_logits_scaled.shape
                d0 = att_logits_scaled.argmax(dim=2, keepdim=True)            # [B,1,1,H,W]
                r_int = torch.clamp(r_map.round().long(),
                                    min=1,
                                    max=min(D_//2, int(self.r_range[1])))     # [B,1,H,W]
                all_d = torch.arange(D_, device=att_logits_scaled.device).view(1,1,D_,1,1)
                win   = (all_d - d0).abs()                                    # [B,1,D,H,W]
                mask  = (win <= r_int.unsqueeze(2))                            # [B,1,D,H,W]
                masked_logits = att_logits_scaled.masked_fill(~mask, -1e9)
            else:
                masked_logits = att_logits_scaled
        else:
            masked_logits = att_logits
            
        T = 1.0
        att_weights_prob = F.softmax(masked_logits/T, dim=2)
        
        
        # T = 1.0
        # att_weights_prob = F.softmax(att_weights/T, dim=2)
        
        
        prob_flat          = att_weights_prob.squeeze(1)
        top2_probs, idx_2  = prob_flat.topk(2, dim=1, largest=True)
        
        disp_diff = idx_2[:,1].float() - idx_2[:,0].float()



        _, ind = att_weights_prob.sort(2, True)
        k = 24
        ind_k = ind[:, :, :k]
        ind_k = ind_k.sort(2, False)[0]



        att_topk = torch.gather(att_weights_prob, 2, ind_k)
        disparity_sample_topk = ind_k.squeeze(1).float()
        
        if not self.att_weights_only:
            concat_features_left = self.concat_feature(features_left_cat)
            concat_features_right = self.concat_feature(features_right_cat)
            
            ## 이 concat volume에서는 right feature를 왼쪽으로 disparity 만큼 이동시킨거랑, left feature가 concat되어있음. 결국 반복인거지.
            ## 그럼 차원은 [batch, 2*channel, disparity topk, h, w] 가 됨. 여기서는 곱하기 2 해서 32
            concat_volume = self.concat_volume_generator(concat_features_left, concat_features_right, disparity_sample_topk)
            volume = att_topk * concat_volume
            volume = self.concat_stem(volume)
            
            ## 여기는 volume에서 sigmoid로 각 채널마다 중요도를 계산을 하고 그걸 앞서 구한 features_left_cat에 곱함.
            ## 그런데 여기는 1x1 convolution을 거치니까 여기서는 새로운 feature의 조합을 계산을 하지는 않는다.
            ## sigmoid를 통해 채널의 중요도를 계산하고 volume에 attention을 주는 것 뿐이니까.
            volume = self.concat_feature_att_4(volume, features_left_cat)
            
            ## 근데 여기서 features_left는 stem을 거치지 않고 feature upsample만 feature map이다. 왜냐면 여기서는 장거리의 context를 반영을 하는 것이 목적인데
            ## stem을 끼게 되면 local한 특징을 너무 많이 포함하게 된다. 근데 애초에 왜 hourglass가 위에서 말한 역할을 수행해야만 하는지는 잘 모르겠다.
            
            ## 여기도 마찬가지 분리를 위해서 떨어뜨려놓음
            features_left_for_hg = [feat.detach() if self.training else feat for feat in features_left]
            cost = self.hourglass(volume, features_left_for_hg)
            ### 여기까지가 1/4 사이즈 prediction하는거고
            
            # if self.enable_lora:
            #     residual_cost = self.lora_module.process_concat_volume(volume)
            #     cost = cost + residual_cost
            
            xspx = self.spx_4(features_left_cat)
            xspx = self.spx_2(xspx, stem_2x)
            spx_pred = self.spx(xspx)
            
            spx_pred = F.softmax(spx_pred, 1)    
            
            occ_up, occ_1_4, occ_logit, _ = self.occ_head(
                features_left_cat,         # [B,80,H/4,W/4]
                att_logits=att_weights,    # [B,1,D/4,H/4,W/4]
                corr_volume=corr_volume_1, # [B,1,D/4,H/4,W/4]
                spx_pred=spx_pred          # [B,9, ...]
            )
            ### 여기가 upsample할 때 필요한 context


        att_prob = torch.gather(att_weights, 2, ind_k).squeeze(1)
        att_prob = F.softmax(att_prob, dim=1)


        pred_att = torch.sum(att_prob * disparity_sample_topk, dim=1)
        pred_att_up = context_upsample(pred_att.unsqueeze(1), spx_pred)


        if self.att_weights_only:
            return [pred_att_up * 4, pred_att * 4]
        
        ## 그러면 여기서 cost에서 disparity의 probability를 구하고, 
        ## 그거를 disparity sample topk에 weight sum해서 하나의 disparity를 결정
        pred, prob = regression_topk(cost.squeeze(1), disparity_sample_topk, 12)
        
        ## 고 다음에 주변과의 유사도를 다시 계산해서 주변 9개의 probability를 구해서
        ## 또 똑같이 가중합을 해서 찐 최종을 구함.
        pred_up = context_upsample(pred, spx_pred)
        prob_up1 = context_upsample(prob[:,0,:,:].unsqueeze(1),spx_pred)
        prob_up2 = context_upsample(prob[:,1,:,:].unsqueeze(1),spx_pred)
        
        confidence = prob_up1 + prob_up2
        confidence_map, _ = att_prob.max(dim=1, keepdim=True)
        return [pred_up * 4, pred.squeeze(1) * 4, pred_att_up * 4, pred_att * 4], \
            [disp_diff.detach(), corr_volume_1.detach(), confidence, corr_volume_1.detach()], \
            [None, att_weights.detach(), cost.detach(), match_left.detach(), match_right.detach()], \
            [occ_up, occ_logit]
            # [feature_left, att_weights.detach(), cost.detach(), match_left.detach(), match_right.detach()]
    
    
    def freeze_original_network(self):
        for name, param in self.named_parameters():
            if 'adaptor' not in name:
                param.requires_grad = False
    
    
    def unfreeze_all(self):
        for param in self.parameters():
            param.requires_grad = True
