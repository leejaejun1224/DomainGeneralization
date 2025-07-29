import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiInputAdaptor(nn.Module):
    """3개 입력(corr_volume_1, features_left_cat, features_left)을 모두 고려한 Adaptor"""
    def __init__(self, rank=4, alpha=0.3):
        super(MultiInputAdaptor, self).__init__()
        self.alpha = alpha
        
        # 1. Correlation Volume용 Adaptor
        self.corr_adaptor = Conv3DAdaptor(1, 8, kernel_size=3, rank=rank, alpha=alpha)
        
        # 2. Feature Left Cat용 Adaptor  
        self.feat_cat_adaptor = ChannelAttAdaptor(8, 80, rank=rank//2, alpha=alpha)
        
        # 3. Hourglass 내부의 features_left 상호작용 Adaptor
        self.hourglass_adaptor = HourglassMultiInputAdaptor(rank=rank, alpha=alpha)
        
    def forward(self, corr_volume_1, features_left_cat, features_left):
        return {
            'corr_adaptor_output': self.corr_adaptor(corr_volume_1),
            'feat_cat_adaptor': self.feat_cat_adaptor,
            'hourglass_adaptor': self.hourglass_adaptor
        }

class HourglassMultiInputAdaptor(nn.Module):
    """Hourglass 내부의 multi-input 처리용 Adaptor"""
    def __init__(self, rank=4, alpha=0.3):
        super(HourglassMultiInputAdaptor, self).__init__()
        self.alpha = alpha
        
        # 각 스케일별 3D conv adaptor
        self.conv1_adaptor = Conv3DAdaptor(16, 16, kernel_size=3, rank=rank, alpha=alpha)  # 8->16 after concat
        self.conv2_adaptor = Conv3DAdaptor(32, 32, kernel_size=3, rank=rank, alpha=alpha)  # 16->32
        self.conv3_adaptor = Conv3DAdaptor(48, 48, kernel_size=3, rank=rank, alpha=alpha)  # 32->48
        
        # Features_left와의 상호작용 adaptor
        self.feat_interaction_8 = FeatureInteractionAdaptor(16, 128, rank=rank//2)   # features_left[1]
        self.feat_interaction_16 = FeatureInteractionAdaptor(32, 320, rank=rank//2)  # features_left[2] 
        self.feat_interaction_32 = FeatureInteractionAdaptor(48, 256, rank=rank//2)  # features_left[3]
        
        # 업샘플링 경로 adaptor
        self.up_conv2_adaptor = Conv3DAdaptor(32, 32, kernel_size=3, rank=rank, alpha=alpha*0.5)
        self.up_conv1_adaptor = Conv3DAdaptor(16, 16, kernel_size=3, rank=rank, alpha=alpha*0.5)

class FeatureInteractionAdaptor(nn.Module):
    """3D 특징과 2D 특징 간의 상호작용을 학습하는 Adaptor"""
    def __init__(self, conv3d_channels, feat2d_channels, rank=4, alpha=0.5):
        super(FeatureInteractionAdaptor, self).__init__()
        self.alpha = alpha
        
        # 2D feature -> 3D attention weight 생성
        self.feat2d_to_att = nn.Sequential(
            nn.Conv2d(feat2d_channels, rank, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(rank, conv3d_channels, kernel_size=1, bias=False)
        )
        
        # 3D feature 자체의 적응
        self.conv3d_self_adaptor = nn.Sequential(
            nn.Conv3d(conv3d_channels, rank, kernel_size=1, bias=False),
            nn.ReLU(inplace=True), 
            nn.Conv3d(rank, conv3d_channels, kernel_size=1, bias=False)
        )
        
        # 초기화
        for m in [self.feat2d_to_att, self.conv3d_self_adaptor]:
            for layer in m:
                if isinstance(layer, (nn.Conv2d, nn.Conv3d)):
                    nn.init.kaiming_uniform_(layer.weight, a=1)
        
        # 마지막 레이어들은 0으로 초기화
        nn.init.zeros_(self.feat2d_to_att[-1].weight)
        nn.init.zeros_(self.conv3d_self_adaptor[-1].weight)
        
    def forward(self, conv3d_feat, feat2d):
        # 2D feature에서 attention weight 생성
        att_weight = self.feat2d_to_att(feat2d).unsqueeze(2)  # (B, C, 1, H, W)
        
        # 3D feature 자체 적응
        self_adapted = self.conv3d_self_adaptor(conv3d_feat)
        
        # 상호작용: attention과 self-adaptation 결합
        interaction = torch.sigmoid(att_weight) * self_adapted
        
        return self.alpha * interaction

class Adaptor(nn.Module):
    """3개 입력을 모두 고려한 완전한 Adapted Model"""
    def __init__(self, original_corr_stem, original_corr_feature_att_4, original_hourglass_att, 
                 adaptor_rank=4, adaptor_alpha=0.3):
        super(Adaptor, self).__init__()
        
        # 기존 모델들 (frozen)
        self.corr_stem = original_corr_stem
        self.corr_feature_att_4 = original_corr_feature_att_4  
        self.hourglass_att = original_hourglass_att
        
        for model in [self.corr_stem, self.corr_feature_att_4, self.hourglass_att]:
            for param in model.parameters():
                param.requires_grad = False
                
        # Multi-input Adaptor
        self.adaptor = MultiInputAdaptor(rank=adaptor_rank, alpha=adaptor_alpha)
        
    def forward(self, corr_volume_1, features_left_cat, features_left):
        # 1. Correlation stem + adaptor
        corr_out = self.corr_stem(corr_volume_1)
        corr_adapted = corr_out + self.adaptor.corr_adaptor(corr_volume_1)
        
        # 2. Correlation-feature attention + adaptor  
        original_att = self.corr_feature_att_4.im_att(features_left_cat).unsqueeze(2)
        adapted_att = self.adaptor.feat_cat_adaptor(original_att, features_left_cat)
        corr_attended = torch.sigmoid(adapted_att) * corr_adapted
        
        # 3. Hourglass attention + multi-input adaptor
        hourglass_out = self._adapted_hourglass_forward(corr_attended, features_left)
        
        return hourglass_out
    
    def _adapted_hourglass_forward(self, x, features_left):
        """Adaptor가 적용된 hourglass forward"""
        # Conv1 + Feature interaction
        conv1 = self.hourglass_att.conv1(x)
        conv1_interaction = self.adaptor.hourglass_adaptor.feat_interaction_8(conv1, features_left[1])
        conv1_adapted = conv1 + conv1_interaction
        
        # 기존 attention 적용
        original_att_8 = self.hourglass_att.feature_att_8.im_att(features_left[1]).unsqueeze(2)
        conv1 = torch.sigmoid(original_att_8) * conv1_adapted
        
        # Conv2 + Feature interaction  
        conv2 = self.hourglass_att.conv2(conv1)
        conv2_interaction = self.adaptor.hourglass_adaptor.feat_interaction_16(conv2, features_left[2])
        conv2_adapted = conv2 + conv2_interaction
        
        original_att_16 = self.hourglass_att.feature_att_16.im_att(features_left[2]).unsqueeze(2)
        conv2 = torch.sigmoid(original_att_16) * conv2_adapted
        
        # Conv3 + Feature interaction
        conv3 = self.hourglass_att.conv3(conv2)  
        conv3_interaction = self.adaptor.hourglass_adaptor.feat_interaction_32(conv3, features_left[3])
        conv3_adapted = conv3 + conv3_interaction
        
        original_att_32 = self.hourglass_att.feature_att_32.im_att(features_left[3]).unsqueeze(2)
        conv3 = torch.sigmoid(original_att_32) * conv3_adapted
        
        # 업샘플링 경로 (기존과 동일하게 adaptor 적용)
        conv3_up = self.hourglass_att.conv3_up(conv3)
        conv2 = torch.cat((conv3_up, conv2), dim=1)
        conv2 = self.hourglass_att.agg_0(conv2)
        conv2 = conv2 + self.adaptor.hourglass_adaptor.up_conv2_adaptor(conv2)
        
        conv2_up = self.hourglass_att.conv2_up(conv2)
        conv1 = torch.cat((conv2_up, conv1), dim=1) 
        conv1 = self.hourglass_att.agg_1(conv1)
        conv1 = conv1 + self.adaptor.hourglass_adaptor.up_conv1_adaptor(conv1)
        
        conv = self.hourglass_att.conv1_up(conv1)
        return conv

# Conv3DAdaptor와 ChannelAttAdaptor는 이전과 동일
class Conv3DAdaptor(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, rank=4, alpha=1.0):
        super(Conv3DAdaptor, self).__init__()
        self.rank = rank
        self.alpha = alpha
        
        self.lora_A = nn.Conv3d(in_channels, rank, kernel_size=1, bias=False)
        self.lora_B = nn.Conv3d(rank, out_channels, kernel_size=kernel_size, 
                               padding=kernel_size//2, bias=False)
        
        nn.init.kaiming_uniform_(self.lora_A.weight, a=1)
        nn.init.zeros_(self.lora_B.weight)
        
    def forward(self, x):
        return self.alpha * self.lora_B(self.lora_A(x))

class ChannelAttAdaptor(nn.Module):
    def __init__(self, cv_channels, im_channels, rank=8, alpha=0.5):
        super(ChannelAttAdaptor, self).__init__()
        self.alpha = alpha
        
        self.att_adaptor = nn.Sequential(
            nn.Conv2d(im_channels, rank, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(rank, cv_channels, kernel_size=1, bias=False)
        )
        
        for m in self.att_adaptor:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
        nn.init.zeros_(self.att_adaptor[-1].weight)
        
    def forward(self, original_att, im_features):
        adaptor_att = self.att_adaptor(im_features).unsqueeze(2)
        return original_att + self.alpha * adaptor_att
