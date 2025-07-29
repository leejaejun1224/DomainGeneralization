import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class LoRAResidualModule(nn.Module):
    def __init__(self, rank=16, maxdisp=192):
        super(LoRAResidualModule, self).__init__()
        self.rank = rank
        self.maxdisp = maxdisp
        
        # LoRA components for correlation processing
        # 근데 이렇게 하면 원래에 비해서 파라미터가 늘어나는거 아니야? 
        self.lora_corr_down = nn.Conv3d(1, rank, kernel_size=1, bias=False)
        self.lora_corr_up = nn.Conv3d(rank, 8, kernel_size=1, bias=False)
        
        # LoRA components for feature attention
        self.lora_feat_att_down = nn.Conv2d(80, rank, kernel_size=1, bias=False)
        self.lora_feat_att_up = nn.Conv2d(rank, 8, kernel_size=1, bias=False)
        
        # Lightweight residual hourglass for attention
        self.residual_att_conv1 = nn.Sequential(
            nn.Conv3d(8, rank, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm3d(rank),
            nn.ReLU(inplace=True)
        )
        self.residual_att_conv2 = nn.Sequential(
            nn.Conv3d(rank, rank*2, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm3d(rank*2),
            nn.ReLU(inplace=True)
        )
        self.residual_att_up = nn.Sequential(
            nn.ConvTranspose3d(rank*2, rank, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(rank),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(rank, 1, kernel_size=4, stride=2, padding=1)
        )
        
        # LoRA components for concat volume processing
        self.lora_concat_down = nn.Conv3d(16, rank, kernel_size=1, bias=False)  # 32 -> 16으로 변경
        self.lora_concat_up = nn.Conv3d(rank, 16, kernel_size=1, bias=False)
        
        # Residual hourglass for cost volume
        self.residual_cost_conv1 = nn.Sequential(
            nn.Conv3d(16, rank, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm3d(rank),
            nn.ReLU(inplace=True)
        )
        self.residual_cost_conv2 = nn.Sequential(
            nn.Conv3d(rank, rank*2, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm3d(rank*2),
            nn.ReLU(inplace=True)
        )
        self.residual_cost_up = nn.Sequential(
            nn.ConvTranspose3d(rank*2, rank, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(rank),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(rank, 1, kernel_size=4, stride=2, padding=1)
        )
        
        # Adaptive scaling factors
        self.att_scale = nn.Parameter(torch.zeros(1))
        self.cost_scale = nn.Parameter(torch.zeros(1))
        
        self._init_weights()
    
    def _init_weights(self):
        # Initialize LoRA weights with small random values
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Conv2d)):
                if 'down' in str(m):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out')
                elif 'up' in str(m):
                    nn.init.zeros_(m.weight)  # Start with zero to maintain original behavior
                else:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out')
    
    def forward(self, corr_volume_1, features_left_cat, att_weights_only=False):
        
        # LoRA residual for correlation processing
        # 원래 코드에서는 여기가 corr_stem에 해당하는 역할
        # 역할은? 
        lora_corr = self.lora_corr_up(self.lora_corr_down(corr_volume_1))
        
        # LoRA residual for feature attention
        lora_feat_att = self.lora_feat_att_up(self.lora_feat_att_down(features_left_cat))
        lora_feat_att = lora_feat_att.unsqueeze(2)  # Add disparity dimension
        
        # Combine LoRA residuals
        residual_corr_volume = lora_corr + lora_feat_att
        
        # Lightweight attention processing
        # 여기가 hourglass_att에 해당하는 부분임.
        res_att_conv1 = self.residual_att_conv1(residual_corr_volume)
        res_att_conv2 = self.residual_att_conv2(res_att_conv1)
        residual_att_weights = self.residual_att_up(res_att_conv2)
        
        if not att_weights_only:
            # LoRA residual for concat volume processing
            # Note: This assumes concat_volume input, you'll need to pass it from main forward
            pass  # Will be handled in main forward function
        
        return residual_att_weights, residual_corr_volume
    
    def process_concat_volume(self, concat_volume):
        """Process concat volume with LoRA residual"""
        lora_concat = self.lora_concat_up(self.lora_concat_down(concat_volume))
        
        # Lightweight cost processing
        res_cost_conv1 = self.residual_cost_conv1(lora_concat)
        res_cost_conv2 = self.residual_cost_conv2(res_cost_conv1)
        residual_cost = self.residual_cost_up(res_cost_conv2)
        
        return residual_cost
