import torch
import torch.nn as nn


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):

        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        return x


class FusionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):

        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)


class MonoDepthDecoder(nn.Module):
    def __init__(self,
                 encoder_channels=[32, 64, 160, 256],
                 decoder_channels=[160, 64, 32],
                 final_channels=32):

        super().__init__()
        self.decoder4 = DecoderBlock(encoder_channels[3], decoder_channels[0])
        self.decoder3 = DecoderBlock(decoder_channels[0] + encoder_channels[2], decoder_channels[1])
        self.decoder2 = DecoderBlock(decoder_channels[1] + encoder_channels[1], decoder_channels[2])
        self.fusion = FusionBlock(decoder_channels[2] + encoder_channels[0], final_channels)
        self.out_conv = nn.Conv2d(final_channels, 1, kernel_size=3, padding=1)
    
    def forward(self, features, pos_encodings):
        feat1 = features[0] + pos_encodings[0]  # stage1: e.g., 32x32
        feat2 = features[1] + pos_encodings[1]  # stage2: e.g., 64x64
        feat3 = features[2] + pos_encodings[2]  # stage3: e.g., 160x160
        feat4 = features[3] + pos_encodings[3]  # stage4: e.g., 256x256
        
        x = self.decoder4(feat4)   # 8×8 → 16×16, 채널 256
        x = torch.cat([x, feat3], dim=1)  # 16×16, 채널: 256 + 256 = 512
        x = self.decoder3(x)       # 16×16 → 32×32, 채널 128
        x = torch.cat([x, feat2], dim=1)  # 32×32, 채널: 128 + 128 = 256
        x = self.decoder2(x)       # 32×32 → 64×64, 채널 64
        x = torch.cat([x, feat1], dim=1)  # 64×64, 채널: 64 + 64 = 128
        x = self.fusion(x)         # Fusion block (64×64, 채널: final_channels=64)
        depth = self.out_conv(x)   # 최종 1채널 monodepth map (64×64)
        return depth
