import torch
import torch.nn as nn

class MonoDepthDecoder(nn.Module):
    def __init__(self,
                 encoder_channels=[64, 128, 320, 512],  # 2*embed_dim for each stage (e.g., [2*32, 2*64, 2*160, 2*256])
                 decoder_channels=[160, 64, 32],
                 final_channels=32):
        super().__init__()
        # decoder block 입력 채널 수는 그대로 사용
        self.decoder4 = DecoderBlock(encoder_channels[3], decoder_channels[0])
        self.decoder3 = DecoderBlock(decoder_channels[0] + encoder_channels[2], decoder_channels[1])
        self.decoder2 = DecoderBlock(decoder_channels[1] + encoder_channels[1], decoder_channels[2])
        self.fusion = FusionBlock(decoder_channels[2] + encoder_channels[0], final_channels)
        self.out_conv = nn.Sequential(
            nn.Conv2d(final_channels, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, features):
        # features: list of feature maps from encoder (already with concatenated pos tokens)
        feat1, feat2, feat3, feat4 = features[0], features[1], features[2], features[3]
        x = self.decoder4(feat4)
        x = torch.cat([x, feat3], dim=1)
        x = self.decoder3(x)
        x = torch.cat([x, feat2], dim=1)
        x = self.decoder2(x)
        x = torch.cat([x, feat1], dim=1)
        x = self.fusion(x)
        depth = self.out_conv(x)
        return depth

#########################################
# Decoder Block & Fusion Block
#########################################
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        in_channels: 채널 결합 후 입력 채널 수.
        out_channels: 해당 Block의 출력 채널.
        """
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
        return self.conv(x)

class FusionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        in_channels: concat 후 채널 수.
        out_channels: fusion 후 출력 채널.
        """
        super().__init__()
<<<<<<< HEAD

        self.up1 = Upsample(encoder_channels[3], decoder_channels[0])
        self.up2 = Upsample(encoder_channels[2], decoder_channels[1])
        self.up3 = Upsample(encoder_channels[1], decoder_channels[2])
        self.out = nn.Conv2d(encoder_channels[0], final_channels, kernel_size=3, padding=1)
        

    def forward(self, pos_encodings):
        pos1, pos2, pos3, pos4 = pos_encodings

        x = pos4

        x = self.up1(x, pos3)
        x = self.up2(x, pos2)
        x = self.up3(x, pos1)
        x = self.out(x)

        return x


# if __name__=="__main__":
#     decoder = MonoDepthDecoder()
#     pos_encodings = [torch.randn(1, 32, 32, 32), torch.randn(1, 64, 16, 16), torch.randn(1, 160, 8, 8), torch.randn(1, 256, 4, 4)]
#     depth = decoder(pos_encodings)
#     print(depth.shape)
=======
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
>>>>>>> b356d1e5c87f6d957bdae21878df4e3ea8538f97
