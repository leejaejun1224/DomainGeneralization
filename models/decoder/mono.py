import torch
import torch.nn as nn
import torch.nn.functional as F
class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.block = DecoderBlock(out_channels*2, out_channels)

    def forward(self, x, skipped_x):
        x = self.upconv(x)
        x = torch.cat([x, skipped_x], dim=1)
        x = self.block(x)
        return x


class DecoderBlock(nn.Module):
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
        x = self.conv(x)
        return x



class MonoDepthDecoder(nn.Module):
    def __init__(self,
                 encoder_channels=[32, 64, 160, 256],
                 decoder_channels=[160, 64, 32],
                 final_channels=48):
        super().__init__()

        self.up1 = Upsample(encoder_channels[3], decoder_channels[0])
        self.up2 = Upsample(encoder_channels[2], decoder_channels[1])
        self.up3 = Upsample(encoder_channels[1], decoder_channels[2])
        self.out_conv = nn.Sequential(
            nn.ConvTranspose2d(decoder_channels[2]*2, decoder_channels[2], kernel_size=2, stride=2),  # now 1/2
            nn.ReLU(inplace=True),
            nn.Conv2d(decoder_channels[2], decoder_channels[2], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(decoder_channels[2],  1,   kernel_size=3, padding=1),
            # nn.Sigmoid()  # [0,1]
        )
        self.out_conv_low = nn.Sequential(
            nn.Conv2d(decoder_channels[2]*2, decoder_channels[2], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(decoder_channels[2],  1,   kernel_size=3, padding=1),
            # nn.Sigmoid()  # [0,1]
        )

    def forward(self, features):
        # pos1, pos2, pos3, pos4 = pos_encodings
        feat1, feat2, feat3, feat4 = features

        x = feat4

        x = self.up1(x, feat3)
        x = self.up2(x, feat2)
        x = self.up3(x, feat1)
        x_low = self.out_conv_low(torch.cat([x, feat1], 1))
        x_low = torch.clamp(x_low, 0.0, 1.0)

        x_half = self.out_conv(torch.cat([x, feat1], 1))  # 1/2
        x_half = torch.clamp(x_half, 0.0, 1.0)
        # 최종: 1/2 → 1/1 (원본 해상도)
        x_full = F.interpolate(
            x_half,
            scale_factor=2,         # 1/2 → 1/1
            mode='bilinear',
            align_corners=True
        )

        return x_low*48.0, x_half*96.0, x_full*192.0


# if __name__=="__main__":
#     decoder = MonoDepthDecoder()
#     pos_encodings = [torch.randn(1, 32, 32, 32), torch.randn(1, 64, 16, 16), torch.randn(1, 160, 8, 8), torch.randn(1, 256, 4, 4)]
#     depth = decoder(pos_encodings)
#     print(depth.shape)