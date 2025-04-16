import torch
import torch.nn as nn


class SigLoss(nn.Module):
    def __init__(self):
        super(SigLoss, self).__init__()

    def forward(self, x):
        return torch.mean(torch.abs(x))


class GradientLoss(nn.Module):
    def __init__(self):
        super(GradientLoss, self).__init__()

    def forward(self, x):
        grad_x = torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])
        grad_y = torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :])
        return torch.mean(grad_x) + torch.mean(grad_y)


class ReassembleBlocks(nn.Module):
    def __init__(self):
        super(ReassembleBlocks, self).__init__()    
    def forward(self, x):
        return x    
    
class ConvModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvModule, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
    def forward(self, x):
        return self.conv(x)




class HeadDepth(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(HeadDepth, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels,128,kernel_size=3,stride=1,padding=1),
            nn.Upsample(scale_factor=2,mode='bilinear',align_corners=False),
            nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=1, stride=1)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class DPTHead(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DPTHead, self).__init__()

        self.align_corners = False
        self.los_decode = nn.ModuleList(
            SigLoss(),
            GradientLoss()
        )

        self.conv_depth = HeadDepth()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        # self.reassemble_blocks 