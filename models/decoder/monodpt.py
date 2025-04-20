import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------------------------------------------------
#  Basic Building‑Blocks (unchanged)
# -----------------------------------------------------------------------------
class ConvModule(nn.Sequential):
    """Conv‑ReLU (bias optional)"""
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, bias=False, act=True):
        layers = [nn.Conv2d(in_ch, out_ch, k, s, p, bias=bias)]
        if act:
            layers.append(nn.ReLU(inplace=True))
        super().__init__(*layers)


class PreActResidualConvUnit(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1 = ConvModule(ch, ch, 3, 1, 1, bias=False, act=True)
        self.conv2 = ConvModule(ch, ch, 3, 1, 1, bias=False, act=True)

    def forward(self, x):
        return x + self.conv2(self.conv1(x))


class FeatureFusionBlock(nn.Module):
    """DPT‑style fusion; upsample previous stage, add skip, residual refine"""
    def __init__(self, ch=256, first=False):
        super().__init__()
        self.project = ConvModule(ch, ch, k=1, p=0, bias=False, act=True)
        if first:
            self.res_conv_unit1 = None
            self.res_conv_unit2 = PreActResidualConvUnit(ch)
        else:
            self.res_conv_unit1 = PreActResidualConvUnit(ch)
            self.res_conv_unit2 = PreActResidualConvUnit(ch)

    def forward(self, x, skip=None):
        x = self.project(x)
        if skip is not None:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = x + skip
        if self.res_conv_unit1 is not None:
            x = self.res_conv_unit1(x)
        x = self.res_conv_unit2(x)
        return x


class HeadDepth(nn.Module):
    def __init__(self, in_ch=256, max_disp=60):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_ch, 128, 3, padding=1),
            nn.ConvTranspose2d(128, 128, 4, stride=2, padding=1),  # x2 upsample
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 128, 4, stride=2, padding=1),  # x2 upsample
            nn.Conv2d(128, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 1)
        )
        self.depth_norm_indexes = torch.linspace(0, 1, 32, dtype=torch.float32, device=torch.device("cuda")).view(1, 32, 1, 1)
        self.register_buffer("_max_disp", torch.tensor(float(max_disp)))
        self._act = nn.Sigmoid()

        self.min_disp = 1.0
        self.max_disp = max_disp
        self.init_disp = 20.0
        ratio = (self.init_disp - self.min_disp) / (self.max_disp - self.min_disp)
        b0 = math.log(ratio / (1.0 - ratio))
        # 2) 마지막 Conv2d 레이어의 bias에 채우기
        # self.head[-1].bias.data.fill_(b0)

    def forward(self, x):
        # result = self._act(self.head(x)) 
        # map = F.interpolate(self.head(x), scale_factor = 4, mode="bilinear", align_corners=False)
        map = self.head(x)
        map = F.softmax(map, dim=1)
        map = torch.sum(map*self.depth_norm_indexes, dim=1, keepdim=True)
        # disp = result * (self.max_disp - self.min_disp) + self.min_disp
        return map


# -----------------------------------------------------------------------------
#  DPT Depth Decoder tailored for **SegFormer encoder outputs**
# -----------------------------------------------------------------------------
class DPTDepthDecoder(nn.Module):
    """
    Expect input list of 4 feature maps from SegFormer stages:
        [B,  32, H/4,  W/4],
        [B,  64, H/8,  W/8],
        [B, 160, H/16, W/16],
        [B, 256, H/32, W/32]
    Returns full‑resolution 1‑channel disparity/depth map (0‒max_disp).
    """

    def __init__(self, in_dims=(32, 64, 160, 256), mid_ch=256, max_disp=192):
        super().__init__()
        # Stage‑wise conv to unify channel to mid_ch (256)
        self.convs = nn.ModuleList([
            ConvModule(in_dims[0], mid_ch, 3, 1, 1, bias=False),
            ConvModule(in_dims[1], mid_ch, 3, 1, 1, bias=False),
            ConvModule(in_dims[2], mid_ch, 3, 1, 1, bias=False),
            ConvModule(in_dims[3], mid_ch, 3, 1, 1, bias=False)
        ])

        # Four fusion blocks (deepest first)
        self.fusion_blocks = nn.ModuleList([
            FeatureFusionBlock(mid_ch, first=True),
            FeatureFusionBlock(mid_ch),
            FeatureFusionBlock(mid_ch),
            FeatureFusionBlock(mid_ch)
        ])

        self.project = ConvModule(mid_ch, mid_ch, 3, 1, 1, bias=False, act=True)
        self.conv_depth = HeadDepth(mid_ch, max_disp=max_disp)
        self.min_disp = 0.1
        self.max_disp = max_disp

    def forward(self, feats):
        """feats: list length 4, order low‑→high res."""
        assert len(feats) == 4, "Need 4 stage features"
        feats = [conv(f) for conv, f in zip(self.convs, feats)]  # all to 256ch

        # Decode: start from deepest (1/32)
        x = feats[-1]
        x = self.fusion_blocks[0](x)               # no skip for first
        for i in range(1, 4):
            skip = feats[-i - 1]
            x = self.fusion_blocks[i](x, skip)

        x = self.project(x)                        # still 1/4 resolution
        disp = self.conv_depth(x)
        return disp
