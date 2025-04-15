import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_

#########################################
# Positional Encoding
#########################################
class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, height, width):
        super().__init__()
        self.embed_dim = embed_dim
        # learnable positional embedding
        self.pos_embedding = nn.Parameter(torch.zeros(1, embed_dim, height, width))
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Parameter):
            trunc_normal_(m, std=0.02)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 이 모듈은 실제 계산에서 사용하지 않고 pos token을 얻기 위한 parameter 보관용으로 활용됩니다.
        return self.pos_embedding

#########################################
# Overlap Patch Embedding (pos token concat)
#########################################
class OverlapPatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, stride, in_chans, embed_dim):
        """
        입력 이미지를 Conv2d로 임베딩한 후, 
        동일한 spatial 크기의 pos token을 얻어 concat하여 (B, 2*embed_dim, H, W)로 만듭니다.
        이후 flatten하여 (B, N, 2*embed_dim) 형태로 반환합니다.
        """
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size // 2, patch_size // 2))
        # pos token을 얻기 위해 PositionalEncoding 모듈을 사용 (더하기 대신 concat할 예정)
        self.pos_encoding = PositionalEncoding(embed_dim, img_size[0] // stride, img_size[1] // stride)
        # norm layer는 최종 채널이 2*embed_dim이 됨을 고려
        self.norm = nn.LayerNorm(2 * embed_dim)
        self.embed_dim = embed_dim
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
    
    def forward(self, x):
        # x: (B, in_chans, H, W)
        x_proj = self.proj(x)    # (B, embed_dim, H, W)
        B, C, H, W = x_proj.shape
        # obtain pos token (shape: (1, embed_dim, H0, W0))
        pos = self.pos_encoding(x_proj)
        if H != pos.shape[2] or W != pos.shape[3]:
            pos = F.interpolate(pos, size=(H, W), mode='bilinear', align_corners=False)
        pos = pos.expand(B, -1, -1, -1)  # (B, embed_dim, H, W)
        # concat along channel dimension → (B, 2*embed_dim, H, W)
        x_cat = torch.cat([x_proj, pos], dim=1)
        # flatten spatial dimensions: (B, N, 2*embed_dim)
        x_cat = x_cat.flatten(2).transpose(1, 2)
        x_cat = self.norm(x_cat)
        return x_cat, H, W

#########################################
# Efficient Self-Attention 및 관련 모듈 (변경 없음)
#########################################
class EfficientSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias, qk_scale, attn_drop, proj_drop, sr_ratio):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.scale = qk_scale or head_dim ** -0.5
        self.sr_ratio = sr_ratio
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        self.proj = nn.Linear(dim, dim)

        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0]*m.kernel_size[1]*m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0/fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.transpose(1, 2).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).transpose(1, 2)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        
        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn_weights = attn.softmax(dim=-1)
        attn = self.attn_drop(attn_weights)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn_weights

class DWConv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
    
    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1,2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1,2)
        return x

class MixFFN(nn.Module):
    def __init__(self, in_features, hidden_features, out_features=None, activation=nn.GELU, drop=0.):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.activation = activation()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(drop)
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0]*m.kernel_size[1]*m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0/fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
    
    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

#########################################
# Transformer Block (입력 dim은 2×embed_dim)
#########################################
class Block(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias, qk_scale, attn_drop, proj_drop, sr_ratio, 
                 drop_path=0., norm_layer=nn.LayerNorm, mlp_ratio=4., drop=0., act_layer=nn.GELU):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attention = EfficientSelfAttention(dim, num_heads, qkv_bias, qk_scale, attn_drop, proj_drop, sr_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MixFFN(dim, mlp_hidden_dim, activation=act_layer, drop=drop)
        self.norm2 = norm_layer(dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0]*m.kernel_size[1]*m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0/fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
    
    def forward(self, x, H, W):
        attn_out, attn_weights = self.attention(self.norm1(x), H, W)
        x = x + self.drop_path(attn_out)
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x, attn_weights

#########################################
# MixVisionTransformer Backbone (SegFormer 스타일)
#########################################
class MixVisionTransformer(nn.Module):
    def __init__(self, img_size, in_chans, embed_dim,
                 depth, num_heads, qkv_bias, qk_scale, sr_ratio,
                 proj_drop, attn_drop, drop_path_rate, norm_layer=nn.LayerNorm):
        super().__init__()
        # 각 patch embedding 모듈는 pos token을 concat하여 2*embed_dim 차원을 출력
        self.patch_embed1 = OverlapPatchEmbedding(img_size=img_size, patch_size=7, stride=4, 
                                                  in_chans=in_chans, embed_dim=embed_dim[0])
        self.patch_embed2 = OverlapPatchEmbedding(img_size=[img_size[0]//4, img_size[1]//4], patch_size=3, stride=2, 
                                                  in_chans=embed_dim[0]*2, embed_dim=embed_dim[1])
        self.patch_embed3 = OverlapPatchEmbedding(img_size=[img_size[0]//8, img_size[1]//8], patch_size=3, stride=2, 
                                                  in_chans=embed_dim[1]*2, embed_dim=embed_dim[2])
        self.patch_embed4 = OverlapPatchEmbedding(img_size=[img_size[0]//16, img_size[1]//16], patch_size=3, stride=2, 
                                                  in_chans=embed_dim[2]*2, embed_dim=embed_dim[3])
        # 각 단계의 출력 차원는 2*embed_dim[i]
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depth))]
        cur = 0

        self.block1 = nn.ModuleList([
            Block(dim=2 * embed_dim[0], num_heads=num_heads[0], qkv_bias=qkv_bias,
                  qk_scale=qk_scale, attn_drop=attn_drop[0], proj_drop=proj_drop[0], sr_ratio=sr_ratio[0],
                  drop_path=dpr[cur + i], norm_layer=norm_layer, mlp_ratio=4., drop=0.,
                  act_layer=nn.GELU)
            for i in range(depth[0])
        ])
        self.norm1 = norm_layer(2 * embed_dim[0])
        cur += depth[0]

        self.block2 = nn.ModuleList([
            Block(dim=2 * embed_dim[1], num_heads=num_heads[1], qkv_bias=qkv_bias,
                  qk_scale=qk_scale, attn_drop=attn_drop[1], proj_drop=proj_drop[1], sr_ratio=sr_ratio[1],
                  drop_path=dpr[cur + i], norm_layer=norm_layer, mlp_ratio=4., drop=0.,
                  act_layer=nn.GELU)
            for i in range(depth[1])
        ])
        self.norm2 = norm_layer(2 * embed_dim[1])
        cur += depth[1]

        self.block3 = nn.ModuleList([
            Block(dim=2 * embed_dim[2], num_heads=num_heads[2], qkv_bias=qkv_bias,
                  qk_scale=qk_scale, attn_drop=attn_drop[2], proj_drop=proj_drop[2], sr_ratio=sr_ratio[2],
                  drop_path=dpr[cur + i], norm_layer=norm_layer, mlp_ratio=4., drop=0.,
                  act_layer=nn.GELU)
            for i in range(depth[2])
        ])
        self.norm3 = norm_layer(2 * embed_dim[2])
        cur += depth[2]

        self.block4 = nn.ModuleList([
            Block(dim=2 * embed_dim[3], num_heads=num_heads[3], qkv_bias=qkv_bias,
                  qk_scale=qk_scale, attn_drop=attn_drop[3], proj_drop=proj_drop[3], sr_ratio=sr_ratio[3],
                  drop_path=dpr[cur + i], norm_layer=norm_layer, mlp_ratio=4., drop=0.,
                  act_layer=nn.GELU)
            for i in range(depth[3])
        ])
        self.norm4 = norm_layer(2 * embed_dim[3])

        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0/fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
    
    def forward_feature(self, x):
        B = x.shape[0]
        outputs = []
        attn_weights_list = []
        # 각 단계에서 pos token은 이미 concat되어 encoder feature로 포함됨.
        # Stage 1
        x, H, W = self.patch_embed1(x)   # (B, N, 2*embed_dim[0])
        for blk in self.block1:
            x, attn_weight = blk(x, H, W)
        x = self.norm1(x)
        x = x.transpose(1,2).reshape(B, -1, H, W).contiguous()
        outputs.append(x)
        attn_weights_list.append(attn_weight)

        # Stage 2
        x, H, W = self.patch_embed2(x)
        for blk in self.block2:
            x, attn_weight = blk(x, H, W)
        x = self.norm2(x)
        x = x.transpose(1,2).reshape(B, -1, H, W).contiguous()
        outputs.append(x)
        attn_weights_list.append(attn_weight)

        # Stage 3
        x, H, W = self.patch_embed3(x)
        for blk in self.block3:
            x, attn_weight = blk(x, H, W)
        x = self.norm3(x)
        x = x.transpose(1,2).reshape(B, -1, H, W).contiguous()
        outputs.append(x)
        attn_weights_list.append(attn_weight)

        # Stage 4
        x, H, W = self.patch_embed4(x)
        for blk in self.block4:
            x, attn_weight = blk(x, H, W)
        x = self.norm4(x)
        x = x.transpose(1,2).reshape(B, -1, H, W).contiguous()
        outputs.append(x)
        attn_weights_list.append(attn_weight)

        return outputs, attn_weights_list

    def forward(self, x):
        return self.forward_feature(x)

#########################################
# RelativeDepthDecoder (encoder의 출력을 그대로 사용)
#########################################
class RelativeDepthDecoder(nn.Module):
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

#########################################
# Relative Depth Estimation Model (전체 모델)
#########################################
class RelativeDepthEstimationModel(nn.Module):
    def __init__(self, img_size=[256, 512], in_chans=3,
                 embed_dim=[32, 64, 160, 256],
                 depth=[2, 2, 2, 2],
                 num_heads=[1, 2, 4, 8],
                 qkv_bias=True,
                 qk_scale=1.0,
                 sr_ratio=[8, 4, 2, 1],
                 proj_drop=[0.0, 0.0, 0.0, 0.0],
                 attn_drop=[0.0, 0.0, 0.0, 0.0],
                 drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.backbone = MixVisionTransformer(
            img_size=img_size, 
            in_chans=in_chans, 
            embed_dim=embed_dim, 
            depth=depth, 
            num_heads=num_heads,
            qkv_bias=qkv_bias, 
            qk_scale=qk_scale, 
            sr_ratio=sr_ratio, 
            proj_drop=proj_drop,
            attn_drop=attn_drop, 
            drop_path_rate=drop_path_rate, 
            norm_layer=norm_layer
        )
        # encoder_channels: 각 단계의 출력 채널은 2×embed_dim
        enc_channels = [2*x for x in embed_dim]
        self.decoder = RelativeDepthDecoder(encoder_channels=enc_channels,
                                            decoder_channels=[160, 64, 32],
                                            final_channels=32)
    
    def forward(self, x):
        features, _ = self.backbone(x)
        depth = self.decoder(features)
        return depth

#########################################
# 테스트
#########################################
# if __name__ == "__main__":
#     model = RelativeDepthEstimationModel(
#         img_size=[256, 512],
#         in_chans=3,
#         embed_dim=[32, 64, 160, 256],
#         depth=[2, 2, 2, 2],
#         num_heads=[1, 2, 4, 8],
#         qkv_bias=True,
#         qk_scale=1.0,
#         sr_ratio=[8, 4, 2, 1],
#         proj_drop=[0.0, 0.0, 0.0, 0.0],
#         attn_drop=[0.0, 0.0, 0.0, 0.0],
#         drop_path_rate=0.1
#     )
    
#     dummy_input = torch.randn(1, 3, 256, 512)
#     pred_depth = model(dummy_input)
#     print("Predicted depth map shape:", pred_depth.shape)  # 예: (1, 1, ~64, ~64)
