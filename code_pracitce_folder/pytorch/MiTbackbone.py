import math
import torch
import torch.nn as nn
import torchsummary
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

import torch.nn.functional as F  # 보간법(interpolation) 사용을 위해 추가

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, height, width):
        super().__init__()
        self.embed_dim = embed_dim
        # 초기에는 고정된 사이즈로 파라미터를 정의합니다.
        self.pos_embedding = nn.Parameter(torch.zeros(1, embed_dim, height, width))
        # Optional: 초기화 (필요에 따라 적용)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Parameter):
            trunc_normal_(m, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x의 shape: (B, C, H, W)
        B, C, H, W = x.shape
        # 만약 입력 feature map의 H, W가 저장된 positional encoding과 다르면 보간법으로 크기를 맞춰줍니다.
        if H != self.pos_embedding.shape[2] or W != self.pos_embedding.shape[3]:
            pos_embedding = F.interpolate(self.pos_embedding, size=(H, W), mode='bilinear', align_corners=False)
        else:
            pos_embedding = self.pos_embedding
        return x + pos_embedding

"""
input : x (batch, width, height, channel)
"""
class EfficientSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias, qk_scale, attn_drop, proj_drop, sr_ratio):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} have to be divided by num_heads{num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim*2, bias=qkv_bias)

        self.scale = qk_scale or head_dim ** -0.5
        self.sr_ratio = sr_ratio
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # who are you?
        self.proj = nn.Linear(dim, dim)
        if sr_ratio > 1:
            # reduction ratio
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        # 이 apply함수는 모델 내부의 모든 레이어를 위에서부터
        # 자동으로 m으로 전달하여 _init_weights를 실행시킨다. 
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

        # He intitialization
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out)) # avg, std
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape
        # output of overlap patch embedding

        # b 8192 24
        # b 2048 32
        # b 512 96
        # b 128 160

        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0,2,1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C//self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C//self.num_heads).permute(2, 0, 3, 1, 4)
        
        k, v = kv[0], kv[1]

        # q : [B, num_heads, N, C/num_head], k : [B, num_heads, N/R, C/num_head]
        attn = (q @ k.transpose(-2,-1))*self.scale
        
        # attn : [B, num_heads, N, N/R]


        attn_weights = attn.softmax(dim=-1)

        attn = self.attn_drop(attn_weights)

        # 아래의 계산은 attention weight를 value의 embedding 벡터에 곱해줘서
        # 각 헤드별로 embedding 벡터를 조정해준다.
        # v : [B, num_head, N/R(sr_ratio^2), C/num_head]
        x = (attn @ v).transpose(1,2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x, attn_weights
    

class EfficientSelfAttentionWithRelPos(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=True, qk_scale=None,
                 attn_drop=0., proj_drop=0., sr_ratio=1, H=16, W=16):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads."
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        # 상대적 위치 바이어스 테이블: (2*H-1) * (2*W-1)개의 위치에 대해 num_heads 차원의 bias
        num_relative_positions = (2 * H - 1) * (2 * W - 1)
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(num_relative_positions, num_heads)
        )

        # 상대적 위치 인덱스 미리 계산 (H*W 패치 기준)
        coords_h = torch.arange(H)
        coords_w = torch.arange(W)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing='ij'))  # [2, H, W]
        coords_flatten = coords.flatten(1)  # [2, H*W]
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # [2, H*W, H*W]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # [H*W, H*W, 2]
        relative_coords[:, :, 0] += H - 1  # shift to start from 0
        relative_coords[:, :, 1] += W - 1
        relative_coords[:, :, 0] *= 2 * W - 1
        relative_position_index = relative_coords.sum(-1)  # [H*W, H*W]
        self.register_buffer("relative_position_index", relative_position_index)

        trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x, H, W):
        """
        x: [B, N, C] 토큰 시퀀스 (N=H*W)
        H, W: 현재 입력의 height, width (patch 개수 기준)
        """
        B, N, C = x.shape

        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            # sr_ratio가 있는 경우 feature map으로 복원 후 downsample
            x_ = x.transpose(1, 2).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).transpose(1, 2)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        k, v = kv[0], kv[1]  # k, v: [B, num_heads, N', C//num_heads]

        # Attention score 계산 (q @ k^T)
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, num_heads, N, N']

        # 상대적 위치 바이어스 추가 (H, W가 초기 설정과 같을 때)
        if H * W == self.relative_position_index.shape[0]:
            # relative_position_index: [N, N]로 펼쳐서 bias table에서 인덱싱
            relative_position_bias = self.relative_position_bias_table[
                self.relative_position_index.view(-1)
            ].view(H * W, H * W, -1).permute(2, 0, 1)  # [num_heads, N, N]
            attn = attn + relative_position_bias.unsqueeze(0)  # [B, num_heads, N, N]

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn

# torch.flatten(n, m) n차원부터 시작해서 m차원까지 포함해서 flatten하라는 뜻(m의 기본값은 -1) 
class DWConv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)
    
    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1,2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1,2)

        # output : B, N', C
        return x



class MixFFN(nn.Module):
    def __init__(self, in_features, hidden_features, out_features=None, activation=nn.GELU, drop=0.):
        super().__init__()
        hidden_features = in_features or hidden_features
        out_features = in_features or out_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.activation_layer = activation()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
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
        x = self.activation_layer(x)
        x = self.fc2(x)
        x = self.dropout(x)

        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias, qk_scale, attn_drop, 
                 proj_drop, sr_ratio, 
                 drop_path=0., norm_layer = nn.LayerNorm, mlp_ratio=4.,
                 drop=0., act_layer=nn.GELU):
        super().__init__()

        self.norm1 = norm_layer(dim)
        self.attention = EfficientSelfAttention(dim, num_heads, qkv_bias, qk_scale, attn_drop, proj_drop, sr_ratio)
        # self.attention = EfficientSelfAttentionWithRelPos(dim, num_heads, qkv_bias, qk_scale, attn_drop, proj_drop, sr_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim*mlp_ratio)
        self.mlp = MixFFN(in_features=dim, hidden_features=mlp_hidden_dim, 
                          activation=act_layer, drop=drop)
        self.norm2 = norm_layer(dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()


    def forward(self, x, H, W):
        output, attn_weights = self.attention(self.norm1(x), H, W)
        x = self.drop_path(output) + x
        x = self.drop_path(self.mlp(self.norm2(x), H, W)) + x

        return x, attn_weights




"""
input : image, kernal, stride, padding

"""
class OverlapPatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, stride, in_chans, embed_dim):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size // 2, patch_size // 2))
        # positional encoding을 추가합니다.
        self.pos_embedding = PositionalEncoding(embed_dim, img_size[0] // stride, img_size[1] // stride)
        self.norm = nn.LayerNorm(embed_dim)
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
        # x: [B, in_chans, H_img, W_img]
        x = self.proj(x)  # [B, embed_dim, H, W]
        _, C, H, W = x.shape
        # pos_embedding.pos_embedding는 [1, embed_dim, H0, W0]로 저장되어 있으므로, H, W가 다를 경우 보간 적용
        pos = self.pos_embedding.pos_embedding
        if H != pos.shape[2] or W != pos.shape[3]:
            pos = F.interpolate(pos, size=(H, W), mode='bilinear', align_corners=False)
        # positional encoding을 feature map에 더함
        x = x + pos
        # flatten하여 (B, N, embed_dim)으로 변환
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        # pos 정보는 추후 positional encoding만을 별도로 학습/활용하기 위해 반환
        return x, pos, H, W

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        in_channels: upsample 후 skip connection과 결합된 채널 수.
        out_channels: 해당 Block의 출력 채널 수.
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
        x = self.conv(x)
        return x

#############################################
# Fusion Block: Skip connection 융합용 (upsample 없이)
#############################################
class FusionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        in_channels: concat 후 채널 수 (예, 64+64=128)
        out_channels: fusion 후 출력 채널 (예, 64)
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

#############################################
# MonoDepth Decoder (최종 해상도: 원본의 1/4)
#############################################
class MonoDepthDecoder(nn.Module):
    def __init__(self,
                 encoder_channels=[64, 128, 256, 512],
                 decoder_channels=[256, 128, 64],
                 final_channels=64):
        """
        encoder_channels: encoder에서 각 단계별 feature map의 채널 수  
          (예: stage1=64, stage2=128, stage3=256, stage4=512)
        decoder_channels: 각 upsampling 단계에서 출력할 채널 수  
          (예: [256, 128, 64])
        final_channels: fusion block 후의 중간 채널 수 (예: 64)
        """
        super().__init__()
        # Block 1: Stage4 (8×8,512) → upsample → 16×16, 채널=256
        self.decoder4 = DecoderBlock(encoder_channels[3], decoder_channels[0])
        # Block 2: Concatenate (decoder4 output + stage3) → (256+256=512) → upsample → 32×32, 채널=128
        self.decoder3 = DecoderBlock(decoder_channels[0] + encoder_channels[2], decoder_channels[1])
        # Block 3: Concatenate (decoder3 output + stage2) → (128+128=256) → upsample → 64×64, 채널=64
        self.decoder2 = DecoderBlock(decoder_channels[1] + encoder_channels[1], decoder_channels[2])
        # Fusion: Concatenate (decoder2 output + stage1) → (64+64=128) → fusion conv → final_channels
        self.fusion = FusionBlock(decoder_channels[2] + encoder_channels[0], final_channels)
        # 최종적으로 1채널의 depth map 산출
        self.out_conv = nn.Conv2d(final_channels, 1, kernel_size=3, padding=1)
    
    def forward(self, features, pos_encodings):
        """
        features: encoder로부터 얻은 feature map 리스트 (길이 4)
        pos_encodings: 각 stage별 positional encoding (길이 4)
        
        (각 stage에서 feature map과 pos encoding은 element‑wise하게 더해집니다.)
        """
        # 각 단계에서 feature와 positional encoding을 결합
        feat1 = features[0] + pos_encodings[0]  # stage1: e.g., 64×64, 64채널
        feat2 = features[1] + pos_encodings[1]  # stage2: e.g., 32×32, 128채널
        feat3 = features[2] + pos_encodings[2]  # stage3: e.g., 16×16, 256채널
        feat4 = features[3] + pos_encodings[3]  # stage4: e.g., 8×8, 512채널
        
        # Decoder 진행
        x = self.decoder4(feat4)   # 8×8 → 16×16, 채널 256
        x = torch.cat([x, feat3], dim=1)  # 16×16, 채널: 256 + 256 = 512
        x = self.decoder3(x)       # 16×16 → 32×32, 채널 128
        x = torch.cat([x, feat2], dim=1)  # 32×32, 채널: 128 + 128 = 256
        x = self.decoder2(x)       # 32×32 → 64×64, 채널 64
        x = torch.cat([x, feat1], dim=1)  # 64×64, 채널: 64 + 64 = 128
        x = self.fusion(x)         # Fusion block (64×64, 채널: final_channels=64)
        depth = self.out_conv(x)   # 최종 1채널 monodepth map (64×64)
        return depth




class MixVisionTransformer(nn.Module):
    def __init__(self, img_size, in_chans, embed_dim,
                 depth, num_heads, qkv_bias, qk_scale, sr_ratio,
                 proj_drop, attn_drop, drop_path_rate, norm_layer=nn.LayerNorm):
        super().__init__()
        
        self.patch_embed1 = OverlapPatchEmbedding(img_size=img_size, patch_size=7, stride=4, 
                                                  in_chans=in_chans, embed_dim=embed_dim[0])
        self.patch_embed2 = OverlapPatchEmbedding(img_size=[img_size[0]//4, img_size[1]//4], patch_size=3, stride=2, 
                                                  in_chans=embed_dim[0], embed_dim=embed_dim[1])
        self.patch_embed3 = OverlapPatchEmbedding(img_size=[img_size[0]//8, img_size[1]//8], patch_size=3, stride=2, 
                                                  in_chans=embed_dim[1], embed_dim=embed_dim[2])
        self.patch_embed4 = OverlapPatchEmbedding(img_size=[img_size[0]//16, img_size[1]//16], patch_size=3, stride=2, 
                                                  in_chans=embed_dim[2], embed_dim=embed_dim[3])

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depth))]
        cur = 0

        self.block1 = nn.ModuleList([
            Block(dim=embed_dim[0], num_heads=num_heads[0], qkv_bias=qkv_bias, 
                  qk_scale=qk_scale, attn_drop=attn_drop[0], proj_drop=proj_drop[0], sr_ratio=sr_ratio[0], 
                  drop_path=dpr[cur + i], 
                  norm_layer=nn.LayerNorm, mlp_ratio=4., drop=0., 
                  act_layer=nn.GELU)
            for i in range(depth[0])
        ])
        self.norm1 = norm_layer(embed_dim[0])
        cur += depth[0]

        self.block2 = nn.ModuleList([
            Block(dim=embed_dim[1], num_heads=num_heads[1], qkv_bias=qkv_bias, 
                  qk_scale=qk_scale, attn_drop=attn_drop[1], proj_drop=proj_drop[1], sr_ratio=sr_ratio[1], 
                  drop_path=dpr[cur + i], 
                  norm_layer=nn.LayerNorm, mlp_ratio=4., drop=0., 
                  act_layer=nn.GELU)
            for i in range(depth[1])
        ])
        self.norm2 = norm_layer(embed_dim[1])
        cur += depth[1]

        self.block3 = nn.ModuleList([
            Block(dim=embed_dim[2], num_heads=num_heads[2], qkv_bias=qkv_bias, 
                  qk_scale=qk_scale, attn_drop=attn_drop[2], proj_drop=proj_drop[2], sr_ratio=sr_ratio[2], 
                  drop_path=dpr[cur + i], 
                  norm_layer=nn.LayerNorm, mlp_ratio=4., drop=0., 
                  act_layer=nn.GELU)
            for i in range(depth[2])
        ])
        self.norm3 = norm_layer(embed_dim[2])
        cur += depth[2]

        self.block4 = nn.ModuleList([
            Block(dim=embed_dim[3], num_heads=num_heads[3], qkv_bias=qkv_bias, 
                  qk_scale=qk_scale, attn_drop=attn_drop[3], proj_drop=proj_drop[3], sr_ratio=sr_ratio[3], 
                  drop_path=dpr[cur + i], 
                  norm_layer=nn.LayerNorm, mlp_ratio=4., drop=0., 
                  act_layer=nn.GELU)
            for i in range(depth[3])
        ])
        self.norm4 = norm_layer(embed_dim[3])

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
        
    def forward_feature(self, x):
        B = x.shape[0]
        outputs = []
        attn_weights_list = []
        pos_encodings = []

        # Stage 1
        x, pos, H, W = self.patch_embed1(x)
        for blk in self.block1:
            x, attn_weight = blk(x, H, W)
        x = self.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outputs.append(x)
        attn_weights_list.append(attn_weight)
        pos_encodings.append(pos)

        # Stage 2
        x, pos, H, W = self.patch_embed2(x)
        for blk in self.block2:
            x, attn_weight = blk(x, H, W)
        x = self.norm2(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outputs.append(x)
        attn_weights_list.append(attn_weight)
        pos_encodings.append(pos)

        # Stage 3
        x, pos, H, W = self.patch_embed3(x)
        for blk in self.block3:
            x, attn_weight = blk(x, H, W)
        x = self.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outputs.append(x)
        attn_weights_list.append(attn_weight)
        pos_encodings.append(pos)

        # Stage 4
        x, pos, H, W = self.patch_embed4(x)
        for blk in self.block4:
            x, attn_weight = blk(x, H, W)
        x = self.norm4(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outputs.append(x)
        attn_weights_list.append(attn_weight)
        pos_encodings.append(pos)

        return outputs, attn_weights_list, pos_encodings
    
    def forward_head(self, x):
        return x

    def forward(self, x):
        features, attn_weights, pos_encodings = self.forward_feature(x)
        # 필요에 따라 head를 추가할 수 있습니다.
        return features, attn_weights, pos_encodings


if __name__=="__main__":
    mitbackbone = MixVisionTransformer(img_size=[384, 1248], in_chans=3, embed_dim=[64, 128, 256, 512],
                                      depth=[2, 2, 2, 2], num_heads=[1, 2, 4, 8], qkv_bias=True,
                                      qk_scale=1.0, sr_ratio=[8, 4, 2, 1], proj_drop=[0.0, 0.0, 0.0, 0.0], attn_drop=[0.0, 0.0, 0.0, 0.0],
                                      drop_path_rate=0.1)
    
    x = torch.randn(1, 3, 384, 1248)
    features, attn_weights, pos_encodings = mitbackbone(x)
    features = [features[0], features[1], features[2], features[3]]
    pos_encodings = [pos_encodings[0], pos_encodings[1], pos_encodings[2], pos_encodings[3]]
    
    decoder = MonoDepthDecoder()
    depth_map = decoder(features, pos_encodings)
    print("Depth map shape:", depth_map.shape) 

    