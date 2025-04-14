import math
import torch
import torch.nn as nn
import torch.nn.functional as F  # 보간법(interpolation) 사용을 위해 추가
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

# (중략)
# 이미 정의된 PositionalEncoding, EfficientSelfAttention, EfficientSelfAttentionWithRelPos,
# DWConv, MixFFN, Block, OverlapPatchEmbedding 등은 그대로 사용합니다.
# 아래는 MixVisionTransformer 클래스에 depth token을 추가한 코드입니다.
class DWConv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)
    
    def forward(self, x, H, W):
        B, N, C = x.shape
        num_spatial = H * W
        # 만약 입력에 depth token이 있다면 마지막 토큰으로 가정
        if N == num_spatial + 1:
            x_spatial = x[:, :num_spatial, :]
            depth_token = x[:, -1:, :]
        else:
            x_spatial = x
            depth_token = None

        # 이제 x_spatial만 가지고 DWConv를 적용합니다.
        B, N_spatial, C = x_spatial.shape  # 여기서 N_spatial = H * W
        # [B, N_spatial, C] → [B, C, H, W]
        x_spatial_conv = x_spatial.transpose(1, 2).reshape(B, C, H, W)
        x_spatial_conv = self.dwconv(x_spatial_conv)  # [B, C, H, W]
        x_spatial_conv = x_spatial_conv.flatten(2).transpose(1, 2)  # [B, H*W, C]

        # depth token이 있다면 다시 concat
        if depth_token is not None:
            out = torch.cat([x_spatial_conv, depth_token], dim=1)  # 최종 [B, H*W+1, C]
        else:
            out = x_spatial_conv

        return out


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
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        if sr_ratio > 1:
            # 공간축소를 위한 conv를 patch 토큰 전용으로 적용할 예정임.
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
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

    def forward(self, x, H, W):
        """
        입력 x: [B, N, C]이고, N = H*W + 1 (첫 번째 토큰이 global token)
        """
        B, N, C = x.shape
        num_patch = H * W  # patch 토큰 개수 (global token 제외)
        
        # global token은 첫 번째 토큰로 가정하고, 나머지는 patch 토큰
        if N == num_patch + 1:
            global_token = x[:, 0:1, :]   # [B, 1, C]
            patch_tokens = x[:, 1:, :]      # [B, H*W, C]
        else:
            # global token이 없는 경우: 전부 patch 토큰으로 처리
            global_token = None
            patch_tokens = x

        # q는 전체 토큰에 대해 계산 (global + patch)
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads)
        q = q.permute(0, 2, 1, 3)  # [B, num_heads, N, C//num_heads]

        # key/value 계산 (patch와 global 분리하여 처리)
        # patch 토큰은 spatial reduction (sr) 적용
        if self.sr_ratio > 1 and patch_tokens is not None:
            # patch_tokens: [B, H*W, C] → [B, C, H, W]
            patch_tokens_2d = patch_tokens.transpose(1, 2).reshape(B, C, H, W)
            patch_tokens_sr = self.sr(patch_tokens_2d)  # [B, C, H', W'], where H'*W' = (H*W)/(sr_ratio^2)
            # reshape back to sequence
            patch_tokens_sr = patch_tokens_sr.reshape(B, C, -1).permute(0, 2, 1)
            patch_tokens_sr = self.norm(patch_tokens_sr)
            # compute kv for patch tokens
            kv_patch = self.kv(patch_tokens_sr)
            kv_patch = kv_patch.reshape(B, -1, 2, self.num_heads, C // self.num_heads)
            kv_patch = kv_patch.permute(2, 0, 3, 1, 4)  # shape: [2, B, num_heads, L_patch, C//num_heads]
        else:
            # sr_ratio <= 1 또는 patch_tokens가 없는 경우: 그대로 사용
            kv_patch = self.kv(patch_tokens)
            kv_patch = kv_patch.reshape(B, -1, 2, self.num_heads, C // self.num_heads)
            kv_patch = kv_patch.permute(2, 0, 3, 1, 4)

        # global token: 바로 kv 계산 (sr 없이)
        if global_token is not None:
            kv_global = self.kv(global_token)
            kv_global = kv_global.reshape(B, -1, 2, self.num_heads, C // self.num_heads)
            kv_global = kv_global.permute(2, 0, 3, 1, 4)  # shape: [2, B, num_heads, 1, C//num_heads]
        else:
            kv_global = None

        # 최종적으로, k, v를 patch와 global을 concat (global token은 앞쪽에 배치)
        if kv_global is not None:
            k = torch.cat([kv_global[0], kv_patch[0]], dim=2)  # global token KV와 patch token KV를 along token dim (2)
            v = torch.cat([kv_global[1], kv_patch[1]], dim=2)
        else:
            k, v = kv_patch[0], kv_patch[1]

        # self-attention 연산
        attn = (q @ k.transpose(-2, -1)) * self.scale   # [B, num_heads, N, L_total] where L_total = 1 + L_patch
        attn_weights = attn.softmax(dim=-1)
        attn = self.attn_drop(attn_weights)
        x_out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x_out = self.proj(x_out)
        x_out = self.proj_drop(x_out)

        print(f"attn_weights shape: {attn_weights.shape}")
        return x_out, attn_weights


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


class OverlapPatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, stride, in_chans, embed_dim):
        super().__init__()


        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size,stride=stride,
                              padding=(patch_size // 2, patch_size // 2))

        # positional encoding
        # self.pos_embedding = PositionalEncoding(embed_dim, img_size[0] // stride, img_size[1] // stride)


        self.norm = nn.LayerNorm(embed_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
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

        x = self.proj(x)
        _, _, H, W = x.shape
        ## positional encoding
        # x = self.pos_embedding(x)
        # B, N(H*W), C(embed_dim)
        x = x.flatten(2).transpose(1,2)
        x = self.norm(x)

        return x, H, W
    





class MixVisionTransformer(nn.Module):
    def __init__(self, img_size, in_chans, embed_dim,
                 depth, num_heads, qkv_bias, qk_scale, sr_ratio,
                 proj_drop, attn_drop, drop_path_rate, norm_layer=nn.LayerNorm):
        super().__init__()
        
        self.patch_embed1 = OverlapPatchEmbedding(img_size=[img_size[0], img_size[1]], patch_size=7, stride=4, 
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
                  norm_layer=norm_layer, mlp_ratio=4., drop=0., 
                  act_layer=nn.GELU)
            for i in range(depth[0])
        ])
        self.norm1 = norm_layer(embed_dim[0])
        cur += depth[0]

        self.block2 = nn.ModuleList([
            Block(dim=embed_dim[1], num_heads=num_heads[1], qkv_bias=qkv_bias, 
                  qk_scale=qk_scale, attn_drop=attn_drop[1], proj_drop=proj_drop[1], sr_ratio=sr_ratio[1], 
                  drop_path=dpr[cur + i], 
                  norm_layer=norm_layer, mlp_ratio=4., drop=0., 
                  act_layer=nn.GELU)
            for i in range(depth[1])
        ])
        self.norm2 = norm_layer(embed_dim[1])
        cur += depth[1]

        self.block3 = nn.ModuleList([
            Block(dim=embed_dim[2], num_heads=num_heads[2], qkv_bias=qkv_bias, 
                  qk_scale=qk_scale, attn_drop=attn_drop[2], proj_drop=proj_drop[2], sr_ratio=sr_ratio[2], 
                  drop_path=dpr[cur + i], 
                  norm_layer=norm_layer, mlp_ratio=4., drop=0., 
                  act_layer=nn.GELU)
            for i in range(depth[2])
        ])
        self.norm3 = norm_layer(embed_dim[2])
        cur += depth[2]

        self.block4 = nn.ModuleList([
            Block(dim=embed_dim[3], num_heads=num_heads[3], qkv_bias=qkv_bias, 
                  qk_scale=qk_scale, attn_drop=attn_drop[3], proj_drop=proj_drop[3], sr_ratio=sr_ratio[3], 
                  drop_path=dpr[cur + i], 
                  norm_layer=norm_layer, mlp_ratio=4., drop=0., 
                  act_layer=nn.GELU)
            for i in range(depth[3])
        ])
        self.norm4 = norm_layer(embed_dim[3])

        # >>> Depth token 관련 추가 부분 <<<
        # 초기 stage (patch_embed1)의 embedding 차원(embed_dim[0])과 동일한 depth token
        self.depth_token0 = nn.Parameter(torch.zeros(1, 1, embed_dim[0]))
        # 각 stage 전환 시 depth token의 embedding 차원을 맞추기 위한 projection layers
        self.depth_proj1 = nn.Linear(embed_dim[0], embed_dim[1])
        self.depth_proj2 = nn.Linear(embed_dim[1], embed_dim[2])
        self.depth_proj3 = nn.Linear(embed_dim[2], embed_dim[3])
        # 초기화
        trunc_normal_(self.depth_token0, std=0.02)
        # >>> 여기까지 추가 <<<

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
                nn.init.constant_(m.bias, 0)

    def forward_feature(self, x):
        B = x.shape[0]
        output = []
        attn_weights = []

        ### Stage 1
        # Patch embedding
        x, H, W = self.patch_embed1(x)  # x: [B, N, embed_dim[0]]
        # depth token 추가 (shape: [B, 1, embed_dim[0]])
        depth_token = self.depth_token0.expand(B, -1, -1)
        x = torch.cat([x, depth_token], dim=1)  # concat → [B, N+1, embed_dim[0]]
        # Transformer blocks (stage 1)
        for blk in self.block1:
            x, attn = blk(x, H, W)
        x = self.norm1(x)
        # 분리: spatial token과 depth token
        x_spatial = x[:, :-1, :]  # [B, N, embed_dim[0]]
        depth_token = x[:, -1:, :]  # [B, 1, embed_dim[0]]
        # spatial token은 2D 형태로 변환하여 다음 stage에 사용
        x_spatial_4D = x_spatial.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        output.append(x_spatial_4D)
        attn_weights.append(attn)

        ### Stage 2
        x, H2, W2 = self.patch_embed2(x_spatial_4D)  # [B, N2, embed_dim[1]]
        # depth token projection → embed_dim[1]
        depth_token = self.depth_proj1(depth_token)
        # depth token 재결합
        x = torch.cat([x, depth_token], dim=1)  # [B, N2+1, embed_dim[1]]
        for blk in self.block2:
            x, attn = blk(x, H2, W2)
        x = self.norm2(x)
        x_spatial = x[:, :-1, :]
        depth_token = x[:, -1:, :]  # [B, 1, embed_dim[1]]
        x_spatial_4D = x_spatial.reshape(B, H2, W2, -1).permute(0, 3, 1, 2).contiguous()
        output.append(x_spatial_4D)
        attn_weights.append(attn)

        ### Stage 3
        x, H3, W3 = self.patch_embed3(x_spatial_4D)  # [B, N3, embed_dim[2]]
        depth_token = self.depth_proj2(depth_token)     # embed_dim[2]
        x = torch.cat([x, depth_token], dim=1)  # [B, N3+1, embed_dim[2]]
        for blk in self.block3:
            x, attn = blk(x, H3, W3)
        x = self.norm3(x)
        x_spatial = x[:, :-1, :]
        depth_token = x[:, -1:, :]  # [B, 1, embed_dim[2]]
        x_spatial_4D = x_spatial.reshape(B, H3, W3, -1).permute(0, 3, 1, 2).contiguous()
        output.append(x_spatial_4D)
        attn_weights.append(attn)

        ### Stage 4
        x, H4, W4 = self.patch_embed4(x_spatial_4D)  # [B, N4, embed_dim[3]]
        depth_token = self.depth_proj3(depth_token)     # embed_dim[3]
        x = torch.cat([x, depth_token], dim=1)  # [B, N4+1, embed_dim[3]]
        for blk in self.block4:
            x, attn = blk(x, H4, W4)
        x = self.norm4(x)
        x_spatial = x[:, :-1, :]
        depth_token = x[:, -1:, :]  # 최종 depth token (B, 1, embed_dim[3])
        x_spatial_4D = x_spatial.reshape(B, H4, W4, -1).permute(0, 3, 1, 2).contiguous()
        output.append(x_spatial_4D)
        attn_weights.append(attn)

        # 최종적으로, output은 각 stage의 spatial feature map 리스트, 
        # 추가로 depth_token을 반환할 수 있으며, disparity head로 연결할 수 있음.
        return output, attn_weights, depth_token

    def forward_head(self, x):


        return x

    def forward(self, x):
        x, attn_weights, depth_token = self.forward_feature(x)
        # x: list of spatial feature maps from each stage
        # depth_token: 최종 stage에서 업데이트된 depth token → disparity distribution 예측 등에 활용 가능
        return x, attn_weights, depth_token


# =============================================================================
# 간단한 테스트 (주석 해제하여 실행해 보세요)
if __name__=="__main__":
    model = MixVisionTransformer(
        img_size=[384, 1248], in_chans=3, embed_dim=[64, 128, 256, 512],
        depth=[2, 2, 2, 2], num_heads=[1, 2, 4, 8], qkv_bias=True,
        qk_scale=1.0, sr_ratio=[8, 4, 2, 1],
        proj_drop=[0.0, 0.0, 0.0, 0.0], attn_drop=[0.0, 0.0, 0.0, 0.0],
        drop_path_rate=0.1
    )
    x = torch.randn(1, 3, 256, 256)
    outs, attn_weights, depth_token = model(x)
    print("Spatial outputs:", [o.shape for o in outs])
    print("Final depth token shape:", depth_token.shape)
# =============================================================================
