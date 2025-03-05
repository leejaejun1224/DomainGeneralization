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
        # self.attention = EfficientSelfAttention(dim, num_heads, qkv_bias, qk_scale, attn_drop, proj_drop, sr_ratio)
        self.attention = EfficientSelfAttentionWithRelPos(dim, num_heads, qkv_bias, qk_scale, attn_drop, proj_drop, sr_ratio)
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
                  norm_layer = nn.LayerNorm, mlp_ratio=4., drop=0., 
                  act_layer=nn.GELU)
        for i in range(depth[0])])
        self.norm1 = norm_layer(embed_dim[0])

        cur += depth[0]
        self.block2 = nn.ModuleList([
            Block(dim=embed_dim[1], num_heads=num_heads[1], qkv_bias=qkv_bias, 
                  qk_scale=qk_scale, attn_drop=attn_drop[1], proj_drop=proj_drop[1], sr_ratio=sr_ratio[1], 
                  drop_path=dpr[cur + i], 
                  norm_layer = nn.LayerNorm, mlp_ratio=4., drop=0., 
                  act_layer=nn.GELU)
        for i in range(depth[1])])
        self.norm2 = norm_layer(embed_dim[1])


        cur += depth[1]
        self.block3 = nn.ModuleList([
            Block(dim=embed_dim[2], num_heads=num_heads[2], qkv_bias=qkv_bias, 
                  qk_scale=qk_scale, attn_drop=attn_drop[2], proj_drop=proj_drop[2], sr_ratio=sr_ratio[2], 
                  drop_path=dpr[cur + i], 
                  norm_layer = nn.LayerNorm, mlp_ratio=4., drop=0., 
                  act_layer=nn.GELU)
        for i in range(depth[2])])
        self.norm3 = norm_layer(embed_dim[2])



        cur += depth[2]
        self.block4 = nn.ModuleList([
            Block(dim=embed_dim[3], num_heads=num_heads[3], qkv_bias=qkv_bias, 
                  qk_scale=qk_scale, attn_drop=attn_drop[3], proj_drop=proj_drop[3], sr_ratio=sr_ratio[3], 
                  drop_path=dpr[cur + i], 
                  norm_layer = nn.LayerNorm, mlp_ratio=4., drop=0., 
                  act_layer=nn.GELU)
        for i in range(depth[3])])
        self.norm4 = norm_layer(embed_dim[3])

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
        


    def forward_feature(self, x):
        B = x.shape[0]
        output = []
        attn_weights = []

        # stage 1
        x, H, W = self.patch_embed1(x)
        
        # output size 
        # x.shape : [B, 8192, 24]
        # H, W : 64, 128


        for i, blk in enumerate(self.block1):
            x, attn_weight = blk(x, H, W)
            
        # x : [B, N, C]
        # attn_weight : [B, num_heads, C, C/num_heads]

        x = self.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        output.append(x)
        attn_weights.append(attn_weight)


        # stage 2
        x, H, W = self.patch_embed2(x)
        for i, blk in enumerate(self.block2):
            x, attn_weight = blk(x, H, W)
        # x : [B, N, C]

        x = self.norm2(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        output.append(x)
        attn_weights.append(attn_weight)

        # stage 3
        x, H, W = self.patch_embed3(x)
        for i, blk in enumerate(self.block3):
            x, attn_weight = blk(x, H, W)

        
        # x : [B, N, C]
        x = self.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        output.append(x)
        attn_weights.append(attn_weight)        

        # stage 4
        x, H, W = self.patch_embed4(x)
        for i, blk in enumerate(self.block4):
            x, attn_weight = blk(x, H, W)    
        


        # x : [B, N, C]
        # attn_weight : [B, num_heads, N, C/num_heads]
        x = self.norm4(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        
        output.append(x)
        attn_weights.append(attn_weight)

        return output, attn_weights
    

    def forward_head(self, x):
        return x


    def forward(self, x):
        x, attn_weights = self.forward_feature(x)
        # x = self.forward_head(x)
        return x, attn_weights


# if __name__=="__main__":
#     mitbackbone = MixVisionTransformer(img_size=256, in_chans=3, embed_dim=[64, 128, 256, 512],
#                                       depth=[2, 2, 2, 2], num_heads=[1, 2, 4, 8], qkv_bias=True,
#                                       qk_scale=1.0, sr_ratio=[8, 4, 2, 1], proj_drop=[0.0, 0.0, 0.0, 0.0], attn_drop=[0.0, 0.0, 0.0, 0.0],
#                                       drop_path_rate=0.1)
    
#     x = torch.randn(1, 3, 256, 256)
#     output = mitbackbone(x)
#     # print(output)

    