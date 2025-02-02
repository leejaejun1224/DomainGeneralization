import math
import torch
import torch.nn as nn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


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

        return x





"""
input : image, kernal, stride, padding

"""
def OverlapPatchEmbedding():
    return


    
"""
input : x (batch, width, height, channel)
"""
class EfficientSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias, qk_scale, attn_drop, proj_drop, sr_ratio):
        super.__init__()
        assert dim % num_heads == 0, f"dim {dim} have to be divided by num_heads{num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim*2, bias=qkv_bias)

        self.scale = qk_scale or head_dim ** -0.5
        self.sr_ratio = sr_ratio
        self.attn_drop = attn_drop
        self.proj_drop = proj_drop
        
        # who are you?
        self.proj = nn.Linear(dim, dim)
        if sr_ratio > 1:
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
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # v : [B, num_head, N/R(sr_ratio^2), C/num_head]
        x = (attn @ v).transpose(1,2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x
    



    










