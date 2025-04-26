import math
import torch
import torch.nn as nn
import torchsummary
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

import torch.nn.functional as F  # 보간법(interpolation) 사용을 위해 추가
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_

class RelPosBias2D(nn.Module):
    """Static relative position bias table."""
    def __init__(self, heads: int, height: int, width: int):
        super().__init__()
        num_rel_y = 2 * height - 1
        num_rel_x = 2 * width - 1
        self.table = nn.Parameter(torch.zeros(num_rel_y * num_rel_x, heads))
        trunc_normal_(self.table, std=0.02)
        # build index map [N, N]
        coords_h = torch.arange(height)
        coords_w = torch.arange(width)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing='ij'))  # [2,H,W]
        coords_flat = coords.reshape(2, -1)                                        # [2, N]
        rel = coords_flat[:, :, None] - coords_flat[:, None, :]                    # [2, N, N]
        rel[0] += height - 1
        rel[1] += width - 1
        self.register_buffer('index_map', rel[0] * (2 * width - 1) + rel[1])      # [N, N]

    def forward(self):
        # returns bias [heads, N, N]
        bias = self.table[self.index_map.view(-1)].view(*self.index_map.shape, -1)  # [N, N, heads]
        return bias.permute(2, 0, 1)

class EfficientSelfAttentionRel(nn.Module):
    """Self-attention with static or continuous relative positional bias."""
    def __init__(
        self,
        dim,
        num_heads,
        qkv_bias=False,
        attn_drop=0.,
        proj_drop=0.,
        sr_ratio=1,
        rel_size=(64, 64),
        mlp_hidden=64,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.sr_ratio = sr_ratio

        # QKV projections
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        # output projection
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        # static bias table for sr_ratio == 1
        self.rel_bias = RelPosBias2D(num_heads, rel_size[0], rel_size[1])
        # continuous MLP for sr_ratio > 1
        self.pos_mlp = nn.Sequential(
            nn.Linear(2, mlp_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden, num_heads)
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels // m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        """
        x: [B, Nq, C], Nq = H * W
        returns: out [B, Nq, C], attn [B, heads, Nq, Nk]
        """
        B, Nq, C = x.shape
        heads = self.num_heads
        head_dim = C // heads

        # Q projection
        q = self.q(x).reshape(B, Nq, heads, head_dim).permute(0, 2, 1, 3)

        # KV projection
        if self.sr_ratio > 1:
            # downsample spatially for kv
            Hk, Wk = H // self.sr_ratio, W // self.sr_ratio
            Nk = Hk * Wk
            feat_map = x.permute(0, 2, 1).reshape(B, C, H, W)
            down = F.avg_pool2d(feat_map, self.sr_ratio, self.sr_ratio)
            kv_tokens = down.reshape(B, C, -1).permute(0, 2, 1)  # [B, Nk, C]
            kv = self.kv(kv_tokens).reshape(B, Nk, 2, heads, head_dim).permute(2, 0, 3, 1, 4)
        else:
            Nk = Nq
            kv = self.kv(x).reshape(B, Nq, 2, heads, head_dim).permute(2, 0, 3, 1, 4)

        k, v = kv[0], kv[1]  # each [B, heads, Nk, head_dim]

        # attention logits
        attn_logits = (q @ k.transpose(-2, -1)) * self.scale  # [B, heads, Nq, Nk]

        # add relative positional bias
        if self.sr_ratio == 1:
            bias = self.rel_bias()  # [heads, Nq, Nq]
            attn_logits = attn_logits + bias.unsqueeze(0)
        else:
            # continuous bias between Q-grid and K-grid
            yq = torch.linspace(-1, 1, H, device=x.device)
            xq = torch.linspace(-1, 1, W, device=x.device)
            yy_q, xx_q = torch.meshgrid(yq, xq, indexing='ij')
            coords_q = torch.stack([yy_q, xx_q], dim=-1).view(Nq, 2)

            Hk, Wk = H // self.sr_ratio, W // self.sr_ratio
            yk = torch.linspace(-1, 1, Hk, device=x.device)
            xk = torch.linspace(-1, 1, Wk, device=x.device)
            yy_k, xx_k = torch.meshgrid(yk, xk, indexing='ij')
            coords_k = torch.stack([yy_k, xx_k], dim=-1).view(Nk, 2)

            offsets = coords_q[:, None, :] - coords_k[None, :, :]  # [Nq, Nk, 2]
            bias = self.pos_mlp(offsets.view(-1, 2))              # [Nq*Nk, heads]
            bias = bias.view(Nq, Nk, heads).permute(2, 0, 1)      # [heads, Nq, Nk]
            attn_logits = attn_logits + bias.unsqueeze(0)

        # softmax + dropout
        attn = F.softmax(attn_logits, dim=-1)
        attn = self.attn_drop(attn)

        # aggregate and project
        out = (attn @ v).transpose(1, 2).reshape(B, Nq, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out, attn


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
    def __init__(
        self,
        dim,
        num_heads,
        qkv_bias,
        attn_drop,
        proj_drop,
        sr_ratio,
        rel_size: tuple,
        drop_path=0.,
        norm_layer=nn.LayerNorm,
        mlp_ratio=4.,
        drop=0.,
        act_layer=nn.GELU,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        # Use EfficientSelfAttentionRel with relative bias
        self.attn = EfficientSelfAttentionRel(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            sr_ratio=sr_ratio,
            rel_size=rel_size,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        mlp_hidden = int(dim * mlp_ratio)
        self.norm2 = norm_layer(dim)
        self.mlp = MixFFN(in_features=dim, hidden_features=mlp_hidden, out_features=dim,
                          activation=act_layer, drop=drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels // m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        # x: [B, N, C]
        shortcut = x
        x_norm = self.norm1(x)
        x_attn, attn_weights = self.attn(x_norm, H, W)
        x = shortcut + self.drop_path(x_attn)
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x, attn_weights



"""
input : image, kernal, stride, padding

"""
class OverlapPatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, stride, in_chans, embed_dim):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim,
                               kernel_size=patch_size, stride=stride,
                               padding=(patch_size // 2, patch_size // 2))
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
        x = self.proj(x)            # [B, embed_dim, H, W]
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, N=H*W, C]
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

        H1, W1 = img_size[0]//4, img_size[1]//4
        H2, W2 = H1//2, W1//2
        H3, W3 = H2//2, W2//2
        H4, W4 = H3//2, W3//2

        # patch embeddings omitted for brevity...
        # Stage1 Blocks
        dpr = list(torch.linspace(0, drop_path_rate, sum(depth)))
        cur = 0
        self.block1 = nn.ModuleList([
            Block(
                dim=embed_dim[0], num_heads=num_heads[0], qkv_bias=qkv_bias,
                attn_drop=attn_drop[0], proj_drop=proj_drop[0], sr_ratio=sr_ratio[0],
                rel_size=(H1, W1), drop_path=dpr[cur+i], norm_layer=norm_layer)
            for i in range(depth[0])
        ])
        self.norm1 = norm_layer(embed_dim[0])
        cur += depth[0]
        # Stage2 Blocks
        self.block2 = nn.ModuleList([
            Block(
                dim=embed_dim[1], num_heads=num_heads[1], qkv_bias=qkv_bias,
                attn_drop=attn_drop[1], proj_drop=proj_drop[1], sr_ratio=sr_ratio[1],
                rel_size=(H2, W2), drop_path=dpr[cur+i], norm_layer=norm_layer)
            for i in range(depth[1])
        ])
        self.norm2 = norm_layer(embed_dim[1])
        cur += depth[1]
        # Stage3
        self.block3 = nn.ModuleList([
            Block(dim=embed_dim[2], num_heads=num_heads[2], qkv_bias=qkv_bias,
                  attn_drop=attn_drop[2], proj_drop=proj_drop[2], sr_ratio=sr_ratio[2],
                  rel_size=(H3, W3), drop_path=dpr[cur+i], norm_layer=norm_layer)
            for i in range(depth[2])
        ])
        self.norm3 = norm_layer(embed_dim[2])
        cur += depth[2]
        # Stage4
        self.block4 = nn.ModuleList([
            Block(dim=embed_dim[3], num_heads=num_heads[3], qkv_bias=qkv_bias,
                  attn_drop=attn_drop[3], proj_drop=proj_drop[3], sr_ratio=sr_ratio[3],
                  rel_size=(H4, W4), drop_path=dpr[cur+i], norm_layer=norm_layer)
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

        # Stage 1
        x, H, W = self.patch_embed1(x)
        for blk in self.block1:
            x, attn_weight = blk(x, H, W)
        x = self.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outputs.append(x)
        attn_weights_list.append(attn_weight)

        # Stage 2
        x, H, W = self.patch_embed2(x)
        for blk in self.block2:
            x, attn_weight = blk(x, H, W)
        x = self.norm2(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outputs.append(x)
        attn_weights_list.append(attn_weight)

        # Stage 3
        x, H, W = self.patch_embed3(x)
        for blk in self.block3:
            x, attn_weight = blk(x, H, W)
        x = self.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outputs.append(x)
        attn_weights_list.append(attn_weight)

        # Stage 4
        x, H, W = self.patch_embed4(x)
        for blk in self.block4:
            x, attn_weight = blk(x, H, W)
        x = self.norm4(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outputs.append(x)
        attn_weights_list.append(attn_weight)

        return outputs, attn_weights_list
    
    def forward_head(self, x):
        return x

    def forward(self, x):
        features, attn_weights = self.forward_feature(x)
        # 필요에 따라 head를 추가할 수 있습니다.
        return features, attn_weights


# if __name__=="__main__":
#     mitbackbone = MixVisionTransformer(img_size=[256, 512], in_chans=3, embed_dim=[32, 64, 160, 256],
#                                       depth=[2, 2, 2, 2], num_heads=[1, 2, 4, 8], qkv_bias=True,
#                                       qk_scale=1.0, sr_ratio=[8, 4, 2, 1], proj_drop=[0.0, 0.0, 0.0, 0.0], attn_drop=[0.0, 0.0, 0.0, 0.0],
#                                       drop_path_rate=0.1)
    
#     x = torch.randn(1, 3, 256, 512)
#     features, attn_weights, pos_encodings = mitbackbone(x)
#     features = [features[0], features[1], features[2], features[3]]
#     print(features[0].shape, pos_encodings[0].shape)
#     print(features[1].shape, pos_encodings[1].shape)
#     print(features[2].shape, pos_encodings[2].shape)
#     print(features[3].shape, pos_encodings[3].shape)

#     pos_encodings = [pos_encodings[0], pos_encodings[1], pos_encodings[2], pos_encodings[3]]
    
#     decoder = MonoDepthDecoder()
#     depth_map = decoder(features, pos_encodings)
#     print("Depth map shape:", depth_map.shape) 

    