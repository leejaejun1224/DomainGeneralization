import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_    



class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, embed_dim, patch_size, stride):
        super(PatchEmbedding, self).__init__()

        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size,
                               stride=patch_size, padding=(patch_size // 2, patch_size // 2))
        self.norm = nn.LayerNorm(embed_dim)
        self.apply(self._init_weights)
        
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)

    def forward(self, x):

        x = self.patch_embed(x)

        _, _, H, W = x.shape

        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)


        return x



class VisionTransformer(nn.Module):
    def __init__(self, img_size, in_chans, embed_dim,
                 depth, num_heads, qkv_bias, qk_scale, sr_ratio,
                 proj_drop, attn_drop, drop_path_rate, norm_layer=nn.LayerNorm):
        super(VisionTransformer, self).__init__()
        self.patch_embed = PatchEmbedding(in_channels=in_chans, embed_dim=embed_dim, patch_size=patch_size, stride=stride)
        
    def forward(self, x):
        x = self.patch_embed(x)

        _, _, H, W = x.shape

        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)        




def vit_small():
    model = VisionTransformer(img_size=224, in_chans=3, embed_dim=384,
                              depth=12, num_heads=6, qkv_bias=True, qk_scale=None,
                              sr_ratio=16, proj_drop=0.1, attn_drop=0.1,
                              drop_path_rate=0.1, norm_layer=nn.LayerNorm)

    return model