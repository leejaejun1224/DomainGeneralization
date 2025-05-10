import torch
import torch.nn as nn
import torch.nn.functional as F



"""
드럽지만 일단 여기서 임베딩 -> 인풋 : (B, C, H, W)
아우풋은 (B, H*W, C)


"""
class PosembSelfAtt(nn.Module):
    def __init__(self, emb_dim, num_heads, patch_size, grid_size, input_channels):
        super(PosembSelfAtt, self).__init__()
        
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.patch_embed = nn.Conv2d(input_channels, emb_dim, kernel_size=patch_size, stride=patch_size)

        self.q_proj = nn.Linear(emb_dim, emb_dim)
        self.k_proj = nn.Linear(emb_dim, emb_dim)
        self.v_proj = nn.Linear(emb_dim, emb_dim)

        ## PROJECTIN
        self.proj = nn.Linear(emb_dim, emb_dim)

        ## rel pose bias
        self.patch_size = patch_size
        self.bias_table = nn.Parameter(torch.zeros(2 * grid_size[0] - 1, 2 * grid_size[1] - 1))
        ## ??????????mesh?


    def forward(self, x):
        
        if len(x.shape) == 4:
            x = self.patch_embed(x) # -> [batch, embed_dim, H/patch_size, W/patch_size]
            ## [batch, N, embed_dim]
            x = x.flatten(2).transpose(1,2)
            B, N, C = x.shape
        else:
            B, N, C = x.shape
            
        ## [batch, num_heads, N, head_dim]
        q = self.q_proj(x).view(B, N, self.num_heads, C // self.num_heads).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.num_heads, C // self.num_heads).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.num_heads, C // self.num_heads).transpose(1, 2)
        
        ## [batch, num_heads, N, N]
        attn = q @ k.transpose(-2, -1) * self.scale
        attn = attn 
        attn = F.softmax(attn, dim=-1)
        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, N, C)
        out = self.proj(out)
        

        return x



if __name__ == "__main__":
    
    emb_dim = 128
    num_heads = 4
    patch_size = 16
    img_size = (1248, 384)
    grid_size = (img_size[0]//patch_size, img_size[1]//patch_size)
    input_channels = 3
    batch_size = 4
    ## [batch, C, H, W]
    x = torch.randn(batch_size, input_channels, img_size[0], img_size[1])


    posemb_selfatt = PosembSelfAtt(emb_dim, num_heads, patch_size, grid_size, input_channels)
    print(posemb_selfatt(x).shape)