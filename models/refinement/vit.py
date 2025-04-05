import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch.nn as nn
from functools import partial
from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import Block
import torchvision.transforms as pth_transforms
import cv2
import os
import colorsys
import requests

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class VisionTransformer(nn.Module):
    """ Vision Transformer """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                proj_drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x[:, 0]

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


def vit_small(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    
    if pretrained:
        url = "https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
        state_dict = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)
    
    return model


def vit_base(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    
    if pretrained:
        url = "https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
        state_dict = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)
    
    return model

def get_attention_maps(model, img):

    # Ensure model is in eval mode
    model.eval()
    
    # Store attention maps
    attention_maps = []
    
    # Hook function to capture attention maps
    def hook_fn(module, input, output):
        # Extract attention weights from the output
        # For most ViT implementations, this is the attention matrix before softmax
        # Shape is typically [B, num_heads, seq_len, seq_len]
        if hasattr(module, 'attn'):
            # Get the attention weights
            with torch.no_grad():
                B, N, C = input[0].shape
                qkv = module.attn.qkv(input[0]).reshape(B, N, 3, module.attn.num_heads, module.attn.head_dim).permute(2, 0, 3, 1, 4)
                q, k, v = qkv.unbind(0)
                q, k = module.attn.q_norm(q), module.attn.k_norm(k)
                
                # Calculate attention weights
                q = q * module.attn.scale
                attn = q @ k.transpose(-2, -1)
                attn = attn.softmax(dim=-1)
                
                # Store the attention weights
                attention_maps.append(attn.detach())
    
    # Register hooks for each transformer block
    hooks = []
    for block in model.blocks:
        hooks.append(block.register_forward_hook(hook_fn))
    
    # Forward pass
    with torch.no_grad():
        _ = model(img)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    return attention_maps

def apply_mask(image, mask, color, alpha=0.7):

    for c in range(3):
        image[:, :, c] = image[:, :, c] * (1 - alpha * mask) + alpha * mask * color[c] * 255
    return image

def random_colors(N, bright=True):

    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    np.random.shuffle(colors)
    return colors

def extract_and_visualize(model, img_path=None, output_dir="vit_attention_output"):

    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    # Load and preprocess image
    if img_path is None:
        url = "https://images.unsplash.com/photo-1600585154340-be6161a56a0c"
        image = Image.open(requests.get(url, stream=True).raw).convert('RGB')
    else:
        image = Image.open(img_path).convert('RGB')
    
    # Save original image
    image.save(os.path.join(output_dir, "original.png"))
    
    # Preprocess image
    transform = pth_transforms.Compose([
        pth_transforms.Resize(224),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    # Get attention maps
    attention_maps = get_attention_maps(model, img_tensor)
    
    # Prepare original image for visualization
    img_np = np.array(image.resize((224, 224)))
    
    # Visualize attention maps for each block and head
    # for block_idx in range(len(attention_maps)):
    block_idx = -1
    attn = attention_maps[block_idx]
    num_heads = attn.shape[1]
    
    for head_idx in range(num_heads):
        # Get attention from CLS token to patches
        attn_map = attn[0, head_idx, 0, 1:].cpu().numpy()
        
        # Reshape to grid
        grid_size = int(np.sqrt(attn_map.shape[0]))
        attn_grid = attn_map.reshape(grid_size, grid_size)
        
        # Normalize
        attn_norm = (attn_grid - attn_grid.min()) / (attn_grid.max() - attn_grid.min() + 1e-8)
        
        # Upsample
        attn_upsampled = cv2.resize(attn_norm, (224, 224), interpolation=cv2.INTER_NEAREST)
        
        # Save heatmap
        plt.figure(figsize=(10, 10))
        plt.imshow(attn_upsampled, cmap='viridis')
        plt.colorbar()
        plt.title(f'Attention Map - Block {block_idx}, Head {head_idx}')
        plt.axis('off')
        plt.savefig(os.path.join(output_dir, f"attn_block{block_idx}_head{head_idx}.png"), bbox_inches='tight')
        plt.close()
        
        # Apply mask to original image
        colors = random_colors(1)
        masked_img = img_np.copy()
        masked_img = apply_mask(masked_img, attn_upsampled, colors[0], alpha=0.7)
        
        # Save masked image
        plt.figure(figsize=(10, 10), frameon=False)
        ax = plt.Axes(plt.gcf(), [0., 0., 1., 1.])
        ax.set_axis_off()
        plt.gcf().add_axes(ax)
        ax.imshow(masked_img.astype(np.uint8))
        plt.savefig(os.path.join(output_dir, f"masked_block{block_idx}_head{head_idx}.png"))
        plt.close()
        
        print(f"Processed attention map for block {block_idx}, head {head_idx}")

if __name__ == "__main__":
    # Test basic model    
    # Test enhanced visualization
    model = vit_base(pretrained=True)
    extract_and_visualize(
        model,
        img_path="dog.bmp",
        output_dir="vit_attention_output"
    )
