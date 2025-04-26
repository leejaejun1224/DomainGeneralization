import torch
import torch.nn as nn
from torch.nn import functional as F
from einops import rearrange

class RelPosBias2D(nn.Module):
    """Static relative position bias table."""
    def __init__(self, heads: int, height: int, width: int):
        super().__init__()
        num_rel_y = 2 * height - 1
        num_rel_x = 2 * width - 1
        self.table = nn.Parameter(torch.zeros(num_rel_y * num_rel_x, heads))
        # build index map [N, N]
        coords_h = torch.arange(height)
        coords_w = torch.arange(width)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing='ij'))  # [2,H,W]
        coords_flat = coords.reshape(2, -1)                                        # [2, N]
        rel = coords_flat[:, :, None] - coords_flat[:, None, :]
        rel[0] += height - 1
        rel[1] += width - 1
        self.register_buffer('index_map', rel[0] * (2 * width - 1) + rel[1])      # [N, N]

    def forward(self):
        # returns bias [heads, N, N]
        print(self.index_map.view(-1))
        bias = self.table[self.index_map.view(-1)].view(*self.index_map.shape, -1)  # [N, N, heads]
        return bias.permute(2, 0, 1)
    

if __name__ == "__main__":
    model = RelPosBias2D(heads=2, height=2, width=3)
    print(model())
