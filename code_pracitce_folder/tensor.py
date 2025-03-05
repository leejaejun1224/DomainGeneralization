import torch
import torch.nn as nn

H, W = 2,3

relative_position_bias_table = nn.Parameter(torch.zeros(2 * W - 1, 2 * H - 1))

x = torch.arange(H)
y = torch.arange(W)

grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')

coord = torch.stack(torch.meshgrid(x, y, indexing='ij'))
coord_flatten = coord.flatten(1)

# 차원 하나가 추가가 되네
rel_coord = coord_flatten[:,:,None] - coord_flatten[:,None,:]
rel_coord = rel_coord.permute(1, 2, 0).contiguous()

rel_coord[:, :, 0] += H - 1
rel_coord[:, :, 1] += W - 1
rel_coord[:, :, 0] *= 2 * W - 1
rel_pos_index = rel_coord.sum(-1)

relative_position_bias = relative_position_bias_table[rel_pos_index.view(-1)].view(
    H * W, H * W, -1
)

print(relative_position_bias)
