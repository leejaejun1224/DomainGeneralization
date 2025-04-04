import torch
import torch.nn as nn
from vision_transformer import vit_small
from resnet import resnet50

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = vit_small()

url = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url, map_location="cpu")
model.state_dict().update(state_dict)


"""
input : left image [B, 3, N, W], initial disparity map [B, 1, H, W] (option : left attention map) 
output : refined disparity map [B, 1, H, W]
"""

class Refine(nn.Module):
    def __init__(self):
        super(Refine, self).__init__()

    def forward(self, left, initial_disp, attention_maps):

        left = left.to(device)
        initial_disp = initial_disp.to(device)
        attention_maps = attention_maps.to(device)

        left = model(left)
        

        return 0
