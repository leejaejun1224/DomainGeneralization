import torch
import torch.nn as nn












"""
input : image, kernal, stride, padding

"""
def OverlapPatchMerging():
    return


    
"""
input : x (batch, width, height, channel)
"""
class EfficientSelfAttention(nn.Module):
    def __init__(self, reduction_ratio):
        super.__init__()
        self.reduction_ratio = reduction_ratio
        self.query = nn.Linear()
        self.key = nn.Linear()
        self.value = nn.Linear()
        
    
    def forward(x):
        batch, width, height, channel = x.shape
        
        N = width * height
        
        
        
        
    










