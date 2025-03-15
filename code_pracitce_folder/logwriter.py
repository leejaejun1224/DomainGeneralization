import numpy as np
import torch
def average_diff(lst):
    return np.mean(np.diff(lst))

data = [0.6, 0.612, 0.621, 0.625, 0.631, 0.635, 0.641, 0.645, 0.651, 0.655]

tensor = torch.tensor([img_id for img_id in data])
print(tensor.float())