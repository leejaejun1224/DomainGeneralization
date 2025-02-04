import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)
