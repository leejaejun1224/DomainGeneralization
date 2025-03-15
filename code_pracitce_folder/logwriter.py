import torch

random_tensor = torch.randn(2, 3, 4)
threshold = torch.tensor([0.2, 0.6])
threshold = threshold.unsqueeze(1).unsqueeze(2)
print("threshold", threshold)
mask = random_tensor > threshold
num_pixels = mask.numel()

print(mask)
print(mask.sum(dim=(0,1,2)))
print(num_pixels)
print(mask.sum(dim=(0,1,2))/num_pixels)