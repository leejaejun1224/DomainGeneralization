import torch
from itertools import starmap

a1 = torch.randn(2, 3, 4)
a2 = torch.randn(2, 3, 4)

data_batch = {
    "label": torch.tensor([10, 20]),
    "meta": [a1, a2],
    "file_path": [
        'leftImg8bit_trainvaltest/leftImg8bit/test/bielefeld/bielefeld_000000_025748_leftImg8bit.png',
        'leftImg8bit_trainvaltest/leftImg8bit/test/bielefeld/bielefeld_000001_025748_leftImg8bit.png'
    ]
}

def split_batch(data_batch, batch_idx):
    batch = {}
    for key, value in data_batch.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value[batch_idx].data  
        elif isinstance(value, list):
            if isinstance(value[0], torch.Tensor):
                batch[key] = [v[batch_idx].data for v in value]  
            else:
                batch[key] = value[batch_idx] 
    return batch

batch_size = 2
for i in range(batch_size):
    batch = split_batch(data_batch, i)
    print(f"batch_idx {i}")
    print(batch['meta'])

