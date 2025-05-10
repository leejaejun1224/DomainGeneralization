import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import OrdinalEncoder

df = pd.DataFrame({
    'A': [0, 0, 2, 4, 2],
    'B': [0, 1, 0, 0, 0],
    'C': [5, 7, 5, 7, 9]
})
encoder = OrdinalEncoder()
df_encoded = pd.DataFrame(
    encoder.fit_transform(df),
    columns=df.columns
).astype(int)

print("인코딩 된 df:\n", df_encoded, "\n")

num_unique_values = df_encoded.nunique(axis=0).astype(int).to_numpy()
print(num_unique_values)

bits = np.ceil(np.log2(num_unique_values)).astype(np.int64)
print("bits", bits)
offsets = np.concatenate(([0], np.cumsum(bits[:-1])))
print("offsets", offsets)
pd_torch = torch.from_numpy(df_encoded.values).to(torch.int64).to('cuda')
offsets_t = torch.from_numpy(offsets).to(torch.int64).to('cuda')
print("offsets_t", offsets_t)
print("pd_torch", pd_torch)
print("pd_torch << offsets_t", pd_torch << offsets_t)
packed_pd = (pd_torch << offsets_t).sum(dim=1)
print("packed_pd", packed_pd)

total_bits = bits.sum()
print(total_bits)
assert total_bits <= 64, f"총 필요 비트 {total_bits}비트 → 64비트 초과!"
