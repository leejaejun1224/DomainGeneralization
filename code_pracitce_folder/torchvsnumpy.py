import time
import numpy as np
import torch

seed = 123
H, W = 2000, 2000
total = H * W
num_diff = 55445

np.random.seed(seed)
torch.manual_seed(seed)

# 랜덤 문자열 만들기
strings = ['ffsdfa','bsdfsd','caagg','sdfsdfd','esdgdfhfg','f','fhgfhs','jkllkjw','gasdsa','dfghfghf','hassdf','fsdfsdfdh']
a_cpu = np.random.choice(strings, size=(H, W))
b_cpu = a_cpu.copy()

# 랜덤으로 바꿀 숫자 뽑기
random_idx = np.random.choice(total, num_diff, replace=False)

xs = random_idx // W
ys = random_idx % W
b_cpu[xs, ys] = 'sdfsdf'

# 여 위까지만 대체하셈

# cpu 시작
start_cpu = time.time()
diff_cpu = np.count_nonzero(a_cpu != b_cpu)
cpu_total_ms   = (time.time() - start_cpu) * 1e3
cpu_diff_rate = diff_cpu / total * 100.0

print(f"CPU 다른거 비율 = {cpu_diff_rate:.3f}% // 계산 시간={cpu_total_ms:.3f} ms")


# 이건 gpu 시작
torch.cuda.synchronize()
start_gpu = time.time()

unique_vals = np.unique(np.concatenate([a_cpu.ravel(), b_cpu.ravel()]))
str2int_a = unique_vals.searchsorted(a_cpu).astype(np.int32)
str2int_b = unique_vals.searchsorted(b_cpu).astype(np.int32)
gpu_a = torch.from_numpy(str2int_a).cuda(non_blocking=True)
gpu_b = torch.from_numpy(str2int_b).cuda(non_blocking=True)
# 이건 gpu에 데이터 변환하고 올리는게 끝난 시점
torch.cuda.synchronize()
on_gpu = time.time()


diff_gpu = (gpu_a != gpu_b).sum()

# 계산 다 끝
torch.cuda.synchronize()
gpu_total_ms   = (time.time() - start_gpu) * 1e3
gpu_only_calc_ms = (time.time() - on_gpu) * 1e3

# 다른거 비율
gpu_diff_rate = diff_gpu.item() / total * 100.0

print(f"GPU 다른거 비율 = {gpu_diff_rate:.3f}% // 계산 시간 = {gpu_only_calc_ms:.3f} ms // 총 시간(메모리 올리는 것도 포함하면) = {gpu_total_ms:.3f} ms")