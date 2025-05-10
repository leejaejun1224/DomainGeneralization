import pandas as pd
import pickle
import time
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
import torch
import cupy as cp
from tqdm import tqdm

# 1) 데이터 로드 및 타입 지정
dtypes = {
    'age':'int','workclass':'category','education':'category',
    'marital_status':'category','occupation':'category',
    'relationship':'category','race':'category','gender':'category',
    'capital_gain':'int','capital_loss':'int','hours_per_week':'int',
    'native_country':'category','income':'category'
}
columns = list(dtypes.keys())

data = (
    pd.read_csv('/home/cvlab/Downloads/adultSalary.csv', names=columns)
      .astype(dtypes)
      .reset_index(drop=True)
)
with open('/home/cvlab/Downloads/df_synthpop.pickle','rb') as fr:
    synth_data = pickle.load(fr)
    
# synth_data = synth_data.reset_index(drop=True)


repeat_times = 4
data = pd.concat([data]*repeat_times, ignore_index=True)
synth_data = pd.concat([synth_data]*repeat_times, ignore_index=True)

# print("=====미친 2중 for문 버전=====")
# data_list = []
# synth_data_list = []
# for i in range(len(synth_data)):
#     data_list.append(data.iloc[i, :].values.flatten().tolist())
#     synth_data_list.append(synth_data.iloc[i, :].values.flatten().tolist())

# start = time.time()
# cnt = 0
# for row1 in tqdm(synth_data_list):
#     for row2 in data_list:
#         if row1 == row2:
#             cnt += 1
#             break
# end = time.time()
# print(f"미친 2중 for문 수행 시간: {end - start:.4f}초")
# print(f"미친 2중 for문 중복 비율 : {cnt / len(synth_data) * 100:.8f}%")


print("=====numpy 버전=====")
data_midx  = pd.MultiIndex.from_frame(data)
synth_midx = pd.MultiIndex.from_frame(synth_data)
start = time.time()
mask_cpu = synth_midx.isin(data_midx)
end   = time.time()
cnt_cpu = mask_cpu.sum()
ratio_cpu = cnt_cpu / len(synth_data) * 100
print(f"cpu isin 수행 시간: {end - start:.4f}초")
print(f"cpu 매칭 비율      : {ratio_cpu:.8f}%")
print("\n")





print("=====torch 버전=====")
# 여기서 데이터가 범주형인 컬럼만 추출하면 13개 중의 컬럼 중에서 9개만 추출이 됨
cat_cols = data.select_dtypes(include=['category','object']).columns.tolist()

## sklearn에서 제공하는 범주형 컬럼 인코딩 라이브러리를 써서 데이터가 숫자가 아니라 범주형이 것들만 인코딩을 해줄거임.
## 일단 고 짓거리를 해주눈 클래스를 하나 불러오고 
encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

##이 클래스에 우리의 데이터를 넣어주고 fit 함수를 돌리면 자기들끼리 데이터를 숫자로 만드는 규칙을 생성하고 
## 그 규칙을 이용해서 데이터를 숫자로 변환해줌
## 근데 왜 두개를 concat을 해서 넣어주냐?
## 왜냐하면 같은 규칙을 적용을 해야하거든. 그렇지 때문에 48842x13크기의 행렬을 아래로 붙여서(axis=0) 
# 97684x13크기의 행렬을 만들어서 그 행렬에 규칙을 만들어줄거임
encoder.fit(pd.concat([data[cat_cols], synth_data[cat_cols]], axis=0, ignore_index=True))

data_enc  = data.copy()
synth_enc = synth_data.copy()

## 그러면 이제 cat_cols안에 들어있는 카테고리 9개들에 대한 그 열의 문자열 데이터만 위의 encoder에서 정한 규칙대로
## 싹 바꿔서 끼워넣어주는 거임.
data_enc[cat_cols]   = encoder.transform(data[cat_cols])
synth_enc[cat_cols]  = encoder.transform(synth_data[cat_cols])

## 자료형을 다 정수형으로 바꿔주고
arr_orig  = data_enc.values.astype(np.int64)
arr_synth = synth_enc.values.astype(np.int64)


## 이제 여기가 핵심임
## 내가 말한 이진수로 바꾸면 안되냐고 했던 부분이 이 부분
## 개념은 비트패킹이라고 하네
unique_values = data_enc.nunique(axis=0).astype(int).to_numpy()
bits = np.ceil(np.log2(unique_values)).astype(np.int64)

## 이 오프셋은 옆으로 밀 칸을 말함.
## 혹은 각 13개의 컬럼이 몇 번째 칸부터 몇 번째 칸까지 들어갈건지 정해둔다고 생각하면 됨
## 첫 번째 열은 첫 번째 칸부터 들어갈거라 맨 처음에 0번째 부터 들어갈거라는 뜻으로 0을 concat
offsets = np.concatenate(([0], np.cumsum(bits[:-1])))
assert bits.sum() <= 64, f"비트패킹에 필요한 비트 {bits.sum()}비트 → 64비트 초과!"


### gpu에 올려주고
device       = 'cuda'
orig_tensor  = torch.from_numpy(arr_orig).to(torch.int64).to('cuda')
synth_tensor = torch.from_numpy(arr_synth).to(torch.int64).to('cuda')
offsets_t    = torch.from_numpy(offsets).to(torch.int64).to('cuda')



### 위에서 정한 offset 칸에다가 데이터를 차곡차곡 집어넣음
packed_orig  = (orig_tensor  << offsets_t).sum(dim=1)
packed_synth = (synth_tensor << offsets_t).sum(dim=1)

## 다시 정리하자면 48842x13 크기의 matrix가 있고,
# 1. 이 13개의 열에서 각각 하나의 열마다 서로 다른 값이 몇 개 있는지를 알아낸 후
# 2. 그 값의 가짓수를 최소 몇 비트가 있으면 모두 표현할 수 있는지 계산하면
# 3. 각 열마다 몇 비트가 있으면 나는 이 열의 모든 조합을 만들 수 있어! 를 알게되고
# 4. 그래서 각 열마다 그 열의 모든 조합을 만들어서 넣을 수 있는 칸을 만들어둠
"""
예를 들면
범주 : A, B, C
1행 : 1, 2, 3
2행 : 2, 3, 4
3행 : 2, 4, 5
4행 : 1, 5, 3
5행 : 1, 6, 4

이렇게 있으면
A는 1,2 이렇게 2가지 0과1 조합만 만들면 되니까 0,1이 들어갈 한 칸만 있으면 됨 -> 1비트
B는 근데 2,3,4,5,6 5가지를 다 만들어야 하니까 2칸으론 부족하고 3칸은 있어야 함 -> 3비트
C는 3,4,5 3가지 조합만 만들면되니까 0,1이 들어갈 2칸만 있으면 됨 -> 2비트

그러면 A는 1비트, B는 3비트, C는 2비트 총 6비트 필요
그러면 offset은?
A는 0번째 칸부터 한 칸 먹고, B는 1번째 칸부터 3칸 먹고, C는 4번째 칸부터 2칸 먹고
그러면 offset은" [[0], [1], [4]] -> 이 [] 안에 숫자 계산해주는 놈이 cumsum

[[0],<요 한 칸에 A채워 넣고> [1], <요 세 칸에 B채워 넣고> [4] <요 두 칸에 C채워 넣고>]

"""


## 48842구요
M = packed_synth.size(0)

torch.cuda.synchronize()
t0 = time.time()

## 킹즈인
mask_gpu = torch.isin(packed_synth, packed_orig)

torch.cuda.synchronize()
gpu_time = time.time() - t0

cnt_gpu   = int(mask_gpu.sum().item())
ratio_torch = cnt_gpu / M * 100

print(f"torch isin 수행 시간: {gpu_time:.4f}초")
print(f"torch 중복 비율 : {ratio_torch:.8f}%")
print("\n")


print("=====수경이가 사랑하는 cupy 버전=====")
orig_cp    = cp.asarray(arr_orig)
synth_cp   = cp.asarray(arr_synth)
offsets_cp = cp.asarray(offsets)

packed_orig  = (orig_cp  << offsets_cp).sum(axis=1)
packed_synth = (synth_cp << offsets_cp).sum(axis=1)

start_evt = cp.cuda.Event(); end_evt = cp.cuda.Event()
start_evt.record()
mask_cp = cp.isin(packed_synth, packed_orig)
end_evt.record()
end_evt.synchronize()
gpu_time_ms = cp.cuda.get_elapsed_time(start_evt, end_evt)

cnt_gpu   = int(mask_cp.sum().item())
ratio_cupy = cnt_gpu / packed_synth.shape[0] * 100.0

print(f"cupy isin 수행 시간: {gpu_time_ms/1000:.4f}초")
print(f"cupy 중복 비율 : {ratio_cupy:.8f}%")