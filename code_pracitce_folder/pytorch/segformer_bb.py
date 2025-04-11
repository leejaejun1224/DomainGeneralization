import torch
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import cv2
from transformers import SegformerModel, SegformerConfig
import torchvision.transforms as transforms
import requests

def get_transform():
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
def unnormalize(tensor, mean, std):
    # tensor: [3, H, W]
    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)
    return tensor * std + mean

image_path = "/home/jaejun/Pictures/product-card-512x512.png"

# url = "http://images.cocodataset.org/val2017/000000039769.jpg"    
img_pil = Image.open(image_path).convert("RGB")

transform = get_transform()
img = transform(img_pil).numpy()

# 원본 높이, 너비
h, w = img.shape[1], img.shape[2]
top_pad = 384 - h
right_pad = 1248 - w
# pad images
# img = np.lib.pad(img, ((0, 0), (top_pad, 0), (0, right_pad)), mode='constant', constant_values=0)
img = torch.from_numpy(img).unsqueeze(0)
print("Input:", img.shape)  # [1, 3, 384, 1248]

config = SegformerConfig.from_pretrained(
    "nvidia/segformer-b0-finetuned-ade-512-512",
    output_attentions=True,
    output_hidden_states=True
)
model = SegformerModel.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512", config=config)
model.eval()

encoder = model.encoder
features = encoder(img, output_hidden_states=True, output_attentions=True)

# hidden states shape (batch, channel, height, width)
print("Hidden states:")
for i, hs in enumerate(features.hidden_states):
    print(f"  hidden_states[{i}]: {hs.shape}")

# attentions shape (batch, head, seq_len_q, seq_len_k)
print("Attentions:")
for i, attn in enumerate(features.attentions):
    print(f"  attentions[{i}]: {attn.shape}")

########################################################################
# (1) 실제로 Query/Key를 2D (H,W) 형태로 되돌릴 때, 
#     stage 0~1 (hidden_states[0]) => 96*312 = 29952 
#                 (hidden_states[1]) => 48*156 = 7488
#     Key => 12*39 = 468
#     Head 수 => 첫 두 어텐션: 1 head, 뒤 두 어텐션: 2 heads
########################################################################

# 예: 첫 두 attentions는 shape (1, 1, 29952, 468)
#     이를 (1, 1, 96, 312, 12, 39)로 바꿀 수 있다
#     뒤 두 attentions는 (1, 2, 7488, 468) -> (1, 2, 48, 156, 12, 39)

# Query 해상도, 어텐션 head 수를 하드코딩 예시
# B0 기준 (stage0, stage1, stage2, stage3) & 각각 블록 2개씩
# stage0 (96,312), stage1 (48,156), stage2 (24,78), stage3 (12,39)
# but each stage has 2 blocks => total 8
q_shapes = [
    (128, 128), (128, 128),   # stage0 block1, block2
    (64, 64), (64, 64),   # stage1 block1, block2
    (32, 32),  (32, 32),    # stage2 block1, block2
    (16, 16),  (16, 16),    # stage3 block1, block2
]



num_heads = [
    1, 1,   # stage0 block1, block2
    2, 2,   # stage1 block1, block2
    5, 5,   # stage2 block1, block2
    8, 8,   # stage3 block1, block2
]


K_h, K_w = 16, 16

reshaped_attentions = []
for i in range(len(features.attentions)):  # = 8
    attn_map = features.attentions[i]
    b, h, q_len, k_len = attn_map.shape
    qH, qW = q_shapes[i]
    real_heads = num_heads[i]
    
    # 실제로 k_len이 12*39 인지 확인 (468) 
    # 또는 stage별 key 해상도가 다른지 체크
    attn_reshaped = attn_map.view(b, real_heads, qH, qW, K_h, K_w)
    reshaped_attentions.append(attn_reshaped)

    print(i, attn_reshaped.shape)

import cv2
import matplotlib.pyplot as plt
selected_block_index = 7   # 원하는 output 리스트 번호 선택
selected_block = reshaped_attentions[selected_block_index]
# 각 블록에 따른 query grid shape 및 헤드 수
q_shape = q_shapes[selected_block_index]
n_heads = num_heads[selected_block_index]

# 지정 query 위치 (예: 해당 블록 내에서 원하는 위치로 선택)
# 각 블록의 query 해상도에 맞게 인덱스를 지정합니다.
# 예: q_shape = (12, 39) 인 경우, 중앙 근처의 위치를 사용
qH_idx, qW_idx = q_shape[0] // 2, q_shape[1] // 2 

# 복원된 원본 이미지 (정규화 되기 전의 색상) 계산 (0~1 범위)
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
img_unnorm = unnormalize(img.squeeze(0), mean, std)
img_unnorm = img_unnorm.permute(1, 2, 0).numpy()
img_unnorm = np.clip(img_unnorm, 0, 1)  # [384, 1248, 3]

# 투명도 설정 및 컬러맵 설정 ('jet' 사용)
alpha = 0.75
colormap = plt.get_cmap('jet')

# 각 head에 대해 오버레이 결과를 이미지로 저장
for head_idx in range(n_heads):
    # 각 head의 지정 query 위치에 해당하는 attention map: shape (12, 39)
    attention_map = selected_block[0, head_idx, qH_idx, qW_idx, :, :].detach().numpy()
    
    # 리사이즈: attention map을 원본 이미지 크기 (1248, 384)로 확장
    attention_map_resized = cv2.resize(attention_map, (512, 512), interpolation=cv2.INTER_LINEAR)
    
    # 정규화: 0 ~ 1
    attention_map_resized = (attention_map_resized - attention_map_resized.min())
    if attention_map_resized.max() != 0:
        attention_map_resized = attention_map_resized / attention_map_resized.max()
    
    # 컬러맵 적용: (H, W, 3), RGB [0,1]
    attention_colored = colormap(attention_map_resized)[:, :, :3]
    
    # 오버레이: 복원된 이미지에 attention heatmap을 알파 혼합
    overlay = img_unnorm + attention_colored * alpha
    overlay = np.clip(overlay, 0, 1)
    
    # 결과 저장: head별로 파일 저장 (matplotlib의 imsave 사용)
    filename = f"attention{selected_block_index}_head{head_idx}.png"
    plt.imsave(filename, overlay)
    print(f"Saved overlay for block {selected_block_index}, head {head_idx} as {filename}")
