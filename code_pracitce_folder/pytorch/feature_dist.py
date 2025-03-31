from transformers import SegformerModel  # MiT 기반 모델
import torch

# MiT 백본 로드 (예: segformer-b0)
model = SegformerModel.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
model.eval()

# 두 이미지 입력
image1 = torch.randn(1, 3, 224, 224)  # 예시 입력
image2 = torch.randn(1, 3, 224, 224)

# 특징 추출
with torch.no_grad():
    outputs1 = model(image1).last_hidden_state  # (B, seq_len, hidden_size)
    outputs2 = model(image2).last_hidden_state

print(outputs1.shape)
# 특징 벡터로 변환 (예: 평균 풀링)
feat1 = outputs1.mean(dim=1)  # (B, hidden_size)
feat2 = outputs2.mean(dim=1)
print(feat1.shape)

# 코사인 거리 계산
cos_dist = 1 - torch.nn.functional.cosine_similarity(feat1, feat2, dim=1)
print("Feature Distance (Cosine):", cos_dist)