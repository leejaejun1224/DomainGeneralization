from transformers import SegformerImageProcessor, SegformerModel
from PIL import Image
import torch

# 1. Image Processor와 모델 로드
# 최신 transformers에서는 feature_extractor_type 생략
processor = SegformerImageProcessor.from_pretrained('nvidia/segformer-b0-finetuned-ade-512-512')
model = SegformerModel.from_pretrained('nvidia/segformer-b0-finetuned-ade-512-512')
encoder = model.encoder

# 2. 이미지 로드
image_path = "/Users/leejaejun/workspace/dataset/etc/minji.jpg"  # 실제 경로로 변경
image = Image.open(image_path).convert("RGB")
print("Original image size:", image.size)  # 원본 크기 확인 (1280, 1824 예상)

# 3. 이미지 전처리 (512x512로 리사이즈)
inputs = processor(images=image, return_tensors="pt")
pixel_values = inputs["pixel_values"]  # [1, 3, 512, 512]
print("Processed input shape:", pixel_values.shape)  # 전처리 후 크기 확인

# 4. Encoder에 입력 전달 (hidden_states 출력 활성화)
with torch.no_grad():
    encoder_outputs = encoder(pixel_values, output_hidden_states=True)

# 5. Encoder 출력 확인
if encoder_outputs.hidden_states is not None:
    for i, feature_map in enumerate(encoder_outputs.hidden_states):
        print(f"Stage {i+1} output shape: {feature_map.shape}")
else:
    print("Error: hidden_states is None. Check encoder configuration.")

# 6. 모델을 평가 모드로 설정
model.eval()