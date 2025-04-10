import cv2
import numpy as np
import torch
import torch.nn.functional as F

# 데이터 로드 (히트맵 데이터를 numpy 배열로 가정)
image = cv2.imread('/home/cvlab/DomainGeneralization/log/2025-04-07_23_10/top_one/000088_10.png', cv2.IMREAD_GRAYSCALE).astype(np.float32)
image = image[17:285, 12:888]

# PyTorch 텐서로 변환 (H, W) -> (1, 1, H, W) 형태로 변환
image_tensor = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

# 커널 크기 설정
kernel_size = 15

# 이미지 패딩 추가 (PyTorch의 pad 사용)
padding = kernel_size // 2
padded_image = F.pad(image_tensor, (padding, padding, padding, padding), mode='reflect')  # (1, 1, H+pad, W+pad)

# 커널 생성 (로컬 평균 계산용)
kernel = torch.ones(1, 1, kernel_size, kernel_size) / (kernel_size * kernel_size)  # (1, 1, kernel_size, kernel_size)

# 로컬 평균 계산
local_mean = F.conv2d(padded_image, kernel, padding=0)  # 패딩 이미 적용됨, (1, 1, H, W)

# 로컬 표준편차 계산 (제곱 차이의 평균의 제곱근)
local_sqr_mean = F.conv2d(padded_image**2, kernel, padding=0)  # (1, 1, H, W)
local_std = torch.sqrt(torch.clamp(local_sqr_mean - local_mean**2, min=0))  # (1, 1, H, W)

# Z-score 계산
z_score = torch.abs((image_tensor - local_mean) / torch.clamp(local_std, min=1e-10))  # (1, 1, H, W)

# 이상치 마스크 생성
outlier_mask = z_score > 1.5  


image_np = image_tensor.squeeze().numpy()  # (H, W)
denoised_image = image_np.copy()

from scipy.ndimage import median_filter
denoised_image[outlier_mask.squeeze().numpy()] = median_filter(image_np, size=kernel_size)[outlier_mask.squeeze().numpy()]

denoised_image = cv2.normalize(denoised_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

colored_denoised = cv2.applyColorMap(denoised_image, cv2.COLORMAP_JET)

normalized_original = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
colored_original = cv2.applyColorMap(normalized_original, cv2.COLORMAP_JET)

cv2.imshow('Original Heatmap', colored_original)
cv2.imshow('Denoised Heatmap', colored_denoised)
cv2.waitKey(0)  # 키 입력 대기
cv2.destroyAllWindows()  # 창 닫기

cv2.imwrite('denoised_heatmap.png', colored_denoised)
cv2.imwrite('original_heatmap.png', colored_original)