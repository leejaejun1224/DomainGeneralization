import numpy as np
import matplotlib.pyplot as plt

# 1) 분포 배열 로드
flying = np.load('./mean_disparity_distribution_flyingthing.npy')
kitti  = np.load('./mean_disparity_distribution_kitti.npy')

# 2) 길이 맞추기 (둘 중 짧은 길이 기준)
common_len = min(len(flying), len(kitti))
flying = flying[:common_len]
kitti  = kitti[:common_len]
disparities = np.arange(common_len)

# 3) 비율 계산
eps = 1e-8
ratio = np.divide(kitti, flying, out=np.full_like(kitti, np.nan), where=flying > eps)
ratio = np.log(ratio+1)  # 로그 변환 (0을 피하기 위해 1을 더함)
# 4) 이동평균(선택)
def moving_average(x, w=7):
    if w <= 1:
        return x
    kernel = np.ones(w) / w
    mask   = np.isnan(x)
    x_fill = np.where(mask, 0, x)
    ma     = np.convolve(x_fill, kernel, mode='same')
    counts = np.convolve((~mask).astype(float), kernel, mode='same')
    with np.errstate(invalid='ignore'):
        ma = np.divide(ma, counts, out=np.full_like(ma, np.nan), where=counts > 0)
    return ma

ratio_smooth = moving_average(ratio, w=1)

# 5) 그래프 1: 분포
plt.figure(figsize=(10, 5))
plt.plot(disparities, flying, label='FlyingThings3D')
plt.plot(disparities, kitti,  label='KITTI')
plt.xlabel('Disparity (pixels)')
plt.ylabel('Proportion')
plt.title('Mean Disparity Distribution')
plt.legend()
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
plt.tight_layout()

# 6) 그래프 2: 비율
plt.figure(figsize=(10, 5))
plt.plot(disparities, ratio_smooth, label='KITTI / Flying (smoothed)')
plt.xlabel('Disparity (pixels)')
plt.ylabel('Ratio')
plt.title('KITTI vs FlyingThings3D Ratio per Disparity')
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
plt.legend()
plt.tight_layout()

plt.show()
