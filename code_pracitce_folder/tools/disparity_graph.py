import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# 제공해주신 disparity 읽기 함수
def load_disp(filename):
    filename = os.path.expanduser(filename)
    data = Image.open(filename)
    data = np.array(data, dtype=np.float32) / 256.
    return data

# disparity map의 분포를 히스토그램으로 출력하는 함수
def plot_disp_distribution(filename):
    # disparity map 불러오기
    disp = load_disp(filename)
    
    # 1차원 배열로 펼치기
    flatten_disp = disp.flatten()
    
    # 히스토그램 그리기
    plt.figure(figsize=(8, 5))
    plt.hist(flatten_disp, bins=256, range=(1, 192))
    plt.title('Disparity Distribution')
    plt.xlabel('Disparity Value')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # 테스트할 disparity map 파일 경로 (예: disparity.png)
    filename = '/home/jaejun/dataset/kitti_2015/training/disp_occ_0/000116_10.png'
    plot_disp_distribution(filename)
