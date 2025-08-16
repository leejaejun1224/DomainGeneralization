import numpy as np
import matplotlib.pyplot as plt

# 0~100초 구간, 1초 간격으로 생성
t = np.arange(0, 101, 1)  # 시간 (초)
y = np.random.rand(len(t))  # 랜덤값 (0~1 사이)

# 그래프 그리기
plt.figure(figsize=(8, 4))
plt.plot(t, y, linewidth=1.5)
plt.xlabel("Time (s)")
plt.ylabel("Random value")
plt.title("Random values over 100 seconds")
plt.grid(True, linestyle="--", alpha=0.5)

# PDF로 저장
plt.savefig("random_plot.png", format="png")
plt.close()
