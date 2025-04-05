import cv2
import numpy as np
import matplotlib.pyplot as plt

# 클릭 이벤트 처리 함수
def onclick(event):
    # 클릭한 좌표
    x, y = int(event.xdata), int(event.ydata)
    
    # 유효한 좌표인지 확인
    if x is not None and y is not None and 0 <= y < disparity.shape[0] and 0 <= x < disparity.shape[1]:
        # 해당 위치의 disparity 값
        disp_value = disparity[y, x]
        print(f"좌표: ({x}, {y}), Disparity 값: {disp_value}")

# Disparity 맵 로드 (DrivingStereo 기준)
disparity_path = "/home/jaejun/dataset/kitti_2015/testing/image_2/000116_10.png"  # 실제 disparity 맵 경로로 변경
disparity = cv2.imread(disparity_path, cv2.IMREAD_UNCHANGED)  # uint16 형식으로 읽기
# disparity = disparity.astype(np.float32) / 256.0  # float로 변환 및 스케일링

# 이미지 표시
fig, ax = plt.subplots()
ax.imshow(disparity, cmap='jet')  # disparity 맵을 컬러맵으로 표시
plt.title("Disparity Map")
# plt.colorbar()  # 값의 범위를 보여주는 컬러바 추가

# 클릭 이벤트 연결
fig.canvas.mpl_connect('button_press_event', onclick)

# 플롯 표시
plt.show()