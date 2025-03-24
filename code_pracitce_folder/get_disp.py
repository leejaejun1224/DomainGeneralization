import os
import random
import numpy as np
import cv2
from PIL import Image

def load_left_image(path):
    """
    1) PIL -> NumPy 변환
    2) RGB -> BGR 순서로 변경(OpenCV 호환)
    """
    pil_img = Image.open(os.path.expanduser(path)).convert('RGB')
    rgb_np = np.array(pil_img, dtype=np.uint8)
    bgr_np = rgb_np[..., ::-1].copy()
    return bgr_np

def load_disparity(path):
    """
    KITTI GT 형식에 맞춰 /256 스케일 후 float32로 반환
    """
    disp_pil = Image.open(os.path.expanduser(path))
    disp_np = np.array(disp_pil, dtype=np.float32)
    disp_np /= 256.0  # KITTI 2015 GT 형식
    return disp_np

def random_crop_3_images(left_bgr, right_bgr, disp_np, crop_w=512, crop_h=256):
    """
    training=True 시, 512x256 랜덤 크롭
    (KITTI2015Dataset처럼)
    - x1: [0, w-crop_w]
    - y1: (0~h-crop_h) 또는 (0.3h~h-crop_h) 중 랜덤
    모든 이미지(left, right, disp)에 동일 좌표로 적용
    """
    h, w, _ = left_bgr.shape
    x1 = random.randint(0, w - crop_w)

    # KITTI2015Dataset 로직과 동일: y 범위 랜덤
    if random.randint(0, 10) >= 8:
        y1 = random.randint(0, h - crop_h)
    else:
        y1 = random.randint(int(0.3 * h), h - crop_h)

    left_cropped = left_bgr[y1:y1+crop_h, x1:x1+crop_w, :]
    right_cropped = right_bgr[y1:y1+crop_h, x1:x1+crop_w, :]
    disp_cropped = disp_np[y1:y1+crop_h, x1:x1+crop_w]

    return left_cropped, right_cropped, disp_cropped

def pad_3_images(left_bgr, right_bgr, disp_np, tgt_w=1248, tgt_h=384):
    """
    training=False 시, (1248×384)에 맞춰 상단+오른쪽 zero-padding
    (KITTI2015Dataset처럼)
    """
    h, w, c = left_bgr.shape
    top_pad = tgt_h - h
    right_pad = tgt_w - w
    if top_pad < 0 or right_pad < 0:
        raise ValueError("원본이 (1248×384)보다 큽니다. 패딩 불가")

    # Left
    left_padded = np.pad(
        left_bgr,
        pad_width=((top_pad, 0), (0, right_pad), (0, 0)),
        mode='constant',
        constant_values=0
    )
    # Right
    right_padded = np.pad(
        right_bgr,
        pad_width=((top_pad, 0), (0, right_pad), (0, 0)),
        mode='constant',
        constant_values=0
    )
    # Disparity
    disp_padded = np.pad(
        disp_np,
        pad_width=((top_pad, 0), (0, right_pad)),
        mode='constant',
        constant_values=0
    )

    return left_padded, right_padded, disp_padded

def shift_left_by_disparity(left_bgr, disp_np):
    """
    disparity가 0이 아닌 픽셀에 대해,
    (disp_val / 255)를 정수화해 x 좌표를 왼쪽으로 이동.
    - 이동된 픽셀만 채워지고, 나머지는 (0,0,0)
    - 반환: shifted_left
    - 또한, 각 픽셀이 (x->new_x)로 이동했다는 매핑 정보를 저장해서 함께 반환
      (나중에 오른쪽 이미지 픽셀과 비교 시 필요)
    """
    h, w, _ = left_bgr.shape
    shifted = np.zeros_like(left_bgr)  # 전부 검정(0)으로 초기화

    # 매핑 정보( (y,x) -> (y,new_x) ) 보관
    # 혹은 (y, new_x) <- 원본 x
    moved_positions = []  # (y, new_x, orig_x)

    for y in range(h):
        for x in range(w):
            disp_val = disp_np[y, x]
            if disp_val != 0:
                shift = int(disp_val)
                new_x = x - shift
                if 0 <= new_x < w:
                    shifted[y, new_x] = left_bgr[y, x]
                    # (y, new_x)에 왼쪽의 (y, x)가 왔다는 정보
                    moved_positions.append((y, new_x, x))

    return shifted, moved_positions

def sample_right_and_diff(shifted_left, right_bgr, moved_positions):
    """
    1) 왼쪽이 disparity만큼 이동되어 (y,new_x)에 픽셀이 놓였다면,
       오른쪽 이미지에서 (y,new_x)를 샘플링.
    2) shifted_left[y,new_x] - right_bgr[y,new_x] = diff
    3) diff_img에 결과를 저장 (음수가 있을 수 있으므로 absdiff를 예시로 사용)
       - '빼줘'라고 하셨으니, raw subtraction도 가능하지만,
         보통 시각화를 위해 absdiff를 많이 씁니다.
         여기서는 raw subtraction을 하되, 음수는 클리핑하겠습니다.
    """
    h, w, _ = shifted_left.shape
    diff_img = np.zeros_like(shifted_left, dtype=np.int16)  
    # int16으로 만들어두면 (음수~양수) 계산 가능

    for (y, new_x, orig_x) in moved_positions:
        left_val = shifted_left[y, new_x].astype(np.int16)  # BGR int16
        right_val = right_bgr[y, new_x].astype(np.int16)    # BGR int16
        diff_bgr = left_val - right_val
        # 사용자 취향에 따라 absdiff 할 수도 있고, 그대로 둬도 됨
        # 여기서는 '빼준다'는 요청대로 raw subtraction 후 시각화 위해 클리핑
        diff_bgr = np.clip(diff_bgr, -255, 255)  
        diff_img[y, new_x] = diff_bgr

    # 이제 diff_img는 int16 범위(-255~255).
    # 시각화를 위해 0~255로 옮길 필요가 있음
    # (음수 -> 어두운 부분을 표현하고 싶다면 별도 컬ormap이 필요.
    #  간단히 absdiff를 쓰면 이런 고민이 적어짐)

    # 여기서는 '빼준다'고 했으니, -값도 표현하려면, 128을 기준으로 가운데 정렬?
    # 간단히 absdiff가 아니라 실제 차이를 보려면:
    #   0   -> 128 (무차이)
    #   -255-> 0, +255->255
    # => scaled_value = (diff_value + 255) / 2
    # (예: -255 -> 0, 0->128, +255->255)
    # 채널별로 해도 되고, 일괄 변환
    diff_img_8u = diff_img + 255  # 범위: 0 ~ 510
    diff_img_8u = diff_img_8u // 2  # 범위: 0 ~ 255 (integer)
    diff_img_8u = diff_img_8u.astype(np.uint8)

    return diff_img_8u

def show_disparity_quality(left_path, right_path, disp_path, training=True):
    """
    1) left, right, disparity 로드
    2) (training=True) 시 랜덤 크롭, (training=False) 시 패딩
    3) disparity로 왼쪽 이미지를 shift
    4) shift된 위치에 대해 오른쪽에서 픽셀 샘플링 -> (shifted_left - right)에 대한 diff 계산
    5) OpenCV로 시각화
    """
    # (1) Load
    left_bgr = load_left_image(left_path)
    right_bgr = load_left_image(right_path)  # 동일하게 BGR 변환
    disp_np  = load_disparity(disp_path)

    # (2) 동일한 방식으로 left, right, disp 처리
    if training:
        left_bgr, right_bgr, disp_np = random_crop_3_images(left_bgr, right_bgr, disp_np)
    else:
        left_bgr, right_bgr, disp_np = pad_3_images(left_bgr, right_bgr, disp_np)

    # (3) 왼쪽 이미지 shift
    shifted_left, moved_positions = shift_left_by_disparity(left_bgr, disp_np)

    # (4) 오른쪽 이미지에서 (3)과 동일 위치를 샘플링 → 차이(diff) 구하기
    diff_img = sample_right_and_diff(shifted_left, right_bgr, moved_positions)
    
    # (Disparity 시각화를 위해)
    disp_visual = cv2.normalize(disp_np, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # (5) 시각화
    cv2.imshow("Left Image", left_bgr)
    cv2.imshow("Right Image", right_bgr)
    cv2.imshow("Disparity (Norm)", disp_visual)
    cv2.imshow("Shifted Left", shifted_left)
    cv2.imshow("Diff (ShiftedLeft - Right)", diff_img)

    print("[INFO] Press any key to continue, ESC to exit.")
    key = cv2.waitKey(0)
    if key == 27:
        cv2.destroyAllWindows()

# -------------------------------------------
# 사용 예시
# -------------------------------------------
if __name__ == "__main__":
    left_path = "/home/jaejun/dataset/kitti_2015/training/image_2/000000_10.png"
    right_path = "/home/jaejun/dataset/kitti_2015/training/image_3/000000_10.png"
    disp_path = "/home/jaejun/dataset/kitti_2015/training/disp_occ_0/000000_10.png"

    # training=True (512x256 랜덤 크롭 후 시각화)
    # show_disparity_quality(left_path, right_path, disp_path, training=True)

    # training=False (1248x384 패딩 후 시각화)
    show_disparity_quality(left_path, right_path, disp_path, training=False)
