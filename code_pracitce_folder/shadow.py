"""
shadow_aug.py  ──────────────────────────────────────────────
사용법
------
python shadow_aug.py input.jpg --out out.jpg --pair input_R.jpg

- input.jpg      : (필수) 왼쪽 이미지 경로
- --pair         : (선택) 오른쪽 이미지 경로까지 주면 '쌍' 증강 → out_R.jpg 로 저장
- --out          : (선택) 저장 파일명 (기본 ⟨입력명⟩_shadow.jpg)

의존 패키지
-----------
pip install opencv-python numpy pillow
"""

import cv2, argparse, os, random
import numpy as np
from PIL import Image


# ───────────────────────── 하이라이트 보조 함수
def _random_polygon_mask(h, w, parts=6):
    """0~1 랜덤 다각형(불규칙 그림자)"""
    cx, cy = random.randint(0, w), random.randint(0, h)
    angles = np.sort(np.random.uniform(0, 2*np.pi, parts))
    rads   = np.random.uniform(min(h, w)*0.2, max(h, w)*0.6, parts)
    pts = np.stack([cx + rads*np.cos(angles), cy + rads*np.sin(angles)], axis=-1)
    pts = np.clip(pts, 0, [[w-1, h-1]])
    mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(mask, [pts.astype(np.int32)], 255)
    return mask.astype(np.float32)/255.0


def _random_gradient_mask(h, w):
    """0~1 대각선 soft gradient"""
    direction = random.choice(['h', 'v', 'd'])
    if direction == 'h':      # left→right
        grad = np.tile(np.linspace(0, 1, w), (h, 1))
    elif direction == 'v':    # top→bottom
        grad = np.tile(np.linspace(0, 1, h), (w, 1)).T
    else:                     # diagonal
        x = np.linspace(0, 1, w)
        y = np.linspace(0, 1, h)[:, None]
        grad = (x + y) / 2
    grad = cv2.GaussianBlur(grad, (0, 0), sigmaX=max(h, w)*0.05)
    return grad.astype(np.float32)


# ───────────────────────── 증강 함수
def apply_shadow(img, prob=0.6):
    """다각형 + 그라디언트 그림자 혼합"""
    if random.random() > prob:
        return img

    h, w = img.shape[:2]
    alpha = random.uniform(0.2,0.5)                      # 어둠
    mask_poly = _random_polygon_mask(h, w, parts=random.randint(5, 9))
    mask_grad = _random_gradient_mask(h, w)
    mask = np.maximum(mask_poly, mask_grad)               # 0~1
    shadow = alpha + (1 - alpha) * mask                   # 0.3~1
    return (img * shadow[..., None]).astype(img.dtype)

def darken_only(img, bright_range=(0.5, 1.0)):
    """채널별 왜곡 없이 전체 밝기만 낮춘다."""
    factor = random.uniform(*bright_range)  # 1.0 = 그대로, 0.4 = 많이 어둡게
    print(factor)
    return (img.astype(np.float32) * factor).clip(0, 255).astype(img.dtype)
# ───────────────────────── 네모 그림자 보조 함수
def _random_rect_shadow(img, prob=0.7,
                        h_ratio=(0.2, 0.5),   # 직사각형 높이 비율
                        w_ratio=(0.1, 0.9),   # 직사각형 너비 비율
                        alpha_range=(0.25, 0.55),  # 어둡게 할 정도
                        blur_sigma=1):
    """
    이미지 하단에 네모난 그림자를 얹어 반환.
    - prob: 1.0 → 항상, 0.0 → 절대 X
    - h_ratio / w_ratio: 직사각형 크기 (전체 대비 비율, 튜플이면 랜덤)
    - alpha_range: 0.0(완전검정)~1.0(원본) 가운데 랜덤
    - blur_sigma: 모서리 부드럽게
    """
    if random.random() > prob:
        return img

    h, w = img.shape[:2]

    # 랜덤 크기‧위치 계산
    rh = int(h * random.uniform(*h_ratio))
    rw = int(w * random.uniform(*w_ratio))
    # y 는 하단부 쪽으로만, x 는 자유롭게
    y0 = random.randint(int(h * 0.55), max(h - rh - 1, int(h * 0.55)))
    x0 = random.randint(0, w - rw - 1)

    # 마스크 생성 (1 = 그대로, 0 = 완전검정)
    mask = np.ones((h, w), np.float32)
    darkness = random.uniform(*alpha_range)
    print(darkness)
    mask[y0:y0 + rh, x0:x0 + rw] = darkness
    mask = cv2.GaussianBlur(mask, (0, 0), blur_sigma)

    # 적용
    out = (img.astype(np.float32) * mask[..., None]).clip(0, 255).astype(img.dtype)
    return out


def augment_pair(imgL, imgR=None):
    """좌·우 쌍 증강. (네모 그림자 → 다각형 그림자 → 전체 감광)"""
    # 1) 하단 네모 그림자
    imgL = _random_rect_shadow(imgL)
    # 2) 기존 다각형/그래디언트 그림자
    # imgL = apply_shadow(imgL)
    # 3) 전체 밝기 낮추기
    # imgL = darken_only(imgL)

    if imgR is not None:
        imgR = _random_rect_shadow(imgR)
        # imgR = apply_shadow(imgR)
        # imgR = darken_only(imgR)
        return imgL, imgR
    return imgL, None





def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--img', default="/home/jaejun/dataset/flyingthing/FlyingThings3D_subset_image_clean/FlyingThings3D_subset/train/image_clean/left/0000001.png", help='left image path')
    ap.add_argument('--pair', help='right image path (optional)')
    ap.add_argument('--out', default='./shadow.png')
    args = ap.parse_args()

    imgL = cv2.imread(args.img, cv2.IMREAD_COLOR)
    if imgL is None:
        raise FileNotFoundError(args.img)
    imgR = cv2.imread(args.pair, cv2.IMREAD_COLOR) if args.pair else None

    augL, augR = augment_pair(imgL, imgR)

    outL = args.out or os.path.splitext(args.img)[0] + '_shadow.jpg'
    cv2.imwrite(outL, augL)
    print('saved', outL)

    if augR is not None:
        outR = os.path.splitext(args.pair)[0] + '_shadow.jpg'
        cv2.imwrite(outR, augR)
        print('saved', outR)


if __name__ == '__main__':
    main()
