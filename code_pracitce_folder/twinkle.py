import cv2
import numpy as np

def add_specular_highlight(img_path,
                           save_path=None,
                           center=None,          # (x,y). None이면 이미지 중앙
                           radius=80,            # 하이라이트 반지름(px)
                           strength=0.7,         # 0~1, 세기
                           feather=0.6,          # 0~1, 가장자리 부드럽게
                           mode="screen",        # "screen" or "add"
                           blobs=1,              # 여러 개 찍고 싶으면 >1
                           seed=None):
    """
    이미지 중간(혹은 지정 위치)에 반짝이는 하이라이트를 얹는다.
    """
    if seed is not None:
        np.random.seed(seed)

    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(img_path)
    h, w = img.shape[:2]
    img_f = img.astype(np.float32) / 255.0

    out = img_f.copy()
    for _ in range(blobs):
        cx = center[0] if center else w // 2
        cy = center[1] if center else h // 2
        if center is None and blobs > 1:
            cx = np.random.randint(int(0.3*w), int(0.7*w))
            cy = np.random.randint(int(0.3*h), int(0.7*h))

        # 가우시안 마스크 생성
        yy, xx = np.mgrid[0:h, 0:w]
        r2 = ((xx - cx) ** 2 + (yy - cy) ** 2).astype(np.float32)
        sigma = radius ** 2
        mask = np.exp(-r2 / (2 * sigma))
        # feather로 테두리 더 부드럽게
        mask = mask ** (1.0 - feather)
        mask = (mask * strength).clip(0, 1)[..., None]  # (H,W,1)

        if mode == "screen":
            # screen blend: 1 - (1-A)*(1-B)
            out = 1 - (1 - out) * (1 - mask)
        elif mode == "add":
            out = np.clip(out + mask, 0, 1)
        else:
            raise ValueError("mode should be 'screen' or 'add'")

    out_uint8 = (out * 255).astype(np.uint8)
    if save_path:
        cv2.imwrite(save_path, out_uint8)
    return out_uint8

# 사용 예시
result = add_specular_highlight("/home/jaejun/dataset/flyingthing/FlyingThings3D_subset_image_clean/FlyingThings3D_subset/train/image_clean/left/0000001.png",
                                "left_glare.png", radius=60, strength=0.6, blobs=3)
