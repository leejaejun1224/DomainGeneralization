# models/selfscan/transform.py
import numpy as np, cv2
from PIL import Image, ImageFilter
import torch, torch.nn.functional as F
from torchvision import transforms

# ───────── 공통 util ─────────
TARGET_W, TARGET_H = 1248, 384

def pad_to_size(t):                       # t:[C,H,W]
    _, h, w = t.shape[-3:]
    return F.pad(t, (0, TARGET_W-w, 0, TARGET_H-h), mode="reflect")

def rgb_with_edge_enhancement(t):         # t:[3,H,W] 0–255
    """RGB 채널에 edge 정보를 혼합하여 3채널 유지"""
    # Sobel edge 계산
    gray = t.mean(0, keepdim=True)
    kx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=t.dtype).view(1,1,3,3)
    ky = kx.transpose(-1,-2)
    gx = F.conv2d(gray[None], kx, padding=1)
    gy = F.conv2d(gray[None], ky, padding=1)
    edge_mag = (gx.abs()+gy.abs()).squeeze(0)  # [1,H,W]
    
    # Edge 정보를 RGB 채널에 혼합 (가중 합)
    edge_weight = 0.1  # edge 강도 조절
    enhanced_rgb = t + edge_weight * edge_mag
    
    # 값 범위 클리핑
    enhanced_rgb = torch.clamp(enhanced_rgb, 0, 255)
    
    return enhanced_rgb  # [3,H,W]

def rgb_only(t):                          # t:[3,H,W] 0–255
    """Edge 정보 없이 RGB만 반환"""
    return t

# ───────── photometric 증강들 ─────────
def pil_clahe(im, clip=2.0, tiles=(8,8)):
    lab = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2LAB)
    l,a,b = cv2.split(lab)
    l = cv2.createCLAHE(clip, tiles).apply(l)
    lab = cv2.merge((l,a,b))
    return Image.fromarray(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB))

def pil_unsharp(im, r=2.0, pct=150):
    return Image.blend(im, im.filter(ImageFilter.UnsharpMask(r, pct)), 0.5)

class HFStereoV2:
    def __init__(self, use_edge_enhancement=True):
        # 🔥 3채널 정규화로 변경
        self.mean = [0.485, 0.456, 0.406]
        self.std  = [0.229, 0.224, 0.225]
        self.normalize = transforms.Normalize(self.mean, self.std)
        self.weak_photo = transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)

        # Edge enhancement 사용 여부
        self.edge_func = rgb_with_edge_enhancement if use_edge_enhancement else rgb_only

        # weak: 거의 원본 → pad·Edge Enhancement·Normalize
        self.weak_pipe = transforms.Compose([
            transforms.Lambda(lambda im: pad_to_size(
                transforms.functional.to_tensor(im)*255.0)),
            transforms.Lambda(self.edge_func),
            self.normalize,
        ])

        # strong: 강한 색·주파수 증강 + 동일 후처리
        self.strong_pipe = transforms.Compose([
            transforms.ColorJitter(0.4,0.4,0.4,0.1),
            transforms.RandomChoice([
                transforms.Lambda(pil_clahe),
                transforms.Lambda(lambda im: pil_unsharp(im)),
            ]),
            transforms.Lambda(lambda im: pad_to_size(
                transforms.functional.to_tensor(im)*255.0)),
            transforms.Lambda(self.edge_func),
            self.normalize,
        ])

    def __call__(self, pil_img):
        weak   = self.weak_pipe(pil_img)    # [3,384,1248]
        strong = self.strong_pipe(pil_img)  # [3,384,1248]
        return weak, strong
