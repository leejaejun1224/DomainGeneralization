from __future__ import annotations
import argparse, math, numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from PIL import Image
import matplotlib.pyplot as plt

# mean/std for ImageNet normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def load_image(path: str, device: str) -> torch.Tensor:
    img = Image.open(path).convert('RGB')
    arr = np.array(img, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1)
    for t, m, s in zip(tensor, IMAGENET_MEAN, IMAGENET_STD):
        t.sub_(m).div_(s)
    return tensor.unsqueeze(0).to(device)

# 1. Thick feature encoder
class ThickEncoder(nn.Module):
    def __init__(self, out_ch: int = 256):
        super().__init__()
        self.backbone = timm.create_model('resnet50', pretrained=True, features_only=True)
        c_in = self.backbone.feature_info[1]['num_chs']
        self.proj = nn.Sequential(
            nn.Conv2d(c_in, out_ch, 1, bias=False),
            nn.SyncBatchNorm(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.backbone(x)[1])

# 2. Cosine cost-volume
@torch.no_grad()
def build_cosine_cost(fL: torch.Tensor, fR: torch.Tensor, radius: int = 3) -> torch.Tensor:
    vol = []
    for d in range(-radius, radius + 1):
        shifted = torch.roll(fR, shifts=d, dims=-1)
        sim = F.cosine_similarity(fL, shifted, dim=1, eps=1e-6).unsqueeze(1)
        vol.append(sim)
    return torch.cat(vol, dim=1)

# 3. Refiner network
class CosineRefiner(nn.Module):
    def __init__(self, radius: int = 3, temperature: float = 0.1):
        super().__init__()
        self.enc = ThickEncoder()
        self.radius = radius
        self.temperature = temperature
        lap = torch.tensor([[0,1,0],[1,-4,1],[0,1,0]], dtype=torch.float32)
        self.register_buffer('lap_kernel', lap.unsqueeze(0).unsqueeze(0))
        

    def forward(self, imgL: torch.Tensor, imgR: torch.Tensor, disp_full: torch.Tensor, smooth: bool = True):
        # 1) downsample the **full-res** disp to quarter res and normalize
        disp0 = F.interpolate(disp_full, scale_factor=0.25, mode='bilinear', align_corners=False) / 4.0
        # 2) build features & cost at quarter res
        fL, fR = self.enc(imgL), self.enc(imgR)
        cost   = build_cosine_cost(fL, fR, self.radius)
        p      = F.softmax(cost / self.temperature, dim=1)
        offsets = torch.arange(-self.radius, self.radius + 1, device=imgL.device) \
                       .view(1, -1, 1, 1)
        delta  = (p * offsets).sum(dim=1, keepdim=True)
        disp_ref = disp0 + delta
        # 3) refine at quarter res
        disp_q_ref = disp0 + delta
        conf   = p.max(dim=1, keepdim=True).values
        if smooth:
            disp_q_ref = disp_q_ref - 0.2 * F.conv2d(disp_q_ref, self.lap_kernel, padding=1)
        # 4) upsample both refined disparity and confidence back to full resolution
        disp_full_ref = F.interpolate(disp_q_ref, scale_factor=4,
                                      mode='bilinear', align_corners=False) * 4.0
        conf_full     = F.interpolate(conf,     scale_factor=4,
                                      mode='bilinear', align_corners=False)
        return disp_full_ref, conf_full

# 4. Upscale utility
def upscale(disp_q: torch.Tensor, scale: int = 4) -> torch.Tensor:
    return F.interpolate(disp_q, scale_factor=scale, mode='bilinear', align_corners=False) * scale

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Cosine-Similarity Stereo Refiner with Error Map')
    parser.add_argument('--left',       type=str, default='/home/jaejun/dataset/kitti_2015/training/image_2/000002_10.png')
    parser.add_argument('--right',      type=str, default='/home/jaejun/dataset/kitti_2015/training/image_3/000002_10.png')
    parser.add_argument('--disp',       type=str, default='/home/jaejun/DomainGeneralization/log/2025-05-25_18_52_baseline/disp/tgt/000002_10.npy')
    parser.add_argument('--gt',         type=str, default='/home/jaejun/dataset/kitti_2015/training/disp_occ_0/000002_10.png')
    parser.add_argument('--out',        type=str, default='/home/jaejun/DomainGeneralization/log/2025-05-25_18_52_baseline/disp/tgt/000002_10_ref.png')
    # parser.add_argument('--error-img',  type=str, default='/home/jaejun/DomainGeneralization/log/2025-05-25_18_52_baseline/disp/tgt/000000_10_ref.png')
    parser.add_argument('--device',     type=str, default='cuda')
    args = parser.parse_args()
    
    

    from datasets.data_io import get_transform
    import torch.nn.functional as F

    # load PIL images
    left_pil  = Image.open(args.left).convert('RGB')
    right_pil = Image.open(args.right).convert('RGB')
    w, h = left_pil.size

    # KITTI test resizing + normalize via get_transform
    # target full size: 1248x384
    processed = get_transform()
    left_t  = processed(left_pil)  # Tensor [3,H,W]
    right_t = processed(right_pil)
    # pad to (1248,384)
    top_pad   = 384 - h
    right_pad = 1248 - w
    assert top_pad >= 0 and right_pad >= 0, 'Input larger than KITTI test resolution'
    # pad format: (padW_left, padW_right, padH_top, padH_bottom)
    left_t  = F.pad(left_t,  (0, right_pad, top_pad, 0))
    right_t = F.pad(right_t, (0, right_pad, top_pad, 0))

    # batchify
    imgL = left_t.unsqueeze(0).to(args.device)
    imgR = right_t.unsqueeze(0).to(args.device)

    # load coarse full-res disparity (H x W), pad similarly
    d0_full = np.load(args.disp).astype(np.float32)
    # ensure 2D array
    # d0_full = np.squeeze(d0_full)
    # # pad to KITTI test resolution
    # d0_pad  = np.pad(d0_full, ((top_pad,0),(0,right_pad)), mode='constant', constant_values=0)
    disp_full = torch.from_numpy(d0_full).unsqueeze(0).to(args.device)  # [1,1,H,W]
    disp0 = disp_full
    # disp0 = F.interpolate(disp_full, scale_factor=0.25, mode='bilinear', align_corners=False) / 4.0
    print(imgL.shape, imgR.shape, disp0.shape)
    # load GT disparity and pad
    # gt_full = np.load(args.gt).astype(np.float32)
    # gt_pad  = np.pad(gt_full, ((top_pad,0),(0,right_pad)), mode='constant', constant_values=0)

    # model inference
    model = CosineRefiner().to(args.device).eval()
    with torch.no_grad():
        disp_q, conf = model(imgL, imgR, disp0)
        # disp_full_ref = upscale(disp_q)
    # crop back to original size
    ref_np = disp_q.squeeze().cpu().numpy()
    ref_crop = ref_np[top_pad:top_pad+h, 0:w]
    np.save(args.out, ref_crop)
    import os
    import matplotlib.pyplot as plt

    # out 경로가 .npy일 경우 .png로 바꿔주거나, 없으면 뒤에 .png 붙임
    base, ext = os.path.splitext(args.out)
    out_png = base + '.png'

    # disparity 값을 0–255 uint8로 정규화
    disp_min, disp_max = ref_crop.min(), ref_crop.max()
    if disp_max > disp_min:
        disp_norm = (ref_crop - disp_min) / (disp_max - disp_min)
    else:
        disp_norm = ref_crop * 0.0
    disp_uint8 = (disp_norm * 255.0).astype('uint8')

    # gray cmap으로 저장
    plt.imsave(out_png, disp_uint8, cmap='gray', vmin=0, vmax=255)
    print(f'Refined disparity saved as NPY → {args.out}')
    print(f'Refined disparity saved as PNG → {out_png}')
   # compute error on original region
    # gt_crop = gt_pad[top_pad:top_pad+h, 0:w]
    # if ref_crop.shape != gt_crop.shape:
    #     raise ValueError(f'Shape mismatch: refined {ref_crop.shape} vs GT {gt_crop.shape}')
    # error_map = np.abs(ref_crop - gt_crop)
    # np.save(args.error_npy, error_map)
    # plt.imsave(args.error_img, error_map, cmap='jet')

    # print(f'Refined disparity saved → {args.out}')
    # print(f'Error map saved: {args.error_npy}, {args.error_img}')
