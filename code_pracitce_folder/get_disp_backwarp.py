import math
import time
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

###################################################
# 1) 데이터 로드 함수
###################################################
def load_image_as_tensor(path):
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"이미지를 불러올 수 없습니다: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    chw = np.transpose(rgb, (2, 0, 1))  # (3,H,W)
    tensor = torch.from_numpy(chw).unsqueeze(0)   # (1,3,H,W)
    return tensor


def load_disparity_as_tensor(path, scale=1.0):
    disp_raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if disp_raw is None:
        raise FileNotFoundError(f"디스패리티를 불러올 수 없습니다: {path}")
    disp = disp_raw.astype(np.float32) * scale
    disp = np.expand_dims(disp, axis=(0,1))  # (1,1,H,W)

    return torch.from_numpy(disp)


def naive_warp_left_to_right(left, disparity):
    left_np = left[0].permute(1,2,0).cpu().numpy()
    disp_np = disparity[0,0].cpu().numpy()     # (H,W)

    H, W, _ = left_np.shape

    warped_np = np.zeros_like(left_np, dtype=np.float32)
    mask_np   = np.zeros((H,W), dtype=np.float32)  

    start = time.time()
    for y in range(H):
        for x in range(W):
            d = disp_np[y, x]
            new_x = int(round(x - d))
            if 0 <= new_x < W and d > 0:
                warped_np[y, new_x, :] = left_np[y, x, :]
                mask_np[y, new_x] = 1.0
    end = time.time()
    print(f"Time taken: {end - start:.2f} seconds")

    warped_t = torch.from_numpy(warped_np).permute(2,0,1).unsqueeze(0)  # (1,3,H,W)
    valid_t  = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0)      # (1,1,H,W)
    return warped_t, valid_t


def photometric_loss(left_warped, right, valid_mask):

    diff = torch.abs(left_warped - right)
    diff_per_pixel = diff.sum(dim=1, keepdim=True)  # RGB합
    masked_diff = diff_per_pixel * valid_mask
    sum_diff = masked_diff.sum()
    sum_mask = valid_mask.sum()
    loss = sum_diff / (sum_mask + 1e-8)
    return loss

def main():
    image = "000000_10.png"
    left_path  = f"/home/jaejun/dataset/kitti_2015/training/image_2/{image}"
    right_path = f"/home/jaejun/dataset/kitti_2015/training/image_3/{image}"
    disp_path  = f"/home/jaejun/dataset/kitti_2015/training/disp_occ_0/{image}"

    left_t = load_image_as_tensor(left_path)       # (1,3,H,W)
    right_t = load_image_as_tensor(right_path)     # (1,3,H,W)
    disp_t = load_disparity_as_tensor(disp_path, scale=1.0 / 256.0)  # (1,1,H,W)

    left_warped_t, valid_mask_t = naive_warp_left_to_right(left_t, disp_t)

    loss_val = photometric_loss(left_warped_t, right_t, valid_mask_t)
    print(f"Photometric Loss (naive warp) = {loss_val.item():.4f}")

    left_warped_np = left_warped_t[0].permute(1,2,0).cpu().numpy()
    right_np       = right_t[0].permute(1,2,0).cpu().numpy()
    mask_np        = valid_mask_t[0,0].cpu().numpy()

    hole_applied = left_warped_np.copy()
    hole_applied[mask_np < 0.5] = 0  # 0=검정

    plt.figure()
    plt.imshow(np.clip(left_t[0].permute(1,2,0).cpu().numpy(), 0,1))
    plt.title("Left (original)")

    plt.figure()
    plt.imshow(np.clip(right_np, 0,1))
    plt.title("Right (original)")

    plt.figure()
    plt.imshow(np.clip(left_warped_np, 0,1))
    plt.title("Warped Left (Naive, integer shift)")

    plt.figure()
    plt.imshow(np.clip(hole_applied, 0,1))
    plt.title("Warped Left + Hole Visualization")

    plt.show()

if __name__ == "__main__":
    main()
