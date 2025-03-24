import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def load_image(image_path, as_grayscale=False):
    image = Image.open(image_path)
    if as_grayscale:
        image = image.convert('L')
    else:
        image = image.convert('RGB')
    # PIL 이미지를 numpy 배열로 변환 후 torch 텐서로
    image_np = np.array(image, dtype=np.float32)
    if as_grayscale:
        image_tensor = torch.from_numpy(image_np).unsqueeze(0).unsqueeze(0)  # [B=1, C=1, H, W]
    else:
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0)  # [B=1, C=3, H, W]
    print(f"Loaded image shape: {image_tensor.shape}, min: {image_tensor.min().item()}, max: {image_tensor.max().item()}")
    return image_tensor

def custom_warp(image, disparity):
    B, C, H, W = image.shape
    device = image.device
    y, x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
    x = x.float()
    y = y.float()
    x_shifted = x - disparity.squeeze(1).squeeze(0)  # 왼쪽 -> 오른쪽 이동
    x_shifted = 2 * x_shifted / (W - 1) - 1
    y_normalized = 2 * y / (H - 1) - 1
    grid = torch.stack([x_shifted, y_normalized], dim=-1).unsqueeze(0)
    warped_image = F.grid_sample(image, grid, mode='bilinear', padding_mode='border', align_corners=True)
    return warped_image

def warp_and_diff(left, right, disp_gt):
    # Disparity 값 통계 출력
    print("Disparity min:", disp_gt.min().item(), "max:", disp_gt.max().item())

    # KITTI disparity 값 보정 (필요 시)
    # KITTI disparity는 [0, 255]로 저장됨, 실제 픽셀 단위로 변환
    # disp_gt = disp_gt / 256.0  # KITTI의 경우, 이 스케일링이 필요할 수 있음 (문서 확인 필요)

    left_warped = custom_warp(left, -disp_gt)
    diff = right - left_warped
    return left_warped, diff

def visualize_results(left, right, disp_gt, left_warped, diff):
    left = left.squeeze().cpu().numpy()
    right = right.squeeze().cpu().numpy()
    disp_gt = disp_gt.squeeze().cpu().numpy()
    left_warped = left_warped.squeeze().cpu().numpy()
    diff = diff.squeeze().cpu().numpy()

    if left.shape[0] == 3:
        left = left.transpose(1, 2, 0)
        right = right.transpose(1, 2, 0)
        left_warped = left_warped.transpose(1, 2, 0)
        diff = diff.transpose(1, 2, 0)
        diff_display = diff.mean(axis=2)
    else:
        diff_display = diff

    plt.figure(figsize=(15, 10))

    plt.subplot(2, 3, 1)
    plt.title("Left Image")
    if left.ndim == 3 and left.shape[2] == 3:
        plt.imshow(left.astype(np.uint8))  # [0, 255]로 표시
    else:
        plt.imshow(left, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 3, 2)
    plt.title("Right Image")
    if right.ndim == 3 and right.shape[2] == 3:
        plt.imshow(right.astype(np.uint8))
    else:
        plt.imshow(right, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 3, 3)
    plt.title("Ground Truth Disparity")
    plt.imshow(disp_gt, cmap='jet')
    plt.colorbar(label='Disparity (pixels)')
    plt.axis('off')

    plt.subplot(2, 3, 4)
    plt.title("Warped Left Image")
    if left_warped.ndim == 3 and left_warped.shape[2] == 3:
        plt.imshow(left_warped.astype(np.uint8))
    else:
        plt.imshow(left_warped, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 3, 5)
    plt.title("Difference (Right - Warped Left)")
    plt.imshow(diff_display, cmap='jet')
    plt.colorbar()
    plt.axis('off')

    plt.tight_layout()
    plt.show()

def main(left_path, right_path, disp_gt_path):
    left = load_image(left_path, as_grayscale=False)
    right = load_image(right_path, as_grayscale=False)
    disp_gt = load_image(disp_gt_path, as_grayscale=True)

    print("Left shape:", left.shape)
    print("Right shape:", right.shape)
    print("Disp_gt shape:", disp_gt.shape)

    min_h = min(left.shape[2], right.shape[2], disp_gt.shape[2])
    min_w = min(left.shape[3], right.shape[3], disp_gt.shape[3])
    left = F.interpolate(left, size=(min_h, min_w), mode='bilinear', align_corners=True)
    right = F.interpolate(right, size=(min_h, min_w), mode='bilinear', align_corners=True)
    disp_gt = F.interpolate(disp_gt, size=(min_h, min_w), mode='bilinear', align_corners=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    left, right, disp_gt = left.to(device), right.to(device), disp_gt.to(device)

    left_warped, diff = warp_and_diff(left, right, disp_gt)
    visualize_results(left, right, disp_gt, left_warped, diff)

if __name__ == "__main__":
    left_path = "/home/jaejun/dataset/kitti_2015/training/image_2/000034_10.png"
    right_path = "/home/jaejun/dataset/kitti_2015/training/image_3/000034_10.png"
    disparity_path = "/home/jaejun/dataset/kitti_2015/training/disp_occ_0/000034_10.png"
    main(left_path, right_path, disparity_path)