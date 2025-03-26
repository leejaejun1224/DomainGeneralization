import time
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt

###################################################
# 1) 데이터 로드 함수
###################################################
def load_image_as_tensor(path):
    """
    - OpenCV(BGR)로 읽고 -> RGB로 변환
    - [H,W,3] -> [3,H,W], float32 [0,1]
    - 배치 차원까지 추가 -> (1,3,H,W)
    """
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"이미지를 불러올 수 없습니다: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    chw = np.transpose(rgb, (2, 0, 1))  # (3,H,W)
    tensor = torch.from_numpy(chw).unsqueeze(0)   # (1,3,H,W)
    return tensor

def load_disparity_as_tensor(path, scale=1.0):
    """
    - OpenCV로 읽어 float32 변환
    - shape: (1,1,H,W)
    - 필요한 경우 scale 곱함
    """
    disp_raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if disp_raw is None:
        raise FileNotFoundError(f"디스패리티를 불러올 수 없습니다: {path}")
    disp = disp_raw.astype(np.float32) * scale
    disp = np.expand_dims(disp, axis=(0,1))  # (1,1,H,W)
    return torch.from_numpy(disp)

###################################################
# 2) Left→Right Warping (이미지용)
###################################################
def warp_left_to_right(left, disparity):
    mask = disparity > 0
    left_masked = left * mask.expand(-1, 3, -1, -1)
    """
    left:      (B,3,H,W)
    disparity: (B,1,H,W)
    반환: warp된 이미지 (B,3,H,W)
    """
    B, C, H, W = left.shape

    # pixel 좌표 grid
    y_base, x_base = torch.meshgrid(
        torch.arange(H, device=left.device),
        torch.arange(W, device=left.device),
        indexing='ij'
    )
    x_base = x_base.unsqueeze(0).expand(B, -1, -1)  # (B,H,W)
    y_base = y_base.unsqueeze(0).expand(B, -1, -1)  # (B,H,W)

    # 예: x_new = x_base + disparity
    x_new = x_base + disparity.squeeze(1)

    # 정규화 ([-1,1] 범위)
    x_norm = 2.0 * x_new / (W - 1) - 1.0
    y_norm = 2.0 * y_base / (H - 1) - 1.0

    grid = torch.stack((x_norm, y_norm), dim=-1)  # (B,H,W,2)

    # bilinear 보간
    left_warped = F.grid_sample(
        left_masked, grid,
        mode='nearest',
        padding_mode='border',
        align_corners=True
    )
    return left_warped, left_masked

###################################################
# 3) Left→Right Warping (마스크용)
###################################################
def warp_mask_left_to_right(mask, disparity):
    """
    mask:      (B,1,H,W) 0/1 형태
    disparity: (B,1,H,W)
    반환: warp된 마스크 (B,1,H,W)
    """
    B, C, H, W = mask.shape

    # pixel 좌표 grid
    y_base, x_base = torch.meshgrid(
        torch.arange(H, device=mask.device),
        torch.arange(W, device=mask.device),
        indexing='ij'
    )
    x_base = x_base.unsqueeze(0).expand(B, -1, -1)
    y_base = y_base.unsqueeze(0).expand(B, -1, -1)

    x_new = x_base + disparity.squeeze(1)

    x_norm = 2.0 * x_new / (W - 1) - 1.0
    y_norm = 2.0 * y_base / (H - 1) - 1.0
    grid = torch.stack((x_norm, y_norm), dim=-1)

    # nearest 보간 (마스크라서 이산값 유지)
    mask_warped = F.grid_sample(
        mask, grid,
        mode='nearest',
        padding_mode='zeros',
        align_corners=True
    )
    return mask_warped

def photometric_loss(left, right, disparity):
    left_warped_t, left_masked = warp_left_to_right(left, disparity)
    valid_mask_t = (disparity > 0).float()
    mask_warped_t = warp_mask_left_to_right(valid_mask_t, disparity)
    left_warped_masked_t = left_warped_t * mask_warped_t
    diff_t = torch.abs(left_warped_t - right)
    diff_masked = diff_t * mask_warped_t
    sum_diff = diff_masked.sum()
    sum_mask = mask_warped_t.sum() * diff_t.shape[1]
    loss = sum_diff / (sum_mask + 1e-8)
    return loss





###################################################
# 4) 전체 시연 + 시각화
###################################################
def main():
    # 예시 경로 (직접 수정하세요)
    image = "000000_10.png"
    left_path = "/home/jaejun/dataset/kitti_2015/training/image_2/" + image
    right_path = "/home/jaejun/dataset/kitti_2015/training/image_3/" + image
    disp_path = "/home/jaejun/dataset/kitti_2015/training/disp_occ_0/" + image

    # 1) 로드
    left_t = load_image_as_tensor(left_path)    # (1,3,H,W)
    right_t = load_image_as_tensor(right_path)  # (1,3,H,W)
    disp_t = load_disparity_as_tensor(disp_path, scale=1.0/256.0)  # (1,1,H,W)

    # loss = photometric_loss(left_t, right_t, disp_t)
    # print(f"Photometric loss = {loss.item():.4f}")


    # 2) 왼쪽이미지 Warp
    start = time.time()
    left_warped_t, left_masked = warp_left_to_right(left_t, disp_t)  # (1,3,H,W)

    # 3) 마스크도 Warp (nearest)
    #    예) "disparity>0"인 지점만 유효하다고 가정
    valid_mask_t = (disp_t > 0).float()  # (1,1,H,W)
    mask_warped_t = warp_mask_left_to_right(valid_mask_t, disp_t)  # (1,1,H,W)
    end = time.time()
    print(f"Time taken: {end - start:.2f} seconds")

    # 4) "warp된 부분만" 보고 싶다면 곱하기
    #    (B,3,H,W) x (B,1,H,W) -> broadcast
    left_warped_masked_t = left_warped_t
    mask  = left_warped_t > 0
    # left_warped_masked_t = left_warped_t * mask_warped_t

    # 5) photometric diff (마스크 영역만)
    diff_t = torch.abs(left_warped_t - right_t)
    # 채널(C=3) 고려 시, mask_warped_t를 브로드캐스팅
    # diff_masked = diff_t 
    diff_masked = diff_t * mask
    sum_diff = diff_masked.sum()
    sum_mask = mask_warped_t.sum() * diff_t.shape[1]  # 채널 수만큼 곱
    loss = sum_diff / (sum_mask + 1e-8)
    print(f"Photometric loss (masked) = {loss.item():.4f}")

    # 6) NumPy 변환 (시각화)
    left_np     = left_masked[0].permute(1,2,0).cpu().numpy()
    right_np    = right_t[0].permute(1,2,0).cpu().numpy()
    warped_np   = left_warped_t[0].permute(1,2,0).cpu().numpy()
    masked_np   = left_warped_masked_t[0].permute(1,2,0).cpu().numpy()
    diff_np     = diff_masked[0].permute(1,2,0).cpu().numpy()  # 마스크 적용된 차이

    left_np   = np.clip(left_np,   0,1)
    right_np  = np.clip(right_np,  0,1)
    warped_np = np.clip(warped_np, 0,1)
    masked_np = np.clip(masked_np, 0,1)
    diff_np   = np.clip(diff_np,   0,1)

    # 7) matplotlib 시각화
    plt.figure()
    plt.imshow(left_np)
    plt.title("Left (original)")

    plt.figure()
    plt.imshow(right_np)
    plt.title("Right (original)")

    plt.figure()
    plt.imshow(warped_np)
    plt.title("Warped Left (full)")

    plt.figure()
    plt.imshow(masked_np)
    plt.title("Warped Left (only valid/moved)")

    plt.figure()
    plt.imshow(diff_np)
    plt.title("Difference (masked)")

    plt.show()

if __name__ == "__main__":
    main()
