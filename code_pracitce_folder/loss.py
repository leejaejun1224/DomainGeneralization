import torch
import torch.nn.functional as F

def calc_reconstruction_loss(data_batch, alpha=0.85):
    """
    data_batch:
      - 'tgt_left':  (B, C, H, W) 형태, 왼쪽 이미지
      - 'tgt_right': (B, C, H, W) 형태, 오른쪽 이미지
      - 'pseudo_disp': 길이 1짜리 리스트 [tensor], 내부 텐서는 (B, 1, H, W) 형태 가정
    alpha: SSIM과 L1을 혼합할 때의 비율 (default=0.85)
    """
    bs, _, height, width = data_batch['tgt_left'].shape
    
    # 픽셀 단위 grid 만들기
    x_base = torch.linspace(0, width-1, width).repeat(bs, height, 1).type_as(data_batch['tgt_left'])
    y_base = torch.linspace(0, height-1, height).repeat(bs, width, 1).transpose(1, 2).type_as(data_batch['tgt_left'])
    
    # pseudo_disp를 왼쪽에서 오른쪽 뷰로 warp
    disp = data_batch['pseudo_disp'][0].squeeze(1)  # (B, H, W)
    x_warped = x_base - disp

    # grid_sample에 맞게 [-1,1] 정규화
    x_norm = 2.0 * x_warped / (width - 1) - 1.0
    y_norm = 2.0 * y_base   / (height - 1) - 1.0

    # (B, 2, H, W) -> (B, H, W, 2)
    flow = torch.stack((x_norm, y_norm), dim=1).permute(0, 2, 3, 1)

    # 왼쪽 이미지를 warp하여 오른쪽 시점으로 재구성
    img_right_reconstructed = F.grid_sample(
        data_batch['tgt_left'],
        flow,
        mode='bilinear',
        padding_mode='zeros',
        align_corners=True
    )

    # SSIM (1 - ssim)/2
    ssim_val = compute_ssim(data_batch['tgt_right'], img_right_reconstructed)
    ssim_loss = (1 - ssim_val) / 2

    # L1
    l1_loss = F.l1_loss(data_batch['tgt_right'], img_right_reconstructed, reduction='none').mean(dim=(1,2,3))

    # alpha 비율로 섞기
    reconstruction_loss = alpha * ssim_loss + (1 - alpha) * l1_loss

    return reconstruction_loss.mean()

def calc_reconstruction_loss(data_batch, alpha=0.85):
    bs, _, height, width = data_batch['tgt_left'].shape
    

    x_base = torch.linspace(0, width-1, width).repeat(bs, height, 1).type_as(data_batch['tgt_left'])
    y_base = torch.linspace(0, height-1, height).repeat(bs, width, 1).transpose(1, 2).type_as(data_batch['tgt_left'])

    flow = torch.stack((x_base - data_batch['pseudo_disp'][0].squeeze(1), y_base), dim=1)

    ### [B, C, H, W]
    img_right_reconstructed = F.grid_sample(data_batch['tgt_left'], flow.permute(0, 2, 3, 1), mode='bilinear', padding_mode='zeros', align_corners=True)

    ssim = compute_ssim(data_batch['tgt_right'], img_right_reconstructed)
    ssim_loss = (1 - ssim)/2

    l1_loss = F.l1_loss(data_batch['tgt_right'], img_right_reconstructed, reduction='none').mean(dim=(1,2,3))

    reconstuction_loss = alpha * ssim_loss + (1 - alpha) * l1_loss

    return reconstuction_loss.mean()


def compute_ssim(img1, img2, window_size=3, channel=3):
    window = torch.ones(1, channel, window_size, window_size) / (window_size**2)
    window = window.type_as(img1)

    mean1 = F.conv2d(img1, window, padding=window_size//2, groups=1)
    mean2 = F.conv2d(img2, window, padding=window_size//2, groups=1)

    mean1_sq = mean1.pow(2)
    mean2_sq = mean2.pow(2)
    mean12 = mean1 * mean2

    sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=1) - mean1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=1) - mean2_sq
    sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=1) - mean12

    C1 = 1e-5
    C2 = 1e-5

    ssim_map = ((2*mean1*mean2 + C1)*(2*sigma12 + C2)) / ((mean1_sq + mean2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean(dim=(1,2,3))



if __name__ == "__main__":
    img1 = torch.tensor([[1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15]]).unsqueeze(0).unsqueeze(0).float()
    img2 = torch.tensor([[0,1,2,3,4],[5,6,7,8,9],[10,11,12,13,14]]).unsqueeze(0).unsqueeze(0).float()
    disparity = torch.tensor([[0,0,1,0,0],[0,0,1,1,1],[1,1,0,0,1]]).unsqueeze(0).float()

    bs, c, h, w = img1.shape

    x_base = torch.linspace(0, w-1, w).repeat(bs, h, 1).type_as(img1)
    print(x_base)
    print("===========") 
    print(x_base-disparity)
    print("----------")
    y_base = torch.linspace(0, h-1, h).repeat(bs, w, 1).transpose(1,2).type_as(img1)
    
    flow = torch.stack((x_base - disparity, y_base), dim=3)
    print(flow.shape)
    print("++++++++++")
    
    flow[...,0] = 2.0 * flow[...,0] / (w-1) - 1.0
    flow[...,1] = 2.0 * flow[...,1] / (h-1) - 1.0
    
    right_reconstructed = F.grid_sample(img1, flow, mode='bilinear', padding_mode='zeros', align_corners=True)
    print(right_reconstructed)
    print(right_reconstructed.shape)