from transformers import SegformerImageProcessor, SegformerModel
from PIL import Image
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from ..uda import __models__
import argparse
from experiment import prepare_cfg

processor = SegformerImageProcessor.from_pretrained('nvidia/segformer-b0-finetuned-ade-512-512')


# model.eval()


def setup_args():
    parser = argparse.ArgumentParser(description="StereoDepth Unsupervised Domain Adaptation")
    parser.add_argument('--dataset_config', default='./config/datasets/cityscapes_to_kitti2015.py', help='source domain and target domain name')
    parser.add_argument('--uda_config', default='./config/uda/kit15_cityscapes.py', help='UDA model preparation')
    parser.add_argument('--seed', default=1, metavar='S', help='random seed(default = 1)')
    parser.add_argument('--log_dir', default='./log', help='log directory')
    parser.add_argument('--ckpt', default='', help='checkpoint')
    parser.add_argument('--compute_metrics', default=True, help='compute error')
    parser.add_argument('--save_disp', default=True, help='save disparity')
    parser.add_argument('--save_att', default=True, help='save attention')
    parser.add_argument('--save_heatmap', default=False, help='save heatmap')
    parser.add_argument('--save_entropy', default=True, help='save entropy')
    parser.add_argument('--save_gt', default=True, help='save gt')
    parser.add_argument('--compare_costvolume', default=True, help='compare costvolume')
    return parser.parse_args()



def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    return inputs["pixel_values"]  # [1, 3, 512, 512]


def plot_attention_score(model_path, image_path, click_pos):
    args = setup_args()
    cfg = prepare_cfg(args, mode='test')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 3. 모델 로드 및 학습된 가중치 적용
    model = __models__['StereoDepthUDA'](cfg)
    checkpoint = torch.load(model_path)  # 모델 초기화
    model.student_model.load_state_dict(checkpoint['student_state_dict'])
    model.teacher_model.load_state_dict(checkpoint['teacher_state_dict'])
    model.to('cuda:0')
    model.eval()
    encoder = model.student_model.feature.encoder  # Fast_ACVNet_plus의 FeatureMiT.encoder
    


    # model = SegformerModel.from_pretrained('nvidia/segformer-b0-finetuned-ade-512-512')
    # encoder = model.encoder
    # model.to('cuda:0')
    # model.eval()

    # 4. FeatureMiT 인코더 접근

    # 5. 이미지 입력
    pixel_values = preprocess_image(image_path).to(device)  # [1, 3, 512, 512]
    with torch.no_grad():
        encoder_outputs = encoder(pixel_values, output_hidden_states=True, output_attentions=True)

    # 6. Attention 맵 처리
    if encoder_outputs.attentions is None:
        raise ValueError("Attention weights not captured. Check encoder configuration.")

    # Sum all attention maps from all stages
    attn_map_all = torch.zeros_like(encoder_outputs.attentions[7][0].cpu())
    for i in range(8):  # Sum attention maps from index 0 to 7
        scale = 2**(i // 2)
        print(scale)
        attn_map = F.interpolate(encoder_outputs.attentions[i][0].cpu(), scale_factor=scale, mode='nearest')
        print(f"Attention map shape: {attn_map.shape}")
        # attn_map_all += attn_map
    
    attn_map_all = attn_map_all.numpy()
    
    # 마지막 스테이지의 첫 번째 attention 맵 사용
    attn_map = encoder_outputs.attentions[7][0].cpu().numpy()  # [B, num_heads, N, N] -> [num_heads, N, N]
    print(f"Attention map shape: {attn_map.shape}")  # 디버깅용

    # 멀티헤드 평균 내기
    attn_map_mean = attn_map.mean(axis=0)  # [N, N], N = (H/32) * (W/32)
    attn_map_sum = attn_map_all.mean(axis=0)  # Sum across all stages and average across heads
    h_patch, w_patch = 512 // 32, 512 // 32  # 마지막 스테이지의 패치 크기 (16x16)
    n_patches = h_patch * w_patch
    print(f"Expected patches: {n_patches}")  # 디버깅용

    # 클릭한 픽셀을 패치 인덱스로 변환
    x, y = click_pos  # (x, y) 좌표, 원본 이미지 기준
    orig_w, orig_h = Image.open(image_path).size  # 원본 크기 (예: 1280x1824)
    x_scaled = int(x * 512 / orig_h)  # 세로 비율 기준으로 조정
    y_scaled = int(y * 512 / orig_w)  # 가로 비율 기준으로 조정
    x_patch, y_patch = x_scaled // 32, y_scaled // 32  # 마지막 스테이지의 패치 단위
    click_idx = y_patch * w_patch + x_patch  # 1D 인덱스
    print(f"Click index: {click_idx}, Patch coords: ({x_patch}, {y_patch})")  # 디버깅용

    if click_idx >= n_patches:
        raise ValueError(f"Click index {click_idx} exceeds patch count {n_patches}")

    # 해당 픽셀의 attention score 추출 (mean & sum)
    attn_scores_mean = attn_map_mean[click_idx].reshape(h_patch, w_patch)  # [H/32, W/32] = [16, 16]
    attn_scores_sum = attn_map_sum[click_idx].reshape(h_patch, w_patch)  # [H/32, W/32] = [16, 16]

    # 7. 원본 크기로 업샘플링
    attn_scores_mean = F.interpolate(
        torch.tensor(attn_scores_mean).unsqueeze(0).unsqueeze(0), 
        size=(512, 512),
        mode='bilinear', 
        align_corners=False
    ).squeeze().numpy()

    attn_scores_sum = F.interpolate(
        torch.tensor(attn_scores_sum).unsqueeze(0).unsqueeze(0), 
        size=(512, 512),
        mode='bilinear', 
        align_corners=False
    ).squeeze().numpy()

    # 8. 시각화
    img = Image.open(image_path).resize((512, 512))  # 입력 크기에 맞게 리사이즈
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    
    # 원본 이미지 + 클릭 위치 표시
    ax1.imshow(img)
    ax1.scatter(y_scaled, x_scaled, c='red', s=50, label='Clicked Point')  # (y, x) 순서 주의
    ax1.legend()
    ax1.set_title("Input Image with Clicked Point")
    ax1.axis('off')

    # Mean Attention 히트맵
    im2 = ax2.imshow(img, alpha=0.5)
    im_mean = ax2.imshow(attn_scores_mean, cmap='jet', alpha=0.5, vmin=0, vmax=0.09)
    ax2.scatter(y_scaled, x_scaled, c='red', s=50)
    ax2.set_title("Mean Attention Score Heatmap")
    ax2.axis('off')
    plt.colorbar(im_mean, ax=ax2, label='Mean Attention Score')

    # Sum Attention 히트맵
    im3 = ax3.imshow(img, alpha=0.5)
    im_sum = ax3.imshow(attn_scores_sum, cmap='jet', alpha=0.5, vmin=0, vmax=0.7)  # Adjusted vmax for sum
    ax3.scatter(y_scaled, x_scaled, c='red', s=50)
    ax3.set_title("Sum Attention Score Heatmap")
    ax3.axis('off')
    plt.colorbar(im_sum, ax=ax3, label='Sum Attention Score')

    plt.show()

# 9. 실행
if __name__ == "__main__":
    model_path = "/home/jaejun/DomainGeneralization/log/2025-03-13_20_41/checkpoint_epoch750.pth"  # 학습된 가중치 파일 경로
    image_path = "/home/jaejun/Pictures/road.jpg"  # 실제 이미지 경로
    click_pos = (500, 700)  # 원본 크기 기준 클릭 좌표 (예: 1280x1824)
    plot_attention_score(model_path, image_path, click_pos)