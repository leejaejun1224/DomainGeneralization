import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import cv2  # 이미지를 읽거나 전처리용
import math


def attention_rollout(all_attn, discard_ratio=0.0):


    # what about different image size?
    result = None
    for attn in all_attn:

        attn_mean = attn.mean(dim=1)

        I = torch.eye(attn_mean.size(-1), device=attn_mean.device).unsqueeze(0)
        attn_res = attn_mean + I  
        attn_res = attn_res / attn_res.sum(dim=-1, keepdim=True)

        if result is None:
            result = attn_res
        else:
            result = torch.bmm(result, attn_res)

    return result  



def save_heatmap(image_tensor, outputs, all_attn):


    last_stage_attn = all_attn[-1]  
    rollout = attention_rollout([last_stage_attn])
    b, c, h4, w4 = outputs[-1].shape
    H, W = h4, w4
    save_path = "attention_heatmap.png"

    assert rollout.size(0) == 1, "배치 크기가 1일 때만 예시로 시각화합니다."
    
    mean_attn = rollout.mean(dim=1).squeeze(0)  # (N,)
    attn_map = mean_attn.reshape(H, W).detach().cpu().numpy()
    

    img_np = image_tensor.squeeze(0).permute(1,2,0).cpu().numpy()
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-5)
    fig, axes = plt.subplots(1, 2, figsize=(8,4))
    axes[0].imshow(img_np)
    axes[0].axis('off')
    axes[0].set_title("Original")
    im = axes[1].imshow(attn_map, cmap='jet')
    axes[1].axis('off')
    axes[1].set_title("Attention Rollout (Stage4)")
    fig.colorbar(im, ax=axes[1])
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)