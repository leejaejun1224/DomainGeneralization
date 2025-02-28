import math
import torch
import torch.nn as nn
import matplotlib.pyplot as plt  # 시각화용
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from MiTbackbone import MixVisionTransformer


def visualize_global_attention(attn_weight, H, W, sr_ratio=1, title="Global Attention"):
    """
    attn_weight : [B, num_heads, N, N'] 형태
       - N = Query 토큰 수 (H×W)
       - N' = Key 토큰 수 (H_k×W_k),  보통 H_k = H/sr_ratio, W_k = W/sr_ratio

    H, W : Query가 펴진 해상도
    sr_ratio: Key가 downsample된 배율
    """
    B, num_heads, N, N_ = attn_weight.shape
    print(attn_weight.shape)

    # 1) 모든 Query와 Head에 대해 평균
    #    - dim=2 (Query 축) 을 평균하면 shape: [B, num_heads, N']
    #    - 다시 dim=1 (Head 축) 을 평균하면 shape: [B, N']
    #    => 결과적으로 [B, N'] 형태가 됨
    attn_mean = attn_weight.mean(dim=2)           # [B, num_heads, N']
    attn_mean = attn_mean.mean(dim=1)            # [B, N']

    # 2) Key 해상도 (H_k, W_k) 계산
    #    sr_ratio>1이면 Key쪽은 H/sr, W/sr
    H_k = H // sr_ratio
    W_k = W // sr_ratio
    assert H_k * W_k == N_, f"Key 해상도 H_k*W_k={H_k*W_k}와 N'={N_}가 맞지 않습니다."

    # 3) 단일 배치만 예시 시각화 (b=0)
    #    실제론 for b in range(B): 로 반복 가능
    b = 0
    scores_1d = attn_mean[b]           # shape: [N']
    print(scores_1d.shape)
    scores_2d = scores_1d.reshape(H_k, W_k).detach().cpu()

    plt.figure(figsize=(5,5))
    plt.title(title + f" (sr={sr_ratio})")
    plt.imshow(scores_2d, cmap='jet')
    plt.colorbar()
    plt.show()



if __name__=="__main__":
    # 1) 모델 생성
    mitbackbone = MixVisionTransformer(
        img_size=256, in_chans=3,
        embed_dim=[64, 128, 256, 512],
        depth=[2, 2, 2, 2],
        num_heads=[1, 2, 4, 8],
        qkv_bias=True,
        qk_scale=1.0,
        sr_ratio=[8, 4, 2, 1],
        proj_drop=[0.0, 0.0, 0.0, 0.0],
        attn_drop=[0.0, 0.0, 0.0, 0.0],
        drop_path_rate=0.1
    )

    # 2) 입력
    x = torch.randn(1, 3, 256, 256)

    # 3) 순전파
    #  output: 5개 스테이지 feature (4개+마지막)
    #  attn_weights: 각 스테이지의 마지막 block attention
    features, all_attn = mitbackbone(x)

    # 4) "마지막 스테이지"의 attn_weight 꺼내기
    #  - last_attn.shape = [B, num_heads, N, N']
    #  - 이 때 N = H*W, N' = (H/sr)*(W/sr) 이 될 것
    last_attn = all_attn[0]

    # [중요] 마지막 스테이지의 sr_ratio는 몇?
    #  코드 상: sr_ratio=[8,4,2,1] 이므로 4번째 스테이지는 sr_ratio=1
    #  => 따라서 N'=N => H_k=H, W_k=W

    # 5) "Query" 해상도 (H, W)
    #  4번째 스테이지 patch_embed4의 출력은 (H=? W=?)
    #  1) Stage1: stride=4 => 256/4=64 (H1)
    #  2) Stage2: stride=2 => 64/2=32 (H2)
    #  3) Stage3: stride=2 => 32/2=16 (H3)
    #  4) Stage4: stride=2 => 16/2=8 (H4)
    #  => 마지막 스테이지의 H, W는 8, 8
    H4, W4 = 64, 64

    # 6) 전역 평균 어텐션 시각화
    visualize_global_attention(
        attn_weight=last_attn,  # [B, num_heads, 64, 64] (N=8*8=64, N'=64)
        H=H4, W=W4,             # Query 해상도
        sr_ratio=8,            # 마지막 stage sr=1
        title="Stage4 Global Attention"
    )
