import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerModel  # pip install transformers>=4.38

# -------------------------------------------------
# Decoder: SegFormer 공식 MLP‑형식을 간략화한 버전
# -------------------------------------------------
class MonoDepthDecoder(nn.Module):
    def __init__(self, embed_dims=(32, 64, 160, 256), out_ch=32, max_disp=48):
        super().__init__()
        self.proj4 = nn.Conv2d(embed_dims[3], 256, 1)
        self.proj3 = nn.Conv2d(embed_dims[2], 256, 1)
        self.proj2 = nn.Conv2d(embed_dims[1], 256, 1)
        self.proj1 = nn.Conv2d(embed_dims[0], 256, 1)

        self.fuse = nn.Sequential(
            nn.Conv2d(256 * 4, 256, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=1, bias=False),
        )
        self.head = nn.Conv2d(128, out_ch, 1)
        self.max_disp = max_disp
        self.depth_norm_indexes = torch.linspace(0, 1, 32, dtype=torch.float32, device=torch.device("cuda")).view(1, 32, 1, 1)
    
    def forward(self, feats):                       # feats = (x1,x2,x3,x4)
        x1, x2, x3, x4 = feats
        # ① 채널 정렬
        p4 = self.proj4(x4)                         # 1/32
        p3 = self.proj3(x3)                         # 1/16
        p2 = self.proj2(x2)                         # 1/8
        p1 = self.proj1(x1)                         # 1/4
        # ② 모두 1/4 해상도로 업샘플
        size = p1.shape[-2:]
        p4 = F.interpolate(p4, size, mode="bilinear", align_corners=False)
        p3 = F.interpolate(p3, size, mode="bilinear", align_corners=False)
        p2 = F.interpolate(p2, size, mode="bilinear", align_corners=False)
        # ③ concat‑fuse
        fused = self.fuse(torch.cat([p1, p2, p3, p4], dim=1))
        disp = self.head(fused)
        map = F.interpolate(disp, scale_factor=4, mode="bilinear", align_corners=False) 
        map = F.softmax(map, dim=1)
        map = torch.sum(map*self.depth_norm_indexes, dim=1, keepdim=True)
        return map
        

# -------------------------------------------------
# 전체 네트워크
# -------------------------------------------------
# class MonoDispSegFormer(nn.Module):
#     def __init__(self, segformer_variant="nvidia/segformer-b0", max_disp=192):
#         super().__init__()
#         self.encoder = SegformerModel.from_pretrained(segformer_variant,
#                                                       num_labels=0, id2label={}, label2id={})
#         embed_dims = self.encoder.config.hidden_sizes   # (64, 128, 320, 512) for B0
#         self.decoder = SegFormerDepthDecoder(embed_dims, max_disp=max_disp)

#     def forward(self, x):
#         # encoder outputs: last_hidden_state (N, H*W/32, C) + hidden_states list
#         enc_out = self.encoder(pixel_values=x, output_hidden_states=True)
#         feats = enc_out.hidden_states[1:]  # four stage features (1/4 ... 1/32)
#         disp = self.decoder(feats)
#         disp = F.interpolate(disp, size=x.shape[-2:], mode="bilinear",
#                              align_corners=False)  # 최종 full‑res 출력
#         return disp
