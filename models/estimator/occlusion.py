import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.estimator.submodules import *   # 이미 사용 중인 블록 재사용

class OcclusionPredictor(nn.Module):
    """
    Occlusion mask head (left view).
    Inputs:
        features_left_cat: [B, 80, H/4, W/4]  # features_left[0] + stem_4x
        att_logits:        [B, 1, D, H/4, W/4]  # hourglass_att 출력(softmax 전)
        corr_volume:       [B, 1 or 8, D, H/4, W/4]  # build_norm_correlation_volume 결과 또는 stem 통과본
        spx_pred:          [B, 9, H/2, W/2] or [B, 9, H, W]  # context_upsample에 사용(선택)

    Outputs:
        occ_up:     [B, 1, H, W]     # 확률(시그모이드 후)
        occ_1_4:    [B, 1, H/4, W/4] # 확률(시그모이드 후)
        occ_logit:  [B, 1, H/4, W/4] # 로짓(손실 계산용)
        aux: dict   # 디버그용 통계
    """
    def __init__(self, feat_ch=80, use_corr=True, use_att=True):
        super().__init__()
        self.use_corr = use_corr
        self.use_att  = use_att

        # 좌측 컨텍스트 축소
        self.reduce_feat = BasicConv(feat_ch, 16, kernel_size=3, stride=1, padding=1)

        # 최종 헤드 (간단한 2D CNN)
        # 입력 채널: 16(컨텍스트) + 4(att 분포 통계) + 2(corr 통계) = 22
        in_ch = 16 + (4 if use_att else 0) + (2 if use_corr else 0)
        self.head = nn.Sequential(
            BasicConv(in_ch, 32, kernel_size=3, stride=1, padding=1),
            BasicConv(32, 32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0)  # logit
        )

    @staticmethod
    def _dist_feats(att_logits):
        # att_logits: [B,1,D,H,W] -> softmax over D
        p = F.softmax(att_logits, dim=2).clamp_min(1e-8)
        # top-2
        top2 = torch.topk(p, k=2, dim=2, largest=True).values  # [B,1,2,H,W]
        p1 = top2[:, :, 0]                                     # [B,1,H,W]
        p2 = top2[:, :, 1]
        gap = p1 - p2
        var = p.var(dim=2)                                     # [B,1,H,W]
        ent = -(p * p.log()).sum(dim=2) / math.log(p.size(2))  # 정규화 엔트로피
        return torch.cat([p1, gap, var, ent], dim=1)           # [B,4,H,W]

    @staticmethod
    def _corr_feats(corr_volume):
        # corr_volume: [B,C(=1 or 8),D,H,W]  -> 채널 평균 후 D-통계
        if corr_volume is None:
            return None
        if corr_volume.size(1) > 1:
            c = corr_volume.mean(dim=1, keepdim=True)
        else:
            c = corr_volume
        cmax = c.max(dim=2).values     # [B,1,H,W]
        cstd = c.std(dim=2)            # [B,1,H,W]
        return torch.cat([cmax, cstd], dim=1)  # [B,2,H,W]

    def forward(self, features_left_cat, att_logits=None, corr_volume=None, spx_pred=None):
        B, _, H4, W4 = features_left_cat.shape

        feats = [self.reduce_feat(features_left_cat)]  # [B,16,H/4,W/4]

        if self.use_att and (att_logits is not None):
            feats.append(self._dist_feats(att_logits))  # [B,4,H/4,W/4]
        if self.use_corr and (corr_volume is not None):
            feats.append(self._corr_feats(corr_volume))  # [B,2,H/4,W/4]

        x = torch.cat(feats, dim=1)               # [B, 16/20/22, H/4, W/4]
        occ_logit = self.head(x)                  # [B,1,H/4,W/4]
        occ_1_4   = torch.sigmoid(occ_logit)      # 확률

        # 업샘플: 제공되면 context_upsample, 없으면 bilinear
        if spx_pred is not None:
            occ_up = context_upsample(occ_1_4, spx_pred)  # [B,1,H,W]
        else:
            occ_up = F.interpolate(occ_1_4, scale_factor=4, mode='bilinear', align_corners=False)

        aux = {}
        return occ_up, occ_1_4, occ_logit, aux
