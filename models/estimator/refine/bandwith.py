import torch
import torch.nn as nn
import torch.nn.functional as F

class BandwidthHead(nn.Module):
    """
    픽셀별 집계 밴드폭 헤드.
    입력:  features_left_cat   [B, 80, H/4, W/4]
          +entropy(H0)         [B,  1, H/4, W/4]
          +top2 gap            [B,  1, H/4, W/4]
    출력: tau_map ∈ [tau_min, tau_max]          [B,1,H/4,W/4]
         (선택) r_map   ∈ [r_min, r_max] (float) [B,1,H/4,W/4]
    """
    def __init__(self, in_ch=82, hidden=64,
                 tau_range=(0.6, 1.6), r_range=(2.0, 8.0),
                 predict_r=True, gn_groups=8):
        super().__init__()
        self.tau_min, self.tau_max = tau_range
        self.predict_r = predict_r
        self.r_min, self.r_max = r_range

        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, 1, 1, bias=False),
                nn.GroupNorm(num_groups=min(gn_groups, cout), num_channels=cout),
                nn.ReLU(inplace=True)
            )
        self.trunk = nn.Sequential(
            block(in_ch, hidden),
            block(hidden, hidden)
        )
        self.head_tau = nn.Conv2d(hidden, 1, 3, 1, 1)
        if self.predict_r:
            self.head_r = nn.Conv2d(hidden, 1, 3, 1, 1)

    def forward(self, feat_s4, ent_s4, gap_s4):
        x = torch.cat([feat_s4, ent_s4, gap_s4], dim=1)  # [B,82,H/4,W/4]
        h = self.trunk(x)
        tau = torch.sigmoid(self.head_tau(h))
        tau = self.tau_min + (self.tau_max - self.tau_min) * tau  # [B,1,H/4,W/4]
        if self.predict_r:
            r = torch.sigmoid(getattr(self, 'head_r')(h))
            r = self.r_min + (self.r_max - self.r_min) * r         # [B,1,H/4,W/4] (float)
            return tau, r
        return tau, None
