import torch
import torch.nn as nn
import torch.nn.functional as F

class GeometricMLP(nn.Module):
    """
    입력: (B, D, H, W)
     - B: batch size
     - D: disparity(또는 cost-volume) 채널 수
     - H, W: 이미지 높이/너비

    각 픽셀의 (x, y) 좌표와 D-채널 값을 MLP로 보내
    최종적으로 (B, out_dim, H, W) 형태의 "기하학 임베딩"을 출력.
    """
    def __init__(self, disparity_dim=32, hidden_dim=64, out_dim=16):
        """
        Args:
            disparity_dim: 입력으로 들어오는 disparity(혹은 cost volume) 채널 수 (D)
            hidden_dim   : MLP의 은닉층 크기
            out_dim      : 최종 출력 임베딩(기하 정보)의 채널 크기
        """
        super(GeometricMLP, self).__init__()
        
        # (x, y) 좌표 2차원 + disparity_dim
        # => MLP에 들어갈 최종 입력차원 = 2 + disparity_dim
        self.in_dim = 2 + disparity_dim
        
        # 간단한 MLP 정의 (3-layer 예시)
        self.mlp = nn.Sequential(
            nn.Linear(self.in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )
        
    def forward(self, x):
        """
        x: (B, D, H, W) 형태의 텐서

        반환:
        out: (B, out_dim, H, W)
        """
        B, D, H, W = x.shape
        
        # 1. (x, y) 좌표생성 + 정규화
        #    x 좌표는 [0..W-1], y 좌표는 [0..H-1]
        #    => [-1..1] 범위로 매핑
        #    shape: coords_x, coords_y => (H, W)
        
        device = x.device  # x와 동일한 device(cpu/gpu) 사용
        
        # coords_x: 0..W-1, coords_y: 0..H-1
        coords_x = torch.linspace(0, W-1, steps=W, device=device).unsqueeze(0).expand(H, W)
        coords_y = torch.linspace(0, H-1, steps=H, device=device).unsqueeze(1).expand(H, W)
        
        # 정규화 -> [-1, 1]
        # 예: x_norm = (x/(W-1))*2 - 1
        coords_x_norm = (coords_x / (W - 1)) * 2.0 - 1.0  # (H, W)
        coords_y_norm = (coords_y / (H - 1)) * 2.0 - 1.0  # (H, W)
        
        # 2. x(입력)을 (B, D, H, W) -> (B, H, W, D) 로 permute
        x_perm = x.permute(0, 2, 3, 1)  # shape: (B, H, W, D)
        
        # 3. coords_x_norm, coords_y_norm을 배치 차원 B만큼 복제할 수도 있으나,
        #    더 간단히, 아래 flatten 과정에서 각 픽셀 위치에 맞춰 사용할 것.
        
        # 4. flatten하여 MLP에 넣을 준비
        #    (B, H, W, D) -> (B*H*W, D)
        x_flat = x_perm.reshape(-1, D)  # shape: (B*H*W, D)
        
        # 5. 좌표도 (H, W) -> (H*W,) 후 B배 분량 만큼 반복
        coords_x_norm_flat = coords_x_norm.reshape(-1)  # (H*W,)
        coords_y_norm_flat = coords_y_norm.reshape(-1)  # (H*W,)
        
        #  (B*H*W,) 크기 맞춰야 하므로
        #   - 한 배치 안에서 (H*W) 픽셀  
        #   - 총 B배 만큼 => B*H*W
        #   => repeat_interleave
        coords_x_norm_flat = coords_x_norm_flat.repeat_interleave(B)  # (B*H*W,)
        coords_y_norm_flat = coords_y_norm_flat.repeat_interleave(B)  # (B*H*W,)
        
        # 6. (x_coord, y_coord, cost_volume) 합치기
        #    => concat dim=1 => shape: (B*H*W, 2 + D)
        coords_xy = torch.stack([coords_x_norm_flat, coords_y_norm_flat], dim=1)  # (B*H*W, 2)
        mlp_input = torch.cat([coords_xy, x_flat], dim=1)  # (B*H*W, 2 + D)
        
        # 7. MLP 적용
        mlp_output = self.mlp(mlp_input)  # (B*H*W, out_dim)
        
        # 8. 원위치로 reshape => (B, H, W, out_dim)
        mlp_output = mlp_output.reshape(B, H, W, -1)
        
        # 9. (B, out_dim, H, W) 로 permute
        out = mlp_output.permute(0, 3, 1, 2).contiguous()
        
        return out

def example_usage():
    import torch
    
    # 하이퍼파라미터
    B, D, H, W = 2, 8, 4, 5
    hidden_dim = 16
    out_dim = 4
    
    # 모델 생성
    model = GeometricMLP(disparity_dim=D, hidden_dim=hidden_dim, out_dim=out_dim)
    
    # 더미 입력: (B, D, H, W)
    # 예) cost volume or disparity feature 등
    x = torch.randn(B, D, H, W)  # 랜덤으로 생성
    
    # forward
    output = model(x)
    
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {output.shape}  (B, out_dim, H, W)")
    # 예: (2, 4, 4, 5)

if __name__ == "__main__":
    example_usage()
