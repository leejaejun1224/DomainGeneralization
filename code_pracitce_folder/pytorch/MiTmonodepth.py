import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerModel

class MonoDepthDecoder(nn.Module):
    def __init__(self, encoder_channels, decoder_channels=256, output_channels=1):
        """
        encoder_channels: 인코더에서 추출한 각 스케일 특성의 채널 수 (예: [64, 128, 320, 512])
        decoder_channels: 디코더 내에서 사용할 통일된 채널 수
        output_channels: 출력 채널 수 (깊이 map은 보통 1채널)
        """
        super().__init__()
        
        # 각 encoder 특성의 채널을 decoder_channels로 맞추기 위한 1x1 Conv layer
        self.conv_layers = nn.ModuleList([
            nn.Conv2d(ch, decoder_channels, kernel_size=1) 
            for ch in encoder_channels
        ])
        
        # 여러 스케일 특성을 병합한 후 추가 연산을 위한 합성(conv) layer
        self.fuse = nn.Conv2d(decoder_channels * len(encoder_channels), decoder_channels, kernel_size=3, padding=1)
        
        # 업샘플링 및 추가 convolution 블록
        self.up1 = nn.ConvTranspose2d(decoder_channels, decoder_channels // 2, kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(decoder_channels // 2, decoder_channels // 2, kernel_size=3, padding=1)
        
        self.up2 = nn.ConvTranspose2d(decoder_channels // 2, decoder_channels // 4, kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(decoder_channels // 4, decoder_channels // 4, kernel_size=3, padding=1)
        
        self.up3 = nn.ConvTranspose2d(decoder_channels // 4, decoder_channels // 8, kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(decoder_channels // 8, decoder_channels // 8, kernel_size=3, padding=1)
        
        # 최종 깊이 map을 예측하기 위한 layer (출력 채널 수 = 1)
        self.depth_pred = nn.Conv2d(decoder_channels // 8, output_channels, kernel_size=3, padding=1)
        
    def forward(self, features):
        """
        features: 인코더에서 나온 다중 스케일 특성들이 담긴 리스트.
                  각 특성은 shape (B, H, W, C)로 가정합니다.
        """
        resized_features = []
        # 첫 번째 특성의 공간 크기를 기준으로 다른 특성들을 맞춤
        target_size = features[0].shape[1:3]  # (H, W)
        
        for i, f in enumerate(features):
            # (B, H, W, C) -> (B, C, H, W)
            # x = f.permute(0, 3, 1, 2)
            # 크기가 맞지 않으면 보간법으로 크기 맞추기
            if f.shape[-2:] != target_size:
                f = F.interpolate(f, size=target_size, mode='bilinear', align_corners=False)
            # 1x1 Conv로 채널 차원 맞추기
            x = self.conv_layers[i](f)
            resized_features.append(x)
        
        # 여러 스케일 특성을 채널 방향으로 병합
        x = torch.cat(resized_features, dim=1)
        x = F.relu(self.fuse(x))
        
        # 업샘플링 및 convolution 연산 진행
        x = self.up1(x)
        x = F.relu(self.conv1(x))
        
        x = self.up2(x)
        x = F.relu(self.conv2(x))
        
        x = self.up3(x)
        x = F.relu(self.conv3(x))
        
        # 최종 깊이 map 예측 (크기는 초기 입력 이미지의 일부 해상도일 수 있음)
        depth = self.depth_pred(x)
        return depth

class MonoDepthEstimationModel(nn.Module):
    def __init__(self, pretrained_encoder=True):
        """
        SegFormer의 인코더를 기반으로 단안 깊이 추정 모델 구성
        """
        super().__init__()
        
        # 사전 학습된 SegFormer 인코더 로드 (Transformers 라이브러리 사용)
        self.encoder = SegformerModel.from_pretrained(
            "nvidia/segformer-b0-finetuned-ade-512-512", 
            output_hidden_states=True  # encoder의 다중 스케일 특징 추출을 위해
        )
        
        # 예시로, SegFormer-B0의 각 스테이지 채널을 [64, 128, 320, 512]로 가정
        encoder_channels = [32, 64, 160, 256]
        
        # MonoDepth estimation을 위한 디코더
        self.decoder = MonoDepthDecoder(encoder_channels)
        
    def forward(self, x):
        """
        x: 입력 이미지 텐서, shape (B, 3, H, W)
           (입력 전 반드시 적절한 정규화가 필요합니다.)
        """
        # 인코더를 통해 다중 스케일 특성 추출
        encoder_outputs = self.encoder(pixel_values=x)
        
        # encoder의 마지막 4개의 스테이지 feature 사용 (모델에 따라 조정 필요)
        hidden_states = encoder_outputs.hidden_states[-4:]
        for i, hs in enumerate(hidden_states):
            print(f"hidden_states[{i}].shape: {hs.shape}")
        # 디코더를 통해 깊이 map 예측
        depth = self.decoder(hidden_states)
        return depth

if __name__ == '__main__':
    # 모델 테스트: 임의의 입력 이미지 (배치 크기 1, 3채널, 512x512)
    model = MonoDepthEstimationModel()
    dummy_input = torch.randn(1, 3, 256, 512)
    depth_output = model(dummy_input)
    print("Depth map shape:", depth_output.shape)
