import torch

# 저장된 .pth 파일 로드
checkpoint = torch.load("log/2025-03-16_13_48/checkpoint_epoch_750.pth", map_location=torch.device("cuda:0"))

modules_upto_desc = [
    "feature.",     # FeatureMiT
    "feature_up.",  # FeatUp
    "stem_2", 
    "stem_4",
    "spx",
    "spx_2",
    "spx_4",
    "conv",         # self.conv
    "desc",         # self.desc
]
# 최상위 키(예: 'state_dict', 'optimizer', etc.) 확인
print("=== Checkpoint Keys ===")
for key in checkpoint.keys():
    if key == "student_state_dict":
        for k, v in checkpoint[key].items():
            if any(k.startswith(prefix) for prefix in modules_upto_desc):
                print(k)

# 만약 모델 가중치가 'state_dict' 안에 있다면, 가중치의 키 확인
if "state_dict" in checkpoint:
    print("\n=== State Dict Keys ===")
    for k, v in checkpoint["state_dict"].items():
        if any(k.startswith(prefix) for prefix in modules_upto_desc):
            print(k)
