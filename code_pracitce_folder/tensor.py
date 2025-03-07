import torch

# 1. 예시 데이터 준비
#    (A) 고른 분포(균일) : 64개의 값을 0~1 사이 균일분포로 생성
#    (B) 하나만 큰 값 : 63개는 0~1 사이 균일분포, 1개만 매우 큰 값

# A: 고른 분포
xA = torch.rand(64)  # [0,1] 사이 균일분포
# B: 하나가 큰 분포
xB = torch.rand(64)
xB[0] = 10  # 첫 번째 요소만 크게 설정
print(xB)

# 2. 통계량을 계산해 주는 함수 정의
def compute_stats(x):
    stats = {}
    
    # (1) 평균, 표준편차
    mean_x = torch.mean(x)
    std_x = torch.std(x, unbiased=True)
    stats['mean'] = mean_x.item()
    stats['std'] = std_x.item()
    
    # (2) 왜도(skewness)
    #     -- 표본 분산 및 중앙화된 3차 모멘트를 직접 계산
    mean_centered = x - mean_x
    skew_num = torch.mean(mean_centered**3)
    skew_den = torch.mean(mean_centered**2)**1.5
    skewness = skew_num / skew_den
    stats['skewness'] = skewness.item()
    
    # (3) 첨도(kurtosis, excess)
    kurt_num = torch.mean(mean_centered**4)
    kurt_den = torch.mean(mean_centered**2)**2
    kurtosis_excess = kurt_num / kurt_den - 3.0
    stats['kurtosis_excess'] = kurtosis_excess.item()
    
    # (4) 엔트로피(Shannon Entropy) 계산
    #     -- 값이 음수가 없도록 min값을 0 이상으로 시프트
    #        (이 예시는 데이터 자체가 0~1 범위이거나, 100처럼 양수이므로 바로 사용 가능)
    x_shifted = x - torch.min(x) + 1e-8  # 음수 없도록
    p = x_shifted / torch.sum(x_shifted)
    entropy = -torch.sum(p * torch.log(p + 1e-8))
    stats['entropy'] = entropy.item()
    
    # (5) 최대/평균 비율
    max_mean_ratio = torch.max(x) / (mean_x + 1e-8)
    stats['max_mean_ratio'] = max_mean_ratio.item()
    
    return stats

# 3. 각각에 대해 계산
statsA = compute_stats(xA)
statsB = compute_stats(xB)

# 4. 결과 출력
print("=== (A) 고른 분포 xA ===")
print(f"Mean          : {statsA['mean']:.4f}")
print(f"Std           : {statsA['std']:.4f}")
print(f"Skewness      : {statsA['skewness']:.4f}")
print(f"Kurtosis(ex)  : {statsA['kurtosis_excess']:.4f}")
print(f"Entropy       : {statsA['entropy']:.4f}")
print(f"Max/Mean      : {statsA['max_mean_ratio']:.4f}")

print("\n=== (B) 하나가 큰 분포 xB ===")
print(f"Mean          : {statsB['mean']:.4f}")
print(f"Std           : {statsB['std']:.4f}")
print(f"Skewness      : {statsB['skewness']:.4f}")
print(f"Kurtosis(ex)  : {statsB['kurtosis_excess']:.4f}")
print(f"Entropy       : {statsB['entropy']:.4f}")
print(f"Max/Mean      : {statsB['max_mean_ratio']:.4f}")