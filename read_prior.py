import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets.kitti2015 import KITTI2015Dataset
from datasets.sceneflow import FlyingThingDataset
import numpy as np
from collections import defaultdict

def calculate_mean_disparity_distribution_per_image(dataset, max_disparity=300):
    """
    각 이미지에서 0~192 사이의 disparity 분포를 계산하고, 
    모든 이미지에 대해서 평균을 계산합니다.
    
    Args:
        dataset: KITTI2015Dataset instance
        max_disparity: Maximum disparity value (default: 192)
    
    Returns:
        np.ndarray: 길이 max_disparity+1인 배열, 각 disparity 값의 평균 분포
        dict: 추가 통계 정보
    """
    distributions = []
    valid_images = 0
    
    print(f"Processing {len(dataset)} images...")
    
    for i in range(len(dataset)):
        try:
            sample = dataset[i]
            disparity = sample.get('disparity')
            
            if disparity is not None:
                # 유효한 disparity 값만 고려 (> 0)
                valid_mask = disparity > 0
                
                if np.any(valid_mask):
                    # 정수로 반올림하고 0~192 범위로 클리핑
                    disp_rounded = np.round(disparity[valid_mask]).astype(int)
                    disp_rounded = np.clip(disp_rounded, 0, max_disparity)
                    
                    # 현재 이미지의 disparity 분포 계산
                    counts = np.bincount(disp_rounded, minlength=max_disparity+1)
                    
                    # 정규화하여 분포로 변환 (합이 1이 되도록)
                    if counts.sum() > 0:
                        distribution = counts / counts.sum()
                        distributions.append(distribution)
                        valid_images += 1
                    
            if i % 10 == 0:
                print(f"Processed {i+1}/{len(dataset)} images")
                
        except Exception as e:
            print(f"Error processing image {i}: {e}")
            continue
    
    if len(distributions) == 0:
        print("No valid disparity data found!")
        return None, None
    
    # 모든 이미지의 분포를 평균내기
    mean_distribution = np.mean(distributions, axis=0)
    
    # 통계 정보
    stats = {
        'total_images': len(dataset),
        'valid_images': valid_images,
        'mean_distribution': mean_distribution,
        'std_distribution': np.std(distributions, axis=0),
        'non_zero_disparities': np.sum(mean_distribution > 0)
    }
    
    print(f"\nProcessed {valid_images} valid images out of {len(dataset)} total images")
    print(f"Non-zero disparity values: {stats['non_zero_disparities']}")
    
    return mean_distribution, stats

def print_distribution_results(mean_distribution, stats, top_k=20):
    """
    결과를 보기 좋게 출력하는 함수
    """
    if mean_distribution is None:
        print("No distribution data available")
        return
    
    # 높은 빈도순으로 정렬
    disparity_freq = [(i, freq) for i, freq in enumerate(mean_distribution) if freq > 0]
    disparity_freq.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\nTop {top_k} most frequent disparity values (averaged across all images):")
    print("Disparity | Avg Frequency | Percentage")
    print("-" * 40)
    
    for disp_val, freq in disparity_freq[:top_k]:
        print(f"{disp_val:8d} | {freq:12.6f} | {freq*100:9.4f}%")
    
    # 전체 통계
    print(f"\nDistribution Statistics:")
    print(f"Sum of distribution: {mean_distribution.sum():.6f}")
    print(f"Min non-zero disparity: {min([d for d, f in disparity_freq])}")
    print(f"Max non-zero disparity: {max([d for d, f in disparity_freq])}")
    print(f"Standard deviation (mean): {np.mean(stats['std_distribution']):.6f}")

def main():
    # Initialize dataset (you'll need to provide the actual paths)
    # dataset = KITTI2015Dataset(
    #     datapath="/home/jaejun/dataset/kitti_2015",
    #     list_filename="/home/jaejun/DomainGeneralization/filenames/target/kitti_2015_train.txt",
    #     training=True,  # Set to True to get disparity ground truth
    #     max_len=None,   # Process all samples
    #     aug=False
    # )
    
    dataset = FlyingThingDataset(
        datapath="/home/jaejun/dataset/flyingthing",
        list_filename="/home/jaejun/DomainGeneralization/filenames/source/flyingthing_train.txt",
        training=True,  # Set to True to get disparity ground truth
        max_len=None,   # Process all samples
        aug=False
    )
    
    
    mean_distribution, stats = calculate_mean_disparity_distribution_per_image(dataset, max_disparity=300)
    
    if mean_distribution is not None:
        # 결과 출력
        print_distribution_results(mean_distribution, stats, top_k=30)
        
        # 결과 저장
        np.save('mean_disparity_distribution_kitti.npy', mean_distribution)
        np.save('distribution_stats.npy', stats)
        print("\nResults saved to 'mean_disparity_distribution.npy' and 'distribution_stats.npy'")
        
        # 분포 시각화 (선택사항)
        try:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(12, 6))
            non_zero_indices = mean_distribution > 0
            plt.bar(np.arange(len(mean_distribution))[non_zero_indices], 
                   mean_distribution[non_zero_indices])
            plt.xlabel('Disparity Value')
            plt.ylabel('Average Frequency')
            plt.title('Mean Disparity Distribution Across All Images')
            plt.grid(True, alpha=0.3)
            plt.savefig('disparity_distribution.png', dpi=300, bbox_inches='tight')
            plt.show()
            
        except ImportError:
            print("matplotlib not available for plotting")

if __name__ == "__main__":
    main()