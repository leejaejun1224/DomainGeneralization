import numpy as np
import cv2
from PIL import Image
import re

def read_pfm(file_path):
    """PFM 파일을 읽어서 numpy 배열로 반환"""
    with open(file_path, 'rb') as file:
        # 헤더 읽기
        header = file.readline().rstrip().decode('utf-8')
        
        if header == 'PF':
            channels = 3
        elif header == 'Pf':
            channels = 1
        else:
            raise Exception('PFM 파일이 아닙니다.')
        
        # 크기 정보 읽기
        dim_line = file.readline().decode('utf-8')
        width, height = map(int, dim_line.split())
        
        # 스케일 정보 읽기
        scale = float(file.readline().decode('utf-8'))
        
        # 엔디안 결정
        if scale < 0:
            endian = '<'  # little endian
            scale = -scale
        else:
            endian = '>'  # big endian
        
        # 이미지 데이터 읽기
        data = np.fromfile(file, endian + 'f')
        
        # 배열 형태로 변환
        if channels == 3:
            data = data.reshape((height, width, 3))
        else:
            data = data.reshape((height, width))
        
        # 상하 반전 (PFM은 bottom-up 형식)
        data = np.flipud(data)
        
        return data

def disparity_to_grayscale(disparity_map, output_path):
    """Disparity map을 흑백 이미지로 변환"""
    print(f"원본 데이터 형태: {disparity_map.shape}")
    print(f"데이터 타입: {disparity_map.dtype}")
    print(f"최솟값: {np.min(disparity_map)}")
    print(f"최댓값: {np.max(disparity_map)}")
    print(f"무한대 값 개수: {np.sum(np.isinf(disparity_map))}")
    print(f"NaN 값 개수: {np.sum(np.isnan(disparity_map))}")
    
    # 무한대와 NaN 값 처리
    disparity_map = np.where(np.isinf(disparity_map), 0, disparity_map)
    disparity_map = np.where(np.isnan(disparity_map), 0, disparity_map)
    
    # 유효한 disparity 값 확인
    valid_disparity = disparity_map[disparity_map > 0]
    print(f"0보다 큰 값의 개수: {len(valid_disparity)}")
    
    if len(valid_disparity) == 0:
        print("경고: 0보다 큰 disparity 값이 없습니다. 전체 범위로 정규화합니다.")
        # 전체 범위로 정규화
        min_disp = np.min(disparity_map)
        max_disp = np.max(disparity_map)
        
        if min_disp == max_disp:
            # 모든 값이 같은 경우
            normalized = np.zeros_like(disparity_map, dtype=np.uint8)
        else:
            normalized = (disparity_map - min_disp) / (max_disp - min_disp) * 255
            normalized = np.clip(normalized, 0, 255).astype(np.uint8)
    else:
        # 원래 로직: 양수 값만으로 정규화
        min_disp = np.min(valid_disparity)
        max_disp = np.max(disparity_map)
        
        normalized = np.where(disparity_map > 0, 
                            (disparity_map - min_disp) / (max_disp - min_disp) * 255, 
                            0)
        normalized = np.clip(normalized, 0, 255).astype(np.uint8)
    
    # PNG로 저장
    cv2.imwrite(output_path, normalized)
    print(f"이미지가 {output_path}에 저장되었습니다.")
    
    return normalized

# 사용 예시
disparity_data = read_pfm('/home/jaejun/dataset/flyingthing/FlyingThings3D_subset_disparity/FlyingThings3D_subset/train/disparity/left/0000000.pfm')
grayscale_image = disparity_to_grayscale(disparity_data, 'disparity_grayscale.png')
