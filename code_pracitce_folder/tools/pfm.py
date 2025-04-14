import numpy as np
import cv2

def load_pfm(file):
    """
    PFM 파일을 읽어 numpy 배열로 반환합니다.
    지원하는 헤더:
      - "Pf": 단일 채널 (grayscale)
      - "PF": 3채널 (RGB)
    """
    with open(file, 'rb') as f:
        # 첫 줄 헤더: 'Pf' 또는 'PF'
        header = f.readline().decode('utf-8').rstrip()
        if header == 'PF':
            channels = 3
        elif header == 'Pf':
            channels = 1
        else:
            raise ValueError("Not a PFM file: 잘못된 헤더: {}".format(header))
        
        # 이미지 크기 정보 읽기 (width height)
        dims_line = f.readline().decode('utf-8').rstrip()
        while dims_line.startswith('#'):  # 주석라인 무시
            dims_line = f.readline().decode('utf-8').rstrip()
        dims = dims_line.split()
        if len(dims) != 2:
            raise ValueError("잘못된 이미지 크기 정보")
        width, height = int(dims[0]), int(dims[1])

        # 스케일 팩터 및 엔디안 설정
        scale = float(f.readline().decode('utf-8').rstrip())
        endian = '<' if scale < 0 else '>'
        scale = abs(scale)

        # 남은 데이터를 읽어 float 배열로 변환
        data = np.fromfile(f, endian + 'f')
        expected_elems = width * height * (channels)
        if data.size != expected_elems:
            raise ValueError("예상한 요소수와 실제 읽은 요소수가 다릅니다: expected {} vs got {}".format(expected_elems, data.size))

        # 배열 리쉐입 및 이미지 플립(상하 반전)
        if channels == 3:
            data = np.reshape(data, (height, width, 3))
        else:
            data = np.reshape(data, (height, width))
        data = np.flipud(data)

    return data

def convert_pfm_to_png(pfm_path, png_path):
    # PFM 파일 읽기
    data = load_pfm(pfm_path)
    
    # 데이터가 부동소수점이므로 PNG에 저장할 수 있도록 정규화가 필요합니다.
    # 최소/최대값 계산 후 0~255 범위로 선형 정규화
    data_min = data.min()
    data_max = data.max()
    
    if data_max - data_min > 0:
        data_norm = (data - data_min) / (data_max - data_min)
    else:
        data_norm = np.zeros_like(data)

    # 0~255 범위의 8비트 정수형으로 변환
    data_8bit = (data_norm * 255).astype(np.uint8)
    
    # PNG 파일로 저장 (컬러 이미지의 경우 OpenCV는 BGR 순서여서 사용 시 주의)
    cv2.imwrite(png_path, data_8bit)
    print(f"'{pfm_path}'를 '{png_path}'로 변환 완료.")

if __name__ == "__main__":
    # 파일 경로 설정 (필요에 따라 경로를 수정하세요)
    input_pfm = "/home/cvlab/dataset/FlyingThings3D_subset_disparity/FlyingThings3D_subset/val/disparity/left/0000077.pfm"
    output_png = "converted.png"
    convert_pfm_to_png(input_pfm, output_png)
