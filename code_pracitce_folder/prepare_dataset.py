import os

# 폴더 경로와 출력 텍스트 파일 경로 설정
folder_path = "/home/cvlab/dataset/driving_stereo/testing/image_2"  # 여기를 실제 폴더 경로로 바꿔줘
output_file = "/home/cvlab/DomainGeneralization/filenames/source/driving_stereo_test.txt"           # 출력할 텍스트 파일 이름

# 폴더에서 .jpg 파일 목록 가져오기
img_files = [f for f in os.listdir(folder_path) if f.endswith('.png')]

# 파일 이름을 알파벳순으로 정렬
img_files.sort()

# 텍스트 파일에 쓰기
i=0
with open(output_file, 'w') as f:
    for img_file in img_files:
        if i == 700:
            break
        # 파일 이름에서 확장자 제거 (선택 사항)
        left = 'testing/image_2/' + img_file
        right = 'testing/image_3/' + img_file
        disparity = 'testing/disp_occ_0/' + img_file
        # 같은 이름을 띄어쓰기로 구분해 3번 반복
        line = f"{left} {right} {disparity}\n"
        f.write(line)
        i+=1

print(f"작업 완료! 결과가 {output_file}에 저장되었습니다.")