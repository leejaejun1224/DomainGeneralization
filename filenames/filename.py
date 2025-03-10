import re

# 변환할 파일 경로
input_file = "./target/cityscapes_train_copy.txt"
output_file = "./target/cityscapes_train.txt"

# 파일 읽기
with open(input_file, "r", encoding="utf-8") as f:
    lines = f.readlines()

updated_lines = []

# 정규식 패턴: 도시명_첫 번째 숫자_두 번째 숫자_파일타입.png
pattern = re.compile(r"(.+?/)([^/]+)_(\d+_\d+)(_leftImg8bit\.png)")

i=0
for line in lines:
    i+=1
    if i>400:
        break
    parts = line.strip().split()  # 공백 기준으로 분리
    if len(parts) != 3:
        continue  # 형식이 맞지 않는 줄은 스킵

    left_img, right_img, disparity_img = parts

    # left 이미지에서 기준이 될 숫자 부분 추출
    match = pattern.search(left_img)
    if not match:
        continue

    base_path, city_name, left_number, suffix = match.groups()  # 예: (경로, bielefeld, 000000_025748, _leftImg8bit.png)

    # right와 disparity의 숫자 부분을 left 기준으로 변경
    right_img_new = re.sub(r"(.+?/)([^/]+)_(\d+_\d+)(_rightImg8bit\.png)",
                           rf"\1\2_{left_number}\4", right_img)
    disparity_img_new = re.sub(r"(.+?/)([^/]+)_(\d+_\d+)(_disparity\.png)",
                               rf"\1\2_{left_number}\4", disparity_img)

    # 새로운 라인 생성
    updated_line = f"{left_img} {right_img_new} {disparity_img_new}\n"
    updated_lines.append(updated_line)

# 변환된 데이터 저장
with open(output_file, "w", encoding="utf-8") as f:
    f.writelines(updated_lines)

print(f"변환 완료! 결과는 {output_file}에 저장됨.")
