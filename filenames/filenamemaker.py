from typing import List, Tuple

def write_flyingthings_list(
    ranges: List[Tuple[int, int]],
    out_txt: str = "output.txt",
    img_root: str = "FlyingThings3D_subset_image_clean/FlyingThings3D_subset/train/image_clean",
    disp_root: str = "FlyingThings3D_subset_disparity/FlyingThings3D_subset/train/disparity",
    zpad: int = 7,
) -> None:
    """
    ranges: [[1249,1258], [1369,1378], ...] 형태의 (포함 구간) 리스트
    out_txt: 결과를 저장할 txt 파일 경로
    img_root, disp_root: 이미지/시차 데이터의 루트 디렉터리
    zpad: 파일명 숫자 0-padding 자리수 (기본 7)
    """
    lines = []
    seen = set()  # 중복 방지(겹치는 구간이 있을 때도 한 번만 기록)

    for s, e in ranges:
        if s > e:
            s, e = e, s  # 잘못 뒤집힌 구간이 오더라도 보정
        for n in range(s, e + 1):
            if n in seen:
                continue
            seen.add(n)
            sid = f"{n:0{zpad}d}"
            left_img  = f"{img_root}/left/{sid}.png"
            right_img = f"{img_root}/right/{sid}.png"
            left_pfm  = f"{disp_root}/left/{sid}.pfm"
            lines.append(f"{left_img} {right_img} {left_pfm}")

    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# 사용 예시
if __name__ == "__main__":
    ranges = [
        [1249, 1258],
        [1369, 1378],
        [1539, 1548],
        [1628, 1638],
        [1952, 1958],
        [2110, 2116],
        [2286, 2295],
        [2476, 2485],
        [2542, 2545],
        [2896, 2904],
        [3519, 3523],
        [4054, 4063],
        [5013, 5018],
        [5113, 5121],
        [8276, 8280],
        [8306, 8310],
        [8717, 8720],
        [12619, 12628],
        [15818, 15827],
        [20000, 20007]
    ]
    write_flyingthings_list(ranges, out_txt="train_list.txt")
    print("Saved to train_list.txt")
