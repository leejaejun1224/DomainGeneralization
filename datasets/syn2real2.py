# -*- coding: utf-8 -*-
"""
KITTI 전역 통계(색/조도/대비/채도) → synthetic 이미지(좌/우)로 주입
- Lab 색공간 기반 Reinhard color transfer + 선택적 L 히스토그램 매칭
- 좌/우 쌍에 동일 변환 적용(스테레오 일관성 유지)
- 딥러닝/네트워크 미사용, Pillow+NumPy만 사용

사용 예:
python align_to_kitti.py \
  --kitti-dirs /path/to/KITTI_2015/training/image_2 /path/to/KITTI_2015/training/image_3 \
  --train-list /path/to/train_list.txt \
  --out-root aligned_output \
  --max-kitti 200 \
  --hist-strength 0.2

주의:
- train_list.txt는 각 줄이 "left right disp" 3필드를 공백으로 구분한다고 가정.
- 경로가 상대/절대 혼재해도 out-root 아래로 미러링 저장.
"""

import os
import sys
import argparse
from typing import List, Tuple, Dict
import numpy as np
from PIL import Image

# ----------------------------
# 유틸: 파일/경로 처리
# ----------------------------

VALID_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def list_images(dirs: List[str]) -> List[str]:
    files = []
    for d in dirs:
        if not os.path.isdir(d):
            continue
        for root, _, names in os.walk(d):
            for n in names:
                if n.lower().endswith(VALID_EXTS):
                    files.append(os.path.join(root, n))
    return files


def load_image_rgb(path: str) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img).astype(np.float32) / 255.0
    return arr


def save_image_rgb(path: str, arr_uint8: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(arr_uint8, mode="RGB").save(path)


def mirror_out_path(p: str, out_root: str) -> str:
    """
    상대/절대 경로 모두 out_root 아래로 미러링.
    Windows 드라이브 문자(:) 제거, 선행 슬래시 제거.
    """
    if os.path.isabs(p):
        safe = p.replace(":", "")
        while safe.startswith(os.sep) or safe.startswith("/") or safe.startswith("\\"):
            safe = safe[1:]
        return os.path.join(out_root, safe)
    else:
        return os.path.join(out_root, p)


# ----------------------------
# 색공간 변환 (sRGB <-> Lab)
# OpenCV/Skimage 없이 NumPy만 사용
# ----------------------------

# D65 white
_XN, _YN, _ZN = 0.95047, 1.0, 1.08883
_DELTA = 6.0 / 29.0
_EPS = _DELTA ** 3
_KAPPA = 903.3  # not used directly here; using standard piecewise


def _srgb_to_linear(c: np.ndarray) -> np.ndarray:
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(c: np.ndarray) -> np.ndarray:
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(np.maximum(c, 0.0), 1.0 / 2.4) - 0.055)


def _rgb_to_xyz(rgb: np.ndarray) -> np.ndarray:
    r = _srgb_to_linear(rgb[..., 0])
    g = _srgb_to_linear(rgb[..., 1])
    b = _srgb_to_linear(rgb[..., 2])

    # sRGB D65
    X = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b
    Y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b
    Z = 0.0193339 * r + 0.1191920 * g + 0.9503041 * b
    return np.stack([X, Y, Z], axis=-1)


def _f_xyz_to_lab(t: np.ndarray) -> np.ndarray:
    return np.where(t > _EPS, np.cbrt(t), (t / (3 * (_DELTA ** 2))) + (4.0 / 29.0))


def _xyz_to_lab(xyz: np.ndarray) -> np.ndarray:
    x = xyz[..., 0] / _XN
    y = xyz[..., 1] / _YN
    z = xyz[..., 2] / _ZN

    fx = _f_xyz_to_lab(x)
    fy = _f_xyz_to_lab(y)
    fz = _f_xyz_to_lab(z)

    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)
    return np.stack([L, a, b], axis=-1)


def _f_lab_to_xyz(u: np.ndarray) -> np.ndarray:
    return np.where(u ** 3 > _EPS, u ** 3, (u - 4.0 / 29.0) * (3 * (_DELTA ** 2)))


def _lab_to_xyz(lab: np.ndarray) -> np.ndarray:
    L = lab[..., 0]
    a = lab[..., 1]
    b = lab[..., 2]

    fy = (L + 16.0) / 116.0
    fx = fy + (a / 500.0)
    fz = fy - (b / 200.0)

    xr = _f_lab_to_xyz(fx)
    yr = _f_lab_to_xyz(fy)
    zr = _f_lab_to_xyz(fz)

    X = xr * _XN
    Y = yr * _YN
    Z = zr * _ZN
    return np.stack([X, Y, Z], axis=-1)


def _xyz_to_rgb(xyz: np.ndarray) -> np.ndarray:
    X = xyz[..., 0]
    Y = xyz[..., 1]
    Z = xyz[..., 2]

    # Inverse matrix for sRGB D65
    r_lin =  3.2404542 * X - 1.5371385 * Y - 0.4985314 * Z
    g_lin = -0.9692660 * X + 1.8760108 * Y + 0.0415560 * Z
    b_lin =  0.0556434 * X - 0.2040259 * Y + 1.0572252 * Z

    r = _linear_to_srgb(r_lin)
    g = _linear_to_srgb(g_lin)
    b = _linear_to_srgb(b_lin)
    return np.clip(np.stack([r, g, b], axis=-1), 0.0, 1.0)


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    return _xyz_to_lab(_rgb_to_xyz(rgb))


def lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    return _xyz_to_rgb(_lab_to_xyz(lab))


# ----------------------------
# 통계 추정 (KITTI 전역, 쌍 단위 소스)
# ----------------------------

def compute_kitti_stats(kitti_img_paths: List[str],
                        max_images: int = None,
                        downsample: int = 2,
                        hist_bins: int = 256) -> Dict[str, np.ndarray]:
    """
    KITTI 전역 Lab 평균/표준편차 및 L 히스토그램/CDF 계산.
    downsample>1 이면 속도 목적으로 stride 샘플링.
    """
    if max_images is not None and max_images > 0:
        kitti_img_paths = kitti_img_paths[:max_images]

    mean = np.zeros(3, dtype=np.float64)
    M2 = np.zeros(3, dtype=np.float64)
    count = 0
    lum_hist = np.zeros(hist_bins, dtype=np.float64)

    for idx, p in enumerate(kitti_img_paths):
        try:
            img = load_image_rgb(p)
        except Exception as e:
            print(f"[WARN] KITTI 이미지 로드 실패: {p} ({e})", file=sys.stderr)
            continue

        if downsample and downsample > 1:
            img = img[::downsample, ::downsample, :]

        lab = rgb_to_lab(img)
        flat = lab.reshape(-1, 3).astype(np.float64)

        n_i = flat.shape[0]
        if n_i == 0:
            continue

        mu_i = flat.mean(axis=0)
        # population variance: divide by n
        var_i = flat.var(axis=0)
        M2_i = var_i * n_i

        # combine means/variances
        delta = mu_i - mean
        tot = count + n_i
        mean = mean + delta * (n_i / tot)
        M2 = M2 + M2_i + (delta * delta) * (count * n_i / tot)
        count = tot

        # luminance histogram (L in [0,100] -> bins)
        L = flat[:, 0]
        idxs = np.clip((L / 100.0) * (hist_bins - 1), 0, hist_bins - 1).astype(np.int32)
        lum_hist += np.bincount(idxs, minlength=hist_bins)

        if (idx + 1) % 200 == 0:
            print(f"[INFO] KITTI 통계 진행: {idx+1}/{len(kitti_img_paths)}")

    sigma = np.sqrt(M2 / max(count, 1))
    cdf = np.cumsum(lum_hist)
    if cdf[-1] > 0:
        cdf = cdf / cdf[-1]
    else:
        cdf = np.linspace(0, 1, hist_bins)

    return {
        "lab_mean": mean.astype(np.float32),
        "lab_std": sigma.astype(np.float32),
        "lum_hist": lum_hist.astype(np.float32),
        "lum_cdf": cdf.astype(np.float32),
        "hist_bins": hist_bins,
        "n_images": len(kitti_img_paths),
        "count_pixels": int(count),
    }


def compute_pair_lab_stats(left_rgb: np.ndarray,
                           right_rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """좌/우 쌍을 세로로 이어붙여 쌍 단위 Lab 통계 계산."""
    assert left_rgb.shape[2] == 3 and right_rgb.shape[2] == 3
    # 스택(세로 방향): 통계만 필요하므로 모양은 중요하지 않음
    stacked = np.concatenate([left_rgb, right_rgb], axis=0)
    lab = rgb_to_lab(stacked)
    flat = lab.reshape(-1, 3).astype(np.float64)
    mu = flat.mean(axis=0).astype(np.float32)
    sigma = flat.std(axis=0).astype(np.float32)
    return mu, sigma


# ----------------------------
# 변환(주입): Reinhard + 옵션 L 히스토그램 매칭
# ----------------------------

def match_luminance(L: np.ndarray, target_cdf: np.ndarray, hist_bins: int) -> np.ndarray:
    """
    L in [0,100]을 대상으로 전역 CDF 매칭. 반환도 [0,100] 범위.
    """
    L_flat = L.reshape(-1)
    src_bins = hist_bins
    src_hist = np.bincount(
        np.clip((L_flat / 100.0) * (src_bins - 1), 0, src_bins - 1).astype(np.int32),
        minlength=src_bins
    ).astype(np.float64)
    src_cdf = np.cumsum(src_hist)
    if src_cdf[-1] > 0:
        src_cdf = src_cdf / src_cdf[-1]
    else:
        src_cdf = np.linspace(0, 1, src_bins)

    bin_vals = np.linspace(0.0, 100.0, src_bins)
    # 타깃 CDF의 역함수에 src_cdf를 대입해 매핑 테이블 생성
    mapping = np.interp(src_cdf, target_cdf, bin_vals)
    idxs = np.clip((L_flat / 100.0) * (src_bins - 1), 0, src_bins - 1).astype(np.int32)
    L_new = mapping[idxs].reshape(L.shape)
    return L_new


def reinhard_color_transfer_lab(img_rgb: np.ndarray,
                                mu_src: np.ndarray, sig_src: np.ndarray,
                                mu_tgt: np.ndarray, sig_tgt: np.ndarray,
                                strength: float = 1.0,
                                lum_hist_strength: float = 0.0,
                                target_cdf: np.ndarray = None,
                                hist_bins: int = 256) -> np.ndarray:
    """
    단일 이미지에 변환 적용.
    - Lab 평균/표준편차 매칭 (L,a,b)
    - 옵션: L 히스토그램 매칭(0~1 블렌딩)
    """
    lab = rgb_to_lab(img_rgb)

    sig_src_safe = np.maximum(sig_src, 1e-6)
    lab_out = (lab - mu_src) / sig_src_safe * sig_tgt + mu_tgt

    # 변환 강도(0~1) 블렌딩
    if 0.0 <= strength < 1.0:
        lab_out = lab * (1.0 - strength) + lab_out * strength

    # 선택적 L 히스토그램 매칭
    if lum_hist_strength > 1e-6 and target_cdf is not None:
        L_matched = match_luminance(np.clip(lab_out[..., 0], 0.0, 100.0), target_cdf, hist_bins)
        lab_out[..., 0] = lab_out[..., 0] * (1.0 - lum_hist_strength) + L_matched * lum_hist_strength

    # 유효 범위 클리핑
    lab_out[..., 0] = np.clip(lab_out[..., 0], 0.0, 100.0)
    lab_out[..., 1] = np.clip(lab_out[..., 1], -128.0, 127.0)
    lab_out[..., 2] = np.clip(lab_out[..., 2], -128.0, 127.0)

    rgb = lab_to_rgb(lab_out)
    rgb = np.clip(rgb, 0.0, 1.0)
    return (rgb * 255.0 + 0.5).astype(np.uint8)


def align_pair_to_kitti(left_rgb: np.ndarray, right_rgb: np.ndarray,
                        kitti_stats: Dict[str, np.ndarray],
                        strength: float = 1.0,
                        lum_hist_strength: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    좌/우 쌍에 동일 파라미터(μ_src, σ_src -> KITTI μ,σ) 적용.
    """
    mu_src, sig_src = compute_pair_lab_stats(left_rgb, right_rgb)
    mu_tgt = kitti_stats["lab_mean"]
    sig_tgt = kitti_stats["lab_std"]

    outL = reinhard_color_transfer_lab(
        left_rgb, mu_src, sig_src, mu_tgt, sig_tgt,
        strength=strength,
        lum_hist_strength=lum_hist_strength,
        target_cdf=kitti_stats["lum_cdf"],
        hist_bins=int(kitti_stats["hist_bins"]),
    )
    outR = reinhard_color_transfer_lab(
        right_rgb, mu_src, sig_src, mu_tgt, sig_tgt,
        strength=strength,
        lum_hist_strength=lum_hist_strength,
        target_cdf=kitti_stats["lum_cdf"],
        hist_bins=int(kitti_stats["hist_bins"]),
    )
    return outL, outR


# ----------------------------
# 파이프라인: train_list.txt 처리
# ----------------------------

def parse_train_list_line(line: str) -> Tuple[str, str, str]:
    parts = line.strip().split()
    if len(parts) < 2:
        return None, None, None
    left = parts[0]
    right = parts[1]
    disp = parts[2] if len(parts) >= 3 else ""
    return left, right, disp


def process_train_list(train_list_path: str,
                       kitti_dirs: List[str],
                       out_root: str,
                       inplace: bool = False,
                       max_kitti: int = None,
                       kitti_downsample: int = 2,
                       strength: float = 1.0,
                       hist_strength: float = 0.0) -> str:
    """
    - KITTI 전역 통계 추정
    - train_list.txt 순회하며 좌/우 정렬 후 저장
    - 새 리스트 파일 경로 반환
    """
    kitti_imgs = list_images(kitti_dirs)
    if len(kitti_imgs) == 0:
        raise RuntimeError("KITTI 이미지가 발견되지 않았습니다. --kitti-dirs를 확인하세요.")

    print(f"[INFO] KITTI 이미지 수: {len(kitti_imgs)} (max_kitti={max_kitti})")
    kstats = compute_kitti_stats(kitti_imgs, max_images=max_kitti,
                                 downsample=kitti_downsample, hist_bins=256)
    print(f"[INFO] KITTI Lab mean: {kstats['lab_mean']}, std: {kstats['lab_std']}")
    print(f"[INFO] KITTI 픽셀 수(샘플 기준): {kstats['count_pixels']}")

    # 출력 리스트 준비
    if inplace:
        out_list_path = train_list_path  # 덮어쓰기
        new_lines = []
    else:
        os.makedirs(out_root, exist_ok=True)
        base = os.path.basename(train_list_path)
        name, ext = os.path.splitext(base)
        out_list_path = os.path.join(out_root, f"{name}_aligned{ext}")
        new_lines = []

    with open(train_list_path, "r", encoding="utf-8") as f:
        lines = [ln for ln in f.readlines() if ln.strip()]

    for i, ln in enumerate(lines):
        
        left, right, disp = parse_train_list_line(ln)
        left = "/home/jaejun/dataset/flyingthing/" + left
        right = "/home/jaejun/dataset/flyingthing/" + right
        disp = "/home/jaejun/dataset/flyingthing/" + disp
        print(left)
        
        if left is None:
            continue

        try:
            l_img = load_image_rgb(left)
            r_img = load_image_rgb(right)
        except Exception as e:
            print(f"[WARN] synthetic 이미지 로드 실패: {left} | {right} ({e})", file=sys.stderr)
            # 실패 시 원본 라인 그대로 유지
            if inplace:
                new_lines.append(ln.strip())
            else:
                # out_root 기준으로 미러링 경로 생성은 못 하므로 라인 유지
                new_lines.append(ln.strip())
            continue

        out_left = left if inplace else mirror_out_path(left, out_root)
        out_right = right if inplace else mirror_out_path(right, out_root)

        # 변환
        aL, aR = align_pair_to_kitti(
            l_img, r_img, kstats, strength=strength, lum_hist_strength=hist_strength
        )

        # 저장
        save_image_rgb(out_left, aL)
        save_image_rgb(out_right, aR)

        # 리스트 라인 갱신
        if inplace:
            # 경로 변화 없음
            new_lines.append(ln.strip())
        else:
            # disparity 경로는 그대로 유지(원본 사용)
            new_lines.append(f"{out_left} {out_right} {disp}".strip())

        if (i + 1) % 100 == 0:
            print(f"[INFO] 진행: {i+1}/{len(lines)}")

    with open(out_list_path, "w", encoding="utf-8") as fw:
        fw.write("\n".join(new_lines) + "\n")

    print(f"[DONE] 새 리스트 저장: {out_list_path}")
    return out_list_path


# ----------------------------
# CLI
# ----------------------------

def main():
    parser = argparse.ArgumentParser(description="Align synthetic images to KITTI (color/illumination/contrast/saturation)")
    parser.add_argument("--kitti-dirs", default=["/home/jaejun/dataset/kitti_2015/training/image_2/"])
    parser.add_argument("--train-list", default="/home/jaejun/DomainGeneralization/filenames/source/flyingthing_train.txt")
    parser.add_argument("--out-root", default="/home/jaejun/dataset_new", help="결과 이미지를 저장할 루트(미러링). --inplace면 무시")
    parser.add_argument("--inplace", action="store_true", help="원본 synthetic 이미지를 덮어쓰기(주의)")
    parser.add_argument("--max-kitti", type=int, default=0, help="KITTI 통계 추정에 사용할 최대 이미지 수(0이면 전부)")
    parser.add_argument("--kitti-downsample", type=int, default=2, help="KITTI 통계 산정 시 stride 샘플링(속도↑, 1이면 미사용)")
    parser.add_argument("--strength", type=float, default=1.0, help="Reinhard 변환 강도(0~1, 1=완전 매칭)")
    parser.add_argument("--hist-strength", type=float, default=0.0, help="L 히스토그램 매칭 강도(0~1, 기본 0)")
    args = parser.parse_args()

    max_kitti = None if args.max_kitti is None or args.max_kitti <= 0 else args.max_kitti

    process_train_list(
        train_list_path=args.train_list,
        kitti_dirs=args.kitti_dirs,
        out_root=args.out_root,
        inplace=bool(args.inplace),
        max_kitti=max_kitti,
        kitti_downsample=max(1, int(args.kitti_downsample)),
        strength=float(args.strength),
        hist_strength=float(args.hist_strength),
    )


if __name__ == "__main__":
    main()
