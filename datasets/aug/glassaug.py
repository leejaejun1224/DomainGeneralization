#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
augment_glass_stereo.py

양안 스테레오 데이터에 '자동차 유리' 유사 반투명 레이어를 합성하고,
유리 영역에 해당하는 disparity GT(유리 평면 시차)를 추가(덮어쓰기)하는 스크립트.

입력:
  - --left_dir  : 좌영상 폴더 (예: .../left)
  - --right_dir : 우영상 폴더 (예: .../right)
  - --disp_dir  : 좌영상 기준 disparity GT 폴더 (pfm/png16/npy 지원)
출력:
  - --out_dir   : 출력 루트 (left, right, disp, mask 하위 폴더 생성)

특징:
  1) 유리 마스크: 사다리꼴/사각형 임의 다각형(윈드실드/사이드윈도 느낌) 생성
  2) 유리 렌더: 살짝 청록 틴팅 + 약한 블러 + 위아래 그라디언트로 유리 느낌
  3) disparity: 유리 마스크 내부는 '유리 평면 시차'로 덮어쓰기(배경은 보이지만 라벨은 유리)
  4) png16 GT의 경우 보편 스케일(×256) 지원 (--png_divisor)
  5) 결과 확인용 마스크도 저장

주의:
  - 학습 시 '이미지의 배경은 보이지만 GT는 유리 시차'라는 설정이므로,
    실제 장면물리와 완전 일치하지는 않습니다(투명체 다층 구조 미모델링).
  - 필요 시 투명체 굴절/쯔위즐(왜곡)까지 모델링하려면 별도 워핑이 필요합니다.
"""

import os, glob, argparse, math, random
import numpy as np
import cv2

# -------------------------
# 유틸: Disparity 입출력
# -------------------------
def read_pfm(path):
    """간단 PFM 리더 (RGB/GRAY 지원). 반환: np.float32, HxW or HxWx3"""
    with open(path, "rb") as f:
        header = f.readline().decode('utf-8').rstrip()
        if header not in ('PF','Pf'):
            raise ValueError("Not a PFM file.")
        dims = f.readline().decode('utf-8')
        while dims.startswith('#'):
            dims = f.readline().decode('utf-8')
        w, h = map(int, dims.strip().split())
        scale = float(f.readline().decode('utf-8').strip())
        data = np.fromfile(f, '<f' if scale < 0 else '>f')
        shape = (h, w, 3) if header == 'PF' else (h, w)
        data = np.reshape(data, shape)
        if scale < 0:  # little-endian
            data = data
        else:
            data = data.byteswap().newbyteorder()
        data = np.flipud(data).astype(np.float32)
        return data

def write_pfm(path, image, scale=1.0):
    """간단 PFM 라이터."""
    image = np.flipud(image)
    if image.dtype.name != 'float32':
        image = image.astype(np.float32)
    color = (len(image.shape) == 3 and image.shape[2] == 3)
    with open(path, 'wb') as f:
        f.write(('PF\n' if color else 'Pf\n').encode('utf-8'))
        f.write(f"{image.shape[1]} {image.shape[0]}\n".encode('utf-8'))
        endian_scale = -scale  # little-endian
        f.write((str(endian_scale) + '\n').encode('utf-8'))
        image.tofile(f)

def read_disp_auto(path, png_divisor=256.0):
    ext = os.path.splitext(path)[1].lower()
    if ext == '.pfm':
        disp = read_pfm(path)
        if disp.ndim == 3:
            disp = cv2.cvtColor(disp, cv2.COLOR_RGB2GRAY)
        return disp.astype(np.float32)
    elif ext == '.png':
        raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if raw is None:
            raise FileNotFoundError(path)
        if raw.dtype == np.uint16:
            return (raw.astype(np.float32) / png_divisor)
        else:
            return raw.astype(np.float32)
    elif ext == '.npy':
        return np.load(path).astype(np.float32)
    else:
        raise ValueError(f"Unsupported disp ext: {ext}")

def write_disp_like(path_in, disp, out_path, png_divisor=256.0):
    ext = os.path.splitext(path_in)[1].lower()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if ext == '.pfm':
        write_pfm(out_path.replace('.png', '.pfm').replace('.npy','.pfm'), disp)
    elif ext == '.png':
        # png16 (×divisor)로 저장
        arr = np.clip(disp * png_divisor, 0, 65535).astype(np.uint16)
        cv2.imwrite(out_path.replace('.pfm','.png').replace('.npy','.png'), arr)
    elif ext == '.npy':
        np.save(out_path.replace('.pfm','.npy').replace('.png','.npy'), disp.astype(np.float32))
    else:
        # 기본은 npy
        np.save(out_path + '.npy', disp.astype(np.float32))

# -------------------------
# 유리 마스크/렌더 생성
# -------------------------
def random_glass_polygon(h, w, rng, min_frac=0.25, max_frac=0.55, windshield_bias=True):
    """
    화면 내 임의 사다리꼴/사각형 다각형 생성.
    - min/max_frac: 화면 대비 유리 면적 비율 범위
    - windshield_bias=True면 상단이 좁고 하단이 넓은 사다리꼴 성향
    """
    area_target = rng.uniform(min_frac, max_frac) * (h * w)

    # 기준 박스 가로세로 비율 추정
    aspect = rng.uniform(1.6, 2.4) if windshield_bias else rng.uniform(1.1, 2.0)
    box_h = math.sqrt(area_target / aspect)
    box_w = area_target / box_h

    # 화면에 맞춰 보정
    box_h = np.clip(box_h, h * 0.2, h * 0.8)
    box_w = np.clip(box_w, w * 0.2, w * 0.95)

    # 중심점
    cy = rng.uniform(h * 0.35, h * 0.65)
    cx = rng.uniform(w * 0.35, w * 0.65)

    # 기본 사각형 꼭짓점
    x0 = int(np.clip(cx - box_w / 2, 0, w-1))
    x1 = int(np.clip(cx + box_w / 2, 0, w-1))
    y0 = int(np.clip(cy - box_h / 2, 0, h-1))
    y1 = int(np.clip(cy + box_h / 2, 0, h-1))

    # 사다리꼴 왜곡 (상단 좁게/하단 넓게)
    top_shave = int((x1 - x0) * rng.uniform(0.05, 0.2)) if windshield_bias else 0
    bot_expand = int((x1 - x0) * rng.uniform(0.0, 0.15)) if windshield_bias else 0

    pts = np.array([
        [x0 + top_shave, y0],
        [x1 - top_shave, y0],
        [x1 + bot_expand, y1],
        [x0 - bot_expand, y1],
    ], dtype=np.int32)

    # 살짝 랜덤 노이즈
    jitter = rng.integers(low=-max(1,int(0.02*w)), high=max(2,int(0.02*w)), size=pts.shape)
    pts = pts + jitter
    pts[:,0] = np.clip(pts[:,0], 0, w-1)
    pts[:,1] = np.clip(pts[:,1], 0, h-1)
    return pts

def make_mask_from_polygon(h, w, poly):
    mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(mask, [poly.astype(np.int32)], 255)
    return mask

def apply_glass_render(img, mask, rng, tint=(190, 220, 220), alpha_range=(0.18, 0.35), blur_ks=(7,11), grad_strength=0.25):
    """
    유리 렌더링:
     - tint 컬러를 알파 블렌딩
     - 마스크 내부만 약한 블러
     - 세로 그라디언트 알파로 위쪽/아래쪽 진하기 변화
    """
    h, w = img.shape[:2]
    alpha = rng.uniform(*alpha_range)

    # 베이스 틴트
    tint_img = np.full_like(img, tint, dtype=np.uint8)

    # 내부 블러
    k = rng.choice(blur_ks)
    blurred = cv2.GaussianBlur(img, (k, k), 0)

    # 그라디언트 알파(세로)
    yy = (np.arange(h, dtype=np.float32) / max(1, h-1)).reshape(h,1)
    grad = (1.0 - grad_strength/2) + grad_strength * (0.5 - np.abs(yy - 0.5)) * 2.0  # 중앙 밝고 위아래 살짝 진함
    grad = np.clip(grad, 0.0, 1.0)

    # 합성
    out = img.copy()
    m = (mask > 0).astype(np.float32)
    m3 = np.dstack([m,m,m])

    # 블러 먼저 살짝
    out = (out * (1 - 0.25*m[...,None]) + blurred * (0.25*m[...,None])).astype(np.uint8)

    # 틴트+그라디언트 알파
    a = (alpha * grad).astype(np.float32)
    a3 = np.dstack([a,a,a]) * m3
    out = (out.astype(np.float32) * (1 - a3) + tint_img.astype(np.float32) * a3).astype(np.uint8)

    # 간단한 하이라이트 선(선택)
    if rng.random() < 0.6:
        hl = out.copy()
        num = rng.integers(1,3)
        for _ in range(num):
            x1 = rng.integers(0, w//2)
            y1 = rng.integers(0, h//2)
            x2 = x1 + rng.integers(w//3, w)
            y2 = y1 + rng.integers(h//6, h//2)
            cv2.line(hl, (x1,y1), (x2,y2), (255,255,255), rng.integers(1,3), lineType=cv2.LINE_AA)
        hl = cv2.GaussianBlur(hl, (0,0), sigmaX=3, sigmaY=3)
        # 마스크 내부에만 소량 가중
        out = np.where(m3>0, ((out.astype(np.float32)*0.95 + hl.astype(np.float32)*0.05)).astype(np.uint8), out)
    return out

# -------------------------
# 유리 평면 disparity 생성
# -------------------------
def sample_glass_plane_disp(disp_in, mask, rng, dmin=None, dmax=None, slope_px=3.0):
    """
    유리 영역의 plane disparity 생성.
    - d0: 유리를 배경 ‘중간대’ 깊이에 두기 위해, 유효 disp의 분위수 범위에서 샘플
    - 기울기: 유리 폴리곤 폭/높이에 대해 ±slope_px 픽셀 정도 변화하도록 설정
    """
    H, W = disp_in.shape[:2]
    valid = disp_in[disp_in > 0]
    if valid.size < 100:
        base_min = 1.0
        base_max = max(64.0, W*0.05)
    else:
        q20, q80 = np.percentile(valid, [20, 80])
        base_min = max(1.0, q20)
        base_max = max(base_min+1.0, q80)

    if dmin is None: dmin = base_min
    if dmax is None: dmax = base_max
    d0 = rng.uniform(dmin, dmax)

    # 마스크의 bbox
    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        return np.zeros_like(disp_in, dtype=np.float32)
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    w = max(1, x1 - x0 + 1)
    h = max(1, y1 - y0 + 1)

    # x,y 방향 기울기 (폴리곤 폭/높이에서 ±slope_px 변동)
    gx = rng.uniform(-slope_px/w, slope_px/w)
    gy = rng.uniform(-slope_px/h, slope_px/h)

    # 중심 기준 평면
    cx = 0.5*(x0+x1)
    cy = 0.5*(y0+y1)
    xx, yy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    plane = d0 + gx*(xx - cx) + gy*(yy - cy)

    # 범위 클램프
    plane = np.clip(plane, max(0.01, dmin*0.5), dmax*1.5).astype(np.float32)

    disp_out = disp_in.copy()
    disp_out[mask>0] = plane[mask>0]
    return disp_out

# -------------------------
# 메인 파이프라인
# -------------------------
def process_pair(left_path, right_path, disp_path, out_dir, args, rng):
    name = os.path.splitext(os.path.basename(left_path))[0]

    # 이미지/disp 읽기
    left  = cv2.imread(left_path, cv2.IMREAD_COLOR)
    right = cv2.imread(right_path, cv2.IMREAD_COLOR)
    if left is None or right is None:
        print(f"[WARN] skip (imread fail): {left_path}")
        return
    disp = read_disp_auto(disp_path, png_divisor=args.png_divisor)

    H, W = left.shape[:2]
    # 유리 폴리곤 & 마스크
    poly = random_glass_polygon(H, W, rng, min_frac=args.min_area_frac, max_frac=args.max_area_frac, windshield_bias=not args.side_window_mode)
    mask = make_mask_from_polygon(H, W, poly)

    # 유리 렌더 합성
    left_aug  = apply_glass_render(left,  mask, rng,
                                   tint=(args.tint_b, args.tint_g, args.tint_r),
                                   alpha_range=(args.alpha_min, args.alpha_max),
                                   blur_ks=(args.blur_kmin|1, args.blur_kmax|1),  # odd kernel
                                   grad_strength=args.grad_strength)
    right_aug = apply_glass_render(right, mask, rng,
                                   tint=(args.tint_b, args.tint_g, args.tint_r),
                                   alpha_range=(args.alpha_min, args.alpha_max),
                                   blur_ks=(args.blur_kmin|1, args.blur_kmax|1),
                                   grad_strength=args.grad_strength)

    # 유리 평면 disparity 생성 및 적용(좌 기준)
    disp_aug = sample_glass_plane_disp(disp, mask, rng, dmin=args.dmin, dmax=args.dmax, slope_px=args.slope_px)

    # 저장 경로
    out_left_dir  = os.path.join(out_dir, "left_real")
    out_right_dir = os.path.join(out_dir, "right_real")
    out_disp_dir  = os.path.join(out_dir, "disp_left")
    out_mask_dir  = os.path.join(out_dir, "mask")
    os.makedirs(out_left_dir, exist_ok=True)
    os.makedirs(out_right_dir, exist_ok=True)
    os.makedirs(out_disp_dir, exist_ok=True)
    os.makedirs(out_mask_dir, exist_ok=True)

    # 파일명 유지
    left_out_path  = os.path.join(out_left_dir,  os.path.basename(left_path))
    right_out_path = os.path.join(out_right_dir, os.path.basename(right_path))
    disp_out_path  = os.path.join(out_disp_dir,  os.path.basename(disp_path))
    mask_out_path  = os.path.join(out_mask_dir,  f"{name}_glass_mask.png")

    cv2.imwrite(left_out_path, left_aug)
    cv2.imwrite(right_out_path, right_aug)
    write_disp_like(disp_path, disp_aug, disp_out_path, png_divisor=args.png_divisor)
    cv2.imwrite(mask_out_path, mask)  # 0/255

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--left_dir',  default="/home/jaejun/dataset/flyingthing/FlyingThings3D_subset_image_clean/FlyingThings3D_subset/train/image_clean/left_real")
    ap.add_argument('--right_dir', default="/home/jaejun/dataset/flyingthing/FlyingThings3D_subset_image_clean/FlyingThings3D_subset/train/image_clean/right_real")
    ap.add_argument('--disp_dir',  default="/home/jaejun/dataset/flyingthing/FlyingThings3D_subset_disparity/FlyingThings3D_subset/train/disparity/left")
    ap.add_argument('--out_dir',   default="/home/jaejun/dataset/flyingthing/FlyingThings3D_subset_image_clean/FlyingThings3D_subset/train/image_clean/glass")

    # 렌더 파라미터
    ap.add_argument('--tint_r', type=int, default=70, help='유리 틴트 R')
    ap.add_argument('--tint_g', type=int, default=70, help='유리 틴트 G')
    ap.add_argument('--tint_b', type=int, default=70, help='유리 틴트 B')
    ap.add_argument('--alpha_min', type=float, default=0.55)
    ap.add_argument('--alpha_max', type=float, default=0.75)
    ap.add_argument('--blur_kmin', type=int, default=7)
    ap.add_argument('--blur_kmax', type=int, default=11)
    ap.add_argument('--grad_strength', type=float, default=0.25)

    # 유리 면적/형상
    ap.add_argument('--min_area_frac', type=float, default=0.02, help='화면 대비 유리 최소 면적비')
    ap.add_argument('--max_area_frac', type=float, default=0.05, help='화면 대비 유리 최대 면적비')
    ap.add_argument('--side_window_mode', action='store_true', help='사이드윈도 성향(사다리꼴 bias 약화)')

    # disparity 평면 파라미터
    ap.add_argument('--dmin', type=float, default=None, help='유리 평면 시차 최소 (기본: 유효 disp 분위수 기반)')
    ap.add_argument('--dmax', type=float, default=None, help='유리 평면 시차 최대 (기본: 유효 disp 분위수 기반)')
    ap.add_argument('--slope_px', type=float, default=3.0, help='유리 폴리곤 폭/높이 대비 시차 변화량(px)')

    # 포맷
    ap.add_argument('--png_divisor', type=float, default=256.0, help='png16 ↔ float 변환 배율(일반적으로 256)')

    # 실행
    ap.add_argument('--seed', type=int, default=2025)

    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    left_list  = sorted(glob.glob(os.path.join(args.left_dir,  '*')))
    right_list = sorted(glob.glob(os.path.join(args.right_dir, '*')))
    disp_list  = sorted(glob.glob(os.path.join(args.disp_dir,  '*')))

    # 파일명 매칭(기본: 동일 파일명 가정)
    # 필요 시 사용자 환경에 맞게 매칭 로직 수정
    base_to_right = {os.path.splitext(os.path.basename(p))[0]: p for p in right_list}
    base_to_disp  = {os.path.splitext(os.path.basename(p))[0]: p for p in disp_list}

    cnt = 0
    for lp in left_list:
        base = os.path.splitext(os.path.basename(lp))[0]
        rp = base_to_right.get(base, None)
        dp = base_to_disp.get(base, None)
        if rp is None or dp is None:
            print(f"[WARN] pair missing: {base}")
            continue
        process_pair(lp, rp, dp, args.out_dir, args, rng)
        cnt += 1
    print(f"[DONE] processed pairs: {cnt}")

if __name__ == '__main__':
    main()
