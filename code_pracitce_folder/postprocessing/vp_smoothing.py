# -*- coding: utf-8 -*-
"""
VP-방향 스무딩/에지 게이팅 진단 스크립트
- 입력:
    --disp : disparity PNG 경로 (H,W), 실수/8bit 모두 허용
    --vp   : "x,y" 형태(예: 320,-96) 또는 --vp-x --vp-y 로 좌표 개별 입력
    --edge : (선택) 에지 확률 PNG 경로(0~1 또는 0~255). 없으면 disparity 기울기로 근사
- 출력:
    *_01_disparity.png
    *_02_edge_prob.png
    *_03_edge_dilated.png
    *_04_mask_M.png
    *_05_mask_M_tilde.png
    *_06_g1_abs.png
    *_07_g2_abs.png
    *_08_overlay_smoothing.png
    *_09_overlay_edges.png
"""

from typing import Optional, Tuple, Dict
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import argparse
import os


# -------------------------------
# 로딩/정규화
# -------------------------------
def load_disparity_png(path: str) -> np.ndarray:
    img = Image.open(path)
    # float/16bit/L 등 다양한 모드 대응
    if img.mode not in ("F", "I;16", "I", "L"):
        img = img.convert("L")
    else:
        img = img.convert("F")
    arr = np.array(img, dtype=np.float32)
    return arr

def load_gray_prob_png(path: str) -> np.ndarray:
    img = Image.open(path).convert("L")
    arr = np.array(img, dtype=np.float32)
    if arr.max() > 1.5:
        arr = arr / 255.0
    return np.clip(arr, 0.0, 1.0).astype(np.float32)

def minmax_normalize(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    xmin, xmax = float(x.min()), float(x.max())
    return (x - xmin) / (xmax - xmin + eps)


# -------------------------------
# 기본 연산(그라디언트, 팽창, 보간)
# -------------------------------
def central_gradients(d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    H, W = d.shape
    dp = np.pad(d, ((1,1),(1,1)), mode="edge")
    gx = 0.5 * (dp[1:-1, 2:] - dp[1:-1, :-2])
    gy = 0.5 * (dp[2:, 1:-1] - dp[:-2, 1:-1])
    return gx.astype(np.float32), gy.astype(np.float32)

def dilate_max(img: np.ndarray, k: int) -> np.ndarray:
    assert k % 2 == 1 and k >= 1
    H, W = img.shape
    pad = k // 2
    p = np.pad(img, ((pad, pad), (pad, pad)), mode="edge")
    patches = [p[dy:dy+H, dx:dx+W] for dy in range(k) for dx in range(k)]
    return np.max(np.stack(patches, axis=0), axis=0)

def bilinear_sample(img: np.ndarray, y: np.ndarray, x: np.ndarray) -> np.ndarray:
    H, W = img.shape
    x = np.clip(x, 0.0, W - 1.0)
    y = np.clip(y, 0.0, H - 1.0)
    x0 = np.floor(x).astype(np.int32); y0 = np.floor(y).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, W - 1);   y1 = np.clip(y0 + 1, 0, H - 1)
    Ia = img[y0, x0]; Ib = img[y0, x1]; Ic = img[y1, x0]; Id = img[y1, x1]
    wa = (x1 - x) * (y1 - y)
    wb = (x - x0) * (y1 - y)
    wc = (x1 - x) * (y - y0)
    wd = (x - x0) * (y - y0)
    out = Ia*wa + Ib*wb + Ic*wc + Id*wd
    return out.astype(np.float32)


# -------------------------------
# VP 방향장/마스크/도함수
# -------------------------------
def vp_unit_field(vp_xy: Tuple[float, float], H: int, W: int, rmin_ratio: float = 1/40) -> Dict[str, np.ndarray]:
    vx, vy = float(vp_xy[0]), float(vp_xy[1])
    yy, xx = np.meshgrid(np.arange(H, dtype=np.float32), np.arange(W, dtype=np.float32), indexing="ij")
    ex = (vx - xx); ey = (vy - yy)
    r = np.sqrt(ex*ex + ey*ey) + 1e-6
    ex = ex / r; ey = ey / r
    rmin = min(H, W) * rmin_ratio
    mask_r = (r > rmin).astype(np.float32)
    return {"ex": ex, "ey": ey, "r": r, "mask_r": mask_r, "rmin": rmin}

def robust_edge_from_disp(d: np.ndarray, q: float = 0.90, eps: float = 1e-6) -> np.ndarray:
    gx, gy = central_gradients(d)
    mag = np.sqrt(gx*gx + gy*gy)
    thr = np.quantile(mag, q)
    return np.clip(mag / (thr + eps), 0.0, 1.0).astype(np.float32)

def compute_masks_and_derivs(
    d: np.ndarray,
    vp_xy: Tuple[float, float],
    pedge: Optional[np.ndarray] = None,
    eta: float = 2.0,
    dilate_ks: int = 5,
    rmin_ratio: float = 1/40,
    beta: float = 0.0,
    delta: float = 1.0
) -> Dict[str, np.ndarray]:
    H, W = d.shape
    vf = vp_unit_field(vp_xy, H, W, rmin_ratio=rmin_ratio)
    ex, ey, mask_r, r = vf["ex"], vf["ey"], vf["mask_r"], vf["r"]

    if pedge is None:
        pedge = robust_edge_from_disp(d, q=0.90)
    pedge = np.clip(pedge.astype(np.float32), 0.0, 1.0)
    pedge_d = dilate_max(pedge, k=dilate_ks)

    # 소프트 게이트
    M = np.power(np.clip(1.0 - pedge_d, 0.0, 1.0), eta) * mask_r

    # 거리 가중(선택)
    if beta > 0.0:
        rmin = vf["rmin"]
        wr = np.power(np.clip(r / max(1.0, rmin), 0.0, 1.0), beta)
    else:
        wr = np.ones_like(M, dtype=np.float32)
    Mtilde = np.clip(M * wr, 0.0, 1.0)

    # 방향 도함수
    gx, gy = central_gradients(d)
    g1 = gx * ex + gy * ey  # 1차: ∇D · e

    yy, xx = np.meshgrid(np.arange(H, dtype=np.float32), np.arange(W, dtype=np.float32), indexing="ij")
    xp = xx + delta * ex; yp = yy + delta * ey
    xm = xx - delta * ex; ym = yy - delta * ey
    d_p = bilinear_sample(d, yp, xp)
    d_m = bilinear_sample(d, ym, xm)
    g2 = d_p - 2.0 * d + d_m      # 2차

    return {
        "pedge": pedge,
        "pedge_dilated": pedge_d,
        "M": M,
        "M_tilde": Mtilde,
        "g1_abs": np.abs(g1),
        "g2_abs": np.abs(g2),
        "mask_r": mask_r
    }


# -------------------------------
# 시각화/저장: 한 그림당 한 플롯
# -------------------------------
def save_fig(img: np.ndarray, title: str, out_path: str, vp_xy: Optional[Tuple[float, float]] = None, draw_rays: bool = False):
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    if vp_xy is not None:
        plt.scatter([vp_xy[0]], [vp_xy[1]], s=20, marker="x")
        if draw_rays:
            H, W = img.shape[:2]
            num_rays = 12
            xs = np.linspace(0, W - 1, num_rays // 4 + 1, dtype=np.float32)
            ys = np.linspace(0, H - 1, num_rays // 4 + 1, dtype=np.float32)
            pts = []
            for x in xs:
                pts.append((x, 0)); pts.append((x, H - 1))
            for y in ys:
                pts.append((0, y)); pts.append((W - 1, y))
            for (x0, y0) in pts[:num_rays]:
                plt.plot([x0, vp_xy[0]], [y0, vp_xy[1]], linewidth=0.8)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def overlay_gray(base: np.ndarray, mask: np.ndarray, alpha: float = 0.35, brighten: bool = True) -> np.ndarray:
    b = base.copy()
    if b.ndim == 2:
        b = np.stack([b, b, b], axis=-1)
    m = np.clip(mask[..., None], 0.0, 1.0)
    overlay = np.ones_like(b) if brighten else np.zeros_like(b)
    out = b * (1.0 - alpha * m) + overlay * (alpha * m)
    return np.clip(out, 0.0, 1.0)


# -------------------------------
# 실행 파이프라인 (경로 기반)
# -------------------------------
def run_vp_smooth_diag_from_path(
    disp_path: str,
    vp_xy: Tuple[float, float],
    pedge_path: Optional[str] = None,
    eta: float = 2.0,
    dilate_ks: int = 5,
    rmin_ratio: float = 1/40,
    beta: float = 0.0,
    delta: float = 1.0,
    thr_smooth: float = 0.5,
    save_prefix: Optional[str] = None
) -> Dict[str, str]:
    d = load_disparity_png(disp_path).astype(np.float32)
    d_norm = minmax_normalize(d)

    pedge = load_gray_prob_png(pedge_path) if pedge_path and os.path.exists(pedge_path) else None

    results = compute_masks_and_derivs(
        d, vp_xy, pedge=pedge, eta=eta, dilate_ks=dilate_ks,
        rmin_ratio=rmin_ratio, beta=beta, delta=delta
    )

    if save_prefix is None:
        base = os.path.splitext(os.path.basename(disp_path))[0]
        save_prefix = os.path.join(os.path.dirname(disp_path) or ".", base)
    paths = {}

    # 1) disparity
    fp = f"{save_prefix}_01_disparity.png"
    save_fig(d_norm, "Disparity (normalized)", fp, vp_xy=vp_xy, draw_rays=True); paths["disparity"] = fp

    # 2) edge prob (input/approx)
    fp = f"{save_prefix}_02_edge_prob.png"
    save_fig(results["pedge"], "Edge prob (input or approx from ∥∇D∥)", fp); paths["edge_prob"] = fp

    # 3) edge prob (dilated)
    fp = f"{save_prefix}_03_edge_dilated.png"
    save_fig(results["pedge_dilated"], "Edge prob (dilated)", fp); paths["edge_dilated"] = fp

    # 4) mask M
    fp = f"{save_prefix}_04_mask_M.png"
    save_fig(results["M"], "Mask M = (1 - edge_dilated)^eta · mask_r", fp); paths["mask_M"] = fp

    # 5) mask M_tilde
    fp = f"{save_prefix}_05_mask_M_tilde.png"
    save_fig(results["M_tilde"], "Mask M_tilde (distance weight included)", fp); paths["mask_M_tilde"] = fp

    # 6) |∇D · e_vp|
    g1n = minmax_normalize(results["g1_abs"])
    fp = f"{save_prefix}_06_g1_abs.png"
    save_fig(g1n, "Directional 1st derivative |∇D · e_vp| (normalized)", fp); paths["g1_abs"] = fp

    # 7) |d²D/de²|
    g2n = minmax_normalize(results["g2_abs"])
    fp = f"{save_prefix}_07_g2_abs.png"
    save_fig(g2n, "Directional 2nd derivative |d²D/de²| (normalized)", fp); paths["g2_abs"] = fp

    # 8) 스무딩 대상 오버레이(밝게)
    smooth_region = (results["M_tilde"] >= thr_smooth).astype(np.float32)
    overlay_bright = overlay_gray(d_norm, smooth_region, alpha=0.35, brighten=True)
    fp = f"{save_prefix}_08_overlay_smoothing.png"
    save_fig(overlay_bright, "Smoothing target overlay (bright regions)", fp, vp_xy=vp_xy, draw_rays=True)
    paths["overlay_smoothing"] = fp

    # 9) 에지 영역 오버레이(어둡게)
    edge_region = (results["pedge_dilated"] >= 0.5).astype(np.float32)
    overlay_dark = overlay_gray(d_norm, edge_region, alpha=0.35, brighten=False)
    fp = f"{save_prefix}_09_overlay_edges.png"
    save_fig(overlay_dark, "Edge region overlay (dark regions, after dilation)", fp, vp_xy=vp_xy, draw_rays=True)
    paths["overlay_edges"] = fp

    return paths


# -------------------------------
# CLI
# -------------------------------
def parse_vp(vp_str: Optional[str], vp_x: Optional[float], vp_y: Optional[float]) -> Tuple[float, float]:
    if vp_str is not None:
        parts = vp_str.split(",")
        if len(parts) != 2:
            raise ValueError("--vp 는 'x,y' 형식이어야 합니다. 예: --vp 320,-96")
        return float(parts[0]), float(parts[1])
    if vp_x is None or vp_y is None:
        raise ValueError("--vp 또는 (--vp-x, --vp-y) 를 지정하십시오.")
    return float(vp_x), float(vp_y)

def main():
    ap = argparse.ArgumentParser(description="VP-방향 스무딩/에지 게이팅 진단")
    ap.add_argument("--disp", default="/home/jaejun/DomainGeneralization/log/2025-08-19_21_56_st/pseudo_disp_smooth/000020_10.png")
    ap.add_argument("--vp", type=str, default=None, help="소실점 'x,y' (예: 320,-96)")
    ap.add_argument("--vp-x", type=float, default=620.0, help="소실점 x")
    ap.add_argument("--vp-y", type=float, default=190.0, help="소실점 y")
    ap.add_argument("--edge", type=str, default=None, help="(선택) 에지 확률 PNG 경로 (0~1 또는 0~255)")
    ap.add_argument("--prefix", type=str, default=None, help="(선택) 저장 파일 접두사(절대/상대 경로)")
    ap.add_argument("--eta", type=float, default=2.0, help="(1 - edge)^eta 게이팅 강도")
    ap.add_argument("--dilate-ks", type=int, default=5, help="에지 팽창 커널 크기(홀 누수 방지)")
    ap.add_argument("--rmin-ratio", type=float, default=1/40, help="VP 특이점 회피 반경 비율")
    ap.add_argument("--beta", type=float, default=0.0, help="거리 가중 지수(>0이면 사용)")
    ap.add_argument("--delta", type=float, default=1.0, help="2차 차분 샘플링 간격(픽셀)")
    ap.add_argument("--thr", type=float, default=0.5, help="스무딩 대상 이진화 임계값")
    args = ap.parse_args()

    vp_xy = parse_vp(args.vp, args.vp_x, args.vp_y)

    paths = run_vp_smooth_diag_from_path(
        disp_path=args.disp,
        vp_xy=vp_xy,
        pedge_path=args.edge,
        eta=args.eta,
        dilate_ks=args.dilate_ks,
        rmin_ratio=args.rmin_ratio,
        beta=args.beta,
        delta=args.delta,
        thr_smooth=args.thr,
        save_prefix=args.prefix
    )

    print("# Saved files")
    for k, v in paths.items():
        print(f"{k:>18s} : {v}")

if __name__ == "__main__":
    main()
