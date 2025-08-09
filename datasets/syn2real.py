import cv2, os, glob
import numpy as np
from tqdm import tqdm
from skimage import color, exposure

# ──────────────── 1) 통계 추출 (Real) ────────────────
def gather_real_stats(real_dir, max_imgs=400):
    print(real_dir)
    Ls, ab = [], []
    gammas = []
    files = sorted(glob.glob(os.path.join(real_dir, '*')))[:max_imgs]
    print(files)
    for fp in tqdm(files, desc='Real stats'):
        bgr = cv2.imread(fp)                 # BGR [0,255]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) / 255.0
        lab = color.rgb2lab(rgb)

        # LAB 통계
        Ls.append(lab[...,0].ravel())
        ab.append(lab[...,1:].reshape(-1,2))

        # luminance 기반 γ 추정  (mean(L_syn^γ)=mean(L_real) → γ = log...)
        mean_lin = rgb.mean()                # RGB 평균 = 밝기 proxy
        mean_labL = lab[...,0].mean() / 100  # [0,1]
        gammas.append(np.log(mean_labL+1e-5)/np.log(mean_lin+1e-5))
    Ls = np.concatenate(Ls);  ab = np.concatenate(ab,0)
    stats = {
        'L_mean': Ls.mean(), 'L_std': Ls.std(),
        'a_mean': ab[:,0].mean(), 'a_std': ab[:,0].std(),
        'b_mean': ab[:,1].mean(), 'b_std': ab[:,1].std(),
        'gamma_mu': np.mean(gammas), 'gamma_std': np.std(gammas)
    }
    return stats

# ──────────────── 2) 변환 함수 (Syn → Real) ────────────────
def adapt_synthetic(img_bgr, stats, clahe=False):
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)/255.0

    # ① 감마 맞추기  (랜덤한 γ 샘플로 augmentation 효과도)
    rng_gamma = np.random.normal(stats['gamma_mu'], stats['gamma_std'])
    rng_gamma = np.clip(rng_gamma, 0.5, 2.0)
    rgb = np.power(rgb, rng_gamma)

    # ② LAB 컬러 트랜스퍼  (Reinhard)
    lab = color.rgb2lab(rgb)
    L,a,b = lab[...,0], lab[...,1], lab[...,2]

    def match(src, m_tgt, s_tgt):
        s_src, std_src = src.mean(), src.std()
        return (src - s_src)/ (std_src + 1e-5) * s_tgt + m_tgt

    L = match(L, stats['L_mean'],  stats['L_std'])
    a = match(a, stats['a_mean'],  stats['a_std'])
    b = match(b, stats['b_mean'],  stats['b_std'])
    lab_t = np.stack([L,a,b], axis=-1)
    rgb_t = color.lab2rgb(lab_t).clip(0,1)

    # ③ CLAHE (선택): 잔여 대비·채도 조정
    if clahe:
        lab8 = cv2.cvtColor((rgb_t*255).astype(np.uint8), cv2.COLOR_RGB2LAB)
        cla = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        lab8[...,0] = cla.apply(lab8[...,0])
        rgb_t = cv2.cvtColor(lab8, cv2.COLOR_LAB2RGB)/255.0

    out = (rgb_t*255).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

# ──────────────── 3) 실행 스크립트 ────────────────
def transform_dataset(real_dir, syn_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    stats = gather_real_stats(real_dir)
    syn_files = glob.glob(os.path.join(syn_dir, '*'))
    for fp in tqdm(syn_files, desc='Transform'):
        num = int(fp.split('/')[-1].split('.')[0])
        # if num < 500:
        img = cv2.imread(fp)
        img_t = adapt_synthetic(img, stats, clahe=True)
        cv2.imwrite(os.path.join(out_dir, os.path.basename(fp)), img_t)

if __name__ == "__main__":
    # REAL_DIR = "/home/jaejun/dataset/kitti_2015/training/image_2"
    # SYN_DIR  = "/home/jaejun/dataset/FlyingThing/FlyingThings3D_subset_image_clean/FlyingThings3D_subset/train/image_clean/right"
    # OUT_DIR  = "/home/jaejun/dataset/FlyingThing/FlyingThings3D_subset_image_clean/FlyingThings3D_subset/train/image_clean/right_real"
    
    REAL_DIR  = "/home/jaejun/dataset/flyingthing/FlyingThings3D_subset_image_clean/FlyingThings3D_subset/train/image_clean/left"
    SYN_DIR = "/home/jaejun/dataset/kitti_2015/training/image_2"
    OUT_DIR = "/home/jaejun/dataset/kitti_2015/training/image_2_syn"
    
    
    transform_dataset(REAL_DIR, SYN_DIR, OUT_DIR)
