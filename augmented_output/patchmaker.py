import numpy as np, cv2, os, math, json

def read_pfm(path):
    with open(path, 'rb') as f:
        header = f.readline().decode('ascii').rstrip()
        if header not in ('PF','Pf'):
            raise ValueError('Not a PFM file.')
        dims = f.readline().decode('ascii')
        while dims.startswith('#'):
            dims = f.readline().decode('ascii')
        w, h = map(int, dims.strip().split())
        scale = float(f.readline().decode('ascii').strip())
        endian = '<' if scale < 0 else '>'
        data = np.fromfile(f, endian+'f')
        data = data.reshape((h, w, 3)) if header=='PF' else data.reshape((h, w))
        data = np.flipud(data)  # bottom-to-top
    return data.astype(np.float32)

def write_disparity_png(disp, path, scale=256.0):
    disp16 = np.clip(disp*scale, 0, 65535).astype(np.uint16)
    cv2.imwrite(path, disp16)

def plane_homography(K, n, d0, baseline):
    t = np.array([baseline,0,0], dtype=np.float64).reshape(3,1)
    H = K @ (np.eye(3) - (t @ n.reshape(1,3)) / d0) @ np.linalg.inv(K)
    return H

def rays_from_pixels(Kinv, us, vs):
    ones = np.ones_like(us, dtype=np.float64)
    pts = np.stack([us, vs, ones], axis=0)  # (3,N)
    r = (Kinv @ pts).T  # (N,3)
    return r

def disparity_depth_on_plane(Kinv, fx, baseline, n, d0, us, vs):
    r = rays_from_pixels(Kinv, us, vs)
    nr = (r @ n)
    Z = (-d0) * r[:,2] / (nr + 1e-12)
    d = fx * baseline / (Z + 1e-12)
    return d, Z

def project(K, P):
    x = (K @ (P / P[2])).reshape(3,)
    return np.array([x[0], x[1]], dtype=np.float64)

def intersect_ground(Kinv, cam_h, u, v):
    r = (Kinv @ np.array([u, v, 1.0], dtype=np.float64)).reshape(3,)
    lam = cam_h / (r[1] + 1e-12)  # y-down
    return lam * r  # (X,Y,Z)

def synthesize_pair(left_path, right_path, disp_pfm_path, out_dir,
                    intrinsics=None, baseline=0.54, cam_h=1.65,
                    ground_poly=None, side_cfg=None,
                    alpha_ground=0.35, alpha_side=0.85):
    os.makedirs(out_dir, exist_ok=True)
    imL = cv2.cvtColor(cv2.imread(left_path), cv2.COLOR_BGR2RGB)
    imR = cv2.cvtColor(cv2.imread(right_path), cv2.COLOR_BGR2RGB)
    H_img, W_img = imL.shape[:2]
    disp0 = read_pfm(disp_pfm_path)
    if disp0.shape[:2] != (H_img, W_img):
        disp0 = cv2.resize(disp0, (W_img, H_img), interpolation=cv2.INTER_LINEAR)

    # Default KITTI 2015 intrinsics scaled to input size
    if intrinsics is None:
        W0, H0 = 1242.0, 375.0
        fx0 = fy0 = 721.5377; cx0 = 609.5593; cy0 = 172.854
        sx, sy = W_img / W0, H_img / H0
        intrinsics = dict(fx=fx0*sx, fy=fy0*sy, cx=cx0*sx, cy=cy0*sy)
    fx, fy, cx, cy = intrinsics['fx'], intrinsics['fy'], intrinsics['cx'], intrinsics['cy']
    K = np.array([[fx,0,cx],[0,fy,cy],[0,0,1.0]], dtype=np.float64)
    Kinv = np.linalg.inv(K)

    # Ground ROI (trapezoid near bottom) if not given
    if ground_poly is None:
        ground_poly = np.array([[40, int(0.70*H_img)],
                                [W_img-40, int(0.70*H_img)],
                                [W_img-10, H_img-10],
                                [10, H_img-10]], dtype=np.int32)
    else:
        ground_poly = np.array(ground_poly, dtype=np.int32)

    # Side plane base segment and height if not given
    if side_cfg is None:
        p1 = (int(0.58*W_img), int(0.68*H_img))
        p2 = (int(0.80*W_img), int(0.66*H_img))
        height_m = 1.4
    else:
        p1 = tuple(side_cfg['p1']); p2 = tuple(side_cfg['p2']); height_m = float(side_cfg.get('height_m', 1.4))

    # Masks
    mask_ground = np.zeros((H_img, W_img), dtype=np.uint8); cv2.fillPoly(mask_ground, [ground_poly], 255)
    P1 = intersect_ground(Kinv, cam_h, p1[0], p1[1])
    P2 = intersect_ground(Kinv, cam_h, p2[0], p2[1])
    P1_top = P1 + np.array([0.0, -height_m, 0.0])
    P2_top = P2 + np.array([0.0, -height_m, 0.0])
    p1_top = np.round(project(K, P1_top)).astype(np.int32)
    p2_top = np.round(project(K, P2_top)).astype(np.int32)
    side_poly = np.array([p1, p2, tuple(p2_top), tuple(p1_top)], dtype=np.int32)
    mask_side = np.zeros((H_img, W_img), dtype=np.uint8); cv2.fillPoly(mask_side, [side_poly], 255)

    # Plane parameters
    n_g = np.array([0.0, 1.0, 0.0], dtype=np.float64); d0_g = -cam_h
    s_dir = P2 - P1; s_dir = s_dir / (np.linalg.norm(s_dir) + 1e-12)
    n_side = np.cross(n_g, s_dir); n_side = n_side / (np.linalg.norm(n_side) + 1e-12)
    d0_side = -float(n_side @ P1)

    # Textures (simple)
    ground_tex = (cv2.cvtColor(cv2.imread(left_path), cv2.COLOR_BGR2RGB) * 0.95).astype(np.uint8)
    side_tex = cv2.flip(ground_tex, 1); side_tex = cv2.GaussianBlur(side_tex, (11,11), 0)
    side_tex = (side_tex * np.array([0.25,0.25,0.25])[None,None,:] + 30).astype(np.uint8)

    # Buffers
    L_syn = ground_tex.copy(); L_syn[:] = imL
    R_syn = imR.copy()
    ZL = np.full((H_img, W_img), np.inf, dtype=np.float64)
    ZR = np.full((H_img, W_img), np.inf, dtype=np.float64)
    disp = disp0.copy()
    validL = np.zeros((H_img, W_img), dtype=np.uint8)
    validR = np.zeros((H_img, W_img), dtype=np.uint8)

    def render_plane(mask, tex, n, d0, alpha):
        nonlocal L_syn, R_syn, ZL, ZR, disp, validL, validR
        ys, xs = np.where(mask>0)
        if len(xs)==0: return
        x0,x1 = xs.min(), xs.max(); y0,y1 = ys.min(), ys.max()
        roi_mask = mask[y0:y1+1, x0:x1+1] > 0
        u,v = np.meshgrid(np.arange(x0, x1+1), np.arange(y0, y1+1))
        us = u[roi_mask].astype(np.float64); vs = v[roi_mask].astype(np.float64)
        d_roi, Z_roi = disparity_depth_on_plane(Kinv, fx, baseline, n, d0, us, vs)
        # Left compose with Z-buffer
        tex_roi = tex[y0:y1+1, x0:x1+1]
        base = L_syn[y0:y1+1, x0:x1+1].astype(np.float32)
        over = tex_roi.astype(np.float32)
        blended = (1-alpha)*base + alpha*over
        zl_old = ZL[y0:y1+1, x0:x1+1][roi_mask]
        upd = (Z_roi < zl_old)
        idx = np.where(roi_mask)
        ys_upd = idx[0][upd]; xs_upd = idx[1][upd]
        L_syn[y0:y1+1, x0:x1+1][ys_upd, xs_upd] = blended[ys_upd, xs_upd].astype(np.uint8)
        ZL[y0:y1+1, x0:x1+1][ys_upd, xs_upd] = Z_roi[upd]
        disp[y0:y1+1, x0:x1+1][ys_upd, xs_upd] = d_roi[upd]
        validL[y0:y1+1, x0:x1+1][ys_upd, xs_upd] = 255

        # Right via plane homography
        Hpl = plane_homography(K, n, d0, baseline)
        mask_roi = np.zeros_like(mask); mask_roi[y0:y1+1, x0:x1+1][roi_mask] = 255
        tex_masked = np.zeros_like(L_syn); tex_masked[y0:y1+1, x0:x1+1][roi_mask] = blended[roi_mask].astype(np.uint8)
        warped = cv2.warpPerspective(tex_masked, Hpl, (L_syn.shape[1], L_syn.shape[0]), flags=cv2.INTER_LINEAR)
        warped_mask = cv2.warpPerspective(mask_roi, Hpl, (L_syn.shape[1], L_syn.shape[0]), flags=cv2.INTER_NEAREST)
        # Depth for right (approx by warping Z)
        ztmp = np.full_like(ZL, np.inf); ztmp[y0:y1+1, x0:x1+1][roi_mask] = Z_roi
        warped_Z = cv2.warpPerspective(ztmp, Hpl, (L_syn.shape[1], L_syn.shape[0]), flags=cv2.INTER_LINEAR)
        updR = (warped_mask>0) & (warped_Z < ZR)
        R_syn[updR] = warped[updR]
        ZR[updR] = warped_Z[updR]
        validR[updR] = 255

    # Render planes
    render_plane(mask_ground, ground_tex, n_g, d0_g, alpha_ground)
    render_plane(mask_side,   side_tex,   n_side, d0_side, alpha_side)

    # Save
    outL = os.path.join(out_dir, "left_synth.png")
    outR = os.path.join(out_dir, "right_synth.png")
    outD = os.path.join(out_dir, "disp_synth_16bit.png")
    outML = os.path.join(out_dir, "mask_left.png")
    outMR = os.path.join(out_dir, "mask_right.png")
    cv2.imwrite(outL, cv2.cvtColor(L_syn, cv2.COLOR_RGB2BGR))
    cv2.imwrite(outR, cv2.cvtColor(R_syn, cv2.COLOR_RGB2BGR))
    write_disparity_png(disp, outD)
    cv2.imwrite(outML, validL)
    cv2.imwrite(outMR, validR)

    # 기록용 설정 저장
    cfg = {
        "intrinsics": intrinsics,
        "baseline": baseline,
        "cam_h": cam_h,
        "ground_poly": ground_poly.tolist(),
        "side_poly": side_poly.tolist(),
        "side_base": {"p1": p1, "p2": p2, "height_m": height_m}
    }
    with open(os.path.join(out_dir, "synth_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    return outL, outR, outD, outML, outMR, os.path.join(out_dir, "synth_config.json")


# ------------------ 사용 예시 ------------------
# (경로를 교체해서 사용하세요)
left_path  = "/home/jaejun/dataset/flyingthing/FlyingThings3D_subset_image_clean/FlyingThings3D_subset/train/image_clean/left/0000165.png"
right_path = "/home/jaejun/dataset/flyingthing/FlyingThings3D_subset_image_clean/FlyingThings3D_subset/train/image_clean/right/0000165.png"
disp_pfm   = "/home/jaejun/dataset/flyingthing/FlyingThings3D_subset_disparity/FlyingThings3D_subset/train/disparity/left/0000165.pfm"
out_dir = "./synth_out"

# (옵션) ROI/사다리꼴을 지정하고 싶으면 다음 형태로 전달
# ground_poly = [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
# side_cfg = {"p1": (xA,yA), "p2": (xB,yB), "height_m": 1.4}

paths = synthesize_pair(left_path, right_path, disp_pfm, out_dir,
                        ground_poly=None, side_cfg=None)
print(paths)
