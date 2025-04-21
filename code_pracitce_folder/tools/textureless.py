import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

def overlay_textureless_map(image_path: str,
                            grad_thr: float = None,
                            blur_sigma: float = 5,
                            alpha: float = 0.6):
    """
    Load an image, compute a texturelessness map via Sobel gradients,
    smooth it, and overlay it with a jet colormap (red=textureless, blue=textured).
    """
    # 1. Read image in BGR, convert to RGB
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # 2. Grayscale [0,1]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    # 3. Sobel gradients
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx*gx + gy*gy)

    # 4. Normalize magnitude to [0,1]
    mag_norm = (mag - mag.min()) / (mag.max() - mag.min() + 1e-6)
    textureless = mag_norm  # 1 = more textureless
    # textureless = 1.0 - mag
    # 5. Optional thresholding
    # if grad_thr is not None:
    #     textureless = np.clip((textureless - grad_thr) / (1.0 - grad_thr), 0, 1)

    # 6. Gaussian smoothing for better visualization
    textureless_sm = gaussian_filter(textureless, sigma=blur_sigma)

    # 7. Plot with overlay and colorbar
    plt.figure(figsize=(12, 8))
    plt.imshow(img_rgb)
    overlay = plt.imshow(textureless_sm, cmap='jet', alpha=alpha)
    plt.axis('off')
    plt.title("Textureless Overlay (red=high, blue=low)")
    # Add colorbar
    cbar = plt.colorbar(overlay, fraction=0.046, pad=0.04)
    cbar.set_label("Textureless Score", rotation=270, labelpad=15)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py path/to/image.jpg [blur_sigma] [alpha]")
        sys.exit(1)
    img_path = sys.argv[1]
    blur = float(sys.argv[2]) if len(sys.argv) > 2 else 5
    alpha = float(sys.argv[3]) if len(sys.argv) > 3 else 0.6
    grad_thr = float(sys.argv[4]) 
    overlay_textureless_map(img_path, blur_sigma=8, alpha=0.5, grad_thr=0.5)
