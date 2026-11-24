import cv2
import numpy as np
from helper import load_gray, show, to_u8, dist_grid, apply_filter

img = load_gray("uneven.png").astype(np.float64)
gamma_l, gamma_h, c, d0 = 0.5, 2.0, 1.0, 30

D = dist_grid(img.shape)
H = (gamma_h - gamma_l) * (1 - np.exp(-c * D ** 2 / d0 ** 2)) + gamma_l

# log -> filter -> exp separates illumination (low freq) from reflectance (high freq)
out = np.expm1(apply_filter(np.log1p(img), H))
lo, hi = np.percentile(out, (1, 99))
out = np.clip((out - lo) / (hi - lo), 0, 1) * 255

show(
    [img, out, cv2.equalizeHist(img.astype(np.uint8)), to_u8(H)],
    ["Original", "Homomorphic filtered", "Plain histogram equalization (comparison)", "Filter response"],
    cols=4,
)