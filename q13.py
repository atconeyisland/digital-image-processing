import numpy as np
from helper import load_gray, show, to_u8, gaussian_lp, butterworth_lp, apply_filter

img = load_gray().astype(np.float64)
D0, k = 30, 1.5

Hg = 1 - gaussian_lp(img.shape, D0)
Hb = 1 - butterworth_lp(img.shape, D0, n=2)

hp_g, hp_b = apply_filter(img, Hg), apply_filter(img, Hb)
sharp_g = np.clip(img + k * hp_g, 0, 255)  # high-frequency emphasis
sharp_b = np.clip(img + k * hp_b, 0, 255)

show(
    [img, to_u8(np.abs(hp_g)), sharp_g, to_u8(Hg),
     img, to_u8(np.abs(hp_b)), sharp_b, to_u8(Hb)],
    ["Original", "Gaussian HPF output", "Gaussian sharpened", "Gaussian HPF response",
     "Original", "Butterworth HPF output", "Butterworth sharpened", "Butterworth HPF response"],
    cols=4,
)