import numpy as np
from helper import load_gray, show, to_u8, gaussian_lp, butterworth_lp, apply_filter

img = load_gray()
D0 = 30

Hg = gaussian_lp(img.shape, D0)
Hb = butterworth_lp(img.shape, D0, n=2)

spectrum = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(img))))

show(
    [img, np.clip(apply_filter(img, Hg), 0, 255), np.clip(apply_filter(img, Hb), 0, 255),
     to_u8(spectrum), to_u8(Hg), to_u8(Hb)],
    ["Original", f"Gaussian LPF (D0={D0})", f"Butterworth LPF (D0={D0}, n=2)",
     "Image spectrum (log)", "Gaussian response", "Butterworth response"],
)