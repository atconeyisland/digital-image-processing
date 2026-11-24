import numpy as np
from helper import load_gray, show

img = load_gray().astype(np.float64)


def psf2otf(psf, shape):
    pad = np.zeros(shape)
    pad[: psf.shape[0], : psf.shape[1]] = psf
    pad = np.roll(pad, (-(psf.shape[0] // 2), -(psf.shape[1] // 2)), axis=(0, 1))
    return np.fft.fft2(pad)


def psnr(ref, test):
    mse = np.mean((ref - np.clip(test, 0, 255)) ** 2)
    return 10 * np.log10(255 ** 2 / mse)


# Degradation: horizontal motion blur (length 21) + Gaussian noise
psf = np.ones((1, 21)) / 21
H = psf2otf(psf, img.shape)
rng = np.random.default_rng(0)
blurred = np.real(np.fft.ifft2(np.fft.fft2(img) * H)) + rng.normal(0, 3, img.shape)
G = np.fft.fft2(blurred)

# Inverse filter (H clamped so division by ~0 doesn't explode completely)
eps = 1e-2
H_safe = np.where(np.abs(H) < eps, eps, H)
inverse = np.real(np.fft.ifft2(G / H_safe))


# Wiener filter: scan K and keep the best PSNR
def wiener(K):
    return np.real(np.fft.ifft2(np.conj(H) / (np.abs(H) ** 2 + K) * G))


ks = np.logspace(-5, 0, 26)
best_k = max(ks, key=lambda k: psnr(img, wiener(k)))
restored = wiener(best_k)

show(
    [img, blurred, inverse, restored],
    [
        "Original",
        f"Degraded (PSNR {psnr(img, blurred):.1f} dB)",
        f"Inverse filter (PSNR {psnr(img, inverse):.1f} dB)",
        f"Wiener K={best_k:.4f} (PSNR {psnr(img, restored):.1f} dB)",
    ],
    cols=4,
)