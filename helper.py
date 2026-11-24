import cv2
import numpy as np
import matplotlib.pyplot as plt

IMG_PATH = "general.png"

def load_gray(path=IMG_PATH):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read '{path}'")
    return img


def load_color(path=IMG_PATH):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read '{path}'")
    return img


def to_u8(a):
    a = np.asarray(a, dtype=np.float64)
    return np.uint8(255 * (a - a.min()) / (a.max() - a.min() + 1e-12))


def show(images, titles, cols=3, size=4):
    rows = -(-len(images) // cols)
    _, axes = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for ax, im, t in zip(axes, images, titles):
        ax.imshow(im, cmap="gray", vmin=0, vmax=255)
        ax.set_title(t)
    plt.tight_layout()
    plt.show()


#(for Q12, Q13, Q15)
def dist_grid(shape):
    """Distance of every frequency sample from the centre of the shifted spectrum."""
    M, N = shape
    u = np.arange(M) - M // 2
    v = np.arange(N) - N // 2
    return np.hypot(u[:, None], v[None, :])


def gaussian_lp(shape, d0):
    return np.exp(-dist_grid(shape) ** 2 / (2 * d0 ** 2))


def butterworth_lp(shape, d0, n=2):
    return 1.0 / (1.0 + (dist_grid(shape) / d0) ** (2 * n))


def apply_filter(img, H):
    F = np.fft.fftshift(np.fft.fft2(img.astype(np.float64)))
    return np.real(np.fft.ifft2(np.fft.ifftshift(F * H)))