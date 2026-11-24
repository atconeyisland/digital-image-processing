import cv2
import numpy as np
import matplotlib.pyplot as plt
from helper import load_gray

img = load_gray("coins.png")


def iterative_threshold(a, tol=0.5):
    t = a.mean()
    while True:
        fg, bg = a[a > t], a[a <= t]
        if fg.size == 0 or bg.size == 0:
            return t
        new_t = (fg.mean() + bg.mean()) / 2
        if abs(new_t - t) < tol:
            return new_t
        t = new_t


def otsu(a):
    p = np.bincount(a.ravel(), minlength=256) / a.size
    w0 = np.cumsum(p)
    mu = np.cumsum(p * np.arange(256))
    with np.errstate(divide="ignore", invalid="ignore"):
        between = (mu[-1] * w0 - mu) ** 2 / (w0 * (1 - w0))
    return int(np.argmax(np.nan_to_num(between, nan=0.0, posinf=0.0, neginf=0.0)))


t_global = 127
t_iter = iterative_threshold(img)
t_otsu = otsu(img)
t_cv, _ = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
print(f"Global={t_global}  Iterative={t_iter:.1f}  Otsu (mine)={t_otsu}  Otsu (OpenCV)={int(t_cv)}")

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
panels = [
    (img, "Original"),
    ((img > t_global) * 255, f"Global T={t_global}"),
    ((img > t_iter) * 255, f"Iterative T={t_iter:.1f}"),
    ((img > t_otsu) * 255, f"Otsu T={t_otsu}"),
]
for ax, (im, t) in zip(axes.ravel()[:4], panels):
    ax.imshow(im, cmap="gray", vmin=0, vmax=255)
    ax.set_title(t)
    ax.axis("off")

ax = axes.ravel()[4]
ax.bar(np.arange(256), np.bincount(img.ravel(), minlength=256), width=1.0, color="gray")
ax.axvline(t_global, color="b", label="Global")
ax.axvline(t_iter, color="g", label="Iterative")
ax.axvline(t_otsu, color="r", label="Otsu")
ax.set_title("Histogram + thresholds")
ax.legend()
axes.ravel()[5].axis("off")
plt.tight_layout()
plt.show()