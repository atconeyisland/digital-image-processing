import numpy as np
import matplotlib.pyplot as plt
from helper import load_gray

img = load_gray("general.png")


def stretch(a):
    lo, hi = int(a.min()), int(a.max())
    return ((a.astype(np.float64) - lo) / (hi - lo) * 255).astype(np.uint8)


def equalize(a):
    cdf = np.bincount(a.ravel(), minlength=256).cumsum()
    cdf_min = cdf[cdf > 0][0]
    lut = np.round((cdf - cdf_min) / (cdf[-1] - cdf_min) * 255)
    return np.clip(lut, 0, 255).astype(np.uint8)[a]


images = [img, stretch(img), equalize(img)]
titles = ["Original", "Histogram stretching", "Histogram equalization"]

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
for c, (im, t) in enumerate(zip(images, titles)):
    axes[0, c].imshow(im, cmap="gray", vmin=0, vmax=255)
    axes[0, c].set_title(t)
    axes[0, c].axis("off")
    axes[1, c].bar(np.arange(256), np.bincount(im.ravel(), minlength=256), width=1.0)
    axes[1, c].set_xlim(0, 255)
    axes[1, c].set_title("Histogram")
plt.tight_layout()
plt.show()