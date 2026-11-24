import numpy as np
from helper import load_gray, show

img = load_gray().astype(np.float64)

negative = 255 - img
log_t = 255 * np.log1p(img) / np.log1p(255)


def power_law(g):
    return 255 * (img / 255) ** g


show(
    [img, negative, log_t, power_law(0.4), power_law(2.5)],
    ["Original", "Negative", "Log", "Power-law γ=0.4 (brightens)", "Power-law γ=2.5 (darkens)"],
)