import cv2
import numpy as np
import matplotlib.pyplot as plt
from helper import load_color

img = load_color("lines.png")
scale = 800 / max(img.shape[:2])
if scale < 1:
    img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)

H, W = edges.shape
diag = int(np.ceil(np.hypot(H, W)))
thetas = np.deg2rad(np.arange(-90, 90))
ys, xs = np.nonzero(edges)

# rho = x*cos(theta) + y*sin(theta) for every edge pixel and every theta at once
rho_idx = np.rint(np.outer(xs, np.cos(thetas)) + np.outer(ys, np.sin(thetas))).astype(int) + diag
acc = np.stack(
    [np.bincount(rho_idx[:, t], minlength=2 * diag + 1) for t in range(len(thetas))], axis=1
).astype(np.float32)

# Non-maximum suppression: keep a cell only if it is the max in its 21x21 neighbourhood
local_max = cv2.dilate(acc, np.ones((21, 21), np.uint8))
peaks = np.argwhere((acc == local_max) & (acc >= 0.6 * acc.max()))
peaks = peaks[np.argsort(-acc[peaks[:, 0], peaks[:, 1]])][:10]

out = img.copy()
L = 2 * diag
for r_i, t_i in peaks:
    rho, th = r_i - diag, thetas[t_i]
    a, b = np.cos(th), np.sin(th)
    x0, y0 = a * rho, b * rho
    p1 = (int(x0 - L * b), int(y0 + L * a))
    p2 = (int(x0 + L * b), int(y0 - L * a))
    cv2.line(out, p1, p2, (0, 0, 255), 2)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
axes[0].imshow(edges, cmap="gray")
axes[0].set_title("Canny edges")
axes[1].imshow(acc, cmap="hot", aspect="auto", extent=[-90, 90, diag, -diag])
axes[1].set_title("Hough accumulator (θ vs ρ)")
axes[1].set_xlabel("θ (deg)")
axes[1].set_ylabel("ρ (px)")
axes[2].imshow(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
axes[2].set_title(f"Detected lines ({len(peaks)})")
for ax in (axes[0], axes[2]):
    ax.axis("off")
plt.tight_layout()
plt.show()