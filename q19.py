import cv2
import numpy as np
from helper import load_color, show, to_u8

img = load_color("coins.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)

_, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Make objects white: if most border pixels are white, the background is bright -> invert
border = np.concatenate([binary[0], binary[-1], binary[:, 0], binary[:, -1]])
if border.mean() > 127:
    binary = cv2.bitwise_not(binary)

kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
sure_bg = cv2.dilate(opening, kernel, iterations=3)

# Distance transform peaks = one seed per object, even when objects touch
dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
sure_fg = (dist > 0.5 * dist.max()).astype(np.uint8) * 255
unknown = cv2.subtract(sure_bg, sure_fg)

n_labels, markers = cv2.connectedComponents(sure_fg)
markers = markers + 1          # background = 1, objects = 2..n
markers[unknown == 255] = 0    # 0 = region watershed must decide

markers = cv2.watershed(img, markers)

result = img.copy()
result[markers == -1] = (0, 0, 255)  # watershed boundaries in red

show(
    [cv2.cvtColor(img, cv2.COLOR_BGR2RGB), to_u8(dist), sure_fg, cv2.cvtColor(result, cv2.COLOR_BGR2RGB)],
    ["Original", "Distance transform", "Sure foreground (seeds)", f"Watershed: {n_labels - 1} objects"],
    cols=4,
)