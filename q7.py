import cv2
import numpy as np
from helper import load_gray, show, to_u8

img = load_gray("general.png").astype(np.float32)


def conv(kernel):
    return cv2.filter2D(img, cv2.CV_32F, np.array(kernel, dtype=np.float32))


laplacian = conv([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
sharpened = np.clip(img - laplacian, 0, 255)  # centre coeff is negative -> subtract

operators = {
    "Sobel": ([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]),
    "Prewitt": ([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], [[-1, -1, -1], [0, 0, 0], [1, 1, 1]]),
}
mags = {name: np.hypot(conv(kx), conv(ky)) for name, (kx, ky) in operators.items()}

sobel_bin = (mags["Sobel"] > 0.25 * mags["Sobel"].max()).astype(np.uint8) * 255

show(
    [img, to_u8(np.abs(laplacian)), sharpened, to_u8(mags["Sobel"]), to_u8(mags["Prewitt"]), sobel_bin],
    ["Original", "Laplacian edges", "Laplacian sharpened", "Sobel magnitude", "Prewitt magnitude", "Sobel thresholded"],
)