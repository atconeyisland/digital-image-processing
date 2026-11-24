import cv2
import numpy as np
from helper import load_gray, show, to_u8

img = cv2.resize(load_gray("general.png"), (256, 256)).astype(np.float64)
M, N = img.shape


def dft_matrix(n):
    k = np.arange(n)
    return np.exp(-2j * np.pi * np.outer(k, k) / n)


Wm, Wn = dft_matrix(M), dft_matrix(N)

# Separable 2D DFT:  F = W_M · f · W_N
F = Wm @ img @ Wn
print("Max difference vs np.fft.fft2:", np.abs(F - np.fft.fft2(img)).max())

spectrum = np.log1p(np.abs(np.fft.fftshift(F)))  # centred, log-scaled magnitude

# Inverse 2D DFT
recon = np.real(np.conj(Wm) @ F @ np.conj(Wn)) / (M * N)
print("Max reconstruction error:", np.abs(recon - img).max())

show(
    [img, to_u8(spectrum), np.clip(recon, 0, 255)],
    ["Input", "Centered magnitude spectrum (log)", "Reconstructed (inverse DFT)"],
)