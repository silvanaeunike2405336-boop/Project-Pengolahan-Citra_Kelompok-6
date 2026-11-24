"""methods.py
Metode pengolahan citra dari Chapter 3 (Intensity Transformations & Spatial Filtering).
Semua fungsi menerima citra grayscale 8-bit (uint8) dan mengembalikan uint8.
"""
import cv2
import numpy as np


# ------------------------------------------------------------------
# A. TRANSFORMASI INTENSITAS
# ------------------------------------------------------------------
def to_grayscale(img_bgr):
    """Grayscale: menyederhanakan citra berwarna menjadi tingkat keabuan."""
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)


def gamma_correction(g, gamma):
    """s = c * r^gamma (c = 1, r dinormalisasi 0..1).
    gamma < 1 mencerahkan, gamma > 1 menggelapkan."""
    r = g.astype(np.float32) / 255.0
    s = np.power(r, gamma)
    return np.clip(s * 255, 0, 255).astype(np.uint8)


def contrast_stretching(g, p_low=1.0, p_high=99.0):
    """s = (r - rmin) / (rmax - rmin) * (L - 1), L = 256.
    rmin/rmax diambil dari persentil agar tidak terpengaruh piksel ekstrem."""
    rmin, rmax = np.percentile(g, p_low), np.percentile(g, p_high)
    if rmax <= rmin:
        return g.copy()
    s = (g.astype(np.float32) - rmin) / (rmax - rmin) * 255
    return np.clip(s, 0, 255).astype(np.uint8)


def histogram_equalization(g):
    """s_k = (L - 1) * sum_{j<=k} p_r(r_j): meratakan distribusi intensitas."""
    return cv2.equalizeHist(g)


# ------------------------------------------------------------------
# B. PEMFILTERAN SPASIAL
# ------------------------------------------------------------------
def average_filter(g, k=3):
    """Filter penghalus: tiap piksel diganti rata-rata jendela k x k."""
    return cv2.blur(g, (int(k), int(k)))


def median_filter(g, k=3):
    """Tiap piksel diganti nilai tengah jendela k x k (efektif untuk noise impuls)."""
    return cv2.medianBlur(g, int(k))


def laplacian_sharpen(g, c=1.0):
    """Citra tajam = f - c * Laplacian(f)."""
    f = g.astype(np.float32)
    lap = cv2.Laplacian(f, cv2.CV_32F, ksize=1)
    return np.clip(f - c * lap, 0, 255).astype(np.uint8)


def sobel_edges(g):
    """Besar gradien = sqrt(Gx^2 + Gy^2)."""
    f = g.astype(np.float32)
    gx = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    return np.clip(mag / (mag.max() + 1e-6) * 255, 0, 255).astype(np.uint8)


# ------------------------------------------------------------------
# C. PENUNJANG (dipakai editor)
# ------------------------------------------------------------------
METHODS = ["Gamma Correction", "Contrast Stretching", "Histogram Equalization",
           "Median Filter", "Laplacian Filter", "Average Filter", "Sobel Filter"]

# Rentang slider tiap parameter: (min, max, step). Dipakai app dan analysis agar konsisten.
PARAM_RANGE = {
    "gamma": (0.1, 3.0, 0.05),
    "p_low": (0.0, 30.0, 0.5),
    "p_high": (70.0, 100.0, 0.5),
    "k": (3, 9, 2),
    "c": (0.1, 3.0, 0.1),
}


def default_params(name):
    return {"Gamma Correction": {"gamma": 0.6},
            "Contrast Stretching": {"p_low": 1.0, "p_high": 99.0},
            "Median Filter": {"k": 3},
            "Average Filter": {"k": 3},
            "Laplacian Filter": {"c": 1.0}}.get(name, {}).copy()


def apply_method(name, g, params):
    """Jalankan satu metode berdasarkan namanya."""
    if name == "Gamma Correction":
        return gamma_correction(g, params["gamma"])
    if name == "Contrast Stretching":
        return contrast_stretching(g, params["p_low"], params["p_high"])
    if name == "Histogram Equalization":
        return histogram_equalization(g)
    if name == "Median Filter":
        return median_filter(g, params["k"])
    if name == "Average Filter":
        return average_filter(g, params["k"])
    if name == "Laplacian Filter":
        return laplacian_sharpen(g, params["c"])
    if name == "Sobel Filter":
        return sobel_edges(g)
    return g


def blend(original, processed, alpha):
    """Intensitas efek 0..1: campuran citra masukan dan hasil metode."""
    out = (1 - alpha) * original.astype(np.float32) + alpha * processed.astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)