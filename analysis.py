"""analysis.py
Analisis kondisi citra dan aturan rekomendasi (rule-based, bukan AI).

Indikator:
  Kecerahan      : rata-rata intensitas grayscale
  Kontras        : simpangan baku intensitas + lebar rentang histogram (P1-P99)
  Noise impuls   : proporsi piksel yang menyimpang jauh dari hasil median filter
  Ketajaman      : varians hasil Laplacian
  Kekuatan tepi  : rata-rata besar gradien Sobel

Urutan saran mengikuti urutan pengolahan yang benar:
  1) noise  2) kecerahan  3) kontras  4) ketajaman
(noise dibersihkan lebih dulu agar tidak ikut dipertajam atau dicerahkan).
Tiap saran membawa nilai parameter awal yang dihitung dari kondisi foto.
"""
import cv2
import numpy as np

DEFAULT_TH = {"dark": 90, "bright": 200, "contrast": 45, "range": 150,
              "noise": 0.01, "sharp": 100}


def analyze(g, noise_dev=40):
    med = cv2.medianBlur(g, 3)
    diff = np.abs(g.astype(np.int16) - med.astype(np.int16))
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    return {
        "brightness": float(g.mean()),
        "contrast": float(g.std()),
        "hist_range": float(np.percentile(g, 99) - np.percentile(g, 1)),
        "noise_ratio": float((diff > noise_dev).mean()),
        "sharpness": float(cv2.Laplacian(g, cv2.CV_64F).var()),
        "edge_strength": float(np.sqrt(gx ** 2 + gy ** 2).mean()),
    }


def _suggest_gamma(mean, lo, hi, target=130):
    """Gamma yang menggeser rata-rata intensitas mendekati target,
    dibulatkan ke kelipatan 0.05 (sama dengan step slider)."""
    mean = float(np.clip(mean, 1, 254))
    g = np.log(target / 255) / np.log(mean / 255)
    g = round(float(g) / 0.05) * 0.05
    return round(float(np.clip(g, lo, hi)), 2)


def find_issues(m, th, lang="id"):
    """Kembalikan daftar kekurangan berurutan: key, title, detail, methods, defaults.
    lang: "id" (Indonesia) atau "en" (English)."""
    def L(i, e):
        return i if lang == "id" else e

    issues = []

    # 1) noise dulu
    if m["noise_ratio"] > th["noise"]:
        r = m["noise_ratio"]
        issues.append(dict(
            key="noise", title=L("Terdapat bintik-bintik (noise impuls)", "Speckles found (impulse noise)"),
            detail=L(f"{r * 100:.2f}% piksel menyimpang jauh dari hasil median.",
                     f"{r * 100:.2f}% of pixels deviate strongly from the median result."),
            methods=["Median Filter"],
            defaults={"k": 3 if r < 0.03 else 5}))     # maks 5 agar huruf kecil tidak menyatu

    # 2) kecerahan
    if m["brightness"] < th["dark"]:
        issues.append(dict(
            key="gelap", title=L("Citra terlalu gelap", "Image is too dark"),
            detail=L(f"Rata-rata intensitas {m['brightness']:.0f} (ambang {th['dark']}).",
                     f"Mean intensity {m['brightness']:.0f} (threshold {th['dark']})."),
            methods=["Gamma Correction"],
            defaults={"gamma": _suggest_gamma(m["brightness"], 0.2, 1.0)}))
    elif m["brightness"] > th["bright"]:
        issues.append(dict(
            key="terang", title=L("Citra terlalu terang", "Image is too bright"),
            detail=L(f"Rata-rata intensitas {m['brightness']:.0f} (ambang {th['bright']}).",
                     f"Mean intensity {m['brightness']:.0f} (threshold {th['bright']})."),
            methods=["Gamma Correction"],
            defaults={"gamma": _suggest_gamma(m["brightness"], 1.0, 3.0)}))

    # 3) kontras
    if m["contrast"] < th["contrast"] or m["hist_range"] < th["range"]:
        issues.append(dict(
            key="kontras", title=L("Kontras rendah", "Low contrast"),
            detail=L(f"Simpangan baku {m['contrast']:.0f}, rentang histogram {m['hist_range']:.0f}.",
                     f"Std. deviation {m['contrast']:.0f}, histogram range {m['hist_range']:.0f}."),
            methods=["Contrast Stretching", "Histogram Equalization"],
            defaults={"p_low": 1.0 if m["hist_range"] > 100 else 2.0, "p_high": 99.0}))

    # 4) ketajaman terakhir
    if m["sharpness"] < th["sharp"]:
        deficit = 1 - m["sharpness"] / max(th["sharp"], 1)
        issues.append(dict(
            key="tajam", title=L("Detail tulisan kurang tajam", "Text details are not sharp"),
            detail=L(f"Varians Laplacian {m['sharpness']:.0f} (ambang {th['sharp']}).",
                     f"Laplacian variance {m['sharpness']:.0f} (threshold {th['sharp']})."),
            methods=["Laplacian Filter"],
            defaults={"c": round(float(np.clip(0.5 + deficit * 1.5, 0.5, 2.0)), 1)}))
    return issues


# ------------------------------------------------------------------
# PEMERIKSAAN DOKUMEN (rule-based): apakah gambar layak dianalisis?
# ------------------------------------------------------------------
MIN_SIDE = 300


def check_document(g):
    """Kembalikan dict(ok, reason, stats). reason: None | "small" | "blank" | "not_doc".
    Ciri halaman bertulis: satu tingkat keabuan latar yang dominan, serta banyak goresan
    kecil (huruf) yang jumlahnya sedikit dibanding luas gambar. Intensitas dinormalisasi
    dulu supaya foto dokumen yang gelap tidak salah ditolak."""
    h, w = g.shape
    if min(h, w) < MIN_SIDE:
        return dict(ok=False, reason="small", stats={})
    if max(h, w) > 900:
        f = 900 / max(h, w)
        g = cv2.resize(g, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    lo, hi = np.percentile(g, (1, 99))
    if hi - lo < 4:
        return dict(ok=False, reason="blank", stats={})
    n = np.clip((g.astype(np.float32) - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
    hist = np.bincount(n.ravel(), minlength=256).astype(np.float32)
    bg = float(np.convolve(hist, np.ones(41), "same").max() / n.size)
    block = max(15, (min(n.shape) // 20) | 1)
    ink = cv2.adaptiveThreshold(cv2.GaussianBlur(n, (3, 3), 0), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV, block, 12)
    ink_ratio = float((ink > 0).mean())
    _, _, st, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    areas = st[1:, cv2.CC_STAT_AREA]
    strokes = int(((areas >= 6) & (areas <= n.size * 0.002)).sum())
    stats = dict(bg=bg, ink=ink_ratio, strokes=strokes)
    if ink_ratio < 0.003 or strokes < 8:
        return dict(ok=False, reason="blank", stats=stats)
    if bg < 0.35 or ink_ratio > 0.35 or strokes < 25:
        return dict(ok=False, reason="not_doc", stats=stats)
    return dict(ok=True, reason=None, stats=stats)