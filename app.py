"""app.py: antarmuka Pixora (Streamlit), dua bahasa (ID/EN).
Jalankan: python -m streamlit run app.py"""
import cv2
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

try:
    import pymupdf as fitz  # PyMuPDF (untuk membaca PDF), nama impor terbaru
    HAS_PDF = True
except ImportError:
    try:
        import fitz  # nama impor lama PyMuPDF
        HAS_PDF = True
    except ImportError:
        HAS_PDF = False

from methods import METHODS, default_params, apply_method, blend, to_grayscale
from analysis import analyze, find_issues, DEFAULT_TH

try:
    from analysis import check_document
except ImportError:          # analysis.py versi lama: lewati pemeriksaan dokumen
    def check_document(g):
        return {"ok": True, "reason": None, "stats": {}}

st.set_page_config(page_title="Pixora", page_icon="🖼️", layout="wide", initial_sidebar_state="collapsed")

# ============================================================ TEKS (ID, EN)
TX = {
    "nav_home": ("Beranda", "Home"), "nav_fix": ("Perbaiki Foto", "Enhance Photo"), "nav_methods": ("Metode", "Methods"),
    "hero_sub": ("Foto catatan atau papan tulismu gelap dan buram? Biar Pixora yang merapikan.",
                 "Dark, blurry photo of your notes or whiteboard? Let Pixora clean it up."),
    "cta": ("Mulai perbaiki foto", "Start enhancing"),
    "chip1": ("Catatan kuliah", "Lecture notes"), "chip2": ("Papan tulis", "Whiteboards"),
    "chip3": ("Materi belajar", "Study material"), "chip4": ("Dokumen", "Documents"),
    "before": ("Sebelum", "Before"), "after": ("Sesudah", "After"),
    "stat1": ("7 metode", "7 methods"), "stat2": ("4 langkah", "4 steps"), "stat3": ("Tanpa akun", "No account"),
    "how": ("Cara kerja", "How it works"),
    "c1": ("Pixora membaca kondisi fotomu.", "Pixora reads your photo's condition."),
    "c2": ("Kamu dapat saran perbaikan yang pas.", "You get fitting fixes suggested."),
    "c3": ("Atur sendiri, lalu unduh hasilnya.", "Fine-tune it, then download."),
    "s1": ("Unggah", "Upload"), "s2": ("Analisis", "Analyze"), "s3": ("Perbaiki", "Enhance"), "s4": ("Hasil", "Result"),
    "step": ("Langkah", "Step"), "of": ("dari", "of"),
    "msg_upload": ("Unggah foto dokumenmu di bawah.", "Upload your document photo below."),
    "msg_pick": ("Lihat saran di bawah, lalu klik Terapkan atau Lewati.", "Check the suggestion below, then click Apply or Skip."),
    "msg_edit": ("Geser slider sampai hasilnya bagus, lalu klik Simpan dan lanjut.", "Drag the sliders until it looks good, then click Save & continue."),
    "msg_done": ("Selesai. Cek hasilnya lalu unduh.", "Done. Check the result and download it."),
    "uploader": ("Foto dokumen", "Document photo"), "photo": ("Foto", "Photo"), "change": ("Ganti foto", "Change photo"),
    "bad_img": ("File bukan gambar yang valid. Coba unggah lagi.", "That file is not a valid image. Please try again."),
    "bright": ("Kecerahan", "Brightness"), "contrast": ("Kontras", "Contrast"), "noise": ("Noise", "Noise"),
    "sharp": ("Ketajaman", "Sharpness"), "orig": ("asli", "orig."),
    "too_dark": ("Terlalu gelap", "Too dark"), "too_bright": ("Terlalu terang", "Too bright"), "normal": ("Normal", "Normal"),
    "low": ("Rendah", "Low"), "enough": ("Cukup", "Good"), "speckled": ("Ada bintik", "Speckles"), "clean": ("Bersih", "Clean"),
    "blurry": ("Kurang tajam", "Not sharp"), "crisp": ("Tajam", "Sharp"),
    "found": ("Ditemukan", "Found"), "suggest": ("Saran", "Suggestion"),
    "still": ("Masih terdeteksi. Tambah perbaikan atau lewati.", "Still detected. Add another fix or skip."),
    "apply": ("Terapkan", "Apply"), "skip": ("Lewati", "Skip"),
    "more_issues": ("masalah lain menunggu.", "more issue(s) waiting."),
    "no_issue": ("Tidak ada masalah lain. Foto sudah baik.", "No other issues. Your photo looks good."),
    "show_skipped": ("Tampilkan lagi yang dilewati", "Show skipped issues again"),
    "history": ("Riwayat langkah", "History"), "tune": ("Atur", "Tune"),
    "gamma": ("Gamma (kecil = cerah)", "Gamma (low = brighter)"), "black": ("Batas hitam", "Black point"),
    "white": ("Batas putih", "White point"), "window": ("Ukuran jendela", "Window size"),
    "weight": ("Bobot penajaman", "Sharpening weight"), "intensity": ("Intensitas (%)", "Intensity (%)"),
    "save": ("Simpan dan lanjut", "Save & continue"), "cancel": ("Batalkan", "Cancel"), "delete": ("Hapus langkah", "Remove step"),
    "res_hint": ("Hasil muncul di sini setelah kamu menerapkan perbaikan.", "Your result appears here once you apply a fix."),
    "hist": ("Histogram", "Histogram"), "download": ("Unduh hasil (PNG)", "Download result (PNG)"),
    "adv": ("Pengaturan lanjutan", "Advanced settings"),
    "tab1": ("Metode manual", "Manual method"), "tab2": ("Ambang deteksi", "Detection thresholds"), "tab3": ("Mulai ulang", "Reset"),
    "method": ("Metode", "Method"), "add": ("Tambahkan", "Add"), "reset": ("Reset semua perbaikan", "Reset all fixes"),
    "th_dark": ("Gelap jika rata-rata <", "Dark if mean <"), "th_bright": ("Terang jika rata-rata >", "Bright if mean >"),
    "th_contrast": ("Kontras rendah jika std <", "Low contrast if std <"), "th_range": ("atau lebar histogram <", "or histogram width <"),
    "th_noise": ("Berbintik jika proporsi >", "Speckled if ratio >"), "th_sharp": ("Kurang tajam jika var Laplacian <", "Not sharp if Laplacian var <"),
    "methods_title": ("Metode", "Methods"),
    "up_hint": ("Untuk dokumen: foto, hasil scan, atau PDF. Foto berwarna diubah ke abu-abu saat diproses.",
                "For documents: photos, scans, or PDFs. Color photos are converted to gray when processed."),
    "pdf_page": ("Halaman", "Page"), "pdf_of": ("dari", "of"),
    "pdf_missing": ("Untuk membaca PDF, pasang dulu dengan perintah: pip install pymupdf", "To read PDFs, first install it with: pip install pymupdf"),
    "pdf_bad": ("PDF tidak bisa dibaca. Coba file lain.", "This PDF could not be read. Try another file."),
    "pdf_all": ("Proses semua halaman", "Process all pages"), "pdf_dl": ("Unduh semua halaman (PDF)", "Download all pages (PDF)"),
    "pdf_cap": ("Hanya 30 halaman pertama yang diproses.", "Only the first 30 pages are processed."),
    "pdf_work": ("Memproses halaman...", "Processing pages..."),
    "pdf_note": ("Pengaturan yang sama diterapkan ke semua halaman.", "The same settings are applied to every page."),
    "dl_page": ("Unduh halaman ini (PNG)", "Download this page (PNG)"),
    "nd_title": ("Pixora tidak menganalisis file ini", "Pixora did not analyze this file"),
    "nd_small": ("Resolusinya terlalu kecil (sisi terpendek di bawah 300 piksel), jadi hasil analisisnya tidak akan bermakna.",
                 "The resolution is too small (shortest side under 300 px), so an analysis would not be meaningful."),
    "nd_blank": ("Tidak ada tulisan yang terdeteksi pada gambar ini.", "No writing was detected in this image."),
    "nd_not_doc": ("Gambar ini tampaknya bukan dokumen. Pixora dirancang untuk halaman bertulis seperti catatan, materi, atau papan tulis.",
                   "This image does not look like a document. Pixora is built for written pages such as notes, study material, or whiteboards."),
    "nd_hint": ("Kalau ini memang dokumenmu, kamu tetap bisa melanjutkan.", "If this really is your document, you can still continue."),
    "nd_force": ("Tetap analisis", "Analyze anyway"),
    "click_hint": ("Klik untuk detail", "Click for details"),
    "foot_course": ("Pengolahan Citra dan Computer Vision", "Image Processing and Computer Vision"),
    "foot_group": ("Kelompok 6", "Group 6"),
    "foot_dev": ("Dikembangkan oleh Silvana dan Asyh", "Developed by Silvana and Asyh"),
    "foot_lab_course": ("Mata kuliah", "Course"), "foot_lab_dev": ("Pengembang", "Developers"),
    "foot_built": ("Dibuat dengan Streamlit, OpenCV, dan NumPy", "Built with Streamlit, OpenCV, and NumPy"),
    "lang_t": ("Bahasa", "Language"),
    "band": ("Siap merapikan fotomu?", "Ready to clean up your photo?"),
    "m_hint": ("Klik tombol pada kartu untuk membalik dan melihat penjelasannya.", "Use the button on a card to flip it and read the explanation."),
    "m_flip": ("Lihat penjelasan ↻", "See explanation ↻"), "m_back": ("Kembali ke rumus ↻", "Back to formula ↻"),
    "m_sym": ("Arti simbol", "What the symbols mean"), "m_how": ("Cara kerja", "How it works"), "m_use": ("Di Pixora", "In Pixora"),
    "about_t": ("Tentang Pixora", "About Pixora"),
    "about_d": ("Pixora adalah aplikasi web untuk menganalisis dan memperbaiki foto dokumen, seperti catatan kuliah, materi belajar, atau tulisan di papan tulis. Pixora membaca kondisi foto dari nilai pikselnya, memberi saran perbaikan yang sesuai, lalu kamu yang memutuskan dan mengatur hasilnya. Saran dibuat dengan aturan (rule-based) dari nilai piksel, bukan kecerdasan buatan.",
                "Pixora is a web app that analyzes and improves document photos such as lecture notes, study material, or whiteboard writing. It reads your photo's condition from its pixel values, suggests fitting fixes, and lets you decide and fine-tune the result. Suggestions are rule-based on pixel values, not artificial intelligence."),
    "steps_t": ("Tahapan", "The steps"),
    "t1": ("Unggah foto", "Upload a photo"), "t1d": ("Pilih foto catatan, materi, atau papan tulis dalam format JPG, PNG, atau BMP.", "Pick a photo of notes, study material, or a whiteboard in JPG, PNG, or BMP."),
    "t2": ("Analisis otomatis", "Automatic analysis"), "t2d": ("Pixora mengukur kecerahan, kontras, noise, dan ketajaman, lalu menandai mana yang bermasalah.", "Pixora measures brightness, contrast, noise, and sharpness, and flags what is off."),
    "t3": ("Rekomendasi dan perbaikan", "Recommend & enhance"), "t3d": ("Masalah muncul satu per satu beserta metode yang cocok. Terapkan atau lewati, lalu atur kekuatannya dengan slider.", "Issues appear one at a time with a fitting method. Apply or skip, then tune the strength with sliders."),
    "t4": ("Hasil dan unduh", "Result & download"), "t4d": ("Bandingkan sebelum dan sesudah, lalu unduh hasilnya sebagai PNG.", "Compare before and after, then download the result as PNG."),
}
SHORT = {
    "Gamma Correction": ("atur kecerahan", "adjust brightness"), "Contrast Stretching": ("lebarkan kontras", "widen contrast"),
    "Histogram Equalization": ("ratakan kontras", "even out contrast"), "Median Filter": ("hilangkan bintik", "remove speckles"),
    "Laplacian Filter": ("pertajam tulisan", "sharpen text"), "Average Filter": ("haluskan citra", "smooth image"),
    "Sobel Filter": ("tampilkan tepi", "show edges"),
}
METHOD_INFO = [
    dict(name="Gamma Correction", group=("Transformasi intensitas", "Intensity transform"),
         latex=r"s = c \cdot r^{\gamma}",
         sym=("r = nilai piksel asli (0 sampai 1), s = nilai piksel hasil, γ (gamma) = pengaturan kecerahan, c = konstanta (1).",
              "r = original pixel value (0 to 1), s = output pixel value, γ (gamma) = brightness setting, c = constant (1)."),
         how=("Tiap piksel dipangkatkan γ. Kalau γ lebih kecil dari 1, foto jadi lebih terang. Kalau lebih besar dari 1, foto jadi lebih gelap.",
              "Every pixel is raised to the power γ. If γ is below 1 the photo gets brighter; above 1 it gets darker."),
         use=("Dipakai saat foto terlalu gelap atau terlalu terang. Nilai γ awal dihitung otomatis dari kecerahan fotomu, dan kamu bisa menggesernya lewat slider Gamma.",
              "Used when a photo is too dark or too bright. The starting γ is computed from your photo's brightness and you can drag the Gamma slider.")),
    dict(name="Contrast Stretching", group=("Transformasi intensitas", "Intensity transform"),
         latex=r"s = \frac{r - r_{min}}{r_{max} - r_{min}}\,(L-1)",
         sym=("r_min dan r_max = piksel paling gelap dan paling terang di foto. L = 256 tingkat keabuan.",
              "r_min and r_max = the darkest and brightest pixels in the photo. L = 256 gray levels."),
         how=("Nilai piksel ditarik melebar: yang paling gelap menjadi 0 (hitam) dan yang paling terang menjadi 255 (putih).",
              "Pixel values are stretched so the darkest becomes 0 (black) and the brightest becomes 255 (white)."),
         use=("Dipakai saat kontras rendah, yaitu tulisan dan latar terlihat mirip. Slider Batas hitam dan Batas putih menentukan piksel mana yang dianggap paling gelap dan terang.",
              "Used for low contrast, when text and background look alike. The Black point and White point sliders decide which pixels count as darkest and brightest.")),
    dict(name="Histogram Equalization", group=("Transformasi intensitas", "Intensity transform"),
         latex=r"s_k = (L-1)\sum_{j=0}^{k} p_r(r_j)",
         sym=("p_r(r_j) = proporsi piksel dengan tingkat keabuan r_j. Jumlah p_r dari 0 sampai k disebut distribusi kumulatif.",
              "p_r(r_j) = share of pixels at gray level r_j. The sum of p_r from 0 to k is the cumulative distribution."),
         how=("Tingkat keabuan dipetakan ulang agar sebarannya merata dari 0 sampai 255, sehingga kontras naik secara otomatis.",
              "Gray levels are remapped so they spread evenly from 0 to 255, which raises contrast automatically."),
         use=("Alternatif Contrast Stretching untuk kontras rendah. Tidak punya parameter, jadi gunakan slider Intensitas kalau efeknya terasa terlalu kuat.",
              "An alternative to Contrast Stretching for low contrast. It has no parameters, so use the Intensity slider if the effect feels too strong.")),
    dict(name="Median Filter", group=("Filter spasial", "Spatial filter"),
         latex=r"g(x,y) = \operatorname{median}\{\,f(s,t)\,\}_{(s,t)\in S_{xy}}",
         sym=("S_xy = jendela k × k di sekitar piksel (x, y). f(s,t) = piksel di dalam jendela itu.",
              "S_xy = the k × k window around pixel (x, y). f(s,t) = the pixels inside that window."),
         how=("Piksel diganti dengan nilai tengah dari semua piksel di jendelanya. Bintik ekstrem terbuang, sedangkan tepi huruf tetap tegas.",
              "Each pixel is replaced by the median of its window. Extreme speckles are discarded while letter edges stay crisp."),
         use=("Dipakai saat foto berbintik (noise). Slider Ukuran jendela: makin besar makin bersih, tetapi makin halus.",
              "Used when a photo is speckled (noisy). Window size slider: bigger is cleaner but softer.")),
    dict(name="Average Filter", group=("Filter spasial", "Spatial filter"),
         latex=r"g(x,y) = \frac{1}{k^{2}} \sum_{(s,t)\in S_{xy}} f(s,t)",
         sym=("k × k = ukuran jendela. f(s,t) = piksel di dalam jendela.",
              "k × k = window size. f(s,t) = the pixels inside the window."),
         how=("Piksel diganti dengan rata-rata tetangganya. Hasilnya halus, tetapi tepi huruf jadi sedikit buram.",
              "Each pixel is replaced by the average of its neighbors. The result is smooth but letter edges get slightly blurred."),
         use=("Tidak disarankan otomatis. Bisa dicoba lewat Metode manual di Pengaturan lanjutan untuk meredam gangguan kecil.",
              "Not suggested automatically. Try it via Manual method in Advanced settings to soften small disturbances.")),
    dict(name="Laplacian Filter", group=("Filter spasial", "Spatial filter"),
         latex=r"g = f - c\,\nabla^{2} f",
         sym=("∇²f = turunan kedua, yaitu seberapa cepat intensitas berubah di sekitar piksel. c = bobot penajaman.",
              "∇²f = second derivative, i.e. how fast intensity changes around a pixel. c = sharpening weight."),
         how=("Perubahan di tepi huruf dihitung, lalu dikurangkan dari foto asli sehingga tepinya lebih menonjol.",
              "Changes at letter edges are computed, then subtracted from the original so the edges stand out."),
         use=("Dipakai saat tulisan kurang tajam. Slider Bobot penajaman: makin besar makin tegas, tetapi terlalu besar bisa memunculkan bintik.",
              "Used when text is not sharp. Sharpening weight slider: bigger is crisper, but too large can create speckles.")),
    dict(name="Sobel Filter", group=("Filter spasial", "Spatial filter"),
         latex=r"|\nabla f| = \sqrt{G_x^{2} + G_y^{2}}",
         sym=("Gx dan Gy = perubahan intensitas ke arah horizontal dan vertikal.",
              "Gx and Gy = intensity change in the horizontal and vertical directions."),
         how=("Besar gradien menunjukkan seberapa kuat tepi di tiap piksel. Hasilnya berupa peta tepi.",
              "The gradient magnitude shows how strong the edge is at each pixel. The output is an edge map."),
         use=("Untuk memeriksa tepi tulisan lewat Metode manual. Karena hasilnya peta tepi, turunkan slider Intensitas agar bercampur dengan foto.",
              "For inspecting text edges via Manual method. Since the output is an edge map, lower the Intensity slider to blend it with the photo.")),
]

ss = st.session_state
ss.setdefault("lang", "id")


def t(key):
    return TX[key][0 if ss.lang == "id" else 1]


def tr(pair):
    return pair[0 if ss.lang == "id" else 1]


# ============================================================ GAYA (palet Starry Night)
import base64
import math
import random


def _uri(svg):
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode()).decode()


def _star(cx, cy, R, r, n=5):
    pts = []
    for k in range(n * 2):
        ang = -math.pi / 2 + k * math.pi / n
        rad = R if k % 2 == 0 else r
        pts.append(f"{cx + rad * math.cos(ang):.1f},{cy + rad * math.sin(ang):.1f}")
    return " ".join(pts)


def contour_svg(color, opacity):
    """Bentuk bintang berlapis-lapis (seperti kontur) untuk hiasan latar."""
    layers = "".join(
        f'<polygon points="{_star(300, 300, R, R * 0.64)}" stroke-opacity="{opacity * (1 - i * 0.08):.3f}"/>'
        for i, R in enumerate(range(50, 300, 34)))
    return _uri('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 600">'
                f'<g fill="none" stroke="{color}" stroke-width="3" stroke-linejoin="round">{layers}</g></svg>')


def sparkle_svg(color, count=16, size=900, seed=7, top=0.55):
    """Kilau bintang kecil yang tersebar (ubin berulang)."""
    rnd = random.Random(seed)
    out = ""
    for _ in range(count):
        x, y, k = rnd.uniform(20, size - 20), rnd.uniform(20, size - 20), rnd.uniform(4, 10)
        out += (f'<path d="M{x:.0f} {y - k:.0f}L{x + k * .25:.0f} {y - k * .25:.0f}L{x + k:.0f} {y:.0f}'
                f'L{x + k * .25:.0f} {y + k * .25:.0f}L{x:.0f} {y + k:.0f}L{x - k * .25:.0f} {y + k * .25:.0f}'
                f'L{x - k:.0f} {y:.0f}L{x - k * .25:.0f} {y - k * .25:.0f}Z" '
                f'fill="{color}" fill-opacity="{rnd.uniform(.3, top):.2f}"/>')
    return _uri(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {size} {size}">{out}</svg>')


CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@500;600;700;800&family=Nunito:wght@400;600;700;800&display=swap');
.stApp *:not([data-testid="stIconMaterial"]):not(.katex):not(.katex *):not(code):not(pre){
  font-family:'Nunito','Segoe UI',system-ui,sans-serif;}
h1,h2,h3,h4,.hero-h,.sec h2,.todo h3,.tile .v,.pg,.band-t,.how,.sc .sn,.sc .stt,.m-name,.notice h3{font-family:'Outfit','Nunito',sans-serif !important;}
:root{--ink:#0C1446;--muted:#6B7BA0;--line:#CFE0EE;--pri:#2B5C92;--pri2:#7FA6CC;--sky:#B3CDE0;--soft:#E9F1F8;
--amber:#B45309;--amber-l:#FEF3C7;--green:#15803D;--green-l:#DCFCE7;}

/* ===== latar halaman: kabut biru, kisi, bintang berlapis, kilau ===== */
.stApp{background:
  url("__CONTOUR_L__") no-repeat right -150px bottom -80px / 600px,
  radial-gradient(ellipse at 0% 0%, rgba(43,92,146,.38), transparent 46%),
  radial-gradient(ellipse at 0% 100%, rgba(12,20,70,.30), transparent 52%),
  radial-gradient(ellipse at 100% 18%, rgba(179,205,224,.55), transparent 46%), #F5F9FD;
  background-attachment:fixed;}
.stApp::before{content:"";position:fixed;inset:0;z-index:0;pointer-events:none;
  background-image:linear-gradient(rgba(43,92,146,.11) 1px,transparent 1px),linear-gradient(90deg,rgba(43,92,146,.11) 1px,transparent 1px);
  background-size:38px 38px;-webkit-mask-image:linear-gradient(135deg,#000 0%,transparent 48%);mask-image:linear-gradient(135deg,#000 0%,transparent 48%);}
.stApp::after{content:"";position:fixed;inset:0;z-index:0;pointer-events:none;background:url("__SPARK_L__") 0 0 / 900px 900px;opacity:.9;}
[data-testid="stAppViewContainer"],[data-testid="stMain"]{position:relative;z-index:1;}
header[data-testid="stHeader"],[data-testid="stDecoration"]{display:none !important;}
.block-container{max-width:1020px;padding-top:1rem;padding-bottom:4rem;}
h1,h2,h3,h4{color:var(--ink);}
[data-testid="stSidebar"],[data-testid="stSidebarCollapsedControl"],[data-testid="collapsedControl"]{display:none !important;}

/* ===== toolbar: satu pill panjang ===== */
.st-key-nav{background:rgba(255,255,255,.93);border:2px solid var(--line);border-radius:999px;padding:.4rem .6rem;
  box-shadow:0 14px 34px -18px rgba(12,20,70,.55);backdrop-filter:blur(8px);margin-bottom:1.3rem;}
.st-key-nav [data-testid="stHorizontalBlock"]{gap:.35rem;align-items:center;}
.st-key-nav button,.st-key-nav [data-testid="stPopover"] button{height:46px !important;min-height:46px !important;border-radius:999px !important;
  border:2px solid transparent !important;padding:0 .6rem !important;transition:all .15s ease;width:100%;}
.st-key-nav button p,.st-key-nav [data-testid="stPopover"] button p{font-weight:800 !important;font-size:1rem;margin:0;}
.st-key-nav button[data-testid="stBaseButton-secondary"],.st-key-nav [data-testid="stPopover"] button{background:transparent !important;color:#34456E !important;}
.st-key-nav button[data-testid="stBaseButton-secondary"]:hover,.st-key-nav [data-testid="stPopover"] button:hover{background:var(--soft) !important;color:var(--pri) !important;}
.st-key-nav button[data-testid="stBaseButton-primary"]{background:#D3E3F1 !important;color:var(--ink) !important;border-color:var(--pri2) !important;
  box-shadow:0 5px 14px -8px rgba(43,92,146,.9);}
.st-key-nav_brand button{position:relative;justify-content:flex-start !important;padding-left:3.1rem !important;background:transparent !important;}
.st-key-nav_brand button:hover{background:transparent !important;}
.st-key-nav_brand button p{font-family:'Outfit',sans-serif !important;font-size:1.5rem !important;font-weight:800 !important;color:var(--ink) !important;letter-spacing:-.02em;}
.st-key-nav_brand button::before{content:"";position:absolute;left:.7rem;top:50%;transform:translateY(-50%);width:28px;height:28px;border-radius:10px;
  background:linear-gradient(135deg,#0C1446,#2B5C92 70%,#B3CDE0);box-shadow:inset 0 0 0 6px rgba(255,255,255,.18);}

/* ===== hero ===== */
.st-key-hero{position:relative;overflow:hidden;border:2px solid rgba(255,255,255,.6);border-radius:32px;padding:2.4rem 2.6rem;
  background:url("__SPARK_W__") 0 0 / 700px 700px, url("__CONTOUR_W__") no-repeat 62% 50% / 440px,
    linear-gradient(115deg,#0C1446 0%,#173A74 28%,#2B5C92 48%,#B3CDE0 76%,#FFFFFF 100%);
  box-shadow:0 30px 60px -32px rgba(12,20,70,.75);}
.st-key-hero::before{content:"";position:absolute;inset:0;z-index:0;pointer-events:none;
  background-image:linear-gradient(rgba(255,255,255,.15) 1px,transparent 1px),linear-gradient(90deg,rgba(255,255,255,.15) 1px,transparent 1px);
  background-size:34px 34px;-webkit-mask-image:linear-gradient(120deg,#000 0%,transparent 62%);mask-image:linear-gradient(120deg,#000 0%,transparent 62%);}
.st-key-hero [data-testid="stHorizontalBlock"]{position:relative;z-index:1;}
.eyebrow{color:#B3CDE0;font-weight:800;letter-spacing:.16em;font-size:.8rem;text-transform:uppercase;}
.hero-h{font-size:4.8rem;font-weight:800;letter-spacing:-.04em;line-height:1;margin:.3rem 0 .6rem;color:#fff;}
.hero-p{font-size:1.1rem;color:#E3EEF8;max-width:430px;margin:0 0 .9rem;line-height:1.55;}
.chips{display:flex;flex-wrap:wrap;gap:.4rem;margin-bottom:1.1rem;}
.chip2{background:rgba(255,255,255,.14);border:1.5px solid rgba(255,255,255,.38);color:#fff;font-weight:700;font-size:.82rem;padding:.25rem .8rem;border-radius:999px;}
.st-key-hero button[data-testid="stBaseButton-primary"]{background:#fff !important;color:var(--ink) !important;border:none !important;font-weight:800;
  box-shadow:0 12px 24px -12px rgba(0,0,0,.6);}
.sp{position:absolute;z-index:0;background:#fff;clip-path:polygon(50% 0,62% 38%,100% 50%,62% 62%,50% 100%,38% 62%,0 50%,38% 38%);animation:twinkle 3.2s ease-in-out infinite;pointer-events:none;}
.mock{display:flex;align-items:center;justify-content:center;gap:.4rem;padding:1.2rem 0;}
.doc{position:relative;width:44%;aspect-ratio:3/4;border-radius:18px;padding:1.4rem 1rem 1rem;display:flex;flex-direction:column;gap:.55rem;box-shadow:0 22px 38px -20px rgba(12,20,70,.7);}
.doc i{display:block;height:8px;border-radius:6px;}
.doc.b{background:#46527E;transform:rotate(-5deg);} .doc.b i{background:#9DB0D3;filter:blur(1.6px);opacity:.85;}
.doc.b::after{content:"";position:absolute;inset:0;border-radius:18px;background-image:radial-gradient(rgba(0,0,0,.4) 1px,transparent 1.2px);background-size:6px 6px;opacity:.5;}
.doc.a{background:#fff;border:2px solid var(--line);transform:rotate(4deg);} .doc.a i{background:#0C1446;} .doc.a i.h{height:11px;background:#2B5C92;}
.doc .lab{position:absolute;top:-12px;left:14px;font-size:.72rem;font-weight:800;padding:.15rem .7rem;border-radius:999px;color:#fff;z-index:2;}
.doc.b .lab{background:#0C1446;} .doc.a .lab{background:var(--pri);}
.mock .arrow{font-size:1.8rem;color:#fff;font-weight:800;text-shadow:0 2px 8px rgba(12,20,70,.5);}
.stats{display:flex;justify-content:center;flex-wrap:wrap;gap:.7rem;margin:1.2rem 0 .4rem;}
.stat{background:#fff;border:2px solid var(--line);border-radius:999px;padding:.4rem 1.1rem;font-weight:800;color:var(--pri);font-size:.92rem;}
.about{background:rgba(255,255,255,.88);border:2px solid var(--line);border-radius:24px;padding:1.4rem 1.6rem;margin:1.2rem 0 0;}
.about h3{margin:0 0 .4rem;font-size:1.3rem;} .about p{margin:0;color:#34456E;line-height:1.7;font-size:1rem;}
.how{text-align:center;font-size:1.5rem;font-weight:800;margin:1.8rem 0 .8rem;color:var(--ink);}

/* ===== kartu: depan = nomor + judul, klik = berbalik & deskripsi muncul (murni di browser) ===== */
.fcgrid{display:grid;grid-template-columns:repeat(4,1fr);gap:1rem;align-items:start;}
details.fc{background:rgba(255,255,255,.92);border:2px solid var(--line);border-radius:24px;padding:1.5rem 1rem 1.3rem;text-align:center;min-height:190px;
  transition:transform .2s ease,box-shadow .2s ease,border-color .2s ease;animation:flipfront .6s ease;}
details.fc summary{list-style:none;cursor:pointer;outline:none;display:block;}
details.fc summary::-webkit-details-marker{display:none;}
details.fc:hover{transform:translateY(-5px);box-shadow:0 22px 34px -22px rgba(12,20,70,.65);border-color:var(--pri2);}
details.fc .sn{width:66px;height:66px;border-radius:50%;margin:0 auto .8rem;display:flex;align-items:center;justify-content:center;font-size:1.9rem;font-weight:800;color:#fff;
  font-family:'Outfit',sans-serif;background:linear-gradient(135deg,#2B5C92,#0C1446);box-shadow:0 12px 24px -10px rgba(12,20,70,.7), inset 0 0 0 5px rgba(255,255,255,.14);}
details.fc .sn.logo{border-radius:22px;background:linear-gradient(135deg,#0C1446,#2B5C92 70%,#B3CDE0);}
details.fc .stt{font-family:'Outfit',sans-serif;font-weight:800;font-size:1.12rem;color:var(--ink);line-height:1.25;}
details.fc .hint{margin-top:.7rem;font-size:.78rem;font-weight:700;color:var(--pri2);letter-spacing:.04em;}
details.fc .sd{margin-top:.8rem;color:#E3EEF8;font-size:.93rem;line-height:1.6;}
details.fc[open]{background:linear-gradient(150deg,#0C1446 0%,#1F4A80 55%,#2B5C92 100%);border-color:#2B5C92;animation:flipback .6s ease;
  box-shadow:0 22px 34px -22px rgba(12,20,70,.85);}
details.fc[open] .sn{width:46px;height:46px;font-size:1.25rem;margin-bottom:.5rem;background:rgba(255,255,255,.16);box-shadow:none;}
details.fc[open] .sn.logo{border-radius:16px;}
details.fc[open] .stt{color:#fff;font-size:1.05rem;}
details.fc[open] .hint{display:none;}
details.fc.wide{min-height:150px;} details.fc.wide .sd{max-width:780px;margin-left:auto;margin-right:auto;}
@keyframes flipfront{from{transform:perspective(900px) rotateY(-90deg);opacity:0;}to{transform:perspective(900px) rotateY(0);opacity:1;}}
@keyframes flipback{from{transform:perspective(900px) rotateY(90deg);opacity:0;}to{transform:perspective(900px) rotateY(0);opacity:1;}}
@media(max-width:900px){.fcgrid{grid-template-columns:1fr 1fr;}}

/* ===== band penutup ===== */
.st-key-band{position:relative;overflow:hidden;background:url("__SPARK_W__") 0 0 / 700px 700px, linear-gradient(120deg,#0C1446 0%,#2B5C92 62%,#B3CDE0 100%);
  border-radius:28px;padding:2.2rem 1.5rem 1.8rem;text-align:center;margin-top:1.8rem;box-shadow:0 24px 50px -30px rgba(12,20,70,.8);}
.band-t{font-size:1.8rem;font-weight:800;color:#fff;margin-bottom:.9rem;}
.st-key-band button[data-testid="stBaseButton-primary"]{background:#fff !important;color:var(--ink) !important;border:none !important;font-weight:800;}

/* ===== halaman perbaiki ===== */
.stepper{display:grid;grid-template-columns:repeat(4,1fr);margin:.4rem 0 .9rem;}
.stp{text-align:center;position:relative;}
.stp .c{width:36px;height:36px;border-radius:50%;margin:0 auto .3rem;display:flex;align-items:center;justify-content:center;font-weight:800;background:#fff;border:2px solid var(--line);color:#9FB0C9;position:relative;z-index:1;}
.stp:not(:last-child)::after{content:"";position:absolute;top:17px;left:calc(50% + 24px);width:calc(100% - 48px);height:2px;background:var(--line);}
.stp.done .c{background:var(--pri2);border-color:var(--pri2);color:#fff;} .stp.done:not(:last-child)::after{background:var(--pri2);}
.stp.active .c{background:var(--pri);border-color:var(--pri);color:#fff;box-shadow:0 0 0 6px rgba(43,92,146,.18);}
.stp .t{font-size:.86rem;font-weight:700;color:#9FB0C9;} .stp.active .t{color:var(--ink);}
.guide{background:#fff;border:2px solid var(--line);border-radius:16px;padding:.8rem 1.1rem;display:flex;gap:.8rem;align-items:center;margin-bottom:.4rem;}
.guide .tag{background:var(--pri);color:#fff;border-radius:999px;padding:.2rem .8rem;font-size:.78rem;font-weight:800;white-space:nowrap;}
.guide .tx{font-weight:700;color:var(--ink);}
.sec{display:flex;gap:.8rem;align-items:center;margin:2rem 0 .7rem;}
.sec .badge{flex:none;width:34px;height:34px;border-radius:50%;background:linear-gradient(135deg,#2B5C92,#0C1446);color:#fff;display:flex;align-items:center;justify-content:center;font-weight:800;}
.sec h2{margin:0;padding:0;font-size:1.3rem;}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:.7rem;margin:.4rem 0;}
.tile{background:rgba(255,255,255,.92);border:2px solid var(--line);border-radius:16px;padding:.7rem .9rem;}
.tile .l{color:var(--muted);font-size:.78rem;font-weight:600;} .tile .v{font-size:1.5rem;font-weight:700;color:var(--ink);line-height:1.3;}
.chip{display:inline-block;font-size:.72rem;font-weight:800;padding:.1rem .6rem;border-radius:999px;}
.chip.ok{background:var(--green-l);color:var(--green);} .chip.warn{background:var(--amber-l);color:var(--amber);} .chip.info{background:var(--soft);color:var(--pri);}
.todo{background:#fff;border:2px solid var(--line);border-left:8px solid var(--pri);border-radius:16px;padding:1rem 1.2rem;margin-bottom:.6rem;}
.todo .k{color:var(--pri);font-size:.75rem;font-weight:800;letter-spacing:.1em;text-transform:uppercase;}
.todo h3{margin:.15rem 0 .2rem;font-size:1.25rem;} .todo p{margin:.1rem 0;color:#4A5A80;font-size:.93rem;}
.todo .rec{margin-top:.6rem;padding:.5rem .8rem;background:var(--soft);border-radius:10px;color:var(--ink);font-size:.95rem;}
.notice{background:#FFF8EC;border:2px solid #F3D9A4;border-left:8px solid #E9A23B;border-radius:16px;padding:1rem 1.2rem;margin:.4rem 0 .8rem;}
.notice .k{color:#B45309;font-size:.75rem;font-weight:800;letter-spacing:.1em;text-transform:uppercase;}
.notice h3{margin:.15rem 0 .3rem;font-size:1.25rem;} .notice p{margin:.15rem 0;color:#5F4B2A;font-size:.95rem;}
.edit-h{color:var(--pri);font-size:.78rem;font-weight:800;letter-spacing:.1em;text-transform:uppercase;}
.muted{color:var(--muted);font-size:.95rem;}
.pg{font-size:1.8rem;font-weight:800;margin:1rem 0 .6rem;color:var(--ink);}
.m-tag{display:inline-block;background:var(--soft);color:var(--pri);font-weight:800;font-size:.72rem;padding:.15rem .7rem;border-radius:999px;}
.m-name{font-size:1.3rem;font-weight:800;color:var(--ink);margin:.4rem 0 .2rem;}
.m-lab{color:var(--pri);font-weight:800;font-size:.74rem;letter-spacing:.08em;text-transform:uppercase;margin-top:.7rem;}
.m-txt{color:#34456E;font-size:.93rem;line-height:1.55;margin:.1rem 0 0;}
[class*="st-key-mc_"]{animation:flipin .5s ease;background:rgba(255,255,255,.92);border-radius:22px !important;}
.stTabs [data-baseweb="tab"]{font-weight:800;} .stTabs [data-baseweb="tab-list"]{gap:.4rem;}
.stButton>button,.stDownloadButton>button{border-radius:999px;font-weight:700;}

/* ===== gerak halus ===== */
@keyframes rise{from{opacity:0;transform:translateY(18px);}to{opacity:1;transform:none;}}
@keyframes flipin{from{transform:perspective(900px) rotateY(88deg);opacity:0;}to{transform:perspective(900px) rotateY(0);opacity:1;}}
@keyframes floata{0%,100%{transform:translateY(0) rotate(-5deg);}50%{transform:translateY(-10px) rotate(-3deg);}}
@keyframes floatb{0%,100%{transform:translateY(0) rotate(4deg);}50%{transform:translateY(8px) rotate(2deg);}}
@keyframes scan{0%{top:-20%;}100%{top:105%;}}
@keyframes twinkle{0%,100%{transform:scale(.6);opacity:.35;}50%{transform:scale(1.15);opacity:1;}}
.eyebrow{animation:rise .6s ease both;} .hero-h{animation:rise .7s .08s ease both;} .hero-p{animation:rise .7s .16s ease both;} .chips{animation:rise .7s .24s ease both;}
.doc.b{animation:floata 6s ease-in-out infinite;} .doc.a{animation:floatb 7s ease-in-out infinite;}
.scanwrap{position:absolute;inset:0;border-radius:18px;overflow:hidden;pointer-events:none;z-index:1;}
.scanwrap::before{content:"";position:absolute;left:0;right:0;height:34px;background:linear-gradient(transparent,rgba(179,205,224,.9),transparent);animation:scan 2.8s ease-in-out infinite;}
.stats,.about,.how{animation:rise .7s ease both;}
@media (prefers-reduced-motion: reduce){*{animation:none !important;transition:none !important;}}
@media(max-width:800px){.hero-h{font-size:3.2rem;}.st-key-hero{padding:1.6rem 1.2rem;}}
</style>
"""
CSS = (CSS.replace("__CONTOUR_L__", contour_svg("#2B5C92", .17)).replace("__CONTOUR_W__", contour_svg("#FFFFFF", .32))
          .replace("__SPARK_L__", sparkle_svg("#2B5C92", 18, 900, 7, .5)).replace("__SPARK_W__", sparkle_svg("#FFFFFF", 22, 700, 3, .85)))
st.markdown(CSS, unsafe_allow_html=True)


def section(num, title):
    st.markdown(f'<div class="sec"><div class="badge">{num}</div><h2>{title}</h2></div>', unsafe_allow_html=True)


def tile(label, value, chip):
    return (f'<div class="tile"><div class="l">{label}</div><div class="v">{value}</div>'
            f'<span class="chip {chip[1]}">{chip[0]}</span></div>')


def hist_fig(g, title, color):
    fig, ax = plt.subplots(figsize=(4, 2.2))
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    ax.hist(g.ravel(), bins=256, range=(0, 255), color=color)
    ax.set_title(title, fontsize=9)
    ax.set_xlim(0, 255)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    return fig


def guide(stage, text):
    items = ""
    for i in range(1, 5):
        cls = "done" if i < stage else "active" if i == stage else ""
        items += f'<div class="stp {cls}"><div class="c">{"✓" if i < stage else i}</div><div class="t">{t(f"s{i}")}</div></div>'
    return (f'<div class="stepper">{items}</div><div class="guide"><span class="tag">{t("step")} {stage} {t("of")} 4</span>'
            f'<span class="tx">{text}</span></div>')


# ============================================================ PDF
PDF_DPI = 150
MAX_PAGES = 30


@st.cache_data(show_spinner=False, max_entries=4)
def pdf_page_count(raw):
    doc = fitz.open(stream=raw, filetype="pdf")
    n = len(doc)
    doc.close()
    return n


def _render(doc, idx, dpi=PDF_DPI):
    pix = doc[idx].get_pixmap(dpi=dpi, colorspace=fitz.csRGB, alpha=False)
    arr = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, 3)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


@st.cache_data(show_spinner=False, max_entries=8)
def render_pdf_page(raw, idx):
    doc = fitz.open(stream=raw, filetype="pdf")
    img_ = _render(doc, idx)
    doc.close()
    return img_


def process_all_pdf(raw, steps):
    """Terapkan langkah yang sama ke semua halaman, lalu susun menjadi PDF baru."""
    src = fitz.open(stream=raw, filetype="pdf")
    out = fitz.open()
    for i in range(min(len(src), MAX_PAGES)):
        g = to_grayscale(_render(src, i))
        for st_ in steps:
            g = blend(g, apply_method(st_["method"], g, st_["defaults"]), st_.get("alpha", 100) / 100)
        ok_, png = cv2.imencode(".png", g)
        rect = src[i].rect
        page = out.new_page(width=rect.width, height=rect.height)
        page.insert_image(page.rect, stream=png.tobytes())
    data = out.tobytes()
    out.close()
    src.close()
    return data


# ============================================================ STATE
ss.setdefault("steps", [])
ss.setdefault("next_id", 1)
ss.setdefault("dismissed", set())
ss.setdefault("attempted", set())
ss.setdefault("file_sig", None)
ss.setdefault("raw", None)
ss.setdefault("fname", None)
ss.setdefault("th_saved", dict(DEFAULT_TH))
ss.setdefault("page", "home")
ss.setdefault("flipped", set())
ss.setdefault("pdf_page", 1)
ss.setdefault("pdf_out", None)
ss.setdefault("force_sig", None)
ss.setdefault("open_step", None)
ss.setdefault("open_about", False)


def add_step(method, defaults=None, issue_key=None):
    for s in ss.steps:
        s["confirmed"] = True
    d = default_params(method)
    d.update(defaults or {})
    ss.steps.append({"id": ss.next_id, "method": method, "defaults": d, "alpha": 100, "confirmed": False})
    ss.next_id += 1
    if issue_key:
        ss.attempted.add(issue_key)


def confirm_step(step_id):
    for s in ss.steps:
        if s["id"] == step_id:
            s["confirmed"] = True


def remove_step(step_id):
    ss.steps = [s for s in ss.steps if s["id"] != step_id]


def dismiss(key):
    ss.dismissed.add(key)


def reset_all():
    ss.steps, ss.dismissed, ss.attempted = [], set(), set()


def restore_skipped():
    ss.dismissed = set()


def clear_photo():
    ss.raw, ss.fname, ss.file_sig = None, None, None
    reset_all()


def force_analyze(sig_):
    ss.force_sig = sig_


def toggle_about():
    ss.open_about = not ss.open_about


def toggle_step(i):
    ss.open_step = None if ss.open_step == i else i


def go(page):
    ss.page = page


def toggle_flip(name):
    if name in ss.flipped:
        ss.flipped.discard(name)
    else:
        ss.flipped.add(name)


def set_lang(lang):
    ss.lang = lang


def get_th():
    return {k: ss.get(f"th_{k}", ss.th_saved[k]) for k in DEFAULT_TH}


def stop_page():
    st.stop()


# ============================================================ TOOLBAR (satu pill: logo + menu + bahasa)
with st.container(key="nav"):
    cols = st.columns([1.35, 1, 1.35, 1, 0.85], vertical_alignment="center")
    cols[0].button("Pixora", key="nav_brand", width="stretch", on_click=go, args=("home",))
    for col, pk in zip(cols[1:4], ["home", "fix", "methods"]):
        col.button(t(f"nav_{pk}"), key=f"nav_{pk}", width="stretch", on_click=go, args=(pk,),
                   type="primary" if ss.page == pk else "secondary")
    try:
        pop = cols[4].popover(ss.lang.upper(), icon=":material/language:", width="stretch")
    except TypeError:
        pop = cols[4].popover("🌐 " + ss.lang.upper(), width="stretch")
    with pop:
        st.markdown(f"**{t('lang_t')}**")
        for lg, name in (("id", "Indonesia"), ("en", "English")):
            st.button(name, key=f"lang_{lg}", width="stretch", on_click=set_lang, args=(lg,),
                      type="primary" if ss.lang == lg else "secondary")
page = ss.page

# ============================================================ BERANDA
if page == "home":
    with st.container(key="hero"):
        _sp = [(5, 10, 0, 12), (46, 7, .8, 9), (53, 72, 1.4, 14), (91, 12, .4, 10), (95, 66, 1.9, 12), (30, 88, 1.1, 8), (72, 42, 2.3, 9), (14, 52, 1.6, 7)]
        st.markdown("".join(f'<i class="sp" style="left:{x}%;top:{y}%;animation-delay:{d}s;width:{z}px;height:{z}px"></i>'
                            for x, y, d, z in _sp), unsafe_allow_html=True)
        left, right = st.columns([1.1, 1], gap="large", vertical_alignment="center")
        chips = "".join(f'<span class="chip2">{t(k)}</span>' for k in ("chip1", "chip2", "chip3", "chip4"))
        left.markdown(
            f'<div class="eyebrow">Analyze · Recommend · Enhance</div><div class="hero-h">Pixora</div>'
            f'<p class="hero-p">{t("hero_sub")}</p><div class="chips">{chips}</div>', unsafe_allow_html=True)
        left.button(t("cta"), type="primary", on_click=go, args=("fix",))
        w = [92, 70, 86, 58, 90, 66, 80, 48]
        lines_b = "".join(f'<i style="width:{x}%"></i>' for x in w)
        lines_a = '<i class="h" style="width:55%"></i>' + "".join(f'<i style="width:{x}%"></i>' for x in w)
        right.markdown(
            f'<div class="mock"><div class="doc b"><span class="lab">{t("before")}</span><div class="scanwrap"></div>{lines_b}</div>'
            f'<div class="arrow">→</div><div class="doc a"><span class="lab">{t("after")}</span>{lines_a}</div></div>',
            unsafe_allow_html=True)
    st.markdown(f'<div class="stats"><span class="stat">{t("stat1")}</span><span class="stat">{t("stat2")}</span>'
                f'<span class="stat">{t("stat3")}</span></div>', unsafe_allow_html=True)
    st.markdown("&nbsp;")
    st.markdown(f'<div class="about"><h3>{t("about_t")}</h3><p>{t("about_d")}</p></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="how">{t("steps_t")}</div>', unsafe_allow_html=True)
    fc_html = "".join(
        f'<details class="fc"><summary><div class="sn">{i}</div><div class="stt">{t(f"t{i}")}</div>'
        f'<div class="hint">{t("click_hint")}</div></summary><div class="sd">{t(f"t{i}d")}</div></details>'
        for i in range(1, 5))
    st.markdown(f'<div class="fcgrid">{fc_html}</div>', unsafe_allow_html=True)
    with st.container(key="band"):
        st.markdown(f'<div class="band-t">{t("band")}</div>', unsafe_allow_html=True)
        _, mid, _ = st.columns([1, 1, 1])
        mid.button(t("cta"), key="cta2", type="primary", width="stretch", on_click=go, args=("fix",))
    stop_page()

# ============================================================ METODE
if page == "methods":
    st.markdown(f'<div class="pg">{t("methods_title")}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="muted">{t("m_hint")}</div>', unsafe_allow_html=True)
    st.markdown("&nbsp;")
    for r in range(0, len(METHOD_INFO), 2):
        for col, info in zip(st.columns(2, gap="medium"), METHOD_INFO[r:r + 2]):
            name = info["name"]
            back = name in ss.flipped
            with col.container(border=True, key=f"mc_{name.replace(' ', '')}_{'b' if back else 'f'}"):
                st.markdown(f'<span class="m-tag">{tr(info["group"])}</span><div class="m-name">{name}</div>',
                            unsafe_allow_html=True)
                if not back:
                    st.latex(info["latex"])
                else:
                    st.markdown(
                        f'<div class="m-lab">{t("m_sym")}</div><p class="m-txt">{tr(info["sym"])}</p>'
                        f'<div class="m-lab">{t("m_how")}</div><p class="m-txt">{tr(info["how"])}</p>'
                        f'<div class="m-lab">{t("m_use")}</div><p class="m-txt">{tr(info["use"])}</p>',
                        unsafe_allow_html=True)
                st.button(t("m_back") if back else t("m_flip"), key=f"flip_{name}", on_click=toggle_flip,
                          args=(name,), width="stretch")
    stop_page()

# ============================================================ PERBAIKI FOTO
guide_box = st.container()
upload_box = st.container()
cond_box = st.container()
work_box = st.container()
res_box = st.container()
adv_box = st.container()

with upload_box:
    section(1, t("s1"))
    st.markdown(f'<div class="muted">{t("up_hint")}</div>', unsafe_allow_html=True)
    file = st.file_uploader(t("uploader"), type=["jpg", "jpeg", "png", "bmp", "pdf"], label_visibility="collapsed")
    if file is not None:
        new_raw = file.getvalue()
        if (file.name, len(new_raw)) != (ss.fname, len(ss.raw or b"")):
            ss.pdf_page = 1
        ss.raw, ss.fname = new_raw, file.name
    elif ss.raw is not None:
        c1, c2 = st.columns([3, 1])
        c1.markdown(f'<span class="muted">{t("photo")}: <b>{ss.fname}</b></span>', unsafe_allow_html=True)
        c2.button(t("change"), on_click=clear_photo)

if ss.raw is None:
    guide_box.markdown(guide(1, t("msg_upload")), unsafe_allow_html=True)
    stop_page()

is_pdf = ss.raw[:4] == b"%PDF"
n_pages = 1
if is_pdf:
    if not HAS_PDF:
        with upload_box:
            st.error(t("pdf_missing"))
        clear_photo()
        stop_page()
    try:
        n_pages = pdf_page_count(ss.raw)
    except Exception:
        with upload_box:
            st.error(t("pdf_bad"))
        clear_photo()
        stop_page()
    ss.pdf_page = min(max(int(ss.pdf_page), 1), n_pages)
    if n_pages > 1:
        with upload_box:
            pc1, pc2 = st.columns([1, 3], vertical_alignment="center")
            ss.pdf_page = int(pc1.number_input(t("pdf_page"), 1, n_pages, ss.pdf_page, key="pdf_page_w"))
            pc2.markdown(f'<span class="muted">{t("pdf_of")} {n_pages}</span>', unsafe_allow_html=True)
    try:
        img = render_pdf_page(ss.raw, ss.pdf_page - 1)
    except Exception:
        with upload_box:
            st.error(t("pdf_bad"))
        clear_photo()
        stop_page()
else:
    img = cv2.imdecode(np.frombuffer(ss.raw, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        with upload_box:
            st.error(t("bad_img"))
        clear_photo()
        stop_page()

sig = (ss.fname, len(ss.raw), ss.pdf_page if is_pdf else 0)
if ss.file_sig != sig:
    ss.file_sig = sig
    reset_all()

gray = to_grayscale(img)

# ---- pemeriksaan: apakah file ini layak dianalisis sebagai dokumen?
chk = check_document(gray)
if not chk["ok"] and ss.force_sig != sig:
    guide_box.markdown(guide(1, t("msg_upload")), unsafe_allow_html=True)
    with cond_box:
        st.markdown(
            f'<div class="notice"><div class="k">{t("found")}</div><h3>{t("nd_title")}</h3>'
            f'<p>{t("nd_" + chk["reason"])}</p><p>{t("nd_hint")}</p></div>', unsafe_allow_html=True)
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), width=260)
        n1, n2, _ = st.columns([1, 1, 2])
        n1.button(t("nd_force"), type="primary", on_click=force_analyze, args=(sig,), width="stretch")
        n2.button(t("change"), key="nd_change", on_click=clear_photo, width="stretch")
    stop_page()

th = get_th()
m0 = analyze(gray)


def status(m):
    br = (t("too_dark"), "warn") if m["brightness"] < th["dark"] else \
         (t("too_bright"), "warn") if m["brightness"] > th["bright"] else (t("normal"), "ok")
    ct = (t("low"), "warn") if (m["contrast"] < th["contrast"] or m["hist_range"] < th["range"]) else (t("enough"), "ok")
    nz = (t("speckled"), "warn") if m["noise_ratio"] > th["noise"] else (t("clean"), "ok")
    sp = (t("blurry"), "warn") if m["sharpness"] < th["sharp"] else (t("crisp"), "ok")
    return br, ct, nz, sp


# ---- 2. Analisis: empat indikator dalam satu baris
with cond_box:
    section(2, t("s2"))
    br, ct, nz, sp = status(m0)
    st.markdown(
        '<div class="tiles">'
        + tile(t("bright"), f"{m0['brightness']:.0f}", br) + tile(t("contrast"), f"{m0['contrast']:.0f}", ct)
        + tile(t("noise"), f"{m0['noise_ratio'] * 100:.2f}%", nz) + tile(t("sharp"), f"{m0['sharpness']:.0f}", sp)
        + '</div>', unsafe_allow_html=True)

# ---- 3. Perbaiki: kiri = pratinjau langsung, kanan = yang harus dilakukan
with work_box:
    section(3, t("s3"))
    left, right = st.columns([1.05, 1], gap="large")
right_top = right.container()
right_hist = right.container()

if any(s_["confirmed"] for s_ in ss.steps):
    with right_hist:
        st.markdown(f"#### {t('history')}")

current = gray
unconfirmed_id = None
for i, s_ in enumerate(ss.steps):
    sid, name, d, done = s_["id"], s_["method"], s_["defaults"], s_["confirmed"]
    before = current
    if done:
        ctx = right_hist.expander(f"{i + 1}. {name}", expanded=False)
    else:
        unconfirmed_id = sid
        ctx = right_top.container(border=True)
    with ctx:
        if not done:
            st.markdown(f'<div class="edit-h">{t("tune")} · {name}</div>', unsafe_allow_html=True)
        p = {}
        if name == "Gamma Correction":
            p["gamma"] = st.slider(t("gamma"), 0.1, 3.0, float(d["gamma"]), 0.05, key=f"g{sid}")
        elif name == "Contrast Stretching":
            p["p_low"] = st.slider(t("black"), 0.0, 30.0, float(d["p_low"]), 0.5, key=f"pl{sid}")
            p["p_high"] = st.slider(t("white"), 70.0, 100.0, float(d["p_high"]), 0.5, key=f"ph{sid}")
        elif name in ("Median Filter", "Average Filter"):
            p["k"] = st.select_slider(t("window"), [3, 5, 7, 9], int(d["k"]), key=f"k{sid}")
        elif name == "Laplacian Filter":
            p["c"] = st.slider(t("weight"), 0.1, 3.0, float(d["c"]), 0.1, key=f"c{sid}")
        alpha = st.slider(t("intensity"), 0, 100, int(s_.get("alpha", 100)), 5, key=f"a{sid}") / 100
        s_["alpha"] = int(alpha * 100)
        d.update(p)
        current = blend(before, apply_method(name, before, p), alpha)
        if not done:
            b1, b2 = st.columns(2)
            b1.button(t("save"), key=f"ok{sid}", type="primary", on_click=confirm_step, args=(sid,), width="stretch")
            b2.button(t("cancel"), key=f"cancel{sid}", on_click=remove_step, args=(sid,), width="stretch")
        else:
            st.button(t("delete"), key=f"del{sid}", on_click=remove_step, args=(sid,))

m1 = analyze(current)
try:
    _found = find_issues(m1, th, ss.lang)
except TypeError:            # analysis.py versi lama (belum dwibahasa)
    _found = find_issues(m1, th)
issues = [x for x in _found if x["key"] not in ss.dismissed]

with right_top:
    if unconfirmed_id is None:
        if issues:
            it = issues[0]
            note = f"<p>{t('still')}</p>" if it["key"] in ss.attempted else ""
            saran = " / ".join(f"<b>{m_}</b> ({tr(SHORT[m_])})" for m_ in it["methods"])
            st.markdown(
                f'<div class="todo"><div class="k">{t("found")}</div><h3>{it["title"]}</h3>'
                f'<p>{it["detail"]}</p>{note}<div class="rec">{t("suggest")}: {saran}</div></div>',
                unsafe_allow_html=True)
            for meth in it["methods"]:
                st.button(f"{t('apply')} {meth}", key=f"fix_{it['key']}_{meth}", type="primary",
                          on_click=add_step, args=(meth, it["defaults"], it["key"]), width="stretch")
            st.button(t("skip"), key=f"skip_{it['key']}", on_click=dismiss, args=(it["key"],), width="stretch")
            if len(issues) > 1:
                st.caption(f"{len(issues) - 1} {t('more_issues')}")
        else:
            st.success(t("no_issue"))
        if ss.dismissed:
            st.button(t("show_skipped"), on_click=restore_skipped)

with left:
    tab_after, tab_before = st.tabs([t("after"), t("before")])
    tab_after.image(current, width="stretch", clamp=True)
    tab_before.image(gray, width="stretch", clamp=True)

# ---- 4. Hasil
with res_box:
    section(4, t("s4"))
    if not ss.steps:
        st.markdown(f'<div class="muted">{t("res_hint")}</div>', unsafe_allow_html=True)
    else:
        kb, kc, kn, ks = status(m1)
        o = t("orig")
        st.markdown(
            '<div class="tiles">'
            + tile(f"{t('bright')} ({o} {m0['brightness']:.0f})", f"{m1['brightness']:.0f}", kb)
            + tile(f"{t('contrast')} ({o} {m0['contrast']:.0f})", f"{m1['contrast']:.0f}", kc)
            + tile(f"{t('noise')} ({o} {m0['noise_ratio'] * 100:.2f}%)", f"{m1['noise_ratio'] * 100:.2f}%", kn)
            + tile(f"{t('sharp')} ({o} {m0['sharpness']:.0f})", f"{m1['sharpness']:.0f}", ks)
            + '</div>', unsafe_allow_html=True)
        with st.expander(t("hist")):
            h1, h2 = st.columns(2)
            h1.pyplot(hist_fig(gray, t("before"), "#B3CDE0"))
            h2.pyplot(hist_fig(current, t("after"), "#2B5C92"))
        ok, buf = cv2.imencode(".png", current)
        if ok:
            st.download_button(t("dl_page") if is_pdf else t("download"), buf.tobytes(),
                               "pixora_hasil.png", "image/png", type="primary")
        if is_pdf:
            st.markdown("&nbsp;")
            st.caption(t("pdf_note"))
            steps_sig = (ss.fname, repr([(x["method"], sorted(x["defaults"].items()), x.get("alpha", 100)) for x in ss.steps]))
            if st.button(t("pdf_all")):
                with st.spinner(t("pdf_work")):
                    ss.pdf_out = (steps_sig, process_all_pdf(ss.raw, ss.steps))
            if ss.pdf_out and ss.pdf_out[0] == steps_sig:
                st.download_button(t("pdf_dl"), ss.pdf_out[1], "pixora_hasil.pdf", "application/pdf")
            if n_pages > MAX_PAGES:
                st.caption(t("pdf_cap"))

with adv_box:
    st.markdown("&nbsp;")
    with st.expander(t("adv")):
        t1, t2, t3 = st.tabs([t("tab1"), t("tab2"), t("tab3")])
        with t1:
            man = st.selectbox(t("method"), METHODS, key="manual_sel")
            st.button(t("add"), on_click=add_step, args=(man,))
        with t2:
            S = ss.th_saved
            st.slider(t("th_dark"), 0, 255, int(S["dark"]), key="th_dark")
            st.slider(t("th_bright"), 0, 255, int(S["bright"]), key="th_bright")
            st.slider(t("th_contrast"), 0, 100, int(S["contrast"]), key="th_contrast")
            st.slider(t("th_range"), 0, 255, int(S["range"]), key="th_range")
            st.slider(t("th_noise"), 0.0, 0.1, float(S["noise"]), 0.001, format="%.3f", key="th_noise")
            st.slider(t("th_sharp"), 0, 1000, int(S["sharp"]), key="th_sharp")
            for k in DEFAULT_TH:
                S[k] = ss[f"th_{k}"]
        with t3:
            st.button(t("reset"), on_click=reset_all)

if unconfirmed_id is not None:
    stage, msg = 3, t("msg_edit")
elif issues:
    stage, msg = (2 if not ss.steps else 3), t("msg_pick")
else:
    stage, msg = 4, t("msg_done")
guide_box.markdown(guide(stage, msg), unsafe_allow_html=True)