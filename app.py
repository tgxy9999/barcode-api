from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64
import zxingcpp
from PIL import Image

app = Flask(__name__)
CORS(app)

MAX_DIM = 1200

# ── Initialize OpenCV Barcode Detector once at startup ──
try:
    cv2_detector = cv2.barcode.BarcodeDetector()
    CV2_AVAILABLE = True
except Exception:
    cv2_detector = None
    CV2_AVAILABLE = False


# ════════════════════════════════════════════════════════
# IMAGE UTILITIES
# ════════════════════════════════════════════════════════

def decode_base64_image(b64_string: str) -> np.ndarray:
    if ',' in b64_string:
        b64_string = b64_string.split(',')[1]
    img_bytes = base64.b64decode(b64_string.strip())
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    return cv2.imdecode(img_array, cv2.IMREAD_COLOR)


def resize_if_large(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    if max(h, w) > MAX_DIM:
        scale = MAX_DIM / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)),
                         interpolation=cv2.INTER_AREA)
    return img


def apply_gamma(gray: np.ndarray, gamma: float) -> np.ndarray:
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in range(256)], dtype=np.uint8)
    return cv2.LUT(gray, table)


def deskew(gray: np.ndarray) -> np.ndarray:
    try:
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)
        if lines is None:
            return gray
        angles = [((l[0][1] * 180 / np.pi) - 90)
                  for l in lines[:20]
                  if -45 < (l[0][1] * 180 / np.pi) - 90 < 45]
        if not angles or abs(float(np.median(angles))) < 0.5:
            return gray
        h, w = gray.shape
        M = cv2.getRotationMatrix2D((w // 2, h // 2),
                                    float(np.median(angles)), 1.0)
        return cv2.warpAffine(gray, M, (w, h),
                              flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE)
    except Exception:
        return gray


# ════════════════════════════════════════════════════════
# DECODERS
# ════════════════════════════════════════════════════════

def try_opencv(img: np.ndarray) -> dict | None:
    """OpenCV ML-based barcode detector."""
    if not CV2_AVAILABLE:
        return None
    try:
        ret, decoded_list, decoded_types, _ = cv2_detector.detectAndDecodeMulti(img)
        if ret and decoded_list:
            for i, val in enumerate(decoded_list):
                if val and val.strip():
                    fmt = decoded_types[i] if decoded_types else "UNKNOWN"
                    return {"value": val.strip(), "format": str(fmt)}
    except Exception:
        pass
    return None


def try_zxing(img: np.ndarray) -> dict | None:
    """zxing-cpp decoder."""
    try:
        pil = Image.fromarray(img if len(img.shape) == 2
                              else cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        results = zxingcpp.read_barcodes(pil)
        if results:
            return {"value": results[0].text, "format": str(results[0].format)}
    except Exception:
        pass
    return None


def try_decoders(img: np.ndarray, variant: str, phase: str) -> dict | None:
    """Try both decoders on one image. Return full result dict or None."""
    for decoder_fn, decoder_name in [
        (try_opencv, "opencv_barcode_detector"),
        (try_zxing,  "zxing_cpp")
    ]:
        result = decoder_fn(img)
        if result:
            return {
                "isReadable":     True,
                "barcodeValue":   result["value"],
                "barcodeFormat":  result["format"],
                "successVariant": variant,
                "decoder":        decoder_name,
                "phase":          phase,
            }
    return None


# ════════════════════════════════════════════════════════
# PHASE 1 — CONTRAST NORMALISATION
# Make dark colours → black, light colours → white
# This is the primary fix for coloured backgrounds
# ════════════════════════════════════════════════════════

def phase1_contrast_variants(img: np.ndarray, gray: np.ndarray,
                              clahe) -> list:
    """
    Variants that push dark → black, light → white.
    Designed to normalise coloured backgrounds before decoding.
    """
    variants = []

    # Raw grayscale (baseline — dark already dark, light already light)
    variants.append(("p1_grayscale",    gray))

    # CLAHE — adaptive contrast, pulls dark bars darker, light bg lighter
    clahe_img = clahe.apply(gray)
    variants.append(("p1_clahe",        clahe_img))

    # Otsu — finds optimal global threshold, black/white split
    _, otsu = cv2.threshold(gray, 0, 255,
                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(("p1_otsu",         otsu))

    # CLAHE + Otsu — best combo for coloured backgrounds
    _, clahe_otsu = cv2.threshold(clahe_img, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(("p1_clahe_otsu",   clahe_otsu))

    # Adaptive threshold — handles uneven lighting across label
    adaptive = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2)
    variants.append(("p1_adaptive",     adaptive))

    # CLAHE + Adaptive
    clahe_adaptive = cv2.adaptiveThreshold(
        clahe_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2)
    variants.append(("p1_clahe_adaptive", clahe_adaptive))

    # Gamma brighten (1.5, 2.0) → makes dark bg lighter, improves contrast
    for gval in [1.5, 2.0]:
        g = apply_gamma(gray, gval)
        g_clahe = clahe.apply(g)
        _, g_otsu = cv2.threshold(g_clahe, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants.append((f"p1_gamma_{gval}_clahe_otsu", g_otsu))

    # Gamma darken (0.7) → makes light bg darker for high-key images
    g_dark = apply_gamma(gray, 0.7)
    _, g_dark_otsu = cv2.threshold(g_dark, 0, 255,
                                   cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(("p1_gamma_0.7_otsu", g_dark_otsu))

    # Bilateral + CLAHE + Otsu — smooth noise while keeping bar edges
    bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
    bil_clahe  = clahe.apply(bilateral)
    _, bil_otsu = cv2.threshold(bil_clahe, 0, 255,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(("p1_bilateral_clahe_otsu", bil_otsu))

    # Original color image (OpenCV detector works well on color)
    variants.append(("p1_original_color", img))

    return variants


# ════════════════════════════════════════════════════════
# PHASE 2 — DEEP PREPROCESSING
# More aggressive techniques for difficult cases
# ════════════════════════════════════════════════════════

def phase2_preprocessing_variants(img: np.ndarray, gray: np.ndarray,
                                   clahe) -> list:
    """
    Advanced preprocessing for barcodes that Phase 1 couldn't read.
    """
    variants = []
    clahe_img = clahe.apply(gray)
    _, otsu = cv2.threshold(gray, 0, 255,
                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, clahe_otsu = cv2.threshold(clahe_img, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kh = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
    kc = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
    ko = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))

    # Morphological erode — thins bleeding/overburnt bars
    eroded = cv2.erode(otsu, kh, iterations=1)
    variants.append(("p2_morph_erode",           eroded))

    # CLAHE + Otsu + Erode — best for overburnt coloured labels
    clahe_eroded = cv2.erode(clahe_otsu, kh, iterations=1)
    variants.append(("p2_clahe_otsu_erode",      clahe_eroded))

    # Morphological close — connects broken bars
    closed = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, kc)
    variants.append(("p2_morph_close",           closed))

    # Morphological open — removes noise blobs between bars
    opened = cv2.morphologyEx(otsu, cv2.MORPH_OPEN, ko)
    variants.append(("p2_morph_open",            opened))

    # Sharpening — recovers soft bar edges
    k_sharp = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(gray, -1, k_sharp)
    _, sharp_otsu = cv2.threshold(sharpened, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(("p2_sharpened_otsu",        sharp_otsu))

    # Aggressive sharpen
    k_sharp2 = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharp2 = cv2.filter2D(gray, -1, k_sharp2)
    _, sharp2_otsu = cv2.threshold(sharp2, 0, 255,
                                   cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(("p2_aggressive_sharp_otsu", sharp2_otsu))

    # Deskew + CLAHE + Otsu — fixes tilted barcode photos
    deskewed = deskew(gray)
    desk_clahe = clahe.apply(deskewed)
    _, desk_otsu = cv2.threshold(desk_clahe, 0, 255,
                                 cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(("p2_deskewed_clahe_otsu",   desk_otsu))

    # Upscale small images
    h, w = gray.shape[:2]
    if w < 800:
        up = cv2.resize(gray, (int(w * 2), int(h * 2)),
                        interpolation=cv2.INTER_CUBIC)
        up_clahe = clahe.apply(up)
        _, up_otsu = cv2.threshold(up_clahe, 0, 255,
                                   cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants.append(("p2_upscaled_clahe_otsu", up_otsu))

    # Denoise + Otsu (expensive — last in phase 2)
    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    den_clahe = clahe.apply(denoised)
    _, den_otsu = cv2.threshold(den_clahe, 0, 255,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(("p2_denoised_clahe_otsu",   den_otsu))

    # Denoise + Erode
    den_eroded = cv2.erode(den_otsu, kh, iterations=1)
    variants.append(("p2_denoised_eroded",       den_eroded))

    return variants


# ════════════════════════════════════════════════════════
# PHASE 3 — INVERTED
# Swap: light → black, dark → white
# For white-bars-on-dark or reversed laser prints
# ════════════════════════════════════════════════════════

def phase3_inverted_variants(img: np.ndarray, gray: np.ndarray,
                              clahe) -> list:
    """
    Invert all Phase 1 + Phase 2 variants.
    Handles barcodes where bars are lighter than background.
    """
    variants = []
    clahe_img  = clahe.apply(gray)
    _, otsu    = cv2.threshold(gray, 0, 255,
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, clahe_otsu = cv2.threshold(clahe_img, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kh = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))

    adaptive = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2)

    bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
    bil_clahe  = clahe.apply(bilateral)
    _, bil_otsu = cv2.threshold(bil_clahe, 0, 255,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Inverted versions of all key Phase 1 variants
    variants.append(("p3_inv_gray",              cv2.bitwise_not(gray)))
    variants.append(("p3_inv_clahe",             cv2.bitwise_not(clahe_img)))
    variants.append(("p3_inv_otsu",              cv2.bitwise_not(otsu)))
    variants.append(("p3_inv_clahe_otsu",        cv2.bitwise_not(clahe_otsu)))
    variants.append(("p3_inv_adaptive",          cv2.bitwise_not(adaptive)))
    variants.append(("p3_inv_bilateral_otsu",    cv2.bitwise_not(bil_otsu)))

    # Inverted gamma
    for gval in [1.5, 2.0, 0.7]:
        g = apply_gamma(gray, gval)
        g_clahe = clahe.apply(g)
        _, g_otsu = cv2.threshold(g_clahe, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants.append((f"p3_inv_gamma_{gval}_clahe_otsu",
                         cv2.bitwise_not(g_otsu)))

    # Inverted morphological
    eroded = cv2.erode(otsu, kh, iterations=1)
    variants.append(("p3_inv_erode",             cv2.bitwise_not(eroded)))
    clahe_eroded = cv2.erode(clahe_otsu, kh, iterations=1)
    variants.append(("p3_inv_clahe_otsu_erode",  cv2.bitwise_not(clahe_eroded)))

    # Inverted denoise
    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    _, den_otsu = cv2.threshold(denoised, 0, 255,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(("p3_inv_denoised_otsu",     cv2.bitwise_not(den_otsu)))

    # Inverted original color
    variants.append(("p3_inv_color",             cv2.bitwise_not(img)))

    return variants


# ════════════════════════════════════════════════════════
# MAIN SCAN FUNCTION
# ════════════════════════════════════════════════════════

def preprocess_and_scan(img: np.ndarray) -> dict:
    """
    3-phase scanning:
    Phase 1 → Contrast normalisation (dark→black, light→white)
    Phase 2 → Deep preprocessing (morphology, sharpen, denoise)
    Phase 3 → Inverted (light→black, dark→white)
    Each variant tried with OpenCV decoder then zxing-cpp.
    """
    img  = resize_if_large(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    phases = [
        ("Phase 1 - Contrast Normalisation", phase1_contrast_variants(img, gray, clahe)),
        ("Phase 2 - Deep Preprocessing",     phase2_preprocessing_variants(img, gray, clahe)),
        ("Phase 3 - Inverted",               phase3_inverted_variants(img, gray, clahe)),
    ]

    total_variants = sum(len(v) for _, v in phases)
    variants_tried = 0

    for phase_name, variants in phases:
        for variant_name, variant_img in variants:
            variants_tried += 1
            result = try_decoders(variant_img, variant_name, phase_name)
            if result:
                result["variantsTried"] = variants_tried
                result["totalVariants"] = total_variants
                result["error"]         = None
                return result

    return {
        "isReadable":     False,
        "barcodeValue":   None,
        "barcodeFormat":  None,
        "successVariant": None,
        "decoder":        None,
        "phase":          None,
        "variantsTried":  variants_tried,
        "totalVariants":  total_variants,
        "error":          f"Not readable after all {total_variants} variants across 3 phases"
    }


# ════════════════════════════════════════════════════════
# ROUTES
# ════════════════════════════════════════════════════════

@app.route('/scan', methods=['POST', 'OPTIONS'])
def scan_barcode():
    """
    POST /scan
    Body: { "image": "<base64 with or without data URI prefix>" }
    """
    if request.method == 'OPTIONS':
        return '', 204
    try:
        data = request.get_json(force=True)
        if not data or 'image' not in data:
            return jsonify({
                "isReadable": False, "barcodeValue": None,
                "barcodeFormat": None, "successVariant": None,
                "decoder": None, "phase": None,
                "variantsTried": 0, "totalVariants": 0,
                "error": "Missing 'image' field in request body"
            }), 400

        img = decode_base64_image(data['image'])
        if img is None:
            return jsonify({
                "isReadable": False, "barcodeValue": None,
                "barcodeFormat": None, "successVariant": None,
                "decoder": None, "phase": None,
                "variantsTried": 0, "totalVariants": 0,
                "error": "Could not decode image"
            }), 400

        return jsonify(preprocess_and_scan(img))

    except Exception as e:
        return jsonify({
            "isReadable": False, "barcodeValue": None,
            "barcodeFormat": None, "successVariant": None,
            "decoder": None, "phase": None,
            "variantsTried": 0, "totalVariants": 0,
            "error": str(e)
        }), 500


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status":           "ok",
        "message":          "Barcode API running — 3-phase pipeline",
        "phases":           [
            "Phase 1: Contrast Normalisation (dark→black, light→white)",
            "Phase 2: Deep Preprocessing (morphology, sharpen, denoise)",
            "Phase 3: Inverted (light→black, dark→white)"
        ],
        "decoders":         ["opencv_barcode_detector", "zxing_cpp"],
        "opencv_available": CV2_AVAILABLE,
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
