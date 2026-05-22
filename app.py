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


def direct_threshold(gray: np.ndarray, value: int) -> np.ndarray:
    """
    Your working method from notebook:
    Direct black/white conversion at fixed threshold.
    No blur, no neighbour mimicking — just hard cut.
    Pixels > value → WHITE (255)
    Pixels <= value → BLACK (0)
    """
    return np.where(gray > value, 255, 0).astype(np.uint8)


def morph_open_cleanup(binary: np.ndarray,
                       ksize: tuple = (2, 2)) -> np.ndarray:
    """
    Your working method from notebook:
    Morphological open with (2,2) kernel.
    Removes tiny noise blobs without affecting bar structure.
    """
    kernel = np.ones(ksize, np.uint8)
    return cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)


# ════════════════════════════════════════════════════════
# DECODERS
# ════════════════════════════════════════════════════════

def try_opencv(img: np.ndarray) -> dict | None:
    if not CV2_AVAILABLE:
        return None
    try:
        ret, decoded_list, decoded_types, _ = \
            cv2_detector.detectAndDecodeMulti(img)
        if ret and decoded_list:
            for i, val in enumerate(decoded_list):
                if val and val.strip():
                    fmt = decoded_types[i] if decoded_types else "UNKNOWN"
                    return {"value": val.strip(), "format": str(fmt)}
    except Exception:
        pass
    return None


def try_zxing(img: np.ndarray) -> dict | None:
    try:
        pil = Image.fromarray(
            img if len(img.shape) == 2
            else cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        )
        results = zxingcpp.read_barcodes(pil)
        if results:
            return {"value": results[0].text,
                    "format": str(results[0].format)}
    except Exception:
        pass
    return None


def try_decoders(img: np.ndarray,
                 variant: str, phase: str) -> dict | None:
    """Try OpenCV then zxing-cpp on one image variant."""
    for fn, name in [(try_opencv, "opencv_barcode_detector"),
                     (try_zxing,  "zxing_cpp")]:
        r = fn(img)
        if r:
            return {
                "isReadable":     True,
                "barcodeValue":   r["value"],
                "barcodeFormat":  r["format"],
                "successVariant": variant,
                "decoder":        name,
                "phase":          phase,
            }
    return None


# ════════════════════════════════════════════════════════
# PHASE 1 — YOUR NOTEBOOK METHOD (direct threshold)
# This is the method confirmed to work on your barcodes.
# Tries multiple threshold values + noise cleanup variants.
# ════════════════════════════════════════════════════════

def phase1_direct_threshold(img: np.ndarray,
                             gray: np.ndarray) -> list:
    """
    Based exactly on your working notebook:
    gray → direct threshold at N → morph_open cleanup

    Tries threshold values from 80 to 200 in steps.
    Lower value  → more pixels become WHITE
    Higher value → more pixels become BLACK

    Your notebook used 120 — we sweep around it.
    """
    variants = []

    # ── Raw first (no threshold) ──
    variants.append(("p1_original_color", img))
    variants.append(("p1_grayscale",      gray))

    # ── Your exact notebook method: threshold=120 + cleanup ──
    t120        = direct_threshold(gray, 120)
    t120_clean  = morph_open_cleanup(t120)
    variants.append(("p1_threshold_120",             t120))
    variants.append(("p1_threshold_120_cleaned",     t120_clean))

    # ── Sweep threshold values around 120 ──
    # Covers different background brightness levels
    for tval in [80, 90, 100, 110, 130, 140, 150, 160, 180, 200]:
        binary      = direct_threshold(gray, tval)
        cleaned     = morph_open_cleanup(binary)
        variants.append((f"p1_threshold_{tval}",         binary))
        variants.append((f"p1_threshold_{tval}_cleaned", cleaned))

    # ── Different kernel sizes for cleanup ──
    for ksize in [(3, 3), (2, 1), (1, 2)]:
        t120_k = morph_open_cleanup(t120, ksize)
        variants.append((f"p1_threshold_120_morph_{ksize[0]}x{ksize[1]}",
                         t120_k))

    return variants


# ════════════════════════════════════════════════════════
# PHASE 2 — INVERTED NOTEBOOK METHOD
# Same method but invert first (for reversed barcodes)
# Light bars on dark background
# ════════════════════════════════════════════════════════

def phase2_inverted_threshold(gray: np.ndarray) -> list:
    """
    Invert the grayscale first, then apply same threshold method.
    Handles barcodes where bars are lighter than background.
    """
    variants = []
    inv_gray = cv2.bitwise_not(gray)

    # ── Inverted + your exact threshold ──
    t120       = direct_threshold(inv_gray, 120)
    t120_clean = morph_open_cleanup(t120)
    variants.append(("p2_inv_threshold_120",         t120))
    variants.append(("p2_inv_threshold_120_cleaned", t120_clean))

    # ── Sweep threshold on inverted image ──
    for tval in [80, 90, 100, 110, 130, 140, 150, 160, 180, 200]:
        binary  = direct_threshold(inv_gray, tval)
        cleaned = morph_open_cleanup(binary)
        variants.append((f"p2_inv_threshold_{tval}",         binary))
        variants.append((f"p2_inv_threshold_{tval}_cleaned", cleaned))

    # ── Also try: threshold first, then invert result ──
    # Different from inverting gray first
    t120_then_inv = cv2.bitwise_not(direct_threshold(gray, 120))
    variants.append(("p2_threshold_120_then_inv",    t120_then_inv))
    t120_clean_inv = cv2.bitwise_not(morph_open_cleanup(
                        direct_threshold(gray, 120)))
    variants.append(("p2_threshold_120_clean_then_inv", t120_clean_inv))

    return variants


# ════════════════════════════════════════════════════════
# PHASE 3 — ENHANCED CONTRAST THEN NOTEBOOK METHOD
# Apply contrast enhancement before your threshold method
# For very dark or very washed-out colored backgrounds
# ════════════════════════════════════════════════════════

def phase3_contrast_then_threshold(img: np.ndarray,
                                    gray: np.ndarray) -> list:
    """
    Improve contrast first, then apply your direct threshold method.
    Covers cases where background color is too close to bar color.
    """
    variants = []
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    # ── CLAHE then direct threshold ──
    clahe_gray = clahe.apply(gray)
    for tval in [100, 110, 120, 130, 140]:
        binary  = direct_threshold(clahe_gray, tval)
        cleaned = morph_open_cleanup(binary)
        variants.append((f"p3_clahe_threshold_{tval}",         binary))
        variants.append((f"p3_clahe_threshold_{tval}_cleaned", cleaned))

    # ── CLAHE inverted then direct threshold ──
    clahe_inv = cv2.bitwise_not(clahe_gray)
    for tval in [100, 120, 140]:
        binary  = direct_threshold(clahe_inv, tval)
        cleaned = morph_open_cleanup(binary)
        variants.append((f"p3_clahe_inv_threshold_{tval}",         binary))
        variants.append((f"p3_clahe_inv_threshold_{tval}_cleaned", cleaned))

    # ── Gamma brighten then threshold ──
    # Helps when background is too dark
    for gamma in [1.5, 2.0]:
        table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255
                          for i in range(256)], dtype=np.uint8)
        g_img = cv2.LUT(gray, table)
        for tval in [110, 120, 130]:
            binary  = direct_threshold(g_img, tval)
            cleaned = morph_open_cleanup(binary)
            variants.append((f"p3_gamma{gamma}_threshold_{tval}_cleaned",
                             cleaned))

    # ── Bilateral filter then threshold ──
    # Smooths noise while keeping bar edges sharp
    bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
    for tval in [110, 120, 130]:
        binary  = direct_threshold(bilateral, tval)
        cleaned = morph_open_cleanup(binary)
        variants.append((f"p3_bilateral_threshold_{tval}_cleaned", cleaned))

    # ── Bilateral inverted then threshold ──
    bil_inv = cv2.bitwise_not(bilateral)
    t120    = direct_threshold(bil_inv, 120)
    cleaned = morph_open_cleanup(t120)
    variants.append(("p3_bilateral_inv_threshold_120_cleaned", cleaned))

    return variants


# ════════════════════════════════════════════════════════
# MAIN SCAN
# ════════════════════════════════════════════════════════

def preprocess_and_scan(img: np.ndarray) -> dict:
    """
    3-phase pipeline based on your working notebook method.

    Phase 1: Direct threshold (your exact method) — normal + sweep
    Phase 2: Inverted then direct threshold — for reversed barcodes
    Phase 3: Contrast enhancement then direct threshold — tough backgrounds
    """
    img  = resize_if_large(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    phases = [
        ("Phase 1 - Direct Threshold (Your Method)",
         phase1_direct_threshold(img, gray)),

        ("Phase 2 - Inverted + Direct Threshold",
         phase2_inverted_threshold(gray)),

        ("Phase 3 - Contrast Enhanced + Direct Threshold",
         phase3_contrast_then_threshold(img, gray)),
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
        "error":          f"Not readable after {total_variants} variants"
    }


# ════════════════════════════════════════════════════════
# ROUTES
# ════════════════════════════════════════════════════════

@app.route('/scan', methods=['POST', 'OPTIONS'])
def scan_barcode():
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
        "status":  "ok",
        "message": "Barcode API — notebook method pipeline",
        "phases": [
            "Phase 1: Direct threshold sweep (your notebook method)",
            "Phase 2: Inverted + direct threshold (reversed barcodes)",
            "Phase 3: Contrast enhanced + direct threshold (tough backgrounds)"
        ],
        "decoders":         ["opencv_barcode_detector", "zxing_cpp"],
        "opencv_available": CV2_AVAILABLE,
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
