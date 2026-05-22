from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64
import zxingcpp
from PIL import Image
import io

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


def array_to_pil(img: np.ndarray) -> Image.Image:
    """
    KEY FIX:
    Encode numpy array to PNG in memory then decode back to PIL.
    This simulates the save/reload cycle in the notebook:
        cv2.imwrite("processed.png", img) → external app reads file
    Without this, raw numpy arrays passed directly to decoders
    can fail even when the same image saved to disk works fine.
    PNG is lossless — preserves exact black/white pixels.
    """
    _, buffer = cv2.imencode('.png', img)
    pil = Image.open(io.BytesIO(buffer.tobytes()))
    pil.load()  # Force full load before BytesIO goes out of scope
    return pil


def direct_threshold(gray: np.ndarray, value: int) -> np.ndarray:
    """
    Your exact notebook method:
    Pixels > value → WHITE (255)
    Pixels <= value → BLACK (0)
    No blur, no neighbour averaging — hard direct cut.
    """
    return np.where(gray > value, 255, 0).astype(np.uint8)


def morph_open_cleanup(binary: np.ndarray,
                       ksize: tuple = (2, 2)) -> np.ndarray:
    """Your notebook cleanup — removes tiny noise with morph open."""
    kernel = np.ones(ksize, np.uint8)
    return cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)


# ════════════════════════════════════════════════════════
# DECODERS
# Both use array_to_pil() to simulate save/reload
# ════════════════════════════════════════════════════════

def try_opencv(img: np.ndarray) -> dict | None:
    if not CV2_AVAILABLE:
        return None
    try:
        # OpenCV detector works best on color/grayscale
        # Use the raw numpy array directly here
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
        # ← KEY FIX: encode to PNG then reload before passing to zxing
        pil = array_to_pil(img)
        results = zxingcpp.read_barcodes(pil)
        if results:
            return {"value": results[0].text,
                    "format": str(results[0].format)}
    except Exception:
        pass
    return None


def try_decoders(img: np.ndarray,
                 variant: str, phase: str) -> dict | None:
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
# PHASE 1 — YOUR EXACT NOTEBOOK METHOD
# grayscale → direct threshold → morph open cleanup
# Sweeps threshold 80–200 to cover all label brightness levels
# ════════════════════════════════════════════════════════

def phase1_direct_threshold(img: np.ndarray,
                             gray: np.ndarray) -> list:
    variants = []

    # Raw originals first
    variants.append(("p1_original_color", img))
    variants.append(("p1_grayscale",      gray))

    # Your exact notebook method: threshold=120 + (2,2) morph open
    t120       = direct_threshold(gray, 120)
    t120_clean = morph_open_cleanup(t120, (2, 2))
    variants.append(("p1_threshold_120",         t120))
    variants.append(("p1_threshold_120_cleaned", t120_clean))  # ← your notebook exact output

    # Sweep threshold values — different labels may need different values
    for tval in [80, 90, 100, 110, 130, 140, 150, 160, 180, 200]:
        binary  = direct_threshold(gray, tval)
        cleaned = morph_open_cleanup(binary, (2, 2))
        variants.append((f"p1_threshold_{tval}",         binary))
        variants.append((f"p1_threshold_{tval}_cleaned", cleaned))

    # Alternative kernel sizes for cleanup
    for ksize in [(3, 3), (2, 1), (1, 2), (3, 1), (1, 3)]:
        t = morph_open_cleanup(t120, ksize)
        variants.append((f"p1_t120_morph_{ksize[0]}x{ksize[1]}", t))

    return variants


# ════════════════════════════════════════════════════════
# PHASE 2 — INVERTED THEN NOTEBOOK METHOD
# For reversed barcodes (light bars on dark background)
# ════════════════════════════════════════════════════════

def phase2_inverted_threshold(gray: np.ndarray) -> list:
    variants = []
    inv_gray = cv2.bitwise_not(gray)

    # Invert first then apply your exact method
    t120       = direct_threshold(inv_gray, 120)
    t120_clean = morph_open_cleanup(t120, (2, 2))
    variants.append(("p2_inv_threshold_120",         t120))
    variants.append(("p2_inv_threshold_120_cleaned", t120_clean))

    # Sweep on inverted
    for tval in [80, 90, 100, 110, 130, 140, 150, 160, 180, 200]:
        binary  = direct_threshold(inv_gray, tval)
        cleaned = morph_open_cleanup(binary, (2, 2))
        variants.append((f"p2_inv_threshold_{tval}",         binary))
        variants.append((f"p2_inv_threshold_{tval}_cleaned", cleaned))

    # Threshold first THEN invert result (different from inverting gray)
    for tval in [100, 120, 140]:
        t     = direct_threshold(gray, tval)
        t_inv = cv2.bitwise_not(morph_open_cleanup(t, (2, 2)))
        variants.append((f"p2_threshold_{tval}_then_inv", t_inv))

    return variants


# ════════════════════════════════════════════════════════
# PHASE 3 — CONTRAST ENHANCEMENT THEN NOTEBOOK METHOD
# For very dark or washed-out colored backgrounds
# ════════════════════════════════════════════════════════

def phase3_contrast_then_threshold(img: np.ndarray,
                                    gray: np.ndarray) -> list:
    variants = []
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    # CLAHE then your threshold method
    clahe_gray = clahe.apply(gray)
    for tval in [100, 110, 120, 130, 140]:
        binary  = direct_threshold(clahe_gray, tval)
        cleaned = morph_open_cleanup(binary, (2, 2))
        variants.append((f"p3_clahe_t{tval}",         binary))
        variants.append((f"p3_clahe_t{tval}_cleaned", cleaned))

    # CLAHE inverted then threshold
    clahe_inv = cv2.bitwise_not(clahe_gray)
    for tval in [100, 120, 140]:
        binary  = direct_threshold(clahe_inv, tval)
        cleaned = morph_open_cleanup(binary, (2, 2))
        variants.append((f"p3_clahe_inv_t{tval}_cleaned", cleaned))

    # Gamma brighten then threshold (dark backgrounds)
    for gamma in [1.5, 2.0]:
        table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255
                          for i in range(256)], dtype=np.uint8)
        g_img = cv2.LUT(gray, table)
        for tval in [110, 120, 130]:
            binary  = direct_threshold(g_img, tval)
            cleaned = morph_open_cleanup(binary, (2, 2))
            variants.append((f"p3_gamma{gamma}_t{tval}_cleaned", cleaned))

    # Bilateral filter then threshold (smooth noise, keep edges)
    bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
    for tval in [110, 120, 130]:
        binary  = direct_threshold(bilateral, tval)
        cleaned = morph_open_cleanup(binary, (2, 2))
        variants.append((f"p3_bilateral_t{tval}_cleaned", cleaned))

    # Bilateral inverted then threshold
    bil_inv = cv2.bitwise_not(bilateral)
    for tval in [110, 120, 130]:
        binary  = direct_threshold(bil_inv, tval)
        cleaned = morph_open_cleanup(binary, (2, 2))
        variants.append((f"p3_bilateral_inv_t{tval}_cleaned", cleaned))

    return variants


# ════════════════════════════════════════════════════════
# MAIN SCAN FUNCTION
# ════════════════════════════════════════════════════════

def preprocess_and_scan(img: np.ndarray) -> dict:
    img  = resize_if_large(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    phases = [
        ("Phase 1 - Direct Threshold (Notebook Method)",
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
        "status":           "ok",
        "message":          "Barcode API — notebook method + PNG encode fix",
        "key_fix":          "array_to_pil() encodes to PNG in memory before decoding — simulates notebook save/reload cycle",
        "phases":           [
            "Phase 1: Direct threshold sweep (your notebook method)",
            "Phase 2: Inverted + direct threshold",
            "Phase 3: Contrast enhanced + direct threshold"
        ],
        "decoders":         ["opencv_barcode_detector", "zxing_cpp"],
        "opencv_available": CV2_AVAILABLE,
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
