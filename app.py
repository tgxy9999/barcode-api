from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64
import zxingcpp
from PIL import Image

app = Flask(__name__)
CORS(app)

MAX_DIM = 1200  # Cap image size to protect free tier RAM

# ── Initialize OpenCV Barcode Detector once at startup ──
# Different algorithm to zxing-cpp — increases detection chances
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
    """Convert base64 string to OpenCV image. Strips data URI prefix if present."""
    if ',' in b64_string:
        b64_string = b64_string.split(',')[1]
    b64_string = b64_string.strip()
    img_bytes = base64.b64decode(b64_string)
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    return img


def resize_if_large(img: np.ndarray) -> np.ndarray:
    """Resize image if larger than MAX_DIM to prevent memory overflow."""
    h, w = img.shape[:2]
    if max(h, w) > MAX_DIM:
        scale = MAX_DIM / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)),
                         interpolation=cv2.INTER_AREA)
    return img


def apply_gamma(gray: np.ndarray, gamma: float) -> np.ndarray:
    """Gamma correction — brightens/darkens backgrounds for better contrast."""
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in range(256)], dtype=np.uint8)
    return cv2.LUT(gray, table)


def deskew(gray: np.ndarray) -> np.ndarray:
    """Correct barcode tilt using Hough line detection."""
    try:
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)
        if lines is None:
            return gray
        angles = []
        for line in lines[:20]:
            rho, theta = line[0]
            angle = (theta * 180 / np.pi) - 90
            if -45 < angle < 45:
                angles.append(angle)
        if not angles:
            return gray
        median_angle = float(np.median(angles))
        if abs(median_angle) < 0.5:
            return gray
        h, w = gray.shape
        M = cv2.getRotationMatrix2D((w // 2, h // 2), median_angle, 1.0)
        return cv2.warpAffine(gray, M, (w, h),
                              flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE)
    except Exception:
        return gray


# ════════════════════════════════════════════════════════
# DECODER 1 — OpenCV BarcodeDetector
# Uses OpenCV's own ML-based barcode detection algorithm
# Completely separate from zxing-cpp — different strengths
# ════════════════════════════════════════════════════════

def try_opencv_decoder(img: np.ndarray) -> dict | None:
    """
    Attempt decode using OpenCV's built-in BarcodeDetector.
    Works best on color or grayscale images.
    Returns result dict or None.
    """
    if not CV2_AVAILABLE or cv2_detector is None:
        return None
    try:
        # detectAndDecodeMulti handles multiple barcodes in one image
        ret, decoded_list, decoded_types, points = cv2_detector.detectAndDecodeMulti(img)
        if ret and decoded_list:
            # Filter out empty results
            for i, value in enumerate(decoded_list):
                if value and value.strip():
                    fmt = decoded_types[i] if decoded_types and i < len(decoded_types) else "UNKNOWN"
                    return {"value": value.strip(), "format": str(fmt)}
    except Exception:
        pass
    return None


# ════════════════════════════════════════════════════════
# DECODER 2 — zxing-cpp
# Pure C++ ZXing port — good for clean/preprocessed images
# ════════════════════════════════════════════════════════

def try_zxing_decoder(img: np.ndarray) -> dict | None:
    """
    Attempt decode using zxing-cpp.
    Returns result dict or None.
    """
    try:
        if len(img.shape) == 2:
            pil_img = Image.fromarray(img)
        else:
            pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        results = zxingcpp.read_barcodes(pil_img)
        if results:
            best = results[0]
            return {"value": best.text, "format": str(best.format)}
    except Exception:
        pass
    return None


def try_both_decoders(img: np.ndarray, variant_name: str) -> dict | None:
    """
    Try both decoders on the same image variant.
    OpenCV first (ML-based), then zxing-cpp fallback.
    Returns full result dict or None.
    """
    # Try OpenCV decoder first
    result = try_opencv_decoder(img)
    if result:
        return {
            "isReadable":     True,
            "barcodeValue":   result["value"],
            "barcodeFormat":  result["format"],
            "successVariant": variant_name,
            "decoder":        "opencv_barcode_detector",
        }

    # Try zxing-cpp decoder
    result = try_zxing_decoder(img)
    if result:
        return {
            "isReadable":     True,
            "barcodeValue":   result["value"],
            "barcodeFormat":  result["format"],
            "successVariant": variant_name,
            "decoder":        "zxing_cpp",
        }

    return None


# ════════════════════════════════════════════════════════
# PREPROCESSING PIPELINE
# 8 tiers ordered fast → expensive
# Each variant tried with BOTH decoders
# ════════════════════════════════════════════════════════

def preprocess_and_scan(img: np.ndarray) -> tuple:
    """
    Run all preprocessing variants.
    Each variant is passed to both decoders.
    Returns (result_dict_or_None, variants_tried, total_variants).
    """
    img = resize_if_large(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(gray)

    variants = []

    # ── TIER 1: Raw / Basic ──
    variants.append(("original",              img))
    variants.append(("grayscale",             gray))
    variants.append(("inverted_gray",         cv2.bitwise_not(gray)))
    variants.append(("clahe",                 clahe_img))
    variants.append(("inverted_clahe",        cv2.bitwise_not(clahe_img)))

    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(("otsu",                  otsu))
    variants.append(("inverted_otsu",         cv2.bitwise_not(otsu)))

    _, clahe_otsu = cv2.threshold(clahe_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(("clahe_otsu",            clahe_otsu))
    variants.append(("inverted_clahe_otsu",   cv2.bitwise_not(clahe_otsu)))

    adaptive = cv2.adaptiveThreshold(gray, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    variants.append(("adaptive",              adaptive))
    variants.append(("inverted_adaptive",     cv2.bitwise_not(adaptive)))

    clahe_adaptive = cv2.adaptiveThreshold(clahe_img, 255,
                     cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    variants.append(("clahe_adaptive",        clahe_adaptive))
    variants.append(("inverted_clahe_adaptive", cv2.bitwise_not(clahe_adaptive)))

    # ── TIER 2: Gamma correction ──
    for gval in [1.5, 2.0, 0.7]:
        g = apply_gamma(gray, gval)
        _, g_otsu = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants.append((f"gamma_{gval}_otsu",          g_otsu))
        variants.append((f"inverted_gamma_{gval}_otsu", cv2.bitwise_not(g_otsu)))
        g_clahe = clahe.apply(g)
        _, g_clahe_otsu = cv2.threshold(g_clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants.append((f"gamma_{gval}_clahe_otsu",    g_clahe_otsu))

    # ── TIER 3: Morphological operations ──
    kh = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
    kc = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
    ko = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))

    eroded = cv2.erode(otsu, kh, iterations=1)
    variants.append(("morph_erode",               eroded))
    variants.append(("inverted_morph_erode",      cv2.bitwise_not(eroded)))

    closed = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, kc)
    variants.append(("morph_close",               closed))
    variants.append(("inverted_morph_close",      cv2.bitwise_not(closed)))

    opened = cv2.morphologyEx(otsu, cv2.MORPH_OPEN, ko)
    variants.append(("morph_open",                opened))
    variants.append(("inverted_morph_open",       cv2.bitwise_not(opened)))

    clahe_eroded = cv2.erode(clahe_otsu, kh, iterations=1)
    variants.append(("clahe_otsu_erode",          clahe_eroded))
    variants.append(("inverted_clahe_otsu_erode", cv2.bitwise_not(clahe_eroded)))

    # ── TIER 4: Bilateral filter ──
    bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
    _, bil_otsu = cv2.threshold(bilateral, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(("bilateral_otsu",            bil_otsu))
    variants.append(("inverted_bilateral_otsu",   cv2.bitwise_not(bil_otsu)))
    bil_clahe = clahe.apply(bilateral)
    _, bil_clahe_otsu = cv2.threshold(bil_clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(("bilateral_clahe_otsu",      bil_clahe_otsu))

    # ── TIER 5: Sharpening ──
    k_sharp = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(gray, -1, k_sharp)
    variants.append(("sharpened",                 sharpened))

    k_sharp2 = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharp2 = cv2.filter2D(gray, -1, k_sharp2)
    _, sharp2_otsu = cv2.threshold(sharp2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(("aggressive_sharp_otsu",          sharp2_otsu))
    variants.append(("inverted_aggressive_sharp_otsu", cv2.bitwise_not(sharp2_otsu)))

    # ── TIER 6: Deskew ──
    deskewed = deskew(gray)
    _, desk_otsu = cv2.threshold(deskewed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(("deskewed_otsu",             desk_otsu))
    variants.append(("inverted_deskewed_otsu",    cv2.bitwise_not(desk_otsu)))
    desk_clahe = clahe.apply(deskewed)
    _, desk_clahe_otsu = cv2.threshold(desk_clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(("deskewed_clahe_otsu",       desk_clahe_otsu))

    # ── TIER 7: Upscaling (only for small images) ──
    h, w = gray.shape[:2]
    if w < 800:
        up = cv2.resize(gray, (int(w * 2), int(h * 2)), interpolation=cv2.INTER_CUBIC)
        up_clahe = clahe.apply(up)
        _, up_otsu = cv2.threshold(up_clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants.append(("upscaled_clahe_otsu",          up_otsu))
        variants.append(("inverted_upscaled_clahe_otsu", cv2.bitwise_not(up_otsu)))
        up_bil = cv2.bilateralFilter(up, 9, 75, 75)
        _, up_bil_otsu = cv2.threshold(up_bil, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants.append(("upscaled_bilateral_otsu",      up_bil_otsu))

    # ── TIER 8: Denoising — most expensive, last resort ──
    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    _, den_otsu = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(("denoised_otsu",                    den_otsu))
    variants.append(("inverted_denoised_otsu",           cv2.bitwise_not(den_otsu)))
    den_clahe = clahe.apply(denoised)
    _, den_clahe_otsu = cv2.threshold(den_clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(("denoised_clahe_otsu",              den_clahe_otsu))
    den_eroded = cv2.erode(den_otsu, kh, iterations=1)
    variants.append(("denoised_eroded",                  den_eroded))
    variants.append(("inverted_denoised_eroded",         cv2.bitwise_not(den_eroded)))

    # ── Try each variant with both decoders ──
    total = len(variants)
    for idx, (name, variant_img) in enumerate(variants):
        result = try_both_decoders(variant_img, name)
        if result:
            result["variantsTried"] = idx + 1
            result["totalVariants"] = total
            return result, idx + 1, total

    return None, total, total


# ════════════════════════════════════════════════════════
# ROUTES
# ════════════════════════════════════════════════════════

@app.route('/scan', methods=['POST', 'OPTIONS'])
def scan_barcode():
    """
    POST /scan
    Body: { "image": "<base64 — with or without data URI prefix>" }
    Response: {
        isReadable, barcodeValue, barcodeFormat,
        successVariant, decoder, variantsTried, totalVariants, error
    }
    """
    if request.method == 'OPTIONS':
        return '', 204

    try:
        data = request.get_json(force=True)

        if not data or 'image' not in data:
            return jsonify({
                "isReadable": False, "barcodeValue": None,
                "barcodeFormat": None, "successVariant": None,
                "decoder": None, "variantsTried": 0,
                "totalVariants": 0,
                "error": "Missing 'image' field in request body"
            }), 400

        img = decode_base64_image(data['image'])
        if img is None:
            return jsonify({
                "isReadable": False, "barcodeValue": None,
                "barcodeFormat": None, "successVariant": None,
                "decoder": None, "variantsTried": 0,
                "totalVariants": 0,
                "error": "Could not decode image — invalid base64 or unsupported format"
            }), 400

        result, tried, total = preprocess_and_scan(img)

        if result:
            result["error"] = None
            return jsonify(result)

        return jsonify({
            "isReadable": False, "barcodeValue": None,
            "barcodeFormat": None, "successVariant": None,
            "decoder": None, "variantsTried": tried,
            "totalVariants": total,
            "error": f"Not readable after {tried} variants with 2 decoders each"
        })

    except Exception as e:
        return jsonify({
            "isReadable": False, "barcodeValue": None,
            "barcodeFormat": None, "successVariant": None,
            "decoder": None, "variantsTried": 0,
            "totalVariants": 0, "error": str(e)
        }), 500


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status":            "ok",
        "message":           "Barcode API running",
        "decoders":          ["opencv_barcode_detector", "zxing_cpp"],
        "opencv_available":  CV2_AVAILABLE,
        "preprocessing_tiers": 8,
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
