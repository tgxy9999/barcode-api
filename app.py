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


def try_decode(img: np.ndarray) -> list:
    """Attempt zxing-cpp decode on an image."""
    try:
        if len(img.shape) == 2:
            pil_img = Image.fromarray(img)
        else:
            pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        return zxingcpp.read_barcodes(pil_img)
    except Exception:
        return []


def apply_gamma(gray: np.ndarray, gamma: float) -> np.ndarray:
    """Gamma correction — brightens dark colored backgrounds."""
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in range(256)], dtype=np.uint8)
    return cv2.LUT(gray, table)


def deskew(gray: np.ndarray) -> np.ndarray:
    """
    Attempt to correct barcode tilt/skew.
    Uses Hough line detection to find dominant angle and rotate.
    """
    try:
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)
        if lines is None:
            return gray
        angles = []
        for line in lines[:20]:  # Use top 20 lines only
            rho, theta = line[0]
            angle = (theta * 180 / np.pi) - 90
            if -45 < angle < 45:
                angles.append(angle)
        if not angles:
            return gray
        median_angle = float(np.median(angles))
        if abs(median_angle) < 0.5:  # Skip if barely tilted
            return gray
        h, w = gray.shape
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        rotated = cv2.warpAffine(gray, M, (w, h),
                                 flags=cv2.INTER_CUBIC,
                                 borderMode=cv2.BORDER_REPLICATE)
        return rotated
    except Exception:
        return gray


def preprocess_variants(img: np.ndarray) -> list:
    """
    Full preprocessing pipeline ordered fast → expensive.
    Each variant targets a specific laser-print problem.
    Exits immediately when a decode succeeds.
    """
    variants = []

    # ── Resize to control memory ──
    img = resize_if_large(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    # ════════════════════════════════════
    # TIER 1 — Fast basic variants
    # ════════════════════════════════════

    # Raw
    variants.append(("original",      img))
    variants.append(("grayscale",     gray))
    variants.append(("inverted_gray", cv2.bitwise_not(gray)))

    # CLAHE — colored backgrounds
    clahe_img = clahe.apply(gray)
    variants.append(("clahe",         clahe_img))
    variants.append(("inverted_clahe",cv2.bitwise_not(clahe_img)))

    # Otsu threshold
    _, otsu = cv2.threshold(gray, 0, 255,
                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(("otsu",          otsu))
    variants.append(("inverted_otsu", cv2.bitwise_not(otsu)))

    # CLAHE + Otsu
    _, clahe_otsu = cv2.threshold(clahe_img, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(("clahe_otsu",          clahe_otsu))
    variants.append(("inverted_clahe_otsu", cv2.bitwise_not(clahe_otsu)))

    # Adaptive threshold — uneven lighting
    adaptive = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, blockSize=11, C=2)
    variants.append(("adaptive",          adaptive))
    variants.append(("inverted_adaptive", cv2.bitwise_not(adaptive)))

    # CLAHE + Adaptive
    clahe_adaptive = cv2.adaptiveThreshold(
        clahe_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, blockSize=11, C=2)
    variants.append(("clahe_adaptive",          clahe_adaptive))
    variants.append(("inverted_clahe_adaptive", cv2.bitwise_not(clahe_adaptive)))

    # ════════════════════════════════════
    # TIER 2 — Gamma correction
    # Targets: dark colored backgrounds
    # ════════════════════════════════════

    for gamma_val in [1.5, 2.0, 0.7]:
        g = apply_gamma(gray, gamma_val)
        _, g_otsu = cv2.threshold(g, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        label = f"gamma_{gamma_val}"
        variants.append((label,                    g_otsu))
        variants.append((f"inverted_{label}",      cv2.bitwise_not(g_otsu)))
        # CLAHE on gamma
        g_clahe = clahe.apply(g)
        _, g_clahe_otsu = cv2.threshold(g_clahe, 0, 255,
                                        cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants.append((f"{label}_clahe_otsu",    g_clahe_otsu))

    # ════════════════════════════════════
    # TIER 3 — Morphological operations
    # Targets: bleeding/merged bars from overburning
    # ════════════════════════════════════

    # Erosion — thins bleeding bars
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
    eroded = cv2.erode(otsu, kernel_h, iterations=1)
    variants.append(("morph_erode",          eroded))
    variants.append(("inverted_morph_erode", cv2.bitwise_not(eroded)))

    # Closing — connects broken bars
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
    morph_close = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, kernel_close)
    variants.append(("morph_close",          morph_close))
    variants.append(("inverted_morph_close", cv2.bitwise_not(morph_close)))

    # Opening — removes noise blobs between bars
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
    morph_open = cv2.morphologyEx(otsu, cv2.MORPH_OPEN, kernel_open)
    variants.append(("morph_open",          morph_open))
    variants.append(("inverted_morph_open", cv2.bitwise_not(morph_open)))

    # CLAHE + Otsu + Erode (best combo for overburnt colored labels)
    clahe_eroded = cv2.erode(clahe_otsu, kernel_h, iterations=1)
    variants.append(("clahe_otsu_erode",          clahe_eroded))
    variants.append(("inverted_clahe_otsu_erode", cv2.bitwise_not(clahe_eroded)))

    # ════════════════════════════════════
    # TIER 4 — Bilateral filter
    # Targets: noisy laser prints, keeps bar edges sharp
    # ════════════════════════════════════

    bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
    _, bilateral_otsu = cv2.threshold(bilateral, 0, 255,
                                      cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(("bilateral_otsu",          bilateral_otsu))
    variants.append(("inverted_bilateral_otsu", cv2.bitwise_not(bilateral_otsu)))

    # Bilateral + CLAHE
    bilateral_clahe = clahe.apply(bilateral)
    _, bilateral_clahe_otsu = cv2.threshold(bilateral_clahe, 0, 255,
                                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(("bilateral_clahe_otsu",    bilateral_clahe_otsu))

    # ════════════════════════════════════
    # TIER 5 — Sharpening
    # Targets: blurry/soft bar edges
    # ════════════════════════════════════

    # Standard sharpen
    kernel_sharp = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(gray, -1, kernel_sharp)
    variants.append(("sharpened", sharpened))

    # Aggressive sharpen
    kernel_sharp2 = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened2 = cv2.filter2D(gray, -1, kernel_sharp2)
    _, sharp2_otsu = cv2.threshold(sharpened2, 0, 255,
                                   cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(("sharpened_aggressive_otsu",          sharp2_otsu))
    variants.append(("inverted_sharpened_aggressive_otsu", cv2.bitwise_not(sharp2_otsu)))

    # ════════════════════════════════════
    # TIER 6 — Deskew
    # Targets: tilted/angled barcode photos
    # ════════════════════════════════════

    deskewed = deskew(gray)
    _, deskew_otsu = cv2.threshold(deskewed, 0, 255,
                                   cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(("deskewed_otsu",          deskew_otsu))
    variants.append(("inverted_deskewed_otsu", cv2.bitwise_not(deskew_otsu)))

    deskew_clahe = clahe.apply(deskewed)
    _, deskew_clahe_otsu = cv2.threshold(deskew_clahe, 0, 255,
                                         cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(("deskewed_clahe_otsu", deskew_clahe_otsu))

    # ════════════════════════════════════
    # TIER 7 — Upscaling (expensive, last resort)
    # Targets: small/low-res barcodes
    # ════════════════════════════════════

    h, w = gray.shape[:2]
    if w < 800:
        up = cv2.resize(gray, (int(w * 2), int(h * 2)),
                        interpolation=cv2.INTER_CUBIC)
        up_clahe = clahe.apply(up)
        _, up_clahe_otsu = cv2.threshold(up_clahe, 0, 255,
                                         cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants.append(("upscaled_clahe_otsu",          up_clahe_otsu))
        variants.append(("inverted_upscaled_clahe_otsu", cv2.bitwise_not(up_clahe_otsu)))

        up_bilateral = cv2.bilateralFilter(up, 9, 75, 75)
        _, up_bilateral_otsu = cv2.threshold(up_bilateral, 0, 255,
                                             cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants.append(("upscaled_bilateral_otsu", up_bilateral_otsu))

    # ════════════════════════════════════
    # TIER 8 — Denoising (most expensive)
    # Targets: heavy laser speckle/noise
    # ════════════════════════════════════

    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    _, denoised_otsu = cv2.threshold(denoised, 0, 255,
                                     cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(("denoised_otsu",     denoised_otsu))
    variants.append(("denoised_inverted", cv2.bitwise_not(denoised_otsu)))

    denoised_clahe = clahe.apply(denoised)
    _, denoised_clahe_otsu = cv2.threshold(denoised_clahe, 0, 255,
                                           cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(("denoised_clahe_otsu",          denoised_clahe_otsu))
    variants.append(("inverted_denoised_clahe_otsu", cv2.bitwise_not(denoised_clahe_otsu)))

    # Denoised + Morph (last resort combo)
    denoised_eroded = cv2.erode(denoised_otsu, kernel_h, iterations=1)
    variants.append(("denoised_eroded",          denoised_eroded))
    variants.append(("inverted_denoised_eroded", cv2.bitwise_not(denoised_eroded)))

    return variants


@app.route('/scan', methods=['POST', 'OPTIONS'])
def scan_barcode():
    """
    POST /scan
    Body: { "image": "<base64 — with or without data URI prefix>" }
    Returns: {
        isReadable, barcodeValue, barcodeFormat,
        successVariant, variantsTried, totalVariants, error
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
                "variantsTried": 0, "totalVariants": 0,
                "error": "Missing 'image' field in request body"
            }), 400

        img = decode_base64_image(data['image'])
        if img is None:
            return jsonify({
                "isReadable": False, "barcodeValue": None,
                "barcodeFormat": None, "successVariant": None,
                "variantsTried": 0, "totalVariants": 0,
                "error": "Could not decode image — invalid base64 or unsupported format"
            }), 400

        variants = preprocess_variants(img)
        total = len(variants)
        variants_tried = 0

        for variant_name, variant_img in variants:
            variants_tried += 1
            results = try_decode(variant_img)
            if results:
                best = results[0]
                return jsonify({
                    "isReadable":     True,
                    "barcodeValue":   best.text,
                    "barcodeFormat":  str(best.format),
                    "successVariant": variant_name,
                    "variantsTried":  variants_tried,
                    "totalVariants":  total,
                    "error":          None
                })

        return jsonify({
            "isReadable": False, "barcodeValue": None,
            "barcodeFormat": None, "successVariant": None,
            "variantsTried": variants_tried, "totalVariants": total,
            "error": "Barcode not readable after all preprocessing attempts"
        })

    except Exception as e:
        return jsonify({
            "isReadable": False, "barcodeValue": None,
            "barcodeFormat": None, "successVariant": None,
            "variantsTried": 0, "totalVariants": 0,
            "error": str(e)
        }), 500


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "ok",
        "message": "Barcode API running",
        "preprocessing_tiers": 8,
        "total_variants": "50+"
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
