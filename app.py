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

# ── OpenCV primary, zxing fallback if OpenCV unavailable ──
try:
    cv2_detector = cv2.barcode.BarcodeDetector()
    CV2_AVAILABLE = True
    ACTIVE_DECODER = "opencv"
except Exception:
    cv2_detector = None
    CV2_AVAILABLE = False
    ACTIVE_DECODER = "zxing"

# ── Threshold sweep order ──
# Always start from your confirmed sweet spot (120),
# then alternate outward in both directions:
# 120 → 110 → 130 → 100 → 140 → 90 → 150 → 80 → 160 → 70 → 170 → 60 → 180 → 50 → 200 → 40 → 220 → 30 → 240 → 20
THRESHOLD_ORDER = [
    120,                          # your best value
    110, 130,                     # ±10
    100, 140,                     # ±20
     90, 150,                     # ±30
     80, 160,                     # ±40
     70, 170,                     # ±50
     60, 180,                     # ±60
     50, 200,                     # wider
     40, 220,                     # wider
     30, 240,                     # extreme
     20,                          # last resort
]


def decode_base64_image(b64_string: str) -> np.ndarray:
    if ',' in b64_string:
        b64_string = b64_string.split(',')[1]
    img_bytes = base64.b64decode(b64_string.strip())
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    return cv2.imdecode(img_array, cv2.IMREAD_COLOR)


def img_to_base64(img: np.ndarray) -> str:
    _, buf = cv2.imencode('.png', img)
    return "data:image/png;base64," + base64.b64encode(
        buf.tobytes()).decode('utf-8')


def to_pil(img: np.ndarray) -> Image.Image:
    _, buf = cv2.imencode('.png', img)
    pil = Image.open(io.BytesIO(buf.tobytes()))
    pil.load()
    return pil


def scan(img: np.ndarray) -> str | None:
    """OpenCV primary. zxing fallback only if OpenCV not available."""
    if CV2_AVAILABLE:
        try:
            ret, decoded_list, _, _ = cv2_detector.detectAndDecodeMulti(img)
            if ret and decoded_list:
                for val in decoded_list:
                    if val and val.strip():
                        return val.strip()
        except Exception:
            pass
        return None
    else:
        try:
            results = zxingcpp.read_barcodes(to_pil(img))
            if results:
                return results[0].text
        except Exception:
            pass
        return None


def threshold_and_scan(gray: np.ndarray,
                       kernel: np.ndarray,
                       tval: int,
                       prefix: str) -> tuple:
    """
    Apply direct threshold + morph open + inverted variants.
    Returns (value, variant_name, processed_img) or (None, None, None).
    """
    # Direct threshold — your notebook method
    binary  = np.where(gray > tval, 255, 0).astype(np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    for img, name in [
        (binary,                      f"{prefix}_t{tval}"),
        (cleaned,                     f"{prefix}_t{tval}_cleaned"),
        (cv2.bitwise_not(binary),     f"{prefix}_inv_t{tval}"),
        (cv2.bitwise_not(cleaned),    f"{prefix}_inv_t{tval}_cleaned"),
    ]:
        val = scan(img)
        if val:
            return val, name, img

    return None, None, cleaned  # return cleaned for debug image


@app.route('/scan', methods=['POST', 'OPTIONS'])
def scan_barcode():
    if request.method == 'OPTIONS':
        return '', 204

    try:
        data = request.get_json(force=True)
        if not data or 'image' not in data:
            return jsonify({
                "isReadable": False, "barcodeValue": None,
                "processedImage": None,
                "error": "Missing 'image' field"
            }), 400

        img = decode_base64_image(data['image'])
        if img is None:
            return jsonify({
                "isReadable": False, "barcodeValue": None,
                "processedImage": None,
                "error": "Could not decode image"
            }), 400

        # ════════════════════════════════════════
        # STEP 1 — original image
        # ════════════════════════════════════════
        val = scan(img)
        if val:
            return jsonify({
                "isReadable": True, "barcodeValue": val,
                "variant": "original", "decoder": ACTIVE_DECODER,
                "processedImage": img_to_base64(img), "error": None
            })

        # ════════════════════════════════════════
        # STEP 2 — grayscale
        # ════════════════════════════════════════
        gray   = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kernel = np.ones((2, 2), np.uint8)

        val = scan(gray)
        if val:
            return jsonify({
                "isReadable": True, "barcodeValue": val,
                "variant": "grayscale", "decoder": ACTIVE_DECODER,
                "processedImage": img_to_base64(gray), "error": None
            })

        # ════════════════════════════════════════
        # STEP 3 — normal image threshold sweep
        # Starts at 120 (your sweet spot) then alternates
        # outward: 110→130→100→140→90→150 etc.
        # Each threshold tries: binary, cleaned, inverted, inv_cleaned
        # ════════════════════════════════════════
        last_debug_img = gray

        for tval in THRESHOLD_ORDER:
            val, variant, debug_img = threshold_and_scan(
                gray, kernel, tval, "normal")
            last_debug_img = debug_img
            if val:
                return jsonify({
                    "isReadable": True, "barcodeValue": val,
                    "variant": variant, "decoder": ACTIVE_DECODER,
                    "processedImage": img_to_base64(debug_img),
                    "error": None
                })

        # ════════════════════════════════════════
        # STEP 4 — CLAHE then same threshold sweep
        # Improves local contrast before thresholding
        # ════════════════════════════════════════
        clahe      = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe_gray = clahe.apply(gray)

        for tval in THRESHOLD_ORDER:
            val, variant, debug_img = threshold_and_scan(
                clahe_gray, kernel, tval, "clahe")
            last_debug_img = debug_img
            if val:
                return jsonify({
                    "isReadable": True, "barcodeValue": val,
                    "variant": variant, "decoder": ACTIVE_DECODER,
                    "processedImage": img_to_base64(debug_img),
                    "error": None
                })

        # Not readable — return last processed image for debugging
        return jsonify({
            "isReadable":     False,
            "barcodeValue":   None,
            "variant":        None,
            "decoder":        ACTIVE_DECODER,
            "brightness":     round(float(np.mean(gray)), 1),
            "processedImage": img_to_base64(last_debug_img),
            "error":          "Barcode not readable"
        })

    except Exception as e:
        return jsonify({
            "isReadable": False, "barcodeValue": None,
            "processedImage": None, "error": str(e)
        }), 500


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status":           "ok",
        "active_decoder":   ACTIVE_DECODER,
        "opencv_available": CV2_AVAILABLE,
        "threshold_order":  THRESHOLD_ORDER,
        "note":             "Always starts from sweet spot 120, sweeps outward both directions"
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
