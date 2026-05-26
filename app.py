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

# ── Try OpenCV decoder — use zxing only if OpenCV library fails to load ──
try:
    cv2_detector = cv2.barcode.BarcodeDetector()
    CV2_AVAILABLE = True
    ACTIVE_DECODER = "opencv"
except Exception:
    cv2_detector = None
    CV2_AVAILABLE = False
    ACTIVE_DECODER = "zxing"


def decode_base64_image(b64_string: str) -> np.ndarray:
    if ',' in b64_string:
        b64_string = b64_string.split(',')[1]
    img_bytes = base64.b64decode(b64_string.strip())
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    return cv2.imdecode(img_array, cv2.IMREAD_COLOR)


def img_to_base64(img: np.ndarray) -> str:
    """Convert numpy image to base64 PNG string."""
    _, buf = cv2.imencode('.png', img)
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode('utf-8')


def to_pil(img: np.ndarray) -> Image.Image:
    """Encode to PNG then reload — simulates notebook save/reload cycle."""
    _, buf = cv2.imencode('.png', img)
    pil = Image.open(io.BytesIO(buf.tobytes()))
    pil.load()
    return pil


def scan(img: np.ndarray) -> str | None:
    """OpenCV primary, zxing fallback if OpenCV not available."""
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


def get_brightness(gray: np.ndarray) -> float:
    """Get average brightness of image (0=black, 255=white)."""
    return float(np.mean(gray))


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
            return jsonify({"isReadable": True, "barcodeValue": val,
                            "variant": "original", "decoder": ACTIVE_DECODER,
                            "processedImage": img_to_base64(img), "error": None})

        # ════════════════════════════════════════
        # STEP 2 — grayscale
        # ════════════════════════════════════════
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        val = scan(gray)
        if val:
            return jsonify({"isReadable": True, "barcodeValue": val,
                            "variant": "grayscale", "decoder": ACTIVE_DECODER,
                            "processedImage": img_to_base64(gray), "error": None})

        # ════════════════════════════════════════
        # Detect image brightness
        # This decides threshold sweep order:
        # Dark image  → start LOW (30-80) to avoid all-black output
        # Light image → start at your sweet spot (110-130)
        # ════════════════════════════════════════
        brightness   = get_brightness(gray)
        is_dark      = brightness < 100   # avg pixel < 100 = dark image
        kernel       = np.ones((2, 2), np.uint8)
        last_image   = gray

        # ── Threshold order based on brightness ──
        if is_dark:
            # Dark image: try LOW thresholds first
            # Low threshold → more pixels become WHITE → clears dark background
            threshold_order = [40, 50, 60, 70, 30, 80, 90, 100, 110, 120, 130, 140, 150, 160, 180, 200, 20]
        else:
            # Normal/light image: your sweet spot first (110-130)
            threshold_order = [120, 110, 130, 100, 140, 80, 150, 160, 180, 200, 90, 60, 40, 30, 20]

        # ════════════════════════════════════════
        # STEP 3 — direct threshold sweep
        # Your exact notebook method
        # ════════════════════════════════════════
        for tval in threshold_order:
            # Your exact notebook threshold
            binary     = np.where(gray > tval, 255, 0).astype(np.uint8)
            last_image = binary

            val = scan(binary)
            if val:
                return jsonify({"isReadable": True, "barcodeValue": val,
                                "variant": f"threshold_{tval}",
                                "decoder": ACTIVE_DECODER,
                                "processedImage": img_to_base64(binary),
                                "error": None})

            # Your exact notebook morph open cleanup
            cleaned    = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            last_image = cleaned

            val = scan(cleaned)
            if val:
                return jsonify({"isReadable": True, "barcodeValue": val,
                                "variant": f"threshold_{tval}_cleaned",
                                "decoder": ACTIVE_DECODER,
                                "processedImage": img_to_base64(cleaned),
                                "error": None})

        # ════════════════════════════════════════
        # STEP 4 — inverted versions
        # For reversed barcodes (light bars on dark background)
        # ════════════════════════════════════════
        inv_gray = cv2.bitwise_not(gray)
        inv_brightness = get_brightness(inv_gray)

        # Sweep thresholds on inverted image
        if inv_brightness < 100:
            inv_order = [40, 50, 60, 70, 30, 80, 90, 100, 110, 120, 130]
        else:
            inv_order = [120, 110, 130, 100, 140, 80, 150, 160, 60, 40]

        for tval in inv_order:
            # Invert first then threshold
            inv_binary  = np.where(inv_gray > tval, 255, 0).astype(np.uint8)
            inv_cleaned = cv2.morphologyEx(inv_binary, cv2.MORPH_OPEN, kernel)
            last_image  = inv_cleaned

            val = scan(inv_binary)
            if val:
                return jsonify({"isReadable": True, "barcodeValue": val,
                                "variant": f"inv_threshold_{tval}",
                                "decoder": ACTIVE_DECODER,
                                "processedImage": img_to_base64(inv_binary),
                                "error": None})

            val = scan(inv_cleaned)
            if val:
                return jsonify({"isReadable": True, "barcodeValue": val,
                                "variant": f"inv_threshold_{tval}_cleaned",
                                "decoder": ACTIVE_DECODER,
                                "processedImage": img_to_base64(inv_cleaned),
                                "error": None})

        # ════════════════════════════════════════
        # STEP 5 — CLAHE then threshold
        # Boosts local contrast before thresholding
        # Helps when background varies across image
        # ════════════════════════════════════════
        clahe      = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe_gray = clahe.apply(gray)

        for tval in [120, 110, 130, 100, 140, 80, 60, 40]:
            binary     = np.where(clahe_gray > tval, 255, 0).astype(np.uint8)
            cleaned    = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            last_image = cleaned

            val = scan(binary)
            if val:
                return jsonify({"isReadable": True, "barcodeValue": val,
                                "variant": f"clahe_threshold_{tval}",
                                "decoder": ACTIVE_DECODER,
                                "processedImage": img_to_base64(binary),
                                "error": None})

            val = scan(cleaned)
            if val:
                return jsonify({"isReadable": True, "barcodeValue": val,
                                "variant": f"clahe_threshold_{tval}_cleaned",
                                "decoder": ACTIVE_DECODER,
                                "processedImage": img_to_base64(cleaned),
                                "error": None})

        # Return last processed image for debugging even on failure
        return jsonify({
            "isReadable":     False,
            "barcodeValue":   None,
            "variant":        None,
            "decoder":        ACTIVE_DECODER,
            "brightness":     round(brightness, 1),
            "imageType":      "dark" if is_dark else "normal",
            "processedImage": img_to_base64(last_image),
            "error":          "Barcode not readable"
        })

    except Exception as e:
        return jsonify({"isReadable": False, "barcodeValue": None,
                        "processedImage": None, "error": str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status":           "ok",
        "active_decoder":   ACTIVE_DECODER,
        "opencv_available": CV2_AVAILABLE,
        "note":             "Auto-detects dark/light image and adjusts threshold sweep order"
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
