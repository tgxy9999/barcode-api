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
    ACTIVE_DECODER = "zxing"  # fallback


def decode_base64_image(b64_string: str) -> np.ndarray:
    if ',' in b64_string:
        b64_string = b64_string.split(',')[1]
    img_bytes = base64.b64decode(b64_string.strip())
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    return cv2.imdecode(img_array, cv2.IMREAD_COLOR)


def to_pil(img: np.ndarray) -> Image.Image:
    """Encode to PNG then reload — simulates save/reload from notebook."""
    _, buf = cv2.imencode('.png', img)
    pil = Image.open(io.BytesIO(buf.tobytes()))
    pil.load()
    return pil


def scan(img: np.ndarray) -> str | None:
    """
    Scan using OpenCV decoder.
    If OpenCV library failed to load, fall back to zxing-cpp.
    """
    if CV2_AVAILABLE:
        # ── Primary: OpenCV BarcodeDetector ──
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
        # ── Fallback: zxing-cpp (only if OpenCV not available) ──
        try:
            results = zxingcpp.read_barcodes(to_pil(img))
            if results:
                return results[0].text
        except Exception:
            pass
        return None


@app.route('/scan', methods=['POST', 'OPTIONS'])
def scan_barcode():
    if request.method == 'OPTIONS':
        return '', 204

    try:
        data = request.get_json(force=True)
        if not data or 'image' not in data:
            return jsonify({"isReadable": False, "barcodeValue": None,
                            "error": "Missing 'image' field"}), 400

        img = decode_base64_image(data['image'])
        if img is None:
            return jsonify({"isReadable": False, "barcodeValue": None,
                            "error": "Could not decode image"}), 400

        # ════════════════════════════════════════
        # STEP 1 — original image
        # ════════════════════════════════════════
        val = scan(img)
        if val:
            return jsonify({"isReadable": True, "barcodeValue": val,
                            "variant": "original",
                            "decoder": ACTIVE_DECODER, "error": None})

        # ════════════════════════════════════════
        # STEP 2 — grayscale (notebook step 1)
        # ════════════════════════════════════════
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        val = scan(gray)
        if val:
            return jsonify({"isReadable": True, "barcodeValue": val,
                            "variant": "grayscale",
                            "decoder": ACTIVE_DECODER, "error": None})

        # ════════════════════════════════════════
        # STEP 3 — direct threshold + morph open + inverted
        # Exactly your notebook code, threshold sweep around 120
        # ════════════════════════════════════════
        kernel = np.ones((2, 2), np.uint8)

        for tval in [120, 100, 110, 130, 140, 80, 150, 160, 180, 200, 90]:

            # Your exact notebook threshold
            binary = np.where(gray > tval, 255, 0).astype(np.uint8)

            val = scan(binary)
            if val:
                return jsonify({"isReadable": True, "barcodeValue": val,
                                "variant": f"threshold_{tval}",
                                "decoder": ACTIVE_DECODER, "error": None})

            # Your exact notebook morph open cleanup
            cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

            val = scan(cleaned)
            if val:
                return jsonify({"isReadable": True, "barcodeValue": val,
                                "variant": f"threshold_{tval}_cleaned",
                                "decoder": ACTIVE_DECODER, "error": None})

            # Inverted binary
            val = scan(cv2.bitwise_not(binary))
            if val:
                return jsonify({"isReadable": True, "barcodeValue": val,
                                "variant": f"inv_threshold_{tval}",
                                "decoder": ACTIVE_DECODER, "error": None})

            # Inverted cleaned
            val = scan(cv2.bitwise_not(cleaned))
            if val:
                return jsonify({"isReadable": True, "barcodeValue": val,
                                "variant": f"inv_threshold_{tval}_cleaned",
                                "decoder": ACTIVE_DECODER, "error": None})

        return jsonify({"isReadable": False, "barcodeValue": None,
                        "variant": None, "decoder": ACTIVE_DECODER,
                        "error": "Barcode not readable"})

    except Exception as e:
        return jsonify({"isReadable": False, "barcodeValue": None,
                        "error": str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status":          "ok",
        "active_decoder":  ACTIVE_DECODER,
        "opencv_available": CV2_AVAILABLE,
        "note": "OpenCV is primary decoder. zxing-cpp is fallback if OpenCV library fails to load."
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
