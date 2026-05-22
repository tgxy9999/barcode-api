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

# ── OpenCV decoder init ──
try:
    cv2_detector = cv2.barcode.BarcodeDetector()
    CV2_AVAILABLE = True
except Exception:
    cv2_detector = None
    CV2_AVAILABLE = False


def decode_base64_image(b64_string: str) -> np.ndarray:
    if ',' in b64_string:
        b64_string = b64_string.split(',')[1]
    img_bytes = base64.b64decode(b64_string.strip())
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    return cv2.imdecode(img_array, cv2.IMREAD_COLOR)


def to_pil(img: np.ndarray) -> Image.Image:
    """Convert numpy array to PIL via PNG encode — simulates save/reload."""
    _, buf = cv2.imencode('.png', img)
    pil = Image.open(io.BytesIO(buf.tobytes()))
    pil.load()
    return pil


def scan_with_zxing(img: np.ndarray) -> str | None:
    try:
        results = zxingcpp.read_barcodes(to_pil(img))
        if results:
            return results[0].text
    except Exception:
        pass
    return None


def scan_with_opencv(img: np.ndarray) -> str | None:
    if not CV2_AVAILABLE:
        return None
    try:
        ret, decoded_list, _, _ = cv2_detector.detectAndDecodeMulti(img)
        if ret and decoded_list:
            for val in decoded_list:
                if val and val.strip():
                    return val.strip()
    except Exception:
        pass
    return None


def try_scan(img: np.ndarray, label: str) -> dict | None:
    """Try both decoders on one image. Return result or None."""
    for fn, name in [(scan_with_opencv, "opencv"),
                     (scan_with_zxing,  "zxing")]:
        val = fn(img)
        if val:
            return {"value": val, "decoder": name, "variant": label}
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

        # ════════════════════════════════════════════════
        # STEP 1 — original image (no processing)
        # ════════════════════════════════════════════════
        result = try_scan(img, "original")
        if result:
            return jsonify({"isReadable": True,
                            "barcodeValue": result["value"],
                            "variant": result["variant"],
                            "decoder": result["decoder"],
                            "error": None})

        # ════════════════════════════════════════════════
        # STEP 2 — grayscale (your notebook step 1)
        # ════════════════════════════════════════════════
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        result = try_scan(gray, "grayscale")
        if result:
            return jsonify({"isReadable": True,
                            "barcodeValue": result["value"],
                            "variant": result["variant"],
                            "decoder": result["decoder"],
                            "error": None})

        # ════════════════════════════════════════════════
        # STEP 3 — direct threshold (your notebook step 2)
        # Sweep threshold values around your working value of 120
        # lower = more white, higher = more black
        # ════════════════════════════════════════════════
        kernel = np.ones((2, 2), np.uint8)  # your notebook kernel

        for tval in [120, 100, 110, 130, 140, 80, 150, 160, 180, 200, 90]:
            # Your exact notebook code
            binary = np.where(gray > tval, 255, 0).astype(np.uint8)

            result = try_scan(binary, f"threshold_{tval}")
            if result:
                return jsonify({"isReadable": True,
                                "barcodeValue": result["value"],
                                "variant": result["variant"],
                                "decoder": result["decoder"],
                                "error": None})

            # ════════════════════════════════════════════
            # STEP 4 — morph open cleanup (your notebook step 3)
            # ════════════════════════════════════════════
            cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

            result = try_scan(cleaned, f"threshold_{tval}_cleaned")
            if result:
                return jsonify({"isReadable": True,
                                "barcodeValue": result["value"],
                                "variant": result["variant"],
                                "decoder": result["decoder"],
                                "error": None})

            # ════════════════════════════════════════════
            # STEP 5 — inverted versions of each threshold
            # ════════════════════════════════════════════
            inv_binary  = cv2.bitwise_not(binary)
            inv_cleaned = cv2.bitwise_not(cleaned)

            result = try_scan(inv_binary, f"inv_threshold_{tval}")
            if result:
                return jsonify({"isReadable": True,
                                "barcodeValue": result["value"],
                                "variant": result["variant"],
                                "decoder": result["decoder"],
                                "error": None})

            result = try_scan(inv_cleaned, f"inv_threshold_{tval}_cleaned")
            if result:
                return jsonify({"isReadable": True,
                                "barcodeValue": result["value"],
                                "variant": result["variant"],
                                "decoder": result["decoder"],
                                "error": None})

        # Nothing worked
        return jsonify({"isReadable": False,
                        "barcodeValue": None,
                        "variant": None,
                        "decoder": None,
                        "error": "Barcode not readable"})

    except Exception as e:
        return jsonify({"isReadable": False, "barcodeValue": None,
                        "error": str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok",
                    "opencv_available": CV2_AVAILABLE})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
