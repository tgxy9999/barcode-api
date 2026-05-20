from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64
import zxingcpp
from PIL import Image

app = Flask(__name__)
CORS(app)

# ── Max image dimension — resize anything larger to save memory ──
MAX_DIM = 1200


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
    """
    Resize image if larger than MAX_DIM.
    Prevents memory overflow on Render free tier (512MB RAM).
    """
    h, w = img.shape[:2]
    if max(h, w) > MAX_DIM:
        scale = MAX_DIM / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
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


def preprocess_variants(img: np.ndarray) -> list:
    """
    Lightweight preprocessing variants ordered from fastest to most expensive.
    Stops as soon as one works — avoids running all 14 if not needed.
    """
    variants = []

    # Resize first to control memory usage
    img = resize_if_large(img)

    # ── Fast variants first ──
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    variants.append(("original",        img))
    variants.append(("grayscale",       gray))

    # Inverted — handles reversed/white-on-dark barcodes
    variants.append(("inverted_gray",   cv2.bitwise_not(gray)))

    # CLAHE — best for colored backgrounds
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(gray)
    variants.append(("clahe",           clahe_img))

    # Otsu threshold
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(("otsu",            otsu))
    variants.append(("inverted_otsu",   cv2.bitwise_not(otsu)))

    # CLAHE + Otsu
    _, clahe_otsu = cv2.threshold(clahe_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(("clahe_otsu",          clahe_otsu))
    variants.append(("inverted_clahe_otsu", cv2.bitwise_not(clahe_otsu)))

    # ── Medium variants ──
    # Adaptive threshold — handles uneven lighting across label
    adaptive = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, blockSize=11, C=2
    )
    variants.append(("adaptive",          adaptive))
    variants.append(("inverted_adaptive", cv2.bitwise_not(adaptive)))

    # Sharpened
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(gray, -1, kernel)
    variants.append(("sharpened", sharpened))

    # ── Expensive variants last (only if nothing else worked) ──
    # Upscale small images
    h, w = gray.shape[:2]
    if w < 800:
        up = cv2.resize(gray, (int(w * 2), int(h * 2)), interpolation=cv2.INTER_CUBIC)
        clahe_up = clahe.apply(up)
        _, otsu_up = cv2.threshold(clahe_up, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants.append(("upscaled_clahe_otsu",          otsu_up))
        variants.append(("upscaled_inverted_clahe_otsu", cv2.bitwise_not(otsu_up)))

    # Denoise — most expensive, last resort
    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    _, denoised_thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(("denoised_otsu",     denoised_thresh))
    variants.append(("denoised_inverted", cv2.bitwise_not(denoised_thresh)))

    return variants


@app.route('/scan', methods=['POST', 'OPTIONS'])
def scan_barcode():
    """
    POST /scan
    Body: { "image": "<base64 — with or without data URI prefix>" }
    Returns: { isReadable, barcodeValue, barcodeFormat, successVariant, variantsTried, error }
    """
    if request.method == 'OPTIONS':
        return '', 204

    try:
        data = request.get_json(force=True)

        if not data or 'image' not in data:
            return jsonify({
                "isReadable": False, "barcodeValue": None,
                "barcodeFormat": None, "successVariant": None,
                "variantsTried": 0,
                "error": "Missing 'image' field in request body"
            }), 400

        img = decode_base64_image(data['image'])
        if img is None:
            return jsonify({
                "isReadable": False, "barcodeValue": None,
                "barcodeFormat": None, "successVariant": None,
                "variantsTried": 0,
                "error": "Could not decode image — invalid base64 or unsupported format"
            }), 400

        variants = preprocess_variants(img)
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
                    "error":          None
                })

        return jsonify({
            "isReadable": False, "barcodeValue": None,
            "barcodeFormat": None, "successVariant": None,
            "variantsTried": variants_tried,
            "error": "Barcode not readable after all preprocessing attempts"
        })

    except Exception as e:
        return jsonify({
            "isReadable": False, "barcodeValue": None,
            "barcodeFormat": None, "successVariant": None,
            "variantsTried": 0, "error": str(e)
        }), 500


@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "message": "Barcode API is running"})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
