from flask import Flask, request, jsonify
import cv2
import numpy as np
from pyzbar.pyzbar import decode as pyzbar_decode
from pyzbar.pyzbar import ZBarSymbol
import base64
import re

app = Flask(__name__)

# ── Supported formats for pyzbar ──
ALL_FORMATS = [
    ZBarSymbol.CODE128,
    ZBarSymbol.CODE39,
    ZBarSymbol.CODE93,
    ZBarSymbol.CODABAR,
    ZBarSymbol.EAN13,
    ZBarSymbol.EAN8,
    ZBarSymbol.UPCA,
    ZBarSymbol.UPCE,
    ZBarSymbol.I25,
    ZBarSymbol.QRCODE,
    ZBarSymbol.PDF417,
    ZBarSymbol.DATAMATRIX,
]


def decode_base64_image(b64_string: str) -> np.ndarray:
    """Convert base64 string to OpenCV image."""
    # Strip data URI prefix if present e.g. "data:image/jpeg;base64,..."
    if ',' in b64_string:
        b64_string = b64_string.split(',')[1]
    img_bytes = base64.b64decode(b64_string)
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    return img


def try_decode(img: np.ndarray) -> list:
    """Attempt pyzbar decode on an image."""
    results = pyzbar_decode(img, symbols=ALL_FORMATS)
    return results


def preprocess_variants(img: np.ndarray) -> list:
    """
    Generate multiple preprocessed versions of the image.
    ZXing/pyzbar is tried on each variant — first successful decode wins.
    
    Variants handle:
    - Colored backgrounds (grayscale)
    - Low contrast (CLAHE enhancement)  
    - Overburnt bars (adaptive threshold)
    - Inverted barcodes (bitwise_not)
    - Slightly rotated (deskew attempt)
    """
    variants = []

    # ── 1. Original image as-is ──
    variants.append(("original", img))

    # ── 2. Grayscale ──
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    variants.append(("grayscale", gray))

    # ── 3. Grayscale + CLAHE (contrast limited adaptive histogram equalization)
    #       Best for colored backgrounds and uneven lighting
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(gray)
    variants.append(("clahe", clahe_img))

    # ── 4. Grayscale + Global threshold (Otsu) ──
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(("otsu_threshold", otsu))

    # ── 5. CLAHE + Otsu threshold ──
    _, clahe_otsu = cv2.threshold(clahe_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(("clahe_otsu", clahe_otsu))

    # ── 6. Adaptive threshold (handles uneven lighting across label) ──
    adaptive = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=11, C=2
    )
    variants.append(("adaptive_threshold", adaptive))

    # ── 7. CLAHE + Adaptive threshold ──
    clahe_adaptive = cv2.adaptiveThreshold(
        clahe_img, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=11, C=2
    )
    variants.append(("clahe_adaptive", clahe_adaptive))

    # ── 8. Inverted versions (handles white bars on dark background) ──
    for name, v in [("grayscale", gray), ("clahe_otsu", clahe_otsu), ("adaptive_threshold", adaptive)]:
        variants.append((f"inverted_{name}", cv2.bitwise_not(v)))

    # ── 9. Sharpened image ──
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(gray, -1, kernel)
    variants.append(("sharpened", sharpened))

    # ── 10. Upscaled image (helps with small/low-res barcodes) ──
    h, w = gray.shape[:2]
    if w < 1000:
        scale = 2.0
        upscaled = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
        variants.append(("upscaled_2x", upscaled))
        # Upscaled + CLAHE
        clahe_up = clahe.apply(upscaled)
        variants.append(("upscaled_clahe", clahe_up))
        # Upscaled + Otsu
        _, otsu_up = cv2.threshold(clahe_up, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants.append(("upscaled_clahe_otsu", otsu_up))

    # ── 11. Denoised + threshold (laser print noise removal) ──
    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    _, denoised_thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(("denoised_otsu", denoised_thresh))

    return variants


@app.route('/scan', methods=['POST'])
def scan_barcode():
    """
    POST /scan
    Body: { "image": "<base64 encoded image>" }
    Returns: {
        "isReadable": bool,
        "barcodeValue": str | null,
        "barcodeFormat": str | null,
        "successVariant": str | null,   (which preprocessing worked)
        "variantsTried": int,
        "error": str | null
    }
    """
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({
                "isReadable": False,
                "barcodeValue": None,
                "barcodeFormat": None,
                "successVariant": None,
                "variantsTried": 0,
                "error": "Missing 'image' field in request body"
            }), 400

        # Decode image
        img = decode_base64_image(data['image'])
        if img is None:
            return jsonify({
                "isReadable": False,
                "barcodeValue": None,
                "barcodeFormat": None,
                "successVariant": None,
                "variantsTried": 0,
                "error": "Could not decode image — check base64 encoding"
            }), 400

        # Try all preprocessing variants
        variants = preprocess_variants(img)
        variants_tried = 0

        for variant_name, variant_img in variants:
            variants_tried += 1
            results = try_decode(variant_img)
            if results:
                best = results[0]
                value = best.data.decode('utf-8', errors='replace')
                fmt   = best.type
                return jsonify({
                    "isReadable":     True,
                    "barcodeValue":   value,
                    "barcodeFormat":  fmt,
                    "successVariant": variant_name,
                    "variantsTried":  variants_tried,
                    "error":          None
                })

        # All variants failed
        return jsonify({
            "isReadable":     False,
            "barcodeValue":   None,
            "barcodeFormat":  None,
            "successVariant": None,
            "variantsTried":  variants_tried,
            "error":          "Barcode not readable after all preprocessing attempts"
        })

    except Exception as e:
        return jsonify({
            "isReadable":     False,
            "barcodeValue":   None,
            "barcodeFormat":  None,
            "successVariant": None,
            "variantsTried":  0,
            "error":          str(e)
        }), 500


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint — Power Automate can ping this to wake the server."""
    return jsonify({"status": "ok"})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
