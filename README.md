# Barcode Preprocessing API
Free barcode scanning API with OpenCV preprocessing — built for Power Apps + Power Automate.

## What It Does
Takes a base64 image, runs 14+ preprocessing variants (grayscale, CLAHE contrast,
adaptive threshold, invert, upscale, denoise) and attempts pyzbar decode on each
until one succeeds. Returns the barcode value or a clear not-readable response.

## Deploy Free on Render.com (No Credit Card)

1. Fork or upload this repo to your GitHub account
2. Go to https://render.com → Sign up free
3. Click **New → Web Service**
4. Connect your GitHub repo
5. Set:
   - **Runtime:** Python 3
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app --bind 0.0.0.0:$PORT --workers 2 --timeout 60`
   - **Instance Type:** Free
6. Click **Deploy**
7. Your API URL: `https://your-app-name.onrender.com/scan`

## API Usage

### Endpoint
POST /scan

### Request Body
```json
{
  "image": "<base64 encoded image string>"
}
```

### Response — Success
```json
{
  "isReadable": true,
  "barcodeValue": "012345678912",
  "barcodeFormat": "CODE128",
  "successVariant": "clahe_otsu",
  "variantsTried": 5,
  "error": null
}
```

### Response — Not Readable
```json
{
  "isReadable": false,
  "barcodeValue": null,
  "barcodeFormat": null,
  "successVariant": null,
  "variantsTried": 14,
  "error": "Barcode not readable after all preprocessing attempts"
}
```

## Power Automate Setup

1. Trigger: PowerApps (receives base64 image string)
2. Action: HTTP POST to `https://your-app.onrender.com/scan`
   - Body: `{ "image": @{triggerBody()['image']} }`
3. Action: Parse JSON response
4. Action: Respond to PowerApps
   - isReadable: body('HTTP')['isReadable']
   - barcodeValue: body('HTTP')['barcodeValue']

## Power Apps Setup

```
// Button OnSelect — take photo and send to flow
Set(varResult, YourFlow.Run(JSON({image: Camera1.Photo})));

// Label Text — show result
If(varResult.isReadable, varResult.barcodeValue, "Not readable")
```

## Supported Barcode Formats
CODE128, CODE39, CODE93, CODABAR, EAN13, EAN8, UPCA, UPCE, ITF, QR Code, PDF417, DataMatrix

## Preprocessing Variants Attempted (in order)
1. Original image
2. Grayscale
3. CLAHE contrast enhancement
4. Otsu threshold
5. CLAHE + Otsu
6. Adaptive threshold
7. CLAHE + Adaptive threshold
8. Inverted grayscale
9. Inverted CLAHE+Otsu
10. Inverted adaptive threshold
11. Sharpened
12. Upscaled 2x + CLAHE
13. Upscaled 2x + CLAHE + Otsu
14. Denoised + Otsu
