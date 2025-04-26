import cv2
import pytesseract
import numpy as np
import requests
from io import BytesIO

# Path to Tesseract OCR executable
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def detect_plate_from_url(image_url):
    # Step 1: Download image from the URL
    response = requests.get(image_url)
    if response.status_code != 200:
        print("❌ Failed to download image from URL!")
        return

    # Step 2: Convert image bytes to numpy array
    image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if image is None:
        print("❌ Failed to decode image!")
        return

    # Step 3: Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Step 4: Increase contrast
    gray = cv2.convertScaleAbs(gray, alpha=1.8, beta=0)

    # Step 5: Gaussian blur to reduce noise
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Step 6: Adaptive Thresholding
    thresholded = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # Step 7: Sharpen the image
    sharpening_kernel = np.array([[-1, -1, -1],
                                  [-1,  9, -1],
                                  [-1, -1, -1]])
    sharpened = cv2.filter2D(thresholded, -1, sharpening_kernel)

    # Step 8: OCR
    custom_config = r'--psm 7 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789KL '
    text = pytesseract.image_to_string(sharpened, config=custom_config)

    # Step 9: Post-processing
    text = text.strip().replace("\n", " ")
    print("🚗 Detected Number Plate Text:", text)

    # Debug view
    cv2.imshow("Sharpened", sharpened)
    cv2.imshow("Thresholded", thresholded)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example call with your image URL
detect_plate_from_url("https://acko-cms.ackoassets.com/Fancy_Number_Plate_in_Kerala_af4c9079b6.png?fm=webp&w=800&q=75")
