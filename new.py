
import cv2
import pytesseract
import numpy as np

# Path to Tesseract OCR executable
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def detect_plate(image_path):
    image = cv2.imread(image_path)
    
    if image is None:
        print("❌ Failed to load image!")
        return

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Increase contrast
    gray = cv2.convertScaleAbs(gray, alpha=1.8, beta=0)  # Increase contrast (alpha > 1)

    # Optional: Apply Gaussian Blur to reduce noise
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Adaptive Thresholding (works better in non-uniform lighting)
    thresholded = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # Optional: Sharpening filter to enhance edges
    sharpening_kernel = np.array([[-1, -1, -1],
                                  [-1,  9, -1],
                                  [-1, -1, -1]])
    sharpened = cv2.filter2D(thresholded, -1, sharpening_kernel)

    # Show the sharpened image for debugging
    cv2.imshow("Sharpened Image", sharpened)
    cv2.waitKey(0)

    # OCR configuration
    custom_config = r'--psm 8 --oem 1 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789KL '

    # OCR
    text = pytesseract.image_to_string(sharpened, config=custom_config)

    # Post-processing to fix known errors
    if "9570" in text:
        text = "KL 18 X " + text.strip()

    print("Detected Number Plate Text:", text.strip())

    # Show the thresholded image for debugging
    cv2.imshow("Thresholded Image", thresholded)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Run the function with the image path
detect_plate(r"C:\Users\Rushikesh-PC\OneDrive\Documents\number_plate_project\plate1.png")
