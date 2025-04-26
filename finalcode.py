import cv2
import pytesseract
import numpy as np
import requests
import pandas as pd
import joblib
from io import BytesIO

# Tesseract executable path
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def detect_plate_from_url(image_url):
    # Step 1: Download image from the URL
    response = requests.get(image_url)
    if response.status_code != 200:
        print("❌ Failed to download image from URL!")
        return None

    # Step 2: Convert image bytes to numpy array
    image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if image is None:
        print("❌ Failed to decode image!")
        return None

    # Step 3-7: Preprocessing
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.convertScaleAbs(gray, alpha=1.8, beta=0)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    thresholded = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    sharpening_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened = cv2.filter2D(thresholded, -1, sharpening_kernel)

    # Step 8: OCR
    custom_config = r'--psm 7 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789KL '
    text = pytesseract.image_to_string(sharpened, config=custom_config)
    text = text.strip().replace("\n", " ").replace(" ", "")

    print("🚗 Detected Number Plate Text:", text)

    # Optional: Show processed image for debugging
    # cv2.imshow("Sharpened", sharpened)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return text

def get_safety_score(number_plate):
    df = pd.read_csv('data.csv')
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df.ffill(inplace=True)

    vehicle_data = df[df['Number Plate'] == number_plate]

    if vehicle_data.empty:
        return "❌ Error: Vehicle not found in database"

    features = vehicle_data.iloc[0][[
        'Vehicle Type', 'Owner Age', 'License Type', 'Violation Type',
        'Severity', 'Repeated Offense', 'Total Violations',
        'Days Since Last Violation', 'Weather'
    ]].to_dict()

    model = joblib.load('vehicle_safety_model.pkl')
    encoders = {
        col: joblib.load(f'encoders/{col}_encoder.pkl') 
        for col in ['Vehicle Type', 'License Type', 'Violation Type',
                    'Severity', 'Weather', 'Repeated Offense']
    }

    encoded_data = {}
    for col, value in features.items():
        if col in encoders:
            try:
                encoded_data[col] = encoders[col].transform([value])[0]
            except ValueError:
                encoded_data[col] = -1
        else:
            encoded_data[col] = value

    model_features = joblib.load('model_features.pkl')
    input_data = pd.DataFrame([encoded_data]).reindex(columns=model_features, fill_value=0)

    safety_prob = model.predict_proba(input_data)[0][1]
    return safety_prob

if __name__ == "__main__":
    image_url = "https://acko-cms.ackoassets.com/Fancy_Number_Plate_in_Kerala_af4c9079b6.png?fm=webp&w=800&q=75"
    number_plate = detect_plate_from_url(image_url)

    if number_plate:
        score = get_safety_score(number_plate)
        if isinstance(score, float):
            print(f"\n🚗 Safety Score for {number_plate}: {score:.0%}")
            print("📊 Interpretation:")
            print("90%-100%: Very Safe")
            print("75%-89%: Safe")
            print("60%-74%: Moderately Safe")
            print("40%-59%: Risky")
            print("0%-39%: Dangerous")
        else:
            print(score)
