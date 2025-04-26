from flask import Flask, request, jsonify
from flask_cors import CORS
import pytesseract
import cv2
import numpy as np
import requests
import pandas as pd
import joblib

# Initialize Flask app with CORS
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Configure Tesseract path (Update for your system)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def detect_plate_from_url(image_url):
    """Process image from URL to detect number plate"""
    try:
        # Download and decode image
        response = requests.get(image_url)
        if response.status_code != 200:
            return None
            
        image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if image is None:
            return None

        # Image processing pipeline
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.convertScaleAbs(gray, alpha=1.8, beta=0)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        thresholded = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        sharpened = cv2.filter2D(thresholded, -1, np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]))

        # OCR processing
        custom_config = r'--psm 7 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        text = pytesseract.image_to_string(sharpened, config=custom_config)
        
        # Clean and return text
        cleaned_text = text.strip()
        cleaned_text = cleaned_text.replace("\n", " ").replace(" ", "")
        return cleaned_text

    except Exception as e:
        print(f"OCR Error: {str(e)}")
        return None

def get_safety_score(number_plate):
    """Calculate safety score using ML model"""
    try:
        df = pd.read_csv('data/data.csv')
        df.ffill(inplace=True)
        vehicle_data = df[df['Number Plate'] == number_plate]
        
        if vehicle_data.empty:
            return {"error": "Vehicle not found in database"}
        
        # Load model and encoders
        model = joblib.load('models/vehicle_safety_model.pkl')
        encoders = {
            col: joblib.load(f'encoders/{col}_encoder.pkl') 
            for col in ['Vehicle Type', 'License Type', 'Violation Type', 
                       'Severity', 'Weather', 'Repeated Offense']
        }
        
        # Prepare and encode features
        features = vehicle_data.iloc[0][[
            'Vehicle Type', 'Owner Age', 'License Type', 'Violation Type',
            'Severity', 'Repeated Offense', 'Total Violations',
            'Days Since Last Violation', 'Weather'
        ]].to_dict()
        
        encoded_data = {}
        for col, value in features.items():
            if col in encoders:
                try:
                    encoded_data[col] = encoders[col].transform([value])[0]
                except ValueError:
                    encoded_data[col] = -1
            else:
                encoded_data[col] = value
                
        # Make prediction
        model_features = joblib.load('models/model_features.pkl')
        input_data = pd.DataFrame([encoded_data]).reindex(columns=model_features, fill_value=0)
        safety_prob = model.predict_proba(input_data)[0][1]
        return round(safety_prob * 100)

    except Exception as e:
        print(f"Prediction Error: {str(e)}")
        return {"error": str(e)}

@app.route('/api/check', methods=['POST'])
def check_plate():
    try:
        data = request.json
        plate = data.get('plate', '').strip().upper()
        
        if not plate:
            return jsonify({"error": "Number plate required"}), 400
        
        # Get vehicle info
        state_codes = {'KL': 'Kerala', 'MH': 'Maharashtra', 'KA': 'Karnataka'}
        state = state_codes.get(plate[:2], 'Unknown')
        rto = f"{plate[2:4]} RTO"
        
        # Get safety score
        score = get_safety_score(plate)
        if isinstance(score, dict):
            return jsonify(score), 404
        
        # Get violations (mock data)
        violations = [
            "Over Speeding (2023-03-15)",
            "Signal Jump (2022-12-01)"
        ]
        
        return jsonify({
            "state": state,
            "rto": rto,
            "score": score,
            "violations": violations
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)