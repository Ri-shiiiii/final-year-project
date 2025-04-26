import pandas as pd
import joblib  # or pickle if you saved the model using pickle
import numpy as np

# Load your trained model
model = joblib.load('vehicle_safety_model.pkl')  # Replace with your model file

# Sample input data for prediction (as DataFrame)
new_vehicle_data = pd.DataFrame([{
    'Vehicle Type': 'Car',
    'Owner Age': 31,
    'License Type': 'LMV',
    'Violation Type': 'Speeding',
    'Severity': 'High',
    'Location': 'Pune',
    'Repeated Offense': 'No',
    'Total Violations': 1,
    'Days Since Last Violation': 60,
    'Weather': 'Clear'
}])

# Preprocessing (e.g., encoding same way as training)
# -- This part should mirror your training pipeline --
# If you used LabelEncoder or OneHotEncoding, do the same here.
# Assuming you have a preprocessor saved (optional but ideal):
preprocessor = joblib.load('preprocessor.pkl')
input_data = preprocessor.transform(new_vehicle_data)

# OR: If you preprocessed manually, do the same steps here again.
# For now, assume it’s already encoded properly for simplicity:
input_data = new_vehicle_data  # Replace with actual preprocessing

# Predict probability of class 1 (Safe)
prob_safe = model.predict_proba(input_data)[0][1]  # Probability it's 'Safe'

# Calculate base score
base_score = prob_safe * 100

# Optional: Rule-based penalty system
penalty = 0
if new_vehicle_data.iloc[0]['Total Violations'] > 3:
    penalty += 10
if new_vehicle_data.iloc[0]['Repeated Offense'] == 'Yes':
    penalty += 15
if new_vehicle_data.iloc[0]['Days Since Last Violation'] < 30:
    penalty += 10
if new_vehicle_data.iloc[0]['Severity'] == 'High':
    penalty += 15

# Final Safety Score
safety_score = round(max(0, base_score - penalty), 2)

# Output
print(f"🧠 Model Prediction Probability (Safe): {prob_safe:.2f}")
print(f"❗ Penalties Applied: -{penalty}")
print(f"✅ Final Safety Score: {safety_score} / 100")
