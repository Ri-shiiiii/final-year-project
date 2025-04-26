

import pandas as pd
import joblib  # for loading model
import numpy as np

# Load the trained model (change name if needed)
model = joblib.load('vehicle_safety_model.pkl')  # <-- use your actual model file

# New input (raw)
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

# Manual preprocessing (just like in training)
categorical_cols = ['Vehicle Type', 'License Type', 'Violation Type', 'Severity', 'Location', 'Repeated Offense', 'Weather']
new_vehicle_data = pd.get_dummies(new_vehicle_data, columns=categorical_cols)

# Align with training features
# Load a sample training set or get feature names from training
required_columns = joblib.load('model_features.pkl')  # Save your training features once and use here

# Add missing columns
for col in required_columns:
    if col not in new_vehicle_data.columns:
        new_vehicle_data[col] = 0

# Ensure column order
new_vehicle_data = new_vehicle_data[required_columns]

# Predict probability
prob_safe = model.predict_proba(new_vehicle_data)[0][1]  # Probability of being 'Safe'

# Base score
base_score = prob_safe * 100

# Rule-based penalties
penalty = 0
if new_vehicle_data['Total Violations'].values[0] > 3:
    penalty += 10
if 'Repeated Offense_Yes' in new_vehicle_data.columns and new_vehicle_data['Repeated Offense_Yes'].values[0] == 1:
    penalty += 15
if new_vehicle_data['Days Since Last Violation'].values[0] < 30:
    penalty += 10
if 'Severity_High' in new_vehicle_data.columns and new_vehicle_data['Severity_High'].values[0] == 1:
    penalty += 15

# Final safety score
safety_score = round(max(0, base_score - penalty), 2)

# Output
print(f"🧠 Model Prediction Probability (Safe): {prob_safe:.2f}")
print(f"❗ Penalties Applied: -{penalty}")
print(f"✅ Final Safety Score: {safety_score} / 100")
#joblib.dump(list(X.columns), 'model_features.pkl')
