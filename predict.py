import pandas as pd
import joblib

def get_safety_score(number_plate):
    # Load data and model
    df = pd.read_csv('data.csv')
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df.ffill(inplace=True)
    
    # Find vehicle by number plate
    vehicle_data = df[df['Number Plate'] == number_plate]
    
    if vehicle_data.empty:
        return "❌ Error: Vehicle not found in database"
    
    # Extract features
    features = vehicle_data.iloc[0][[
        'Vehicle Type', 'Owner Age', 'License Type', 'Violation Type',
        'Severity', 'Repeated Offense', 'Total Violations',
        'Days Since Last Violation', 'Weather'
    ]].to_dict()
    
    # Load model and encoders
    model = joblib.load('vehicle_safety_model.pkl')
    encoders = {
        col: joblib.load(f'encoders/{col}_encoder.pkl') 
        for col in ['Vehicle Type', 'License Type', 'Violation Type',
                    'Severity', 'Weather', 'Repeated Offense']
    }
    
    # Encode features
    encoded_data = {}
    for col, value in features.items():
        if col in encoders:
            try:
                encoded_data[col] = encoders[col].transform([value])[0]
            except ValueError:
                encoded_data[col] = -1  # Handle unseen categories
        else:
            encoded_data[col] = value
    
    # Create feature array
    model_features = joblib.load('model_features.pkl')
    input_data = pd.DataFrame([encoded_data]).reindex(columns=model_features, fill_value=0)
    
    # Get probability score
    safety_prob = model.predict_proba(input_data)[0][1]
    return safety_prob

if __name__ == "__main__":
    number_plate = input("Enter vehicle number plate: ").strip()
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