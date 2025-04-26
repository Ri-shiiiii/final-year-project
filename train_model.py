import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# Create directories for artifacts
os.makedirs('encoders', exist_ok=True)

# Load and clean data
df = pd.read_csv('data.csv')
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
df.ffill(inplace=True)  # Handle missing values

# Initialize and save encoders
encoders = {
    'Vehicle Type': LabelEncoder(),
    'License Type': LabelEncoder(),
    'Violation Type': LabelEncoder(),
    'Severity': LabelEncoder(),
    'Weather': LabelEncoder(),
    'Repeated Offense': LabelEncoder(),
    'Safe': LabelEncoder()
}

for col, encoder in encoders.items():
    df[col] = encoder.fit_transform(df[col])
    joblib.dump(encoder, f'encoders/{col}_encoder.pkl')

# Define features and target
X = df[['Vehicle Type', 'Owner Age', 'License Type', 'Violation Type',
        'Severity', 'Repeated Offense', 'Total Violations',
        'Days Since Last Violation', 'Weather']]
y = df['Safe']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(f"✅ Model Accuracy: {accuracy_score(y_test, y_pred):.2%}")
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

# Save artifacts
joblib.dump(model, 'vehicle_safety_model.pkl')
joblib.dump(X.columns.tolist(), 'model_features.pkl')
print("\n✅ Saved model, features, and encoders")