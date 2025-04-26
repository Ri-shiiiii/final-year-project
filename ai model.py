import pandas as pd

# Load the CSV file, skipping the first row (which is the original header)
df = pd.read_csv('data.csv', skiprows=0)

# Drop any unwanted 'Unnamed' index columns if they exist
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# Preview the cleaned data
print("✅ Cleaned Data Preview:")
print(df.head())

# Optional: Check column names
print("\n📋 Column Names:")
print(df.columns.tolist())

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Step 1: Handle missing values (if any)
df.fillna(method='ffill', inplace=True)

# Step 2: Encode categorical columns
label_encoder = LabelEncoder()

df['Vehicle Type'] = label_encoder.fit_transform(df['Vehicle Type'])
df['License Type'] = label_encoder.fit_transform(df['License Type'])
df['Violation Type'] = label_encoder.fit_transform(df['Violation Type'])
df['Severity'] = label_encoder.fit_transform(df['Severity'])
df['Weather'] = label_encoder.fit_transform(df['Weather'])
df['Repeated Offense'] = label_encoder.fit_transform(df['Repeated Offense'])
df['Safe'] = label_encoder.fit_transform(df['Safe'])  # Target column

# Optional: You can encode Location and Number Plate too, but not necessary unless useful.

# Step 3: Define feature set (X) and target (y)
X = df[['Vehicle Type', 'Owner Age', 'License Type', 'Violation Type', 
        'Severity', 'Repeated Offense', 'Total Violations', 
        'Days Since Last Violation', 'Weather']]
y = df['Safe']

# Step 4: Train-test split (70% training, 30% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Check shape of datasets
print("✅ Training set size:", X_train.shape)
print("✅ Testing set size:", X_test.shape)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Step 1: Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 2: Make predictions
y_pred = model.predict(X_test)

# Step 3: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("✅ Model Accuracy:", round(accuracy * 100, 2), "%")

print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

print("🔍 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Step 1: Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 2: Make predictions
y_pred = model.predict(X_test)

# Step 3: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("✅ Model Accuracy:", round(accuracy * 100, 2), "%")

print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

print("🔍 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Example new vehicle input (same order of columns as X)
new_data = pd.DataFrame([{
    'Vehicle Type': 'Car',
    'Owner Age': 29,
    'License Type': 'LMV',
    'Violation Type': 'Speeding',
    'Severity': 'High',
    'Repeated Offense': 'No',
    'Total Violations': 1,
    'Days Since Last Violation': 100,
    'Weather': 'Clear'
}])

# Encode using the same encoders used during training
new_data_encoded = pd.get_dummies(new_data)
new_data_encoded = new_data_encoded.reindex(columns=X.columns, fill_value=0)

# Predict
prediction = model.predict(new_data_encoded)[0]
print("🚗 Prediction for New Vehicle is:", "✅ Safe" if prediction == 1 else "❌ Not Safe")

import joblib
joblib.dump(model, 'vehicle_safety_model.pkl')

import joblib
# Assuming your features are stored in X (as in your training script)
joblib.dump(list(X.columns), 'model_features.pkl')
print("✅ Saved model features to 'model_features.pkl'")

from sklearn.preprocessing import LabelEncoder
import joblib

# Initialize encoders for each categorical column
encoders = {
    'Vehicle Type': LabelEncoder(),
    'License Type': LabelEncoder(),
    'Violation Type': LabelEncoder(),
    'Severity': LabelEncoder(),
    'Weather': LabelEncoder(),
    'Repeated Offense': LabelEncoder(),
    'Safe': LabelEncoder()
}

# Fit and transform each column
for col in encoders:
    df[col] = encoders[col].fit_transform(df[col])
    # Save the encoder
    joblib.dump(encoders[col], f'{col}_encoder.pkl')

    # Load encoders (do this once)
encoders = {
    col: joblib.load(f'{col}_encoder.pkl') 
    for col in ['Vehicle Type', 'License Type', 'Violation Type', 
                'Severity', 'Weather', 'Repeated Offense']
}

# Encode new data
new_data = pd.DataFrame([{
    'Vehicle Type': 'Car',
    'Owner Age': 29,
    'License Type': 'LMV',
    'Violation Type': 'Speeding',
    'Severity': 'High',
    'Repeated Offense': 'No',
    'Total Violations': 1,
    'Days Since Last Violation': 100,
    'Weather': 'Clear'
}])

# Apply encoding using the saved encoders
for col in encoders:
    # Handle unseen categories (if any)
    if new_data[col].iloc[0] not in encoders[col].classes_:
        new_data[col] = -1  # Assign a default value
    else:
        new_data[col] = encoders[col].transform([new_data[col].iloc[0]])

        # Reorder columns to match training data
new_data_encoded = new_data[X.columns]