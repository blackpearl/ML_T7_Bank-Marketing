import pandas as pd
import joblib
import pickle

# Load models and preprocessing artifacts
rf_model = joblib.load("rf_model.pkl")
dt_model = joblib.load("dt_model.pkl")
scaler = pickle.load(open("scaler.pkl", "rb"))
label_encoders = pickle.load(open("encoders.pkl", "rb"))

# Sample input
sample = {
    'age': 35,
    'job': 'admin.',
    'marital': 'married',
    'education': 'tertiary',
    'default': 'no',
    'housing': 'yes',
    'loan': 'no',
    'contact': 'cellular',
    'month': 'may',
    'day': 5,
    'duration': 180,
    'campaign': 1,
    'pdays': 999,
    'previous': 0,
    'poutcome': 'unknown',
    'emp.var.rate': 1.1,
    'cons.price.idx': 93.994,
    'cons.conf.idx': -36.4,
    'euribor3m': 4.857,
    'nr.employed': 5191,
    'balance': 1500
}

# Convert to DataFrame
df = pd.DataFrame([sample])

# Encode categorical variables
for col, encoder in label_encoders.items():
    if col in df.columns:
        try:
            df[col] = encoder.transform(df[col])
        except ValueError as e:
            print(f"⚠️ Unknown label in '{col}': {df[col].values[0]}")
            print(f"Allowed: {list(encoder.classes_)}")
            exit(1)

# Apply scaling to numerical columns
numeric_cols = scaler.feature_names_in_
df[numeric_cols] = scaler.transform(df[numeric_cols])

# Use only the expected features (match training set)
expected_features = [
    'age', 'job', 'marital', 'education', 'default', 'housing', 'loan',
    'contact', 'month', 'day', 'duration', 'campaign', 'pdays', 'previous',
    'poutcome', 'balance'
]
df = df[expected_features]

# Predict using both models
rf_pred = rf_model.predict(df.values)[0]
dt_pred = dt_model.predict(df.values)[0]

# Output results
print(f"🔍 Random Forest Prediction: {'Subscribed' if rf_pred == 1 else 'Not Subscribed'}")
print(f"🔍 Decision Tree Prediction: {'Subscribed' if dt_pred == 1 else 'Not Subscribed'}")
