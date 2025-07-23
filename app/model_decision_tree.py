import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
import joblib
import pickle

# Load data
df = pd.read_csv("bank-full.csv", sep=';')

# Encode categorical variables
label_encoders = {}
for col in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Feature scaling
features = df.drop('y', axis=1)
target = df['y']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, target, test_size=0.2, random_state=42)

# Train Decision Tree model
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

# Save model and preprocessing tools
joblib.dump(dt_model, "dt_model.pkl")
pickle.dump(scaler, open("scaler.pkl", "wb"))
pickle.dump(label_encoders, open("encoders.pkl", "wb"))

print("✅ Decision Tree model training complete and saved.")
