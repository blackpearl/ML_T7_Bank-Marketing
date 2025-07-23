import pandas as pd
import numpy as np
import joblib
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("bank-full.csv", sep=';')

# Define target column
target_column = 'y'

# Load encoders and scaler
label_encoders = pickle.load(open("encoders.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Encode target column first ✅
target_encoder = label_encoders[target_column]
y = target_encoder.transform(df[target_column])  # now y = [0, 1]
X = df.drop(columns=[target_column])

# Encode categorical features only
for col, encoder in label_encoders.items():
    if col in X.columns:  # avoid 'y'
        X[col] = encoder.transform(X[col])

# Scale numeric features
X[scaler.feature_names_in_] = scaler.transform(X[scaler.feature_names_in_])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load models
rf_model = joblib.load("rf_model.pkl")
dt_model = joblib.load("dt_model.pkl")

# Predict
rf_preds = rf_model.predict(X_test)
dt_preds = dt_model.predict(X_test)

# Target names for readable labels
target_names = target_encoder.classes_

# === Random Forest Evaluation ===
print("🌲 Random Forest - Confusion Matrix:")
print(confusion_matrix(y_test, rf_preds))
print(classification_report(y_test, rf_preds, target_names=target_names))

sns.heatmap(confusion_matrix(y_test, rf_preds), annot=True, fmt='d', cmap='Blues')
plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# === Decision Tree Evaluation ===
print("🌳 Decision Tree - Confusion Matrix:")
print(confusion_matrix(y_test, dt_preds))
print(classification_report(y_test, dt_preds, target_names=target_names))

sns.heatmap(confusion_matrix(y_test, dt_preds), annot=True, fmt='d', cmap='Greens')
plt.title("Decision Tree Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
