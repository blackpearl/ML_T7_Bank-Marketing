import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib

# Load and prepare data
df = pd.read_csv('app/bank-full.csv', sep=';')

# Initialize encoders dictionary
encoders = {}
categorical_cols = ['job', 'marital', 'education', 'default', 'housing', 
                   'loan', 'contact', 'month', 'poutcome']

# Convert categorical variables
for col in categorical_cols:
    encoders[col] = LabelEncoder()
    df[col] = encoders[col].fit_transform(df[col])

# Handle target variable separately
target_encoder = LabelEncoder()
y = target_encoder.fit_transform(df['y'])

# Split features and target
X = df.drop('y', axis=1)
y = df['y']

# Scale numerical features
scaler = StandardScaler()
numerical_cols = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])



# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train simple model
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Print results
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save everything we need
joblib.dump(model, 'app/model.pkl')
joblib.dump(encoders, 'app/encoders.pkl')
joblib.dump(scaler, 'app/scaler.pkl')
print("\nModel and preprocessing components have been saved")
