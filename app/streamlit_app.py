import os
import joblib
import pandas as pd
import streamlit as st
from collections import OrderedDict
from sklearn.preprocessing import LabelEncoder, StandardScaler

st.set_page_config(layout="wide")

# Feature names from our dataset
ALL_FEATURE_NAMES = [
    'age', 'job', 'marital', 'education', 'default',
    'balance', 'housing', 'loan', 'contact', 'day', 'month', 'duration',
    'campaign', 'pdays', 'previous', 'poutcome'
]

# Possible values for categorical features
JOBS = (
    "admin.", "blue-collar", "entrepreneur", "housemaid", "management", "retired",
    "self-employed", "services", "student", "technician", "unemployed", "unknown"
)
MARITAL = ("married", "single", "divorced")
EDUCATIONS = ("primary", "secondary", "tertiary", "unknown")
DEFAULTS = ("yes", "no")
HOUSING = ("yes", "no")
LOAN = ("yes", "no")
CONTACTS = ("cellular", "telephone", "unknown")
MONTHS = ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec")
POUTCOME = ("unknown", "other", "failure", "success")

def get_user_input_features():
    with st.sidebar.form(key="customer_features_form"):
        st.header("Input Customer Features")

        age = st.slider("Age", 18, 98, 30)
        job = st.selectbox("Job", JOBS)
        marital = st.selectbox("Marital Status", MARITAL)
        education = st.selectbox("Education", EDUCATIONS)
        default = st.radio("Has credit in default?", DEFAULTS)
        balance = st.slider("Balance", -5000, 100000, 0, step=100)
        housing = st.radio("Has housing loan?", HOUSING)
        loan = st.radio("Has personal loan?", LOAN)
        contact = st.selectbox("Contact communication type", CONTACTS)
        month = st.selectbox("Last contact month", MONTHS)
        day = st.slider("Last contact day of month", 1, 31, 15)
        duration = st.slider("Last contact duration (seconds)", 0, 5000, 300)
        campaign = st.slider("Number of contacts during this campaign", 1, 50, 1)
        pdays = st.slider("Days since client was last contacted (-1 means never)", -1, 999, -1)
        previous = st.slider("Number of contacts before this campaign", 0, 50, 0)
        poutcome = st.selectbox("Previous campaign outcome", POUTCOME)

        submitted = st.form_submit_button("Predict Subscription Likelihood")

    if submitted:
        features = OrderedDict({
            "age": age,
            "job": job,
            "marital": marital,
            "education": education,
            "default": default,
            "balance": balance,
            "housing": housing,
            "loan": loan,
            "contact": contact,
            "day": day,
            "month": month,
            "duration": duration,
            "campaign": campaign,
            "pdays": pdays,
            "previous": previous,
            "poutcome": poutcome,
        })
        return features
    return None

def encode_features(raw_features):
    # Create DataFrame and encode categorical variables
    df = pd.DataFrame([raw_features])
    
    # Load encoders and scaler
    encoders = joblib.load('app/encoders.pkl')
    scaler = joblib.load('app/scaler.pkl')
    
    # Transform categorical variables using saved encoders
    categorical_cols = ['job', 'marital', 'education', 'default', 'housing', 
                       'loan', 'contact', 'month', 'poutcome']
    
    for col in categorical_cols:
        df[col] = encoders[col].transform(df[col])
    
    # Transform numerical features using saved scaler
    numerical_cols = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
    df[numerical_cols] = scaler.transform(df[numerical_cols])
    
    return df

def predict_subscription_proba(features_df):
    # Load the model
    model = joblib.load('app/model.pkl')
    probas = model.predict_proba(features_df)
    return probas[0][1]  # Probability for 'yes'

# Main area
col1, col2 = st.columns([2, 1])
with col1:
    st.title("Bank Term Deposit Subscription Predictor")
    st.write("""
    Enter customer information on the left and click **Predict** to assess the likelihood 
    this customer will subscribe to a term deposit (based on historical marketing data).
    """)

# Inputs
user_input = get_user_input_features()

if user_input:
    st.header("User Input Features")
    st.json(user_input)
    
    # Feature engineering/encoding
    try:
        df_input = encode_features(user_input)
        st.header("Encoded Features (for Model Input)")
        st.dataframe(df_input)
        
        # Predict
        st.header("Model Prediction")
        proba = predict_subscription_proba(df_input)
        st.metric("Subscription Probability", f"{proba:.2%}")
        
        if proba > 0.6:
            st.success("High likelihood! Consider contacting this customer.")
        else:
            st.info("Low to moderate likelihood.")
        
        # Model explanation
        with st.expander("Show Feature Importance"):
            st.write("This prediction is based on customer demographics and previous marketing contact history.")
            
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
else:
    st.info("👈 Fill out the form in the sidebar and click Predict!")
