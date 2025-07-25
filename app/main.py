import os
import pickle
import logging
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import streamlit as st
from datetime import datetime
from collections import OrderedDict

# =====================================
# Configuration and Paths
# =====================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, 'models', 'neural_network')

# Model artifacts paths
MODEL_PATHS = {
    'model': os.path.join(MODEL_DIR, 'model_nn.h5'),
    'preprocessor': os.path.join(MODEL_DIR, 'preprocessor_nn.pkl'),
    'feature_names': os.path.join(MODEL_DIR, 'feature_names.pkl'),
}

# Verify paths exist
for name, path in MODEL_PATHS.items():
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find {name} at {path}")

# =====================================
# Logging Configuration
# =====================================
LOG_DIR = os.path.join(BASE_DIR, 'app', 'logs')
os.makedirs(LOG_DIR, exist_ok=True)

# Configure logging
log_file = os.path.join(LOG_DIR, f'app_{datetime.now().strftime("%Y%m%d")}.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# =====================================
# Feature Constants
# =====================================
ALL_FEATURE_NAMES = [
    'age', 'job', 'marital', 'education', 'default',
    'balance', 'housing', 'loan', 'contact', 'day', 'month', 'duration',
    'campaign', 'pdays', 'previous', 'poutcome'
]

# Numeric features that should not be encoded
NUMERIC_FEATURES = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']

# Categorical features that need encoding
CATEGORICAL_FEATURES = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']

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

MONTH_MAPPING = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12
}

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
        campaign = st.slider("Number of contacts performed", 1, 50, 1)
        pdays = st.slider("Days since last contact", -1, 999, -1, 
                         help="-1 means client was not previously contacted")
        previous = st.slider("Previous contacts", 0, 50, 0)
        poutcome = st.selectbox("Outcome of previous campaign", POUTCOME)

        submitted = st.form_submit_button("Predict")

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

def preprocess_features(features_df):
    """Preprocess features to match model requirements."""
    # Create a copy to avoid modifying the original
    df = features_df.copy()
    
    # Keep original month as month_text
    df['month_text'] = df['month'].str.lower()
    
    # Convert month to numeric using mapping
    df['month'] = df['month'].str.lower().map(MONTH_MAPPING)
    
    # Ensure numeric features are float
    for col in NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Create derived features
    df['was_previously_contacted'] = (df['pdays'] != -1).astype(float)
    
    # Create contact categories
    df['last_contact_category'] = pd.cut(
        df['pdays'],
        bins=[-float('inf'), -1, 7, 30, float('inf')],
        labels=['never', 'recent', 'moderate', 'long_ago']
    )
    
    # Create previous contact categories
    df['previous_contact_category'] = pd.cut(
        df['previous'],
        bins=[-float('inf'), 0, 1, 5, float('inf')],
        labels=['never_contacted', 'contacted_once', 'few_times', 'many_times']
    )
    
    # Ensure categorical features are strings
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype(str)
    
    return df

def predict_subscription_proba(features_df):
    try:
        logger.info("Loading neural network model and preprocessor")
        model = tf.keras.models.load_model(MODEL_PATHS['model'])
        logger.info("Model loaded successfully")
        
        with open(MODEL_PATHS['preprocessor'], 'rb') as f:
            preprocessor = pickle.load(f)
        logger.info("Preprocessor loaded successfully")
        
        # Debug: Log preprocessor configuration
        if hasattr(preprocessor, 'transformers_'):
            logger.info("Preprocessor transformers:")
            for name, transformer, columns in preprocessor.transformers_:
                logger.info(f"Transformer: {name}")
                logger.info(f"Columns: {columns}")
                logger.info(f"Type: {type(transformer).__name__}")
        
        with open(MODEL_PATHS['feature_names'], 'rb') as f:
            expected_features = pickle.load(f)
        
        logger.info(f"Expected features: {expected_features}")
        logger.info(f"Input features before preprocessing: {features_df.columns.tolist()}")
        logger.info("Input month value:", features_df['month'].iloc[0])
        
        # Apply custom preprocessing first
        features_df = preprocess_features(features_df)
        logger.info(f"Features after custom preprocessing: {features_df.columns.tolist()}")
        logger.info("Month value after preprocessing:", features_df['month'].iloc[0])
        logger.info("Month_text value after preprocessing:", features_df['month_text'].iloc[0])
        
        # Add debug logging for feature types
        logger.info("Feature types before preprocessor:")
        for col in features_df.columns:
            logger.info(f"{col}: {features_df[col].dtype} = {features_df[col].iloc[0]}")
        
        logger.info("Applying saved preprocessor")
        X_processed = preprocessor.transform(features_df)
        X_processed = X_processed.astype('float32')
        
        logger.info("Making prediction")
        prob = float(model.predict(tf.convert_to_tensor(X_processed), verbose=0)[0][0])
        logger.info(f"Prediction completed: {prob:.4f}")
        
        return prob
        
    except Exception as e:
        logger.error(f"Error in prediction pipeline: {str(e)}", exc_info=True)
        st.error(f"Error loading model or making prediction: {str(e)}")
        raise

# Main area
col1, col2 = st.columns([2, 1])
with col1:
    st.title("Bank Term Deposit Subscription Predictor")
    st.write("""
    Enter customer information on the left and click **Predict** to assess the likelihood 
    this customer will subscribe to a term deposit (based on historical marketing data).
    """)

# Get user input
user_input = get_user_input_features()

if user_input:
    st.header("User Input Features")
    st.json(user_input)
    
    try:
        # Create DataFrame from input
        df_input = pd.DataFrame([user_input])
        
        # Get prediction probability
        proba = predict_subscription_proba(df_input)
        
        # Display results
        st.header("Prediction")
        st.metric("Subscription Probability", f"{proba:.2%}")
        
        if proba > 0.6:
            st.success("High likelihood! Consider contacting this customer.")
        else:
            st.info("Low to moderate likelihood.")
            
        # Model info
        with st.expander("Model Information"):
            st.write("Using Neural Network model for predictions")
    
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
else:
    st.info("👈 Fill out the form in the sidebar and click Predict!")