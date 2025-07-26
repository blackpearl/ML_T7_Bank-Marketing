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
    """Get user input features from the sidebar."""
    with st.sidebar:
        st.header("Input Customer Features")
        
        # Numeric inputs
        age = st.slider("Age", min_value=18, max_value=98, value=30)
        
        # Categorical inputs
        job = st.selectbox("Job", options=JOBS, index=0)
        marital = st.selectbox("Marital Status", options=MARITAL, index=0)
        education = st.selectbox("Education", options=EDUCATIONS, index=0)
        
        # Binary inputs with radio buttons
        default = st.radio("Has credit in default?", options=DEFAULTS, horizontal=True)
        
        # Numeric input with slider
        balance = st.slider("Balance", min_value=-5000, max_value=100000, value=0)
        
        # More binary inputs
        housing = st.radio("Has housing loan?", options=HOUSING, horizontal=True)
        loan = st.radio("Has personal loan?", options=LOAN, horizontal=True)
        
        # Contact information
        contact = st.selectbox("Contact communication type", options=CONTACTS, index=0)
        month = st.selectbox("Last contact month", options=MONTHS, index=0)
        day = st.slider("Last contact day of month", min_value=1, max_value=31, value=15)
        
        # Campaign information
        duration = st.slider("Last contact duration (seconds)", min_value=0, max_value=5000, value=0)
        campaign = st.slider("Number of contacts performed", min_value=1, max_value=50, value=1)
        pdays = st.slider("Days since last contact", min_value=-1, max_value=999, value=-1)
        previous = st.slider("Previous contacts", min_value=0, max_value=50, value=0)
        poutcome = st.selectbox("Outcome of previous campaign", options=POUTCOME, index=0)
    
    # Create features dictionary with all required fields
    features = {
        'age': age,
        'job': job,
        'marital': marital,
        'education': education,
        'default': default,
        'balance': balance,
        'housing': housing,
        'loan': loan,
        'contact': contact,
        'month': month,  # Original month text
        'day': day,
        'duration': duration,
        'campaign': campaign,
        'pdays': pdays,
        'previous': previous,
        'poutcome': poutcome
    }
    
    return features

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

def main():
    st.title("Bank Term Deposit Subscription Predictor")
    st.write("Enter customer information on the left and click Predict to assess the likelihood this customer will subscribe to a term deposit (based on historical marketing data).")
    
    # Get user input features
    features = get_user_input_features()
    
    # Create columns for layout
    col1, col2 = st.columns([2, 3])
    
    with col1:
        st.subheader("Single Customer Prediction")
        st.write(features)
        
        if st.button("Predict"):
            try:
                # Create DataFrame and ensure month is handled correctly
                df_input = pd.DataFrame([features])
                prob = predict_subscription_proba(df_input)
                
                st.success(f"Probability of subscription: {prob:.1%}")
                
                if prob > 0.6:
                    st.info("High probability customer! Recommended for contact.")
                else:
                    st.info("Low probability customer. Consider other prospects first.")
                    
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
                logger.error(f"Error making prediction: {str(e)}", exc_info=True)
    
    # Batch prediction section - moved outside the col1 context
    st.divider()
    st.subheader("Batch Prediction")
    
    # Create two columns for batch section
    batch_col1, batch_col2 = st.columns([2, 3])
    
    with batch_col1:
        uploaded_file = st.file_uploader("Upload customer dataset (CSV)", type=['csv'])
        
        if uploaded_file is not None:
            try:
                # Read CSV file
                df = pd.read_csv(uploaded_file, sep=';')
                
                # Make predictions for all customers
                probabilities = []
                for _, row in df.iterrows():
                    prob = predict_subscription_proba(pd.DataFrame([row]))
                    probabilities.append(prob)
                
                # Add predictions to DataFrame
                df['subscription_probability'] = probabilities
                
                # Filter high-probability customers
                high_prob_customers = df[df['subscription_probability'] > 0.6].copy()
                high_prob_customers['subscription_probability'] = high_prob_customers['subscription_probability'].apply(lambda x: f"{x:.1%}")
                
                # Display results in the right column
                with batch_col2:
                    st.subheader("Dataset Summary")
                    st.write(f"Total customers in dataset: {len(df)}")
                    st.write(f"High-probability customers (>60%): {len(high_prob_customers)}")
                    
                    # Display key feature distributions
                    st.subheader("Key Feature Distributions")
                    dist_col1, dist_col2 = st.columns(2)
                    
                    with dist_col1:
                        st.write("Job Distribution")
                        st.write(df['job'].value_counts())
                        
                        st.write("Month Distribution")
                        st.write(df['month'].value_counts())
                    
                    with dist_col2:
                        st.write("Housing Loan Status")
                        st.write(df['housing'].value_counts())
                        
                        st.write("Previous Campaign Outcome")
                        st.write(df['poutcome'].value_counts())
                    
                    # Display high-probability customers
                    st.subheader("High-Probability Customers")
                    if len(high_prob_customers) > 0:
                        st.dataframe(
                            high_prob_customers[['age', 'job', 'education', 'balance', 'housing', 'month', 'subscription_probability']],
                            use_container_width=True
                        )
                    else:
                        st.info("No customers with subscription probability >60% found in the dataset.")
                    
            except Exception as e:
                st.error(f"Error processing CSV file: {str(e)}")
                logger.error(f"Error processing CSV file: {str(e)}", exc_info=True)
    
    # Show placeholder in batch results area when no file is uploaded
    if uploaded_file is None:
        with batch_col2:
            st.info("Upload a CSV file to see batch predictions and dataset summary.")

if __name__ == "__main__":
    main()