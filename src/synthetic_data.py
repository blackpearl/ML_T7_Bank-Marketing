import pandas as pd
import numpy as np
from datetime import datetime
import os

# Define value ranges and distributions based on original dataset
JOBS = ['management', 'technician', 'entrepreneur', 'blue-collar', 'unknown', 'retired', 'admin.', 'services', 'self-employed', 'unemployed', 'housemaid', 'student']
MARITAL = ['married', 'single', 'divorced']
EDUCATION = ['tertiary', 'secondary', 'unknown', 'primary']
BINARY = ['yes', 'no']
CONTACT = ['cellular', 'telephone', 'unknown']
MONTHS = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
POUTCOME = ['unknown', 'other', 'failure', 'success']

def generate_synthetic_data(num_records=100, seed=42):
    """Generate synthetic customer data matching the bank marketing dataset format."""
    np.random.seed(seed)
    
    # Generate data with emphasis on key predictive features
    data = {
        'age': np.random.randint(18, 95, num_records),
        'job': np.random.choice(JOBS, num_records, p=[0.15, 0.1, 0.1, 0.2, 0.05, 0.1, 0.1, 0.05, 0.05, 0.05, 0.025, 0.025]),  # Higher weight for blue-collar
        'marital': np.random.choice(MARITAL, num_records),
        'education': np.random.choice(EDUCATION, num_records),
        'default': np.random.choice(BINARY, num_records, p=[0.05, 0.95]),  # Most customers don't default
        'balance': np.random.normal(1500, 3000, num_records).astype(int),  # Normal distribution around 1500
        'housing': np.random.choice(BINARY, num_records, p=[0.6, 0.4]),  # Higher weight for 'no' housing loan
        'loan': np.random.choice(BINARY, num_records),
        'contact': np.random.choice(CONTACT, num_records),
        'day': np.random.randint(1, 31, num_records),
        'month': np.random.choice(MONTHS, num_records, p=[0.05, 0.05, 0.05, 0.05, 0.3, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05]),  # Higher weight for May
        'duration': np.random.randint(0, 5000, num_records),
        'campaign': np.random.randint(1, 50, num_records),
        'pdays': np.random.choice([-1] + list(range(1, 999)), num_records, p=[0.8] + [0.2/(999-1)]*(999-1)),  # 80% never contacted before
        'previous': np.random.randint(0, 50, num_records),
        'poutcome': np.random.choice(POUTCOME, num_records, p=[0.4, 0.2, 0.2, 0.2])  # Equal weights for non-unknown outcomes
    }
    
    df = pd.DataFrame(data)
    
    # Create output directory if it doesn't exist
    output_dir = 'data/synthetic'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save to CSV with semicolon separator
    output_path = os.path.join(output_dir, 'synthetic_customers.csv')
    df.to_csv(output_path, sep=';', index=False)
    print(f"Generated {num_records} synthetic customer records and saved to {output_path}")
    
    return df

if __name__ == '__main__':
    # Generate 100 synthetic customer records
    df = generate_synthetic_data(100)
    
    # Print summary statistics
    print("\nDataset Summary:")
    print(f"Total records: {len(df)}")
    print("\nKey feature distributions:")
    print("\nJob distribution:")
    print(df['job'].value_counts())
    print("\nHousing loan distribution:")
    print(df['housing'].value_counts())
    print("\nMonth distribution:")
    print(df['month'].value_counts())
    print("\nPrevious outcome distribution:")
    print(df['poutcome'].value_counts())
    print("\nBalance statistics:")
    print(df['balance'].describe())
