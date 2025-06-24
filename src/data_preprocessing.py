import pandas as pd
import numpy as np
import csv
import sys
import os
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('data_preprocessing.log')
    ]
)

def load_and_preprocess(filepath):
    """Load and preprocess insurance data with robust error handling"""
    logger = logging.getLogger(__name__)
    logger.info(f"Loading data from: {filepath}")
    
    # Handle problematic CSV rows
    rows = []
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        try:
            header = next(reader)  # Save header
        except StopIteration:
            logger.error("CSV file is empty")
            return pd.DataFrame()
            
        expected_columns = len(header)
        logger.info(f"Expected columns: {expected_columns}")
        
        row_count = 0
        for i, row in enumerate(reader):
            row_count += 1
            # Fix rows with extra fields
            if len(row) > expected_columns:
                row = row[:expected_columns]  # Trim extra fields
            # Pad rows with missing fields
            elif len(row) < expected_columns:
                row += [''] * (expected_columns - len(row))
            rows.append(row)
    
    # Create DataFrame from valid rows
    df = pd.DataFrame(rows, columns=header)
    logger.info(f"Processed {row_count} rows")
    
    # Specify data types for problematic columns
    dtype_spec = {
        'mmcode': str,
        'VehicleIntroDate': str,
        'AlarmImmobiliser': str,
        'TrackingDevice': str,
        'NewVehicle': str,
        'WrittenOff': str,
        'Rebuilt': str,
        'Converted': str,
        'CrossBorder': str
    }
    
    # Apply type conversions
    for col, dtype in dtype_spec.items():
        if col in df.columns:
            df[col] = df[col].astype(dtype)
    
    # Convert numeric columns
    numeric_cols = [
        'TotalPremium', 'TotalClaims', 'CalculatedPremiumPerTerm', 
        'RegistrationYear', 'Cylinders', 'cubiccapacity', 'kilowatts',
        'NumberOfDoors', 'CustomValueEstimate', 'CapitalOutstanding',
        'SumInsured', 'ExcessSelected'
    ]
    
    for col in numeric_cols:
        if col in df.columns:
            # Convert to string first to handle mixed types
            df[col] = df[col].astype(str).str.replace(',', '')
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            logger.warning(f"Column {col} not found in dataset")
    
    # Handle dates
    if 'TransactionMonth' in df.columns:
        df['TransactionMonth'] = pd.to_datetime(df['TransactionMonth'], errors='coerce')
    else:
        logger.warning("TransactionMonth column missing")
    
    # Create essential features
    if 'RegistrationYear' in df.columns:
        current_year = datetime.now().year
        df['VehicleAge'] = current_year - df['RegistrationYear']
        # Handle invalid registration years
        df['VehicleAge'] = df['VehicleAge'].clip(lower=0, upper=100)
    else:
        logger.warning("RegistrationYear column missing")
    
    if 'TotalClaims' in df.columns and 'TotalPremium' in df.columns:
        # Handle zero premiums before calculating loss ratio
        df['TotalPremium'] = df['TotalPremium'].replace(0, np.nan)
        df['LossRatio'] = df['TotalClaims'] / df['TotalPremium']
        
        # Create claim flag
        df['HasClaim'] = np.where(df['TotalClaims'] > 0, 1, 0)
    else:
        logger.warning("TotalClaims or TotalPremium columns missing")
    
    # Clean categoricals
    for col in ['Gender', 'Province', 'make', 'Model', 'bodytype']:
        if col in df.columns:
            df[col] = df[col].str.strip().replace('', 'Unknown')
        else:
            logger.warning(f"{col} column missing")
    
    return df

def handle_missing(df):
    """Handle missing values safely"""
    logger = logging.getLogger(__name__)
    logger.info("Handling missing values...")
    
    # Fill numeric missing with median
    num_cols = df.select_dtypes(include=np.number).columns
    for col in num_cols:
        if df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            logger.info(f"Filled {df[col].isnull().sum()} missing values in {col} with median {median_val}")
    
    # Fill categorical missing with mode
    cat_cols = df.select_dtypes(include='object').columns
    for col in cat_cols:
        if df[col].isnull().sum() > 0:
            mode_val = df[col].mode()[0] if not df[col].mode().empty else ''
            df[col] = df[col].fillna(mode_val)
            logger.info(f"Filled {df[col].isnull().sum()} missing values in {col} with mode '{mode_val}'")
    
    return df

def save_processed_data(df, output_path):
    """Save processed data with optimized types"""
    logger = logging.getLogger(__name__)
    
    # Convert to appropriate types to reduce size
    for col in df.select_dtypes(include='integer').columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')
    
    for col in df.select_dtypes(include='float').columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
    
    # Save processed data
    df.to_csv(output_path, index=False)
    logger.info(f"Processed data saved to {output_path}")
    logger.info(f"Final dataset shape: {df.shape}")
    
    # Basic statistics
    if 'HasClaim' in df.columns:
        claim_rate = df['HasClaim'].mean() * 100
        logger.info(f"Rows with claims: {df['HasClaim'].sum()} ({claim_rate:.2f}%)")
    
    if 'LossRatio' in df.columns:
        valid_loss_ratio = df['LossRatio'].replace([np.inf, -np.inf], np.nan).dropna()
        if not valid_loss_ratio.empty:
            logger.info(f"Average Loss Ratio: {valid_loss_ratio.mean():.2f}")
        else:
            logger.warning("No valid loss ratio values to calculate average")
    
    return df

if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logger.info("Starting data preprocessing")
    
    try:
        # Get project root directory
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Input and output paths
        input_path = os.path.join(project_root, 'data', 'insurance_data.csv')
        output_path = os.path.join(project_root, 'data', 'processed_data.csv')
        
        # Validate input file
        if not os.path.exists(input_path):
            logger.error(f"Input file not found: {input_path}")
            sys.exit(1)
            
        # Load and preprocess
        df = load_and_preprocess(input_path)
        
        if df.empty:
            logger.error("Data loading failed. Exiting.")
            sys.exit(1)
            
        df = handle_missing(df)
        
        # Save processed data
        save_processed_data(df, output_path)
        
        logger.info("Data preprocessing completed successfully!")
        
    except Exception as e:
        logger.exception(f"Error during data preprocessing: {e}")
        sys.exit(1)