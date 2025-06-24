import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge  # More stable than LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os
import logging
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.impute import SimpleImputer
import gc  # Garbage collection

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('modeling.log')
    ]
)

def load_data():
    """Load processed data with optimized types"""
    logger = logging.getLogger(__name__)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(project_root, 'data', 'processed_data.csv')
    
    try:
        # Load with optimized dtypes
        dtype_spec = {
            'VehicleAge': 'int16',
            'SumInsured': 'float32',
            'CustomValueEstimate': 'float32',
            'ExcessSelected': 'float32',
            'TotalClaims': 'float32',
            'CalculatedPremiumPerTerm': 'float32'
        }
        df = pd.read_csv(data_path, dtype=dtype_spec)
        logger.info(f"Successfully loaded data with shape: {df.shape}")
        
        # Downcast other numeric columns
        num_cols = df.select_dtypes(include=np.number).columns
        for col in num_cols:
            if col not in dtype_spec:
                df[col] = pd.to_numeric(df[col], downcast='integer')
        
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return pd.DataFrame()

def prepare_severity_data(df):
    """Prepare data for claim severity prediction"""
    logger = logging.getLogger(__name__)
    
    # Filter to policies with claims
    severity_df = df[df['HasClaim'] == 1].copy()
    
    if severity_df.empty:
        logger.error("No claim data available for severity modeling")
        return None, None, None, None
    
    # Define features and target
    X = severity_df[[
        'VehicleAge', 'SumInsured', 'CustomValueEstimate',
        'Province', 'VehicleType', 'make',
        'ExcessSelected', 'CoverType'
    ]]
    
    y = severity_df['TotalClaims']
    
    # Apply log transform to target to handle scale
    y = np.log1p(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    logger.info(f"Severity dataset: {X_train.shape[0]} train, {X_test.shape[0]} test samples")
    return X_train, X_test, y_train, y_test

def prepare_premium_data(df):
    """Prepare data for premium optimization with reduced features"""
    logger = logging.getLogger(__name__)
    
    # Use all data
    premium_df = df.copy()
    
    # Define features and target
    X = premium_df[[
        'VehicleAge', 'SumInsured', 'CustomValueEstimate',
        'Province', 'VehicleType', 'make',
        'ExcessSelected', 'CoverType'
    ]]
    
    y = premium_df['CalculatedPremiumPerTerm']
    
    # Apply log transform to target
    y = np.log1p(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    logger.info(f"Premium dataset: {X_train.shape[0]} train, {X_test.shape[0]} test samples")
    return X_train, X_test, y_train, y_test

def build_model_pipeline(model, use_feature_selection=False):
    """Build preprocessing and modeling pipeline"""
    # Define numeric and categorical features
    numeric_features = [
        'VehicleAge', 'SumInsured', 'CustomValueEstimate', 'ExcessSelected'
    ]
    
    categorical_features = [
        'Province', 'VehicleType', 'make', 'CoverType'
    ]
    
    # Create transformers
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Use sparse output to save memory
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=True))
    ])
    
    # Create preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Create full pipeline
    steps = [('preprocessor', preprocessor)]
    
    if use_feature_selection:
        steps.append(('feature_selection', SelectKBest(f_regression, k=20)))
    
    steps.append(('regressor', model))
    
    return Pipeline(steps=steps)

def train_evaluate_model(name, model, X_train, X_test, y_train, y_test, is_severity=False):
    """Train and evaluate a model"""
    logger = logging.getLogger(__name__)
    
    # Create pipeline
    pipeline = build_model_pipeline(model)
    
    # Train model
    pipeline.fit(X_train, y_train)
    
    # Evaluate
    y_pred = pipeline.predict(X_test)
    
    # Convert back from log scale
    if is_severity:
        y_test = np.expm1(y_test)
        y_pred = np.expm1(y_pred)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    logger.info(f"{name} Results:")
    logger.info(f"  RMSE: {rmse:.4f}")
    logger.info(f"  R²: {r2:.4f}")
    
    # Save model
    os.makedirs("models", exist_ok=True)
    model_path = f"models/{name.replace(' ', '_').lower()}_model.pkl"
    joblib.dump(pipeline, model_path)
    logger.info(f"Model saved to {model_path}")
    
    # Clean up memory
    del pipeline
    gc.collect()
    
    return rmse, r2

def train_models():
    """Train and evaluate all models"""
    logger = logging.getLogger(__name__)
    
    # Load data
    df = load_data()
    if df.empty:
        return
    
    logger.info("\n" + "="*70)
    logger.info("TRAINING CLAIM SEVERITY MODELS")
    logger.info("="*70)
    
    # Claim severity models
    X_train_s, X_test_s, y_train_s, y_test_s = prepare_severity_data(df)
    if X_train_s is not None:
        # Initialize models
        models = {
            "Ridge Regression": Ridge(alpha=1.0),
            "Random Forest": RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1),
            "XGBoost": XGBRegressor(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)
        }
        
        # Train and evaluate
        for name, model in models.items():
            rmse, r2 = train_evaluate_model(
                f"Severity {name}",
                model,
                X_train_s, X_test_s, y_train_s, y_test_s,
                is_severity=True
            )
    
    logger.info("\n" + "="*70)
    logger.info("TRAINING PREMIUM OPTIMIZATION MODELS")
    logger.info("="*70)
    
    # Premium optimization models
    X_train_p, X_test_p, y_train_p, y_test_p = prepare_premium_data(df)
    if X_train_p is not None:
        # Initialize models
        models = {
            "Ridge Regression": Ridge(alpha=1.0),
            "Random Forest": RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1),
            "XGBoost": XGBRegressor(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)
        }
        
        # Train and evaluate
        for name, model in models.items():
            rmse, r2 = train_evaluate_model(
                f"Premium {name}",
                model,
                X_train_p, X_test_p, y_train_p, y_test_p
            )
    
    logger.info("\n" + "="*70)
    logger.info("MODEL TRAINING COMPLETED")
    logger.info("="*70)

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    train_models()