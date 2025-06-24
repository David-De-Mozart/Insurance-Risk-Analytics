# src/shap_analysis.py
import joblib
import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import logging
from sklearn.compose import ColumnTransformer
from scipy.sparse import issparse
import warnings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('shap_analysis.log')
    ]
)

# Suppress warnings
warnings.filterwarnings("ignore")
shap.initjs()

def convert_to_dense_if_sparse(data):
    """Convert sparse matrix to dense array if needed"""
    if issparse(data):
        return data.toarray()
    return data

def analyze_model(model_path, sample_data, model_type="premium"):
    """Perform SHAP analysis on a trained model"""
    logger = logging.getLogger(__name__)
    
    try:
        # Load model
        logger.info(f"Loading model from {model_path}")
        model = joblib.load(model_path)
        
        # Preprocess sample data
        preprocessor = model.named_steps['preprocessor']
        sample_processed = preprocessor.transform(sample_data)
        
        # Convert to dense array for SHAP compatibility
        sample_processed = convert_to_dense_if_sparse(sample_processed)
        
        # Get feature names
        if isinstance(preprocessor, ColumnTransformer):
            feature_names = preprocessor.get_feature_names_out()
        else:
            feature_names = [f"feature_{i}" for i in range(sample_processed.shape[1])]
        
        # Create explainer
        logger.info("Creating SHAP explainer...")
        if "xgb" in model_path.lower() or "xgboost" in model_path.lower():
            explainer = shap.TreeExplainer(model.named_steps['regressor'])
            shap_values = explainer.shap_values(sample_processed)
        else:
            explainer = shap.KernelExplainer(model.predict, shap.sample(sample_processed, 100))
            shap_values = explainer.shap_values(sample_processed)
        
        # Create output directory
        os.makedirs("reports/figures", exist_ok=True)
        
        # Summary plot
        logger.info("Generating SHAP summary plot...")
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, sample_processed, 
                          feature_names=feature_names,
                          show=False)
        plt.title(f"SHAP Summary Plot ({model_type.capitalize()} Model)")
        plt.tight_layout()
        summary_path = f'reports/figures/shap_summary_{model_type}.png'
        plt.savefig(summary_path, dpi=300)
        plt.close()
        logger.info(f"SHAP summary plot saved to {summary_path}")
        
        # Feature importance
        logger.info("Generating feature importance plot...")
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, sample_processed,
                          feature_names=feature_names,
                          plot_type='bar', show=False)
        plt.title(f"Feature Importance ({model_type.capitalize()} Model)")
        plt.tight_layout()
        importance_path = f'reports/figures/shap_feature_importance_{model_type}.png'
        plt.savefig(importance_path, dpi=300)
        plt.close()
        logger.info(f"Feature importance plot saved to {importance_path}")
        
        # Waterfall plot for a single example
        logger.info("Generating waterfall plot...")
        plt.figure(figsize=(12, 8))
        sample_idx = 0  # First sample
        
        # Create waterfall plot
        shap.plots.waterfall(
            shap.Explanation(
                values=shap_values[sample_idx],
                base_values=explainer.expected_value,
                data=sample_processed[sample_idx],
                feature_names=feature_names
            ),
            max_display=15,
            show=False
        )
        plt.title(f"SHAP Waterfall Plot - Sample {sample_idx} ({model_type.capitalize()} Model)")
        plt.tight_layout()
        waterfall_path = f'reports/figures/shap_waterfall_{model_type}.png'
        plt.savefig(waterfall_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Waterfall plot saved to {waterfall_path}")
        
        # Create SHAP values report
        mean_abs_shap = pd.DataFrame({
            'Feature': feature_names,
            'Mean |SHAP|': np.abs(shap_values).mean(axis=0)
        }).sort_values('Mean |SHAP|', ascending=False)
        
        report_path = f'reports/shap_report_{model_type}.csv'
        mean_abs_shap.to_csv(report_path, index=False)
        logger.info(f"SHAP values report saved to {report_path}")
        
        return True
        
    except Exception as e:
        logger.exception(f"Error during SHAP analysis: {e}")
        return False

def load_data():
    """Load processed data with proper dtypes"""
    logger = logging.getLogger(__name__)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(project_root, 'data', 'processed_data.csv')
    
    try:
        # Specify dtypes for problematic columns
        dtype_spec = {
            'mmcode': 'str',
            'VehicleIntroDate': 'str',
            'AlarmImmobiliser': 'str',
            'TrackingDevice': 'str'
        }
        
        # Load only necessary columns to reduce memory
        cols_to_keep = [
            'VehicleAge', 'SumInsured', 'CustomValueEstimate',
            'Province', 'PostalCode', 'VehicleType', 'make',
            'bodytype', 'ExcessSelected', 'CoverType', 'HasClaim'
        ]
        
        df = pd.read_csv(data_path, dtype=dtype_spec, usecols=cols_to_keep)
        logger.info(f"Successfully loaded data with shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return pd.DataFrame()

def prepare_sample(df, model_type="premium"):
    """Prepare sample data for SHAP analysis"""
    logger = logging.getLogger(__name__)
    
    # For premium model
    if model_type == "premium":
        sample = df.sample(100, random_state=42).drop(columns=['HasClaim'], errors='ignore')
        logger.info("Prepared sample for premium model")
        return sample
    
    # For severity model (only claims)
    elif model_type == "severity":
        severity_df = df[df['HasClaim'] == 1]
        if severity_df.empty:
            logger.warning("No claim data available for severity analysis")
            return None
            
        sample = severity_df.sample(100, random_state=42).drop(columns=['HasClaim'], errors='ignore')
        logger.info("Prepared sample for severity model")
        return sample
    
    else:
        logger.error(f"Unknown model type: {model_type}")
        return None

if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logger.info("Starting SHAP analysis")
    
    # Load data
    df = load_data()
    if df.empty:
        logger.error("No data available for SHAP analysis")
        exit(1)
    
    # Analyze premium model
    logger.info("\n" + "="*70)
    logger.info("ANALYZING PREMIUM OPTIMIZATION MODEL")
    logger.info("="*70)
    premium_sample = prepare_sample(df, "premium")
    if premium_sample is not None:
        analyze_model('models/premium_xgboost_model.pkl', premium_sample, "premium")
    
    # Analyze severity model
    logger.info("\n" + "="*70)
    logger.info("ANALYZING CLAIM SEVERITY MODEL")
    logger.info("="*70)
    severity_sample = prepare_sample(df, "severity")
    if severity_sample is not None:
        analyze_model('models/severity_xgboost_model.pkl', severity_sample, "severity")
    
    logger.info("\n" + "="*70)
    logger.info("SHAP ANALYSIS COMPLETED")
    logger.info("="*70)