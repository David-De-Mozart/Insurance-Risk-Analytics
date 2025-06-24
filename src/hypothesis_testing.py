import pandas as pd
import numpy as np
import os
import logging
import sys

try:
    from scipy import stats
    has_scipy = True
except ImportError:
    has_scipy = False
    print("Scipy not installed - statistical tests will be skipped")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('hypothesis_testing.log')
    ]
)

def load_data():
    """Load processed data"""
    logger = logging.getLogger(__name__)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(project_root, 'data', 'processed_data.csv')
    
    try:
        df = pd.read_csv(data_path)
        logger.info(f"Successfully loaded data with shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return pd.DataFrame()

def risk_difference_report(df, group_col, metric='LossRatio'):
    """Generate risk difference report for a grouping column"""
    logger = logging.getLogger(__name__)
    
    if df.empty or group_col not in df.columns:
        logger.warning(f"Cannot analyze risk differences - {group_col} not in data")
        return
    
    # Calculate group metrics
    group_stats = df.groupby(group_col).agg(
        TotalPolicies=('PolicyID', 'count'),
        TotalClaims=('TotalClaims', 'sum'),
        TotalPremium=('TotalPremium', 'sum')
    )
    
    group_stats['ClaimRate'] = group_stats['TotalClaims'] / group_stats['TotalPolicies']
    group_stats['LossRatio'] = group_stats['TotalClaims'] / group_stats['TotalPremium']
    group_stats['Margin'] = (group_stats['TotalPremium'] - group_stats['TotalClaims']) / group_stats['TotalPremium']
    
    # Sort by risk
    sorted_risk = group_stats.sort_values('LossRatio', ascending=False)
    
    logger.info(f"\nRisk Analysis by {group_col}:")
    logger.info(sorted_risk[['TotalPolicies', 'ClaimRate', 'LossRatio', 'Margin']].to_string())
    
    return sorted_risk

def test_province_risk(df):
    """Test risk differences across provinces"""
    logger = logging.getLogger(__name__)
    logger.info("\n" + "="*50)
    logger.info("Testing risk differences across provinces")
    logger.info("="*50)
    
    # Generate report
    province_stats = risk_difference_report(df, 'Province')
    
    # Statistical test if scipy is available
    if has_scipy and not df.empty and 'Province' in df.columns:
        # Prepare data
        provinces = df['Province'].unique()
        groups = [df[df['Province'] == prov]['LossRatio'] for prov in provinces]
        
        # ANOVA test
        try:
            f_val, p_val = stats.f_oneway(*groups)
            logger.info(f"ANOVA for provinces: F-value = {f_val:.4f}, p-value = {p_val:.4g}")
            
            # Interpretation
            alpha = 0.05
            if p_val < alpha:
                logger.info("REJECT H₀: Significant risk differences exist across provinces")
                # Calculate mean loss ratio per province
                province_loss = df.groupby('Province')['LossRatio'].mean().sort_values(ascending=False)
                logger.info(f"Loss Ratio by Province:\n{province_loss}")
            else:
                logger.info("FAIL TO REJECT H₀: No significant risk differences across provinces")
        except Exception as e:
            logger.error(f"ANOVA test failed: {e}")
    else:
        logger.warning("Skipping statistical test - scipy not available or data missing")

def test_zipcode_risk(df):
    """Test risk differences between zip codes"""
    logger = logging.getLogger(__name__)
    logger.info("\n" + "="*50)
    logger.info("Testing risk differences between zip codes")
    logger.info("="*50)
    
    # Generate report
    zip_stats = risk_difference_report(df, 'PostalCode')
    
    # Statistical test if scipy is available
    if has_scipy and not df.empty and 'PostalCode' in df.columns:
        # Focus on zip codes with sufficient data
        zip_counts = df['PostalCode'].value_counts()
        top_zips = zip_counts[zip_counts > 100].index[:10]  # Top 10 zips with >100 policies
        
        if len(top_zips) < 2:
            logger.warning("Insufficient zip codes for comparison")
            return
            
        # ANOVA test
        groups = [df[df['PostalCode'] == zipcode]['LossRatio'] for zipcode in top_zips]
        
        try:
            f_val, p_val = stats.f_oneway(*groups)
            logger.info(f"ANOVA for top zip codes: F-value = {f_val:.4f}, p-value = {p_val:.4g}")
            
            # Interpretation
            alpha = 0.05
            if p_val < alpha:
                logger.info("REJECT H₀: Significant risk differences exist between zip codes")
            else:
                logger.info("FAIL TO REJECT H₀: No significant risk differences between zip codes")
        except Exception as e:
            logger.error(f"ANOVA test failed: {e}")
    else:
        logger.warning("Skipping statistical test - scipy not available or data missing")

def test_gender_risk(df):
    """Test risk differences between genders"""
    logger = logging.getLogger(__name__)
    logger.info("\n" + "="*50)
    logger.info("Testing risk differences between genders")
    logger.info("="*50)
    
    # Clean gender data
    if 'Gender' not in df.columns:
        logger.warning("Gender column missing")
        return
        
    df['Gender'] = df['Gender'].str.strip().replace({'': 'Unknown'})
    
    # Filter relevant genders
    valid_genders = ['Male', 'Female']
    gender_df = df[df['Gender'].isin(valid_genders)].copy()
    
    if gender_df.empty:
        logger.warning("No valid gender data for comparison")
        return
        
    # Generate report
    gender_stats = risk_difference_report(gender_df, 'Gender')
    
    # Statistical test if scipy is available
    if has_scipy:
        # T-test for loss ratio
        male = gender_df[gender_df['Gender'] == 'Male']['LossRatio']
        female = gender_df[gender_df['Gender'] == 'Female']['LossRatio']
        
        try:
            t_val, p_val = stats.ttest_ind(male, female, equal_var=False, nan_policy='omit')
            logger.info(f"T-test for gender: t-value = {t_val:.4f}, p-value = {p_val:.4g}")
            
            # Interpretation
            alpha = 0.05
            if p_val < alpha:
                logger.info("REJECT H₀: Significant risk difference between men and women")
                logger.info(f"Male Loss Ratio: {male.mean():.4f}, Female Loss Ratio: {female.mean():.4f}")
            else:
                logger.info("FAIL TO REJECT H₀: No significant risk difference between genders")
        except Exception as e:
            logger.error(f"T-test failed: {e}")
    else:
        logger.warning("Skipping statistical test - scipy not available")

def main():
    logger = logging.getLogger(__name__)
    logger.info("="*70)
    logger.info("STARTING HYPOTHESIS TESTING FOR INSURANCE RISK ANALYSIS")
    logger.info("="*70)
    
    # Check scipy availability
    if not has_scipy:
        logger.warning("Scipy not installed - statistical tests will be skipped")
        logger.warning("Run 'pip install scipy statsmodels' to enable full functionality")
    
    try:
        # Load processed data
        df = load_data()
        
        if df.empty:
            logger.error("No data available for testing")
            return
            
        # Run tests
        test_province_risk(df)
        test_zipcode_risk(df)
        test_gender_risk(df)
        
        logger.info("\n" + "="*70)
        logger.info("HYPOTHESIS TESTING COMPLETED")
        logger.info("="*70)
        
    except Exception as e:
        logger.exception(f"Critical error during hypothesis testing: {e}")

if __name__ == "__main__":
    main()