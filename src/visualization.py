import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_monthly_trends(df, save_path=None):
    """Plot monthly premium, claims and loss ratio trends"""
    if 'TransactionMonth' not in df.columns:
        print("Skipping monthly trends - TransactionMonth missing")
        return None
    
    # Resample to monthly frequency
    monthly = df.resample('M', on='TransactionMonth').agg({
        'TotalPremium': 'sum',
        'TotalClaims': 'sum'
    })
    monthly['LossRatio'] = monthly['TotalClaims'] / monthly['TotalPremium']
    
    fig, ax = plt.subplots(figsize=(12, 6))
    monthly[['TotalPremium', 'TotalClaims']].plot(ax=ax, secondary_y=True)
    monthly['LossRatio'].plot(ax=ax, color='red', label='Loss Ratio')
    ax.set_title('Monthly Premium, Claims & Loss Ratio')
    ax.set_ylabel('Amount (ZAR)')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved monthly trends to {save_path}")
    return fig

def plot_province_risk(df, save_path=None):
    """Plot loss ratio by province"""
    if 'Province' not in df.columns:
        print("Skipping province risk - Province missing")
        return None
    
    province_loss = df.groupby('Province').agg(
        TotalPremium=('TotalPremium', 'sum'),
        TotalClaims=('TotalClaims', 'sum')
    )
    province_loss['LossRatio'] = province_loss['TotalClaims'] / province_loss['TotalPremium']
    province_loss = province_loss.sort_values('LossRatio')
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x=province_loss.index, y='LossRatio', data=province_loss)
    plt.title('Loss Ratio by Province')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Loss Ratio')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved province risk to {save_path}")
    return plt.gcf()

def plot_vehicle_risk(df, top_n=10, save_path=None):
    """Plot claim frequency and severity by vehicle make"""
    if 'make' not in df.columns:
        print("Skipping vehicle risk - make column missing")
        return None
    
    # Filter to makes with sufficient data
    claim_counts = df[df['TotalClaims'] > 0]['make'].value_counts()
    valid_makes = claim_counts[claim_counts >= 10].index[:top_n]
    
    if len(valid_makes) == 0:
        print("No vehicle makes with sufficient claim data")
        return None
    
    make_risk = df[df['make'].isin(valid_makes)].groupby('make').agg(
        ClaimFrequency=('HasClaim', 'mean'),
        AvgClaimSeverity=('TotalClaims', lambda x: x[x>0].mean())
    ).sort_values('ClaimFrequency', ascending=False)
    
    fig, ax = plt.subplots(1, 2, figsize=(16, 8))
    sns.barplot(y=make_risk.index, x='ClaimFrequency', data=make_risk, ax=ax[0])
    ax[0].set_title('Claim Frequency by Make')
    ax[0].set_xlabel('Proportion of Policies with Claims')
    
    sns.barplot(y=make_risk.index, x='AvgClaimSeverity', data=make_risk, ax=ax[1])
    ax[1].set_title('Average Claim Severity by Make')
    ax[1].set_xlabel('Average Claim Amount (ZAR)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved vehicle risk to {save_path}")
    return fig

def plot_financial_outliers(df, save_path=None):
    """Plot financial metrics distribution with outliers"""
    plt.figure(figsize=(12, 6))
    
    # Plot premiums
    plt.subplot(1, 2, 1)
    sns.boxplot(y=df['TotalPremium'])
    plt.title('Premium Distribution')
    
    # Plot claims
    plt.subplot(1, 2, 2)
    # Only show claims where claim occurred
    claim_data = df[df['TotalClaims'] > 0]['TotalClaims'] if 'TotalClaims' in df.columns else pd.Series()
    sns.boxplot(y=claim_data)
    plt.title('Claim Distribution (Claims > 0)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved financial outliers to {save_path}")
    return plt.gcf()

def generate_eda():
    """Generate all EDA visualizations"""
    # Get project root directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Create output directory
    figures_dir = os.path.join(project_root, 'reports', 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    print(f"Generating EDA visualizations in: {figures_dir}")
    
    # Load processed data
    data_path = os.path.join(project_root, 'data', 'processed_data.csv')
    if not os.path.exists(data_path):
        print(f"Error: Processed data not found at {data_path}")
        return
    
    try:
        df = pd.read_csv(data_path)
    except Exception as e:
        print(f"Error loading processed data: {e}")
        return
    
    # Convert date column if exists
    if 'TransactionMonth' in df.columns:
        df['TransactionMonth'] = pd.to_datetime(df['TransactionMonth'])
    
    # Generate all visualizations
    plot_monthly_trends(df, os.path.join(figures_dir, 'monthly_trends.png'))
    plot_province_risk(df, os.path.join(figures_dir, 'province_risk.png'))
    plot_vehicle_risk(df, save_path=os.path.join(figures_dir, 'vehicle_risk.png'))
    plot_financial_outliers(df, os.path.join(figures_dir, 'financial_outliers.png'))
    
    print("EDA visualizations generated successfully!")