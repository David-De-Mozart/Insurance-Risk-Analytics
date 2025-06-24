# Insurance Risk Analytics Project

## Project Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Initialize DVC: `dvc init`
3. Add data: `dvc add data/insurance_data.csv`
4. Run pipeline: `dvc repro`

## Key Files
- `notebooks/01_EDA.ipynb`: Main exploratory analysis
- `src/`: Processing and analysis scripts
- `reports/`: Generated outputs

## Business Insights

### 🚗 Key Risk Drivers
1. **Regional Risk Variation**  
   Gauteng province shows 25% higher loss ratio (0.35) compared to Western Cape (0.28)
   ![Province Risk](reports/figures/province_risk.png)

2. **Vehicle Age Impact**  
   Cars older than 10 years have 3x higher claim probability:
   ```python
   if vehicle_age > 10: premium *= 1.3  # Price adjustment recommendation