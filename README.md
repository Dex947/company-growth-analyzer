# Company Growth Analyzer

ML-powered system for analyzing and predicting company growth potential using multi-source financial data, market signals, and explainable AI.

## Features

- **Multi-Source Data Ingestion**: Financial metrics (Yahoo Finance), market data, and sentiment analysis
- **Sector-Specific Analysis**: Specialized models for different industries (semiconductors, cloud/SaaS, consumer staples)
- **Temporal Data Collection**: Quarterly snapshots over 3+ years for robust training datasets
- **Factor-Based Modeling**: Dimensionality reduction from 70+ features to 5-8 interpretable factors
- **Machine Learning Models**: Random Forest, XGBoost, LightGBM with cross-validation
- **Explainable AI**: SHAP and LIME for model interpretability

## Project Structure

```
company-growth-analyzer/
├── config/               # Configuration files
│   ├── config.py        # Main configuration
│   └── sector_config.py # Sector-specific settings (49 semiconductor companies)
├── src/
│   ├── data_ingestion/  # Data collection modules
│   │   ├── simple_temporal_collector.py  # Temporal data collection
│   │   ├── financial_data.py
│   │   ├── sentiment_data.py
│   │   └── market_data.py
│   ├── preprocessing/   # Feature engineering
│   │   ├── sector_factor_model.py         # Factor score computation
│   │   └── sector_relative_target.py      # Sector-relative targets
│   ├── models/          # ML models
│   ├── explainability/  # SHAP/LIME explainers
│   └── visualization/   # Plotting and charts
├── outputs/             # Generated results
│   ├── visualizations/  # Charts and plots
│   └── reports/         # Rankings and predictions
└── demo_semiconductor_sector.py  # Main demonstration

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

Run the semiconductor sector analysis:

```bash
python demo_semiconductor_sector.py
```

This will:
1. Collect data for 47 semiconductor companies
2. Generate 508 temporal samples (quarterly snapshots)
3. Train ML models
4. Generate predictions and visualizations

## Results

### Dataset Statistics
- **Companies**: 47 semiconductor companies across market caps
- **Temporal Samples**: 508 (47 companies × ~10 quarters)
- **Features**: 5 sector-specific factors
- **Samples-to-Features Ratio**: 101.6:1

### Model Performance

| Model          | Train Score | Test Score | CV Score (±std) |
|----------------|-------------|------------|-----------------|
| **LightGBM**   | 0.970       | 0.853      | 0.845 ± 0.017   |
| XGBoost        | 0.988       | 0.853      | 0.847 ± 0.022   |
| Random Forest  | 0.998       | 0.863      | 0.832 ± 0.017   |

**Model Quality Indicators**:
- ✅ Test accuracy ~85% (realistic performance)
- ✅ Low CV variance (±1.7-2.2%) indicates stability
- ✅ Small generalization gap (11.8%) acceptable for financial data
- ✅ Balanced class distribution (50/50 split)

### Top Predicted Companies

| Rank | Ticker | Company              | Success Probability |
|------|--------|----------------------|---------------------|
| 1    | MPWR   | Monolithic Power     | 99.9%               |
| 2    | SMCI   | Super Micro Computer | 99.9%               |
| 3    | NVDA   | NVIDIA               | 99.8%               |
| 4    | ALGM   | Allegro Micro        | 99.7%               |
| 5    | LRCX   | Lam Research         | 99.6%               |

## Sector-Specific Factors

The system uses 5 research-backed factors for semiconductor analysis:

1. **Innovation Intensity** (20%)
   - R&D as % of revenue
   - Revenue growth
   - Gross margin

2. **Profitability Quality** (25%)
   - Operating margin
   - Return on equity
   - Free cash flow margin

3. **Market Position** (20%)
   - Market capitalization
   - Sector revenue ranking
   - Price momentum

4. **Financial Health** (15%)
   - Debt-to-equity ratio
   - Current ratio
   - Quick ratio

5. **Growth Momentum** (20%)
   - Revenue growth rate
   - Earnings growth
   - Stock returns (3m, 6m)

## Configuration

### Adding More Companies

Edit `config/sector_config.py`:

```python
SEMICONDUCTOR_COMPANIES = {
    'NVDA': {'name': 'NVIDIA Corporation', 'segment': 'GPU/AI Chips'},
    # Add more companies...
}
```

### Adjusting Temporal Range

Modify `quarters_back` in the demo:

```python
collector = SimpleTemporalCollector(quarters_back=16)  # 4 years instead of 3
```

## Data Sources

- **Yahoo Finance**: Stock prices, basic fundamentals via yfinance
- **Sector Configuration**: Research-based factor definitions
- **Temporal Sampling**: Quarterly historical data collection

## Technical Notes

### Temporal Data Collection

The system collects quarterly snapshots to increase sample size:
- Single snapshot: 47 samples
- Temporal approach: 47 companies × 12 quarters = 564 potential samples
- Actual collected: 508 samples (some companies lack full history)

This 10x increase in samples prevents overfitting and enables reliable ML.

### Sector-Relative Targets

Instead of absolute thresholds (e.g., ">15% return"), the system uses sector-relative performance:
- Target = 1 if company outperforms sector median
- Target = 0 if company underperforms sector median
- Accounts for sector-wide trends and volatility

## License

MIT License

## Acknowledgments

- Factor definitions based on Visible Alpha semiconductor KPIs and McKinsey research
- Data sourced from Yahoo Finance via yfinance library
- ML frameworks: scikit-learn, XGBoost, LightGBM
- Explainability: SHAP, LIME
