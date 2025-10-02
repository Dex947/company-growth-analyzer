# Changelog

All notable changes to the Company Growth Analyzer project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-10-02

### Added

#### Data Ingestion
- Multi-source data collection framework with `BaseDataCollector` abstract class
- Yahoo Finance integration for financial metrics (free, no API key required)
- Alpha Vantage connector for fundamental data (optional)
- News sentiment analysis using Google News RSS and NewsAPI
- VADER and TextBlob sentiment analyzers
- Market data collector for sector-relative metrics and competitive positioning
- Response caching system to reduce API calls (7-day default expiry)
- Retry logic for failed API requests (3 retries with exponential backoff)
- Data aggregator to combine all sources into unified dataset

#### Preprocessing & Feature Engineering
- Robust data preprocessor with domain-aware missing value imputation
- Outlier handling via winsorization (clip at 1st and 99th percentiles)
- RobustScaler and StandardScaler support for normalization
- Feature engineering module creating 30+ derived features:
  - Financial health scores (profitability, efficiency, debt burden)
  - Growth and momentum indicators (acceleration, stable growth)
  - Valuation features (sector-relative, undervalued detection)
  - Competitive position features (dominant player, sector outperformance)
  - Sentiment-based features (sentiment-return alignment, consensus)
  - Interaction features (growth×sentiment, valuation×momentum, size×growth)
- Sector-based rankings and percentiles for all key metrics

#### Machine Learning Models
- Model factory supporting multiple model types:
  - Logistic Regression (interpretable baseline)
  - Random Forest (handles non-linearity)
  - XGBoost (high performance gradient boosting)
  - LightGBM (fast, memory-efficient)
- Model trainer with cross-validation (5-fold stratified)
- Automated train/test splitting with stratification support
- Model persistence (save/load trained models)
- Best model selection based on configurable metrics

#### Explainability
- SHAP (SHapley Additive exPlanations) integration:
  - TreeExplainer for tree-based models (fast)
  - KernelExplainer fallback for any model type
  - Global feature importance calculation
  - Per-prediction feature attributions
  - Interaction effect detection
- LIME (Local Interpretable Model-agnostic Explanations):
  - Local linear approximations around predictions
  - Tabular data support
  - Configurable number of features
- Permutation importance (model-agnostic feature ranking)
- Built-in feature importance for tree models
- Natural language explanation generator:
  - Prediction score with confidence
  - Top 5 driving factors
  - Caveats and limitations
  - Comparative analysis for multiple companies

#### Evaluation
- Comprehensive classification metrics:
  - Accuracy, Precision, Recall, F1-score
  - ROC-AUC for probability-based evaluation
  - Confusion matrices
  - Full classification reports
- Ranking metrics:
  - NDCG@K (Normalized Discounted Cumulative Gain)
  - Measures quality of company ranking
- Model comparison framework
- Automated evaluation report generation (JSON format)

#### Visualization
- Feature importance bar charts
- SHAP summary plots
- Confusion matrix heatmaps
- Model performance comparison plots
- Company ranking visualizations with color gradients
- Dashboard creation for comprehensive results
- High-resolution exports (300 DPI PNG)

#### CLI & Interface
- Click-based command-line interface
- Commands:
  - `collect`: Gather multi-source company data
  - `preprocess`: Clean and engineer features
  - `train`: Train and evaluate models
  - `predict`: Make predictions on new data
  - `visualize`: Generate visual reports
  - `demo`: End-to-end pipeline demonstration
- Progress logging with loguru
- File-based and console logging (10 MB rotation, 30-day retention)

#### Configuration & Infrastructure
- Environment-based configuration (.env support)
- Centralized settings in `config/config.py`
- Automatic directory creation
- Parquet/Feather support for efficient data storage
- JSON for reports and metadata
- Caching system for API responses
- Error handling and logging throughout

#### Documentation
- Comprehensive README with:
  - Installation instructions
  - Quick start guide
  - Usage examples
  - Project structure explanation
  - Model details and assumptions
  - Known limitations
  - API key setup
- Inline code comments explaining methodology
- Docstrings for all classes and functions
- Critical assumption questioning in comments
- CHANGELOG for tracking iterations

### Design Decisions

#### Why These Models?
- **Logistic Regression**: Transparent, fast, good baseline
- **Random Forest**: Balance of performance and interpretability
- **XGBoost/LightGBM**: State-of-the-art performance, with SHAP for interpretation

#### Why Explainability Focus?
- Financial decisions require justification
- Hidden patterns need to be surfaced and validated
- Multiple explanation methods provide robustness (SHAP + LIME + permutation)

#### Why Multi-Source Data?
- Single source (e.g., financials only) misses qualitative factors
- Sentiment captures forward-looking information
- Competitive positioning provides context

#### Trade-offs Made
- **Accuracy vs Speed**: Caching and sampling for SHAP (configurable)
- **Completeness vs Complexity**: Start with interpretable models, add complexity as needed
- **Data Quality vs Coverage**: Accept some missing data with smart imputation

### Known Issues
- SHAP calculation can be slow for large datasets (mitigated by sampling)
- Some API endpoints have rate limits (handled by caching and retries)
- Sentiment analysis is English-only currently

### Dependencies
See `requirements.txt` for full list. Key libraries:
- pandas, numpy, scikit-learn
- xgboost, lightgbm
- shap, lime
- yfinance, requests, beautifulsoup4
- textblob, vaderSentiment
- matplotlib, seaborn, plotly
- click, loguru, python-dotenv

## Future Roadmap

### [1.1.0] - Planned
- Real-time data update scheduling
- Additional data sources (Crunchbase, social media)
- Deep learning models (LSTM for time series, transformers for text)
- Interactive web dashboard
- Automated hyperparameter tuning
- A/B testing framework for model comparison

### [1.2.0] - Planned
- Cloud deployment templates (AWS, GCP, Azure)
- REST API for predictions
- Database integration (PostgreSQL, MongoDB)
- Real-time streaming data support
- Multi-language sentiment analysis
- Sector-specific models

### [2.0.0] - Vision
- End-to-end AutoML pipeline
- Causal inference capabilities
- Counterfactual explanations
- Model monitoring and drift detection
- Multi-modal data (images, audio from earnings calls)

---

## Version History

- **1.0.0** (2025-10-02): Initial release with full pipeline
