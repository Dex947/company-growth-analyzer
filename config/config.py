"""
Configuration management for the company growth analyzer.
Loads settings from environment variables and provides defaults.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_RAW_PATH = PROJECT_ROOT / os.getenv("DATA_RAW_PATH", "data/raw")
DATA_PROCESSED_PATH = PROJECT_ROOT / os.getenv("DATA_PROCESSED_PATH", "data/processed")
MODELS_PATH = PROJECT_ROOT / os.getenv("MODELS_PATH", "models")
LOGS_PATH = PROJECT_ROOT / os.getenv("LOGS_PATH", "logs")
OUTPUTS_PATH = PROJECT_ROOT / os.getenv("OUTPUTS_PATH", "outputs")

# Create directories if they don't exist
for path in [DATA_RAW_PATH, DATA_PROCESSED_PATH, MODELS_PATH, LOGS_PATH,
             OUTPUTS_PATH / "reports", OUTPUTS_PATH / "visualizations"]:
    path.mkdir(parents=True, exist_ok=True)

# API Keys
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "")
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")
CRUNCHBASE_API_KEY = os.getenv("CRUNCHBASE_API_KEY", "")
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "")

# Model settings
RANDOM_SEED = int(os.getenv("RANDOM_SEED", 42))
TEST_SIZE = float(os.getenv("TEST_SIZE", 0.2))
CV_FOLDS = int(os.getenv("CV_FOLDS", 5))

# Data collection settings
MAX_RETRIES = int(os.getenv("MAX_RETRIES", 3))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", 30))
CACHE_EXPIRY_DAYS = int(os.getenv("CACHE_EXPIRY_DAYS", 7))

# Feature engineering settings
SENTIMENT_WINDOW_DAYS = int(os.getenv("SENTIMENT_WINDOW_DAYS", 90))
MIN_NEWS_ARTICLES = int(os.getenv("MIN_NEWS_ARTICLES", 5))

# Model training settings
ENABLE_SHAP = os.getenv("ENABLE_SHAP", "true").lower() == "true"
ENABLE_LIME = os.getenv("ENABLE_LIME", "true").lower() == "true"
MAX_SHAP_SAMPLES = int(os.getenv("MAX_SHAP_SAMPLES", 1000))

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Model configurations
MODEL_CONFIGS = {
    "logistic_regression": {
        "max_iter": 1000,
        "random_state": RANDOM_SEED,
        "solver": "liblinear"
    },
    "random_forest": {
        "n_estimators": 100,
        "max_depth": 10,
        "random_state": RANDOM_SEED,
        "n_jobs": -1
    },
    "xgboost": {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "random_state": RANDOM_SEED,
        "n_jobs": -1
    },
    "lightgbm": {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "random_state": RANDOM_SEED,
        "n_jobs": -1,
        "verbose": -1
    }
}
