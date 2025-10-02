# Data Directory

This directory stores data at various stages of the pipeline.

## Structure

```
data/
├── raw/              # Raw data from APIs
│   ├── cache/       # Cached API responses
│   └── *.parquet    # Combined raw data files
├── processed/       # Preprocessed and feature-engineered data
│   └── *.parquet    # Ready-for-modeling datasets
```

## File Formats

- **Parquet**: Primary format (compressed, fast, type-safe)
- **Feather**: Alternative for cross-language compatibility
- **CSV**: Export format for human readability

## Data Not Tracked

Per `.gitignore`, actual data files are excluded from version control to:
- Avoid repository bloat
- Protect potentially sensitive financial data
- Respect API terms of service

## Generating Data

Use the data collection commands to populate this directory:

```bash
# Collect data
python main.py collect --tickers "AAPL,GOOGL,MSFT" --period 1y

# Preprocess
python main.py preprocess --input data/raw/combined_data_*.parquet
```

## Cache

The `cache/` subdirectory stores API responses for 7 days (configurable in `.env`).
This reduces API calls and improves performance.

To clear cache:
```bash
rm -rf data/raw/cache/*  # Unix/Mac
rd /s data\raw\cache  # Windows
```
