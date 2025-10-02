# Outputs Directory

This directory stores analysis outputs, reports, and visualizations.

## Structure

```
outputs/
├── reports/              # JSON evaluation and explanation reports
│   ├── *_evaluation_*.json
│   └── *_explanation_*.json
└── visualizations/       # Generated plots and charts
    ├── feature_importance.png
    ├── confusion_matrix.png
    ├── company_ranking.png
    └── shap_summary.png
```

## Reports

JSON reports contain:
- Model evaluation metrics
- Feature importance rankings
- SHAP values and explanations
- Timestamps and metadata

## Visualizations

Generated plots include:
- Feature importance bar charts
- SHAP summary plots
- Confusion matrices
- ROC curves
- Company ranking visualizations
- Model comparison charts

All visualizations are saved as high-resolution PNGs (300 DPI).

## Accessing Outputs

```python
from src.utils.helpers import load_json

# Load evaluation report
report = load_json('outputs/reports/xgboost_evaluation_20251002_143052.json')
print(f"Accuracy: {report['accuracy']}")
```

## Note

Output files are excluded from Git. Generate your own by running the pipeline.
