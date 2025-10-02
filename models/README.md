# Models Directory

This directory stores trained model artifacts.

## Contents

- `*.pkl` - Serialized model files (scikit-learn, XGBoost, LightGBM)
- `preprocessor.pkl` - Fitted data preprocessor
- Model metadata and configurations

## File Naming Convention

- `{model_name}_model.pkl` - Trained model
- `preprocessor.pkl` - Data preprocessor
- `{model_name}_config.json` - Model configuration (optional)

## Loading Models

```python
from src.models.model_trainer import ModelTrainer

# Load a saved model
model_data = ModelTrainer.load_model('models/xgboost_model.pkl')
model = model_data['model']
feature_names = model_data['feature_names']

# Make predictions
predictions = model.predict(X_new)
```

## Model Versioning

For production use, consider:
- Timestamping model files
- Storing model metadata (training date, metrics, data version)
- Using MLflow or similar for experiment tracking

## Note

Model files are excluded from Git (.gitignore) due to size.
Retrain models using your own data.
