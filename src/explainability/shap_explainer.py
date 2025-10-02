"""
SHAP (SHapley Additive exPlanations) explainer for model interpretation.
"""

import pandas as pd
import numpy as np
from typing import Any, Dict, Optional
import shap

from src.utils.logger import log
from config.config import MAX_SHAP_SAMPLES


class SHAPExplainer:
    """
    Uses SHAP to explain model predictions.

    SHAP provides:
    - Global feature importance
    - Per-prediction feature attributions
    - Interaction effects

    Why SHAP? It's theoretically grounded (Shapley values from game theory)
    and model-agnostic. But it's computationally expensive for large datasets.
    """

    def __init__(self, model: Any, X_train: pd.DataFrame, max_samples: int = MAX_SHAP_SAMPLES):
        """
        Initialize SHAP explainer.

        Args:
            model: Trained model
            X_train: Training data for background distribution
            max_samples: Maximum samples for SHAP calculation (for performance)
        """
        self.model = model
        self.feature_names = X_train.columns.tolist()

        # Sample training data if too large
        if len(X_train) > max_samples:
            log.info(f"Sampling {max_samples} from {len(X_train)} training samples for SHAP")
            X_train_sample = X_train.sample(n=max_samples, random_state=42)
        else:
            X_train_sample = X_train

        # Create appropriate explainer based on model type
        try:
            # Try TreeExplainer first (fast for tree-based models)
            self.explainer = shap.TreeExplainer(model)
            log.info("Using SHAP TreeExplainer")
        except:
            # Fall back to KernelExplainer (slower but works for any model)
            if hasattr(model, 'predict_proba'):
                predict_fn = lambda x: model.predict_proba(x)[:, 1]
            else:
                predict_fn = model.predict

            self.explainer = shap.KernelExplainer(predict_fn, X_train_sample)
            log.info("Using SHAP KernelExplainer")

    def explain_global(self, X: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate global feature importance.

        Args:
            X: Data to explain

        Returns:
            Dictionary with SHAP values and feature importance
        """
        log.info("Computing global SHAP values")

        shap_values = self.explainer.shap_values(X)

        # Handle multi-class output
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Use positive class for binary classification

        # Calculate mean absolute SHAP values for feature importance
        feature_importance = np.abs(shap_values).mean(axis=0)

        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)

        return {
            'shap_values': shap_values,
            'feature_importance': importance_df,
            'feature_names': self.feature_names
        }

    def explain_prediction(self, X: pd.DataFrame, index: int = 0) -> Dict[str, Any]:
        """
        Explain a single prediction in detail.

        Args:
            X: Input data
            index: Index of sample to explain

        Returns:
            Dictionary with feature contributions
        """
        shap_values = self.explainer.shap_values(X.iloc[[index]])

        # Handle multi-class
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        feature_contributions = pd.DataFrame({
            'feature': self.feature_names,
            'value': X.iloc[index].values,
            'shap_value': shap_values[0]
        }).sort_values('shap_value', key=abs, ascending=False)

        return {
            'contributions': feature_contributions,
            'base_value': self.explainer.expected_value if hasattr(self.explainer, 'expected_value') else 0,
            'prediction': self.model.predict(X.iloc[[index]])[0]
        }

    def get_top_features(self, X: pd.DataFrame, n: int = 10) -> pd.DataFrame:
        """
        Get top N most important features globally.

        Args:
            X: Data to analyze
            n: Number of top features

        Returns:
            DataFrame with top features and their importance
        """
        global_explanation = self.explain_global(X)
        return global_explanation['feature_importance'].head(n)
