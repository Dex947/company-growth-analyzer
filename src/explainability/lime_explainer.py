"""
LIME (Local Interpretable Model-agnostic Explanations) explainer.
"""

import pandas as pd
import numpy as np
from typing import Any, Dict
from lime.lime_tabular import LimeTabularExplainer

from src.utils.logger import log


class LIMEExplainer:
    """
    Uses LIME for local explanations of predictions.

    LIME explains individual predictions by fitting a simple,
    interpretable model locally around the prediction.

    Difference from SHAP: LIME is faster but less theoretically
    rigorous. SHAP gives consistent global importance, LIME doesn't.
    We use both for robustness.
    """

    def __init__(self, model: Any, X_train: pd.DataFrame, mode: str = 'classification'):
        """
        Initialize LIME explainer.

        Args:
            model: Trained model
            X_train: Training data
            mode: 'classification' or 'regression'
        """
        self.model = model
        self.feature_names = X_train.columns.tolist()
        self.mode = mode

        # Create LIME explainer
        self.explainer = LimeTabularExplainer(
            training_data=X_train.values,
            feature_names=self.feature_names,
            mode=mode,
            discretize_continuous=True
        )

        log.info(f"Initialized LIME explainer in {mode} mode")

    def explain_prediction(
        self,
        X: pd.DataFrame,
        index: int = 0,
        num_features: int = 10
    ) -> Dict[str, Any]:
        """
        Explain a single prediction.

        Args:
            X: Input data
            index: Index of sample to explain
            num_features: Number of features to include in explanation

        Returns:
            Dictionary with explanation details
        """
        instance = X.iloc[index].values

        # Get prediction function
        if self.mode == 'classification':
            if hasattr(self.model, 'predict_proba'):
                predict_fn = self.model.predict_proba
            else:
                # Wrap predict in proba-like format
                predict_fn = lambda x: np.column_stack([1 - self.model.predict(x), self.model.predict(x)])
        else:
            predict_fn = self.model.predict

        # Generate explanation
        explanation = self.explainer.explain_instance(
            instance,
            predict_fn,
            num_features=num_features
        )

        # Extract feature contributions
        contributions = explanation.as_list()

        feature_contributions = pd.DataFrame(contributions, columns=['feature_condition', 'weight'])

        return {
            'contributions': feature_contributions,
            'prediction': self.model.predict(X.iloc[[index]])[0],
            'explanation_object': explanation
        }

    def explain_multiple(
        self,
        X: pd.DataFrame,
        indices: list,
        num_features: int = 10
    ) -> Dict[int, Dict[str, Any]]:
        """
        Explain multiple predictions.

        Args:
            X: Input data
            indices: List of sample indices
            num_features: Number of features per explanation

        Returns:
            Dictionary mapping indices to explanations
        """
        explanations = {}

        for idx in indices:
            explanations[idx] = self.explain_prediction(X, idx, num_features)

        return explanations
