"""
Feature importance analysis using permutation importance.
"""

import pandas as pd
import numpy as np
from typing import Any, Dict
from sklearn.inspection import permutation_importance

from src.utils.logger import log
from config.config import RANDOM_SEED


class FeatureImportanceAnalyzer:
    """
    Analyzes feature importance using multiple methods.

    Methods:
    1. Built-in feature importance (for tree models)
    2. Permutation importance (model-agnostic)

    Permutation importance shows impact by shuffling each feature.
    More reliable than built-in importance for high-cardinality features.
    """

    @staticmethod
    def get_builtin_importance(model: Any, feature_names: list) -> pd.DataFrame:
        """
        Get feature importance from model (if available).

        Args:
            model: Trained model with feature_importances_ attribute
            feature_names: List of feature names

        Returns:
            DataFrame with features and importance scores
        """
        if not hasattr(model, 'feature_importances_'):
            log.warning("Model does not have built-in feature importance")
            return pd.DataFrame()

        importance = model.feature_importances_

        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)

        return importance_df

    @staticmethod
    def get_permutation_importance(
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
        n_repeats: int = 10
    ) -> pd.DataFrame:
        """
        Calculate permutation importance.

        Args:
            model: Trained model
            X: Features
            y: Target
            n_repeats: Number of times to permute each feature

        Returns:
            DataFrame with permutation importance scores
        """
        log.info("Calculating permutation importance")

        result = permutation_importance(
            model, X, y,
            n_repeats=n_repeats,
            random_state=RANDOM_SEED,
            n_jobs=-1
        )

        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance_mean': result.importances_mean,
            'importance_std': result.importances_std
        }).sort_values('importance_mean', ascending=False)

        return importance_df

    @staticmethod
    def compare_importance_methods(
        model: Any,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Dict[str, pd.DataFrame]:
        """
        Compare feature importance across methods.

        Args:
            model: Trained model
            X: Features
            y: Target

        Returns:
            Dictionary with importance from different methods
        """
        results = {}

        # Built-in importance
        builtin = FeatureImportanceAnalyzer.get_builtin_importance(model, X.columns.tolist())
        if not builtin.empty:
            results['builtin'] = builtin

        # Permutation importance
        perm = FeatureImportanceAnalyzer.get_permutation_importance(model, X, y)
        results['permutation'] = perm

        return results
