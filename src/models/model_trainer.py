"""
Model training orchestration with cross-validation and evaluation.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional, List
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, roc_auc_score
import pickle
from pathlib import Path

from src.models.model_factory import ModelFactory
from src.utils.logger import log
from config.config import TEST_SIZE, RANDOM_SEED, CV_FOLDS, MODELS_PATH


class ModelTrainer:
    """
    Handles model training, validation, and persistence.

    Methodology:
    1. Train/test split for final evaluation
    2. Cross-validation on training set for hyperparameter selection
    3. Multiple models trained for comparison
    """

    def __init__(self):
        self.models = {}
        self.results = {}
        self.feature_names = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def prepare_data(
        self,
        df: pd.DataFrame,
        target_col: str,
        feature_cols: Optional[List[str]] = None,
        test_size: float = TEST_SIZE
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Prepare data for training.

        Args:
            df: Input DataFrame
            target_col: Name of target column
            feature_cols: List of feature column names. If None, uses all numeric columns.
            test_size: Proportion of data for test set

        Returns:
            X_train, X_test, y_train, y_test
        """
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found")

        # Select features
        if feature_cols is None:
            # Auto-select numeric features, excluding target and identifiers
            exclude_cols = [target_col, 'ticker', 'company_name', 'data_collection_date']
            feature_cols = [col for col in df.select_dtypes(include=[np.number]).columns
                           if col not in exclude_cols]

        self.feature_names = feature_cols

        X = df[feature_cols]
        y = df[target_col]

        log.info(f"Preparing data: {len(X)} samples, {len(feature_cols)} features")

        # Handle categorical features if any
        X = pd.get_dummies(X, drop_first=True)

        # Update feature names after encoding
        self.feature_names = X.columns.tolist()

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=RANDOM_SEED,
            stratify=y if pd.api.types.is_categorical_dtype(y) or y.nunique() < 20 else None
        )

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        log.info(f"Split data: Train={len(X_train)}, Test={len(X_test)}")

        return X_train, X_test, y_train, y_test

    def train_models(
        self,
        model_names: Optional[List[str]] = None,
        use_cv: bool = True
    ) -> Dict[str, Any]:
        """
        Train multiple models and evaluate them.

        Args:
            model_names: List of model names to train. If None, trains all.
            use_cv: Whether to perform cross-validation

        Returns:
            Dictionary with training results for each model
        """
        if self.X_train is None:
            raise ValueError("Data not prepared. Call prepare_data first.")

        log.info("Starting model training")

        # Create models
        self.models = ModelFactory.create_all_models(model_names)

        for name, model in self.models.items():
            log.info(f"Training {name}...")

            try:
                # Train model
                model.fit(self.X_train, self.y_train)

                # Evaluate on training set
                train_score = model.score(self.X_train, self.y_train)

                # Evaluate on test set
                test_score = model.score(self.X_test, self.y_test)

                # Cross-validation if requested
                cv_scores = None
                if use_cv:
                    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
                    cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=cv, scoring='accuracy')

                # Store results
                self.results[name] = {
                    'model': model,
                    'train_score': train_score,
                    'test_score': test_score,
                    'cv_scores': cv_scores,
                    'cv_mean': cv_scores.mean() if cv_scores is not None else None,
                    'cv_std': cv_scores.std() if cv_scores is not None else None
                }

                log.info(f"{name} - Train: {train_score:.4f}, Test: {test_score:.4f}" +
                        (f", CV: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})" if cv_scores is not None else ""))

            except Exception as e:
                log.error(f"Failed to train {name}: {e}")
                continue

        return self.results

    def get_predictions(self, model_name: str) -> Dict[str, np.ndarray]:
        """
        Get predictions from a trained model.

        Args:
            model_name: Name of the model

        Returns:
            Dictionary with predictions and probabilities
        """
        if model_name not in self.results:
            raise ValueError(f"Model {model_name} not trained")

        model = self.results[model_name]['model']

        predictions = {
            'train_pred': model.predict(self.X_train),
            'test_pred': model.predict(self.X_test),
        }

        # Add probabilities if available
        if hasattr(model, 'predict_proba'):
            predictions['train_proba'] = model.predict_proba(self.X_train)
            predictions['test_proba'] = model.predict_proba(self.X_test)

        return predictions

    def save_model(self, model_name: str, filepath: Optional[Path] = None):
        """Save a trained model to disk."""
        if model_name not in self.results:
            raise ValueError(f"Model {model_name} not trained")

        if filepath is None:
            filepath = MODELS_PATH / f"{model_name}_model.pkl"

        model_data = {
            'model': self.results[model_name]['model'],
            'feature_names': self.feature_names,
            'results': {k: v for k, v in self.results[model_name].items() if k != 'model'}
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

        log.info(f"Saved {model_name} model to {filepath}")

    @staticmethod
    def load_model(filepath: Path) -> Dict[str, Any]:
        """Load a trained model from disk."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        log.info(f"Loaded model from {filepath}")
        return model_data

    def get_best_model(self, metric: str = 'test_score') -> Tuple[str, Any]:
        """
        Get the best performing model based on specified metric.

        Args:
            metric: Metric to use ('test_score', 'cv_mean')

        Returns:
            Tuple of (model_name, model_instance)
        """
        if not self.results:
            raise ValueError("No models trained")

        best_name = max(self.results.keys(), key=lambda k: self.results[k][metric] or 0)
        best_model = self.results[best_name]['model']

        log.info(f"Best model: {best_name} ({metric}={self.results[best_name][metric]:.4f})")

        return best_name, best_model
