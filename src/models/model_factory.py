"""
Factory for creating and configuring ML models.
"""

from typing import Dict, Any
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from config.config import MODEL_CONFIGS, RANDOM_SEED
from src.utils.logger import log


class ModelFactory:
    """
    Creates configured model instances.

    Supports both interpretable baselines and advanced models.
    Trade-off: Logistic regression is transparent but limited.
    XGBoost is powerful but needs SHAP for interpretation.
    """

    AVAILABLE_MODELS = {
        "logistic_regression": LogisticRegression,
        "random_forest": RandomForestClassifier,
        "xgboost": XGBClassifier,
        "lightgbm": LGBMClassifier
    }

    @classmethod
    def create_model(cls, model_name: str, custom_params: Dict[str, Any] = None):
        """
        Create a model instance with configuration.

        Args:
            model_name: Name of the model type
            custom_params: Override default parameters

        Returns:
            Configured model instance
        """
        if model_name not in cls.AVAILABLE_MODELS:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(cls.AVAILABLE_MODELS.keys())}")

        # Get default config
        config = MODEL_CONFIGS.get(model_name, {}).copy()

        # Override with custom params
        if custom_params:
            config.update(custom_params)

        # Create model instance
        model_class = cls.AVAILABLE_MODELS[model_name]
        model = model_class(**config)

        log.info(f"Created {model_name} model with config: {config}")

        return model

    @classmethod
    def create_all_models(cls, model_names: list = None) -> Dict[str, Any]:
        """
        Create multiple models for comparison.

        Args:
            model_names: List of model names. If None, creates all available models.

        Returns:
            Dictionary mapping model names to instances
        """
        if model_names is None:
            model_names = list(cls.AVAILABLE_MODELS.keys())

        models = {}
        for name in model_names:
            try:
                models[name] = cls.create_model(name)
            except Exception as e:
                log.error(f"Failed to create model {name}: {e}")

        log.info(f"Created {len(models)} models: {list(models.keys())}")

        return models
