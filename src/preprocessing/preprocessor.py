"""
Data preprocessing module for cleaning and preparing company data for modeling.
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Tuple
from sklearn.preprocessing import StandardScaler, RobustScaler
import pickle

from src.utils.logger import log
from src.utils.helpers import save_dataframe
from config.config import DATA_PROCESSED_PATH, RANDOM_SEED


class DataPreprocessor:
    """
    Handles data cleaning, missing value imputation, and normalization.

    Philosophy: Garbage in, garbage out. But over-cleaning can remove
    signal. Missing P/E ratio might mean company is unprofitable - that's
    information, not noise. We need to be thoughtful about imputation.
    """

    def __init__(self, scaler_type: str = "robust"):
        """
        Initialize preprocessor.

        Args:
            scaler_type: Type of scaler ('standard', 'robust', 'none')
                        Robust scaler is less sensitive to outliers.
        """
        self.scaler_type = scaler_type
        self.scaler = None
        self.feature_columns = None
        self.numeric_columns = None

        if scaler_type == "standard":
            self.scaler = StandardScaler()
        elif scaler_type == "robust":
            self.scaler = RobustScaler()
        elif scaler_type == "none":
            self.scaler = None
        else:
            raise ValueError(f"Unknown scaler type: {scaler_type}")

    def fit_transform(self, df: pd.DataFrame, target_col: Optional[str] = None) -> pd.DataFrame:
        """
        Fit preprocessor on data and transform it.

        Args:
            df: Input DataFrame
            target_col: Name of target column (will not be scaled)

        Returns:
            Preprocessed DataFrame
        """
        log.info(f"Fitting preprocessor on {len(df)} samples")

        df = df.copy()

        # 1. Handle missing values
        df = self._handle_missing_values(df)

        # 2. Remove outliers (optional, can be disabled)
        df = self._handle_outliers(df, method="clip")

        # 3. Identify numeric columns for scaling
        self.numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

        if target_col and target_col in self.numeric_columns:
            self.numeric_columns.remove(target_col)

        # Remove identifier columns
        self.numeric_columns = [col for col in self.numeric_columns
                               if col not in ['ticker', 'data_collection_date']]

        # 4. Scale numeric features
        if self.scaler:
            df[self.numeric_columns] = self.scaler.fit_transform(df[self.numeric_columns])
            log.info(f"Scaled {len(self.numeric_columns)} numeric features")

        self.feature_columns = df.columns.tolist()

        log.info("Preprocessing fit complete")
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using fitted preprocessor."""
        if self.scaler is None and self.scaler_type != "none":
            raise ValueError("Preprocessor not fitted. Call fit_transform first.")

        df = df.copy()

        # Apply same transformations
        df = self._handle_missing_values(df)
        df = self._handle_outliers(df, method="clip")

        if self.scaler:
            df[self.numeric_columns] = self.scaler.transform(df[self.numeric_columns])

        return df

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values with domain-appropriate strategies.

        Strategy:
        - Financial ratios (P/E, PEG, etc.): Median imputation
        - Returns: Forward/backward fill (time-series nature)
        - Sentiment: Neutral (0.0) if missing
        - Categorical: 'Unknown' category
        """
        df = df.copy()

        # Sentiment features: fill with neutral
        sentiment_cols = [col for col in df.columns if 'sentiment' in col.lower()]
        for col in sentiment_cols:
            df[col] = df[col].fillna(0.0)

        # Ratio/percentile features: median within sector
        ratio_cols = [col for col in df.columns
                     if any(x in col for x in ['ratio', 'percentile', 'margin', 'return_on'])]

        for col in ratio_cols:
            if 'sector' in df.columns:
                df[col] = df.groupby('sector')[col].transform(
                    lambda x: x.fillna(x.median())
                )
            df[col] = df[col].fillna(df[col].median())

        # Returns and growth: fill with 0 (neutral performance)
        return_cols = [col for col in df.columns if 'return' in col.lower() or 'growth' in col.lower()]
        for col in return_cols:
            df[col] = df[col].fillna(0.0)

        # Categorical: fill with 'Unknown'
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            df[col] = df[col].fillna('Unknown')

        # Remaining numeric: median imputation
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].median())

        missing_after = df.isna().sum().sum()
        if missing_after > 0:
            log.warning(f"Still have {missing_after} missing values after imputation")

        return df

    def _handle_outliers(self, df: pd.DataFrame, method: str = "clip") -> pd.DataFrame:
        """
        Handle outliers in numeric features.

        Args:
            method: 'clip' (winsorize) or 'remove'

        Note: Financial data has legitimate outliers. Apple's market cap
        is not an error. We clip at extreme percentiles rather than remove.
        """
        if method == "none":
            return df

        df = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        if method == "clip":
            # Clip at 1st and 99th percentiles
            for col in numeric_cols:
                lower = df[col].quantile(0.01)
                upper = df[col].quantile(0.99)
                df[col] = df[col].clip(lower=lower, upper=upper)

        elif method == "remove":
            # Remove rows with values outside 3 standard deviations
            for col in numeric_cols:
                mean = df[col].mean()
                std = df[col].std()
                df = df[(df[col] >= mean - 3*std) & (df[col] <= mean + 3*std)]

        return df

    def save(self, filepath: str):
        """Save fitted preprocessor to disk."""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        log.info(f"Saved preprocessor to {filepath}")

    @staticmethod
    def load(filepath: str) -> 'DataPreprocessor':
        """Load fitted preprocessor from disk."""
        with open(filepath, 'rb') as f:
            preprocessor = pickle.load(f)
        log.info(f"Loaded preprocessor from {filepath}")
        return preprocessor
