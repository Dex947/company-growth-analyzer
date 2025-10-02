"""
Helper utilities for data processing and file operations.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, Dict, Any, List
import json
import time
from functools import wraps
from src.utils.logger import log


def retry_on_failure(max_retries: int = 3, delay: int = 1):
    """
    Decorator to retry a function on failure.

    Args:
        max_retries: Maximum number of retry attempts
        delay: Delay in seconds between retries
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        log.error(f"Failed after {max_retries} attempts: {e}")
                        raise
                    log.warning(f"Attempt {attempt + 1} failed: {e}. Retrying...")
                    time.sleep(delay)
            return None
        return wrapper
    return decorator


def calculate_growth_rate(values: pd.Series, periods: int = 1) -> pd.Series:
    """
    Calculate period-over-period growth rate.

    Args:
        values: Series of values
        periods: Number of periods for growth calculation

    Returns:
        Series of growth rates

    Note: Returns NaN for initial periods where growth can't be calculated.
    This is intentional - we can't infer past performance from current data.
    """
    return values.pct_change(periods=periods)


def normalize_column(series: pd.Series, method: str = "minmax") -> pd.Series:
    """
    Normalize a pandas Series using specified method.

    Args:
        series: Input series
        method: Normalization method ('minmax', 'zscore')

    Returns:
        Normalized series
    """
    if method == "minmax":
        min_val = series.min()
        max_val = series.max()
        if max_val - min_val == 0:
            return pd.Series([0.5] * len(series), index=series.index)
        return (series - min_val) / (max_val - min_val)
    elif method == "zscore":
        return (series - series.mean()) / series.std()
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def save_dataframe(df: pd.DataFrame, filepath: Union[str, Path], format: str = "parquet"):
    """
    Save DataFrame to disk in specified format.

    Args:
        df: DataFrame to save
        filepath: Destination path
        format: File format ('parquet', 'feather', 'csv')

    Note: Parquet and Feather offer better compression and faster I/O than CSV.
    This matters when dealing with large company datasets.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    if format == "parquet":
        df.to_parquet(filepath, compression="snappy", index=False)
    elif format == "feather":
        df.to_feather(filepath)
    elif format == "csv":
        df.to_csv(filepath, index=False)
    else:
        raise ValueError(f"Unsupported format: {format}")

    log.info(f"Saved DataFrame to {filepath} ({format} format)")


def load_dataframe(filepath: Union[str, Path]) -> pd.DataFrame:
    """
    Load DataFrame from disk, auto-detecting format.

    Args:
        filepath: Path to file

    Returns:
        Loaded DataFrame
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    suffix = filepath.suffix.lower()

    if suffix == ".parquet":
        df = pd.read_parquet(filepath)
    elif suffix == ".feather":
        df = pd.read_feather(filepath)
    elif suffix == ".csv":
        df = pd.read_csv(filepath)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")

    log.info(f"Loaded DataFrame from {filepath} with shape {df.shape}")
    return df


def save_json(data: Dict[Any, Any], filepath: Union[str, Path]):
    """Save dictionary to JSON file."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

    log.info(f"Saved JSON to {filepath}")


def load_json(filepath: Union[str, Path]) -> Dict[Any, Any]:
    """Load JSON file to dictionary."""
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    with open(filepath, 'r') as f:
        data = json.load(f)

    log.info(f"Loaded JSON from {filepath}")
    return data


def validate_company_data(df: pd.DataFrame, required_columns: list) -> bool:
    """
    Validate that company data contains required columns.

    Args:
        df: DataFrame to validate
        required_columns: List of required column names

    Returns:
        True if valid, raises ValueError otherwise
    """
    missing_columns = set(required_columns) - set(df.columns)

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    return True
