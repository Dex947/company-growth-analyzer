"""
Sector-Relative Target Variable Generator.

Creates meaningful target variables that compare companies within their sector,
not across different industries.

Key insight: "Did NVDA outperform other semiconductor companies?"
not "Did NVDA outperform Coca-Cola?" (meaningless comparison)
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from loguru import logger

from config.sector_config import SECTOR_TARGET_CONFIG


class SectorRelativeTarget:
    """
    Generate sector-relative performance targets.

    Instead of arbitrary absolute thresholds (returns > 15%),
    use sector-relative metrics (returns > sector median).

    Gateway Arch principle: Don't force cross-sector comparison.
    Apples to apples, not apples to oranges.

    Methods:
    1. Sector-relative returns: (company_return - sector_median) / sector_std
    2. Percentile ranking: Where does company rank within sector?
    3. Binary outperformance: Above or below sector median?

    Attributes:
        sector: Sector name
        config: Target configuration from sector_config
        method: Target calculation method
    """

    def __init__(self, sector: str = 'semiconductors'):
        """
        Initialize sector-relative target generator.

        Args:
            sector: Sector name matching sector_config
        """
        self.sector = sector
        self.config = SECTOR_TARGET_CONFIG.get(sector, {})
        self.method = self.config.get('method', 'sector_relative')
        self.lookback = self.config.get('lookback_period', '6M')
        self.threshold_type = self.config.get('threshold_type', 'median')
        self.volatility_adjust = self.config.get('volatility_adjust', True)

        logger.info(
            f"Initialized SectorRelativeTarget for {sector}: "
            f"method={self.method}, lookback={self.lookback}"
        )

    def create_target(
        self,
        df: pd.DataFrame,
        return_column: str = 'returns_6m',
        sector_column: str = 'sector'
    ) -> Tuple[pd.Series, Dict]:
        """
        Create sector-relative target variable.

        Args:
            df: DataFrame with returns and sector info
            return_column: Column name for returns
            sector_column: Column name for sector

        Returns:
            Tuple of (target_series, metadata_dict)

        Example:
            target, meta = creator.create_target(df)
            # target: Binary (1=outperformer, 0=underperformer)
            # meta: {'threshold': 0.15, 'sector_median': 0.12, 'method': 'binary'}
        """
        if return_column not in df.columns:
            raise ValueError(f"Return column '{return_column}' not found in DataFrame")

        logger.info(f"Creating {self.method} target from {return_column}")

        if self.method == 'sector_relative':
            return self._create_sector_relative_binary(df, return_column, sector_column)
        elif self.method == 'percentile':
            return self._create_percentile_target(df, return_column, sector_column)
        elif self.method == 'continuous':
            return self._create_continuous_target(df, return_column, sector_column)
        else:
            raise ValueError(f"Unknown target method: {self.method}")

    def _create_sector_relative_binary(
        self,
        df: pd.DataFrame,
        return_column: str,
        sector_column: str
    ) -> Tuple[pd.Series, Dict]:
        """
        Binary target: Did company outperform sector median?

        Process:
        1. Calculate sector median return
        2. Optional: Adjust for sector volatility (Sharpe-like)
        3. Binary: 1 if above median, 0 if below

        This ensures balanced classes within each sector.
        """
        returns = df[return_column].copy()

        # If sector column exists, use sector-specific median
        # Otherwise, use overall median
        if sector_column in df.columns and df[sector_column].nunique() > 1:
            logger.info(f"Using sector-specific thresholds ({df[sector_column].nunique()} sectors)")
            sector_medians = df.groupby(sector_column)[return_column].transform('median')
            threshold = sector_medians
        else:
            # Single sector - use overall median
            threshold = returns.median()
            logger.info(f"Using overall median threshold: {threshold:.4f}")

        # Volatility adjustment (optional)
        if self.volatility_adjust:
            if sector_column in df.columns and df[sector_column].nunique() > 1:
                sector_std = df.groupby(sector_column)[return_column].transform('std')
            else:
                sector_std = returns.std()

            # Risk-adjusted excess return
            excess_return = returns - threshold
            risk_adjusted = excess_return / (sector_std + 1e-6)  # Avoid division by zero

            # Positive risk-adjusted excess = outperformer
            target = (risk_adjusted > 0).astype(int)

            metadata = {
                'method': 'sector_relative_volatility_adjusted',
                'threshold': threshold if isinstance(threshold, float) else threshold.mean(),
                'sector_volatility': sector_std if isinstance(sector_std, float) else sector_std.mean(),
                'positive_class_pct': target.mean(),
                'n_outperformers': target.sum(),
                'n_underperformers': (1 - target).sum(),
            }
        else:
            # Simple binary: above/below median
            target = (returns > threshold).astype(int)

            metadata = {
                'method': 'sector_relative_binary',
                'threshold': threshold if isinstance(threshold, float) else threshold.mean(),
                'positive_class_pct': target.mean(),
                'n_outperformers': target.sum(),
                'n_underperformers': (1 - target).sum(),
            }

        logger.info(
            f"Created binary target: {target.sum()} outperformers, "
            f"{(1-target).sum()} underperformers ({target.mean():.1%} positive class)"
        )

        return target, metadata

    def _create_percentile_target(
        self,
        df: pd.DataFrame,
        return_column: str,
        sector_column: str
    ) -> Tuple[pd.Series, Dict]:
        """
        Percentile ranking target (continuous 0-1).

        Company's rank within sector / total companies in sector

        Benefits:
        - Continuous target (more information than binary)
        - Robust to outliers
        - Interpretable (0.9 = top 10%, 0.1 = bottom 10%)
        """
        returns = df[return_column].copy()

        if sector_column in df.columns and df[sector_column].nunique() > 1:
            # Rank within each sector
            target = df.groupby(sector_column)[return_column].rank(pct=True)
        else:
            # Overall ranking
            target = returns.rank(pct=True)

        metadata = {
            'method': 'percentile_ranking',
            'min': target.min(),
            'max': target.max(),
            'mean': target.mean(),
            'median': target.median(),
        }

        logger.info(
            f"Created percentile target: "
            f"range [{target.min():.3f}, {target.max():.3f}], "
            f"mean={target.mean():.3f}"
        )

        return target, metadata

    def _create_continuous_target(
        self,
        df: pd.DataFrame,
        return_column: str,
        sector_column: str
    ) -> Tuple[pd.Series, Dict]:
        """
        Continuous excess return target.

        (company_return - sector_median) / sector_std

        This is a Sharpe-like metric:
        - Positive = outperformed on risk-adjusted basis
        - Magnitude indicates degree of out/underperformance
        """
        returns = df[return_column].copy()

        if sector_column in df.columns and df[sector_column].nunique() > 1:
            sector_median = df.groupby(sector_column)[return_column].transform('median')
            sector_std = df.groupby(sector_column)[return_column].transform('std')
        else:
            sector_median = returns.median()
            sector_std = returns.std()

        # Excess return
        excess = returns - sector_median

        # Risk-adjusted
        target = excess / (sector_std + 1e-6)

        # Clip extreme values
        target = np.clip(target, -5, 5)

        metadata = {
            'method': 'continuous_excess_return',
            'mean': target.mean(),
            'std': target.std(),
            'min': target.min(),
            'max': target.max(),
            'sector_median_return': sector_median if isinstance(sector_median, float) else sector_median.mean(),
        }

        logger.info(
            f"Created continuous target: "
            f"mean={target.mean():.3f}, std={target.std():.3f}, "
            f"range=[{target.min():.2f}, {target.max():.2f}]"
        )

        return target, metadata

    def validate_target_distribution(
        self,
        target: pd.Series,
        metadata: Dict
    ) -> Dict:
        """
        Validate target variable has good properties for ML.

        Checks:
        1. Class balance (binary): Should be roughly 40-60% split
        2. Variance (continuous): Should have sufficient spread
        3. Outliers: Check for extreme values
        4. Missing values: Should have none

        Args:
            target: Target variable
            metadata: Metadata from target creation

        Returns:
            Dict with validation results
        """
        validation = {
            'n_samples': len(target),
            'n_missing': target.isna().sum(),
            'pct_missing': target.isna().mean(),
        }

        if metadata['method'] in ['sector_relative_binary', 'sector_relative_volatility_adjusted']:
            # Binary target
            positive_pct = target.mean()
            validation['positive_class_pct'] = positive_pct
            validation['class_balance_ok'] = 0.3 <= positive_pct <= 0.7
            validation['recommendation'] = (
                'GOOD: Classes are balanced'
                if validation['class_balance_ok']
                else f'WARNING: Imbalanced classes ({positive_pct:.1%} positive)'
            )
        else:
            # Continuous target
            validation['mean'] = target.mean()
            validation['std'] = target.std()
            validation['skewness'] = target.skew()
            validation['kurtosis'] = target.kurtosis()
            validation['has_variance'] = target.std() > 0.1
            validation['recommendation'] = (
                'GOOD: Continuous target has variance'
                if validation['has_variance']
                else 'WARNING: Low variance in target'
            )

        if validation['n_missing'] > 0:
            logger.warning(f"Target has {validation['n_missing']} missing values")

        return validation


def create_multi_period_target(
    df: pd.DataFrame,
    sector: str,
    periods: list = ['3m', '6m', '12m']
) -> pd.DataFrame:
    """
    Create targets for multiple time horizons.

    Averaging across periods reduces noise and captures
    persistent outperformers (not just lucky in one period).

    Args:
        df: DataFrame with return columns
        sector: Sector name
        periods: List of period suffixes (e.g., ['3m', '6m', '12m'])

    Returns:
        DataFrame with multi-period targets
    """
    target_generator = SectorRelativeTarget(sector)
    result = df[['ticker']].copy()

    targets = []

    for period in periods:
        return_col = f'returns_{period}'
        if return_col in df.columns:
            target, metadata = target_generator.create_target(
                df,
                return_column=return_col
            )
            result[f'target_{period}'] = target
            targets.append(target)
            logger.info(f"Created target for {period}: {metadata}")
        else:
            logger.warning(f"Column {return_col} not found, skipping period {period}")

    # Average across periods (if multiple exist)
    if len(targets) > 1:
        result['target_avg'] = pd.DataFrame(targets).T.mean(axis=1)
        logger.info(f"Created averaged target across {len(targets)} periods")

    return result


def compare_absolute_vs_relative_targets(
    df: pd.DataFrame,
    return_column: str = 'returns_6m',
    absolute_threshold: float = 0.15
) -> pd.DataFrame:
    """
    Compare absolute threshold vs sector-relative targets.

    Demonstrates why sector-relative is better.

    Args:
        df: DataFrame with returns and sector
        return_column: Return column name
        absolute_threshold: Absolute return threshold (e.g., 15%)

    Returns:
        Comparison DataFrame
    """
    # Absolute target
    absolute_target = (df[return_column] > absolute_threshold).astype(int)

    # Sector-relative target
    if 'sector' in df.columns:
        sector_target_gen = SectorRelativeTarget('semiconductors')
        relative_target, _ = sector_target_gen.create_target(df, return_column)
    else:
        relative_target = (df[return_column] > df[return_column].median()).astype(int)

    comparison = pd.DataFrame({
        'ticker': df['ticker'],
        'company': df.get('company', ''),
        'sector': df.get('sector', ''),
        'return': df[return_column],
        'absolute_target': absolute_target,
        'relative_target': relative_target,
        'agreement': (absolute_target == relative_target).astype(int),
    })

    # Add sector median for context
    if 'sector' in df.columns:
        comparison['sector_median'] = df.groupby('sector')[return_column].transform('median')

    agreement_rate = comparison['agreement'].mean()
    logger.info(f"Absolute vs Relative agreement: {agreement_rate:.1%}")

    return comparison
