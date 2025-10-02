"""
Sector Factor Model for dimensionality reduction.

This module reduces 75+ features to 5-8 sector-specific factors,
improving sample efficiency and interpretability.

Inspired by Fama-French factor models in quantitative finance.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from loguru import logger
from sklearn.preprocessing import StandardScaler

from config.sector_config import SectorFactorDefinition


class SectorFactorModel:
    """
    Transforms raw features into sector-specific composite factors.

    Gateway Arch principle: Don't force 75 features on 15 samples.
    Instead, compress to 5-8 interpretable factors based on domain knowledge.

    Example:
        For semiconductors:
        - 75 features → 5 factors (innovation, profitability, market_position, health, momentum)
        - Each factor = weighted combination of 3-5 related metrics
        - 15:5 ratio much better than 15:75 for ML

    Attributes:
        sector: Sector name ('semiconductors', 'cloud_saas', etc.)
        factor_definitions: Dict of factor specs from sector_config
        scaler: StandardScaler for normalization
        factor_scores_: DataFrame of computed factor scores
    """

    def __init__(self, sector: str = 'semiconductors'):
        """
        Initialize sector factor model.

        Args:
            sector: Sector name matching sector_config definitions
        """
        self.sector = sector
        self.factor_definitions = SectorFactorDefinition.get_factors(sector)
        self.scaler = StandardScaler()
        self.factor_scores_ = None
        self.feature_to_factor_map_ = {}

        if not self.factor_definitions:
            raise ValueError(f"No factor definitions found for sector: {sector}")

        logger.info(f"Initialized SectorFactorModel for {sector} with {len(self.factor_definitions)} factors")

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute factor scores from raw features.

        Process:
        1. Extract metrics needed for each factor
        2. Normalize within each factor (z-scores)
        3. Combine metrics using factor weights
        4. Return DataFrame with factor scores

        Args:
            df: DataFrame with raw features and 'ticker' column

        Returns:
            DataFrame with columns: ticker, company, sector, factor1, factor2, ...

        Example:
            innovation_factor = (
                0.33 * z_score(rd_pct_revenue) +
                0.33 * z_score(revenue_growth) +
                0.34 * z_score(gross_margin)
            )
        """
        logger.info(f"Computing factor scores for {len(df)} companies")

        if 'ticker' not in df.columns:
            raise ValueError("DataFrame must have 'ticker' column")

        # Start with identifier columns
        result = df[['ticker']].copy()
        if 'company' in df.columns:
            result['company'] = df['company']
        if 'sector' in df.columns:
            result['sector'] = df['sector']

        # Compute each factor
        for factor_name, factor_spec in self.factor_definitions.items():
            try:
                factor_score = self._compute_factor(df, factor_name, factor_spec)
                result[factor_name] = factor_score
                logger.info(f"Computed {factor_name}: mean={factor_score.mean():.3f}, std={factor_score.std():.3f}")
            except Exception as e:
                logger.warning(f"Failed to compute {factor_name}: {e}")
                result[factor_name] = 0.0

        self.factor_scores_ = result

        logger.info(f"Factor scores computed: {result.shape[1] - 1} factors")
        return result

    def _compute_factor(
        self,
        df: pd.DataFrame,
        factor_name: str,
        factor_spec: Dict
    ) -> pd.Series:
        """
        Compute a single factor score.

        Args:
            df: DataFrame with raw features
            factor_name: Name of the factor
            factor_spec: Factor specification from config

        Returns:
            Series of factor scores (one per company)
        """
        metrics = factor_spec['metrics']
        weight = factor_spec.get('weight', 1.0)

        # Extract available metrics
        available_metrics = []
        metric_values = []

        for metric in metrics:
            # Try to find metric in DataFrame (handle various naming conventions)
            metric_col = self._find_metric_column(df, metric)
            if metric_col:
                available_metrics.append(metric)
                metric_values.append(df[metric_col])
                self.feature_to_factor_map_[metric_col] = factor_name

        if not available_metrics:
            logger.warning(f"No metrics found for factor {factor_name}. Metrics needed: {metrics}")
            return pd.Series(0.0, index=df.index)

        # Stack metrics into a matrix
        metric_matrix = np.column_stack(metric_values)

        # Handle missing values
        metric_matrix = self._handle_missing(metric_matrix)

        # Normalize each metric (z-score within factor)
        normalized = self._normalize_metrics(metric_matrix)

        # Equal-weighted combination (can be made more sophisticated)
        n_metrics = len(available_metrics)
        equal_weight = 1.0 / n_metrics
        factor_score = normalized.mean(axis=1)

        # Apply factor-level weight
        factor_score = factor_score * weight

        logger.debug(
            f"{factor_name}: used {len(available_metrics)}/{len(metrics)} metrics "
            f"(weight={weight:.2f})"
        )

        return pd.Series(factor_score, index=df.index)

    def _find_metric_column(self, df: pd.DataFrame, metric: str) -> Optional[str]:
        """
        Find column in DataFrame matching metric name.

        Handles various naming conventions:
        - rd_pct_revenue → rd_pct_revenue, r_and_d_pct_revenue, rdToRevenue
        - revenue_growth → revenue_growth, revenueGrowth, revenue_growth_yoy

        Args:
            df: DataFrame
            metric: Metric name from factor definition

        Returns:
            Column name if found, None otherwise
        """
        # Direct match
        if metric in df.columns:
            return metric

        # Try case-insensitive match
        metric_lower = metric.lower()
        for col in df.columns:
            if col.lower() == metric_lower:
                return col

        # Try removing underscores
        metric_no_underscore = metric.replace('_', '')
        for col in df.columns:
            if col.lower().replace('_', '') == metric_no_underscore:
                return col

        # Common aliases
        aliases = {
            'rd_pct_revenue': ['r_and_d_pct_revenue', 'research_development_ratio'],
            'revenue_growth': ['revenueGrowth', 'revenue_growth_yoy', 'revenue_growth_rate'],
            'operating_margin': ['operatingMargin', 'operating_income_margin', 'ebit_margin'],
            'gross_margin': ['grossMargin', 'grossProfitMargin'],
            'free_cash_flow_margin': ['fcf_margin', 'freeCashFlowMargin'],
            'return_on_equity': ['roe', 'returnOnEquity'],
            'return_on_assets': ['roa', 'returnOnAssets'],
            'debt_to_equity': ['debtToEquity', 'debt_equity_ratio', 'leverage'],
            'earnings_growth': ['earningsGrowth', 'eps_growth', 'earnings_growth_yoy'],
        }

        if metric in aliases:
            for alias in aliases[metric]:
                if alias in df.columns:
                    return alias
                # Try case-insensitive on aliases
                for col in df.columns:
                    if col.lower() == alias.lower():
                        return col

        return None

    def _handle_missing(self, matrix: np.ndarray) -> np.ndarray:
        """
        Handle missing values in metric matrix.

        Strategy:
        1. Column mean imputation (within-metric)
        2. If entire column is NaN, fill with 0
        3. Clip extreme outliers (>5 std)

        Args:
            matrix: 2D numpy array (companies x metrics)

        Returns:
            Matrix with missing values handled
        """
        matrix = matrix.copy()

        for col_idx in range(matrix.shape[1]):
            col = matrix[:, col_idx]

            # Get finite values
            finite_mask = np.isfinite(col)

            if finite_mask.sum() == 0:
                # Entire column is NaN/inf
                matrix[:, col_idx] = 0.0
            else:
                # Impute with column mean
                col_mean = np.mean(col[finite_mask])
                col[~finite_mask] = col_mean
                matrix[:, col_idx] = col

                # Clip outliers (beyond ±5 std)
                col_std = np.std(col[finite_mask])
                if col_std > 0:
                    lower_bound = col_mean - 5 * col_std
                    upper_bound = col_mean + 5 * col_std
                    matrix[:, col_idx] = np.clip(col, lower_bound, upper_bound)

        return matrix

    def _normalize_metrics(self, matrix: np.ndarray) -> np.ndarray:
        """
        Normalize metrics to z-scores.

        Each metric is standardized to mean=0, std=1
        within the factor group.

        Args:
            matrix: 2D array (companies x metrics)

        Returns:
            Normalized matrix
        """
        normalized = np.zeros_like(matrix)

        for col_idx in range(matrix.shape[1]):
            col = matrix[:, col_idx]
            col_mean = np.mean(col)
            col_std = np.std(col)

            if col_std > 0:
                normalized[:, col_idx] = (col - col_mean) / col_std
            else:
                # No variation - all same value
                normalized[:, col_idx] = 0.0

        return normalized

    def get_factor_composition(self) -> pd.DataFrame:
        """
        Get human-readable description of what each factor measures.

        Returns:
            DataFrame with columns: factor, description, metrics, weight
        """
        compositions = []

        for factor_name, factor_spec in self.factor_definitions.items():
            compositions.append({
                'factor': factor_name,
                'description': factor_spec.get('description', ''),
                'metrics': ', '.join(factor_spec['metrics']),
                'weight': factor_spec.get('weight', 1.0),
                'n_metrics': len(factor_spec['metrics']),
            })

        return pd.DataFrame(compositions)

    def explain_factor_score(
        self,
        ticker: str,
        factor_name: str
    ) -> Dict:
        """
        Explain why a company has a particular factor score.

        Args:
            ticker: Company ticker
            factor_name: Name of factor to explain

        Returns:
            Dict with metric contributions
        """
        if self.factor_scores_ is None:
            raise ValueError("Must call fit_transform() first")

        if ticker not in self.factor_scores_['ticker'].values:
            raise ValueError(f"Ticker {ticker} not found in factor scores")

        if factor_name not in self.factor_definitions:
            raise ValueError(f"Factor {factor_name} not defined for sector {self.sector}")

        # Get company's factor score
        company_row = self.factor_scores_[self.factor_scores_['ticker'] == ticker]
        factor_score = company_row[factor_name].values[0]

        # Get metric breakdown
        factor_spec = self.factor_definitions[factor_name]
        metrics = factor_spec['metrics']

        explanation = {
            'ticker': ticker,
            'factor': factor_name,
            'score': factor_score,
            'description': factor_spec.get('description', ''),
            'metrics': metrics,
        }

        return explanation

    def get_top_factors_for_company(
        self,
        ticker: str,
        top_n: int = 3
    ) -> pd.DataFrame:
        """
        Get top N factors (strengths) for a company.

        Args:
            ticker: Company ticker
            top_n: Number of top factors to return

        Returns:
            DataFrame sorted by factor score (descending)
        """
        if self.factor_scores_ is None:
            raise ValueError("Must call fit_transform() first")

        company_row = self.factor_scores_[self.factor_scores_['ticker'] == ticker]

        if len(company_row) == 0:
            raise ValueError(f"Ticker {ticker} not found")

        # Extract factor scores (exclude ticker/company/sector columns)
        factor_cols = [
            col for col in self.factor_scores_.columns
            if col not in ['ticker', 'company', 'sector']
        ]

        scores = company_row[factor_cols].T
        scores.columns = ['score']
        scores = scores.sort_values('score', ascending=False)

        return scores.head(top_n)


def compare_feature_importance_reduction(
    original_df: pd.DataFrame,
    factor_model: SectorFactorModel
) -> Dict:
    """
    Compare original features vs factor model for sample efficiency.

    Args:
        original_df: DataFrame with all raw features
        factor_model: Fitted SectorFactorModel

    Returns:
        Dict with comparison statistics
    """
    n_samples = len(original_df)
    n_original_features = len(original_df.columns) - 3  # Exclude ticker, company, sector
    n_factors = len(factor_model.factor_definitions)

    original_ratio = n_samples / n_original_features
    factor_ratio = n_samples / n_factors

    improvement = factor_ratio / original_ratio if original_ratio > 0 else float('inf')

    comparison = {
        'n_samples': n_samples,
        'n_original_features': n_original_features,
        'n_factors': n_factors,
        'original_sample_per_feature': original_ratio,
        'factor_sample_per_feature': factor_ratio,
        'efficiency_improvement': improvement,
        'meets_10_to_1_rule': factor_ratio >= 10,  # Need 10+ samples per feature
        'recommendation': (
            'GOOD: Factor model provides sufficient samples per feature'
            if factor_ratio >= 10
            else f'WARNING: Need {int(10 * n_factors - n_samples)} more samples or fewer factors'
        )
    }

    return comparison
