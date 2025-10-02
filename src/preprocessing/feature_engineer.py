"""
Feature engineering module for creating derived features and interactions.
"""

import pandas as pd
import numpy as np
from typing import List, Optional

from src.utils.logger import log


class FeatureEngineer:
    """
    Creates engineered features from raw company data.

    Goal: Surface hidden patterns that raw metrics miss.
    Example: A company with declining margins but increasing market
    share might be executing a growth-over-profit strategy.
    """

    def __init__(self):
        self.created_features = []

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create engineered features.

        Args:
            df: Input DataFrame with raw features

        Returns:
            DataFrame with additional engineered features
        """
        log.info("Engineering features")

        df = df.copy()
        initial_features = len(df.columns)

        # 1. Financial health indicators
        df = self._create_financial_health_features(df)

        # 2. Growth and momentum features
        df = self._create_growth_features(df)

        # 3. Valuation features
        df = self._create_valuation_features(df)

        # 4. Competitive position features
        df = self._create_competitive_features(df)

        # 5. Sentiment-based features
        df = self._create_sentiment_features(df)

        # 6. Interaction features
        df = self._create_interaction_features(df)

        new_features = len(df.columns) - initial_features
        log.info(f"Created {new_features} engineered features")

        return df

    def _create_financial_health_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features indicating financial health.

        Question: Does profitability predict success?
        Not always - Amazon was unprofitable for years.
        But combined with growth, it might matter.
        """
        # Profitability score
        if all(col in df.columns for col in ['profit_margin', 'operating_margin', 'return_on_equity']):
            df['profitability_score'] = (
                df['profit_margin'].fillna(0) +
                df['operating_margin'].fillna(0) +
                df['return_on_equity'].fillna(0)
            ) / 3

        # Efficiency ratio
        if all(col in df.columns for col in ['return_on_assets', 'return_on_equity']):
            df['efficiency_ratio'] = df['return_on_assets'] / (df['return_on_equity'] + 0.001)

        # Debt burden
        if 'debt_to_equity' in df.columns:
            df['high_debt'] = (df['debt_to_equity'] > df['debt_to_equity'].median()).astype(int)

        return df

    def _create_growth_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features capturing growth and momentum."""

        # Recent growth acceleration
        if 'returns_3m' in df.columns and 'returns_6m' in df.columns:
            df['growth_acceleration'] = df['returns_3m'] - (df['returns_6m'] / 2)

        # Consistent growth (low volatility + positive returns)
        if 'volatility' in df.columns and 'returns_1y' in df.columns:
            df['stable_growth'] = (df['returns_1y'] / (df['volatility'] + 0.01))

        # Revenue momentum
        if 'revenue_growth' in df.columns:
            df['revenue_growth_strong'] = (df['revenue_growth'] > df['revenue_growth'].median()).astype(int)

        return df

    def _create_valuation_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create valuation-related features."""

        # Undervalued vs sector
        if 'pe_ratio_vs_sector_median' in df.columns:
            df['undervalued_vs_sector'] = (df['pe_ratio_vs_sector_median'] < 1.0).astype(int)

        # Price momentum vs valuation
        if 'returns_6m' in df.columns and 'pe_ratio' in df.columns:
            # High returns with low P/E might indicate undervaluation
            df['momentum_value_combo'] = df['returns_6m'] / (df['pe_ratio'] + 1)

        # Market cap category
        if 'market_cap' in df.columns:
            df['market_cap_category'] = pd.cut(
                df['market_cap'],
                bins=[0, 2e9, 10e9, 100e9, float('inf')],
                labels=['small', 'mid', 'large', 'mega']
            ).astype(str)

        return df

    def _create_competitive_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features related to competitive positioning."""

        # Dominant player (high market cap concentration)
        if 'market_cap_concentration' in df.columns:
            df['is_dominant'] = (df['market_cap_concentration'] > 0.2).astype(int)

        # Growth vs sector
        if 'returns_1y' in df.columns and 'sector' in df.columns:
            sector_avg_return = df.groupby('sector')['returns_1y'].transform('mean')
            df['outperforming_sector'] = (df['returns_1y'] > sector_avg_return).astype(int)

        return df

    def _create_sentiment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create sentiment-derived features."""

        # Sentiment momentum (positive sentiment with positive returns)
        if 'avg_sentiment_vader' in df.columns and 'returns_3m' in df.columns:
            df['sentiment_return_alignment'] = (
                (df['avg_sentiment_vader'] > 0) & (df['returns_3m'] > 0)
            ).astype(int)

        # Sentiment consensus
        if 'positive_ratio' in df.columns and 'negative_ratio' in df.columns:
            df['sentiment_consensus'] = abs(df['positive_ratio'] - df['negative_ratio'])

        # News coverage (more articles might mean more interest)
        if 'article_count' in df.columns:
            df['high_coverage'] = (df['article_count'] > df['article_count'].median()).astype(int)

        return df

    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features between key variables.

        Interactions can reveal non-linear patterns. Example:
        Growth + good sentiment might be more predictive than either alone.
        """
        # Growth × Sentiment
        if 'revenue_growth' in df.columns and 'avg_sentiment_vader' in df.columns:
            df['growth_sentiment_interaction'] = df['revenue_growth'] * df['avg_sentiment_vader']

        # Valuation × Momentum
        if 'pe_ratio' in df.columns and 'returns_6m' in df.columns:
            df['valuation_momentum_interaction'] = df['pe_ratio'] * df['returns_6m']

        # Size × Growth
        if 'market_cap' in df.columns and 'returns_1y' in df.columns:
            # Log scale for market cap to reduce skew
            df['size_growth_interaction'] = np.log1p(df['market_cap']) * df['returns_1y']

        return df
