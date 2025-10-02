"""
Market data collector for gathering sector trends, competitive positioning,
and macroeconomic indicators.
"""

import pandas as pd
from typing import List, Dict, Any
from datetime import datetime

from src.data_ingestion.base_collector import BaseDataCollector
from src.utils.logger import log


class MarketDataCollector(BaseDataCollector):
    """
    Collect market and competitive data.

    This includes:
    - Sector performance and trends
    - Competitive positioning within industry
    - Market share estimates (where available)

    Note: Quantifying "pricing strategy" and "market approach" is challenging.
    We approximate through relative metrics like P/E ratios within sector,
    market cap ranking, and growth rates compared to peers.
    """

    def collect(self, companies_df: pd.DataFrame) -> pd.DataFrame:
        """
        Enrich company data with market context.

        Args:
            companies_df: DataFrame with company financial data

        Returns:
            DataFrame with added market-relative metrics
        """
        if companies_df.empty:
            return companies_df

        log.info(f"Collecting market data for {len(companies_df)} companies")

        # Add sector-relative metrics
        df = companies_df.copy()

        # Calculate sector rankings and percentiles
        df = self._add_sector_rankings(df)

        # Add competitive positioning metrics
        df = self._add_competitive_metrics(df)

        log.info("Market data collection complete")
        return df

    def _add_sector_rankings(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add sector-relative rankings for key metrics.

        This helps answer: Is this company outperforming peers?
        A high P/E ratio might be bad in isolation, but if it's
        lower than sector average, it could signal value.
        """
        metrics_to_rank = [
            'market_cap',
            'revenue',
            'revenue_growth',
            'profit_margin',
            'return_on_equity',
            'pe_ratio'
        ]

        for metric in metrics_to_rank:
            if metric not in df.columns:
                continue

            # Rank within sector (handle NaN values)
            df[f'{metric}_sector_rank'] = df.groupby('sector')[metric].rank(
                method='dense',
                ascending=False,
                na_option='bottom'
            )

            # Percentile within sector
            df[f'{metric}_sector_percentile'] = df.groupby('sector')[metric].rank(
                pct=True,
                na_option='bottom'
            )

            # Ratio to sector median
            sector_median = df.groupby('sector')[metric].transform('median')
            df[f'{metric}_vs_sector_median'] = df[metric] / sector_median

        return df

    def _add_competitive_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add metrics that capture competitive positioning.

        These include:
        - Market cap concentration (are they a dominant player?)
        - Growth relative to industry
        - Valuation premium/discount
        """
        # Market cap concentration within sector
        sector_total_cap = df.groupby('sector')['market_cap'].transform('sum')
        df['market_cap_concentration'] = df['market_cap'] / sector_total_cap

        # Is this a market leader? (top 3 in sector by market cap)
        df['is_sector_leader'] = (df['market_cap_sector_rank'] <= 3).astype(int)

        # Growth momentum: recent returns vs 1-year returns
        if 'returns_3m' in df.columns and 'returns_1y' in df.columns:
            df['growth_momentum'] = df['returns_3m'] - (df['returns_1y'] / 4)

        # Valuation relative to growth (PEG-like metric)
        if 'pe_ratio' in df.columns and 'revenue_growth' in df.columns:
            # Avoid division by zero
            df['valuation_vs_growth'] = df.apply(
                lambda row: row['pe_ratio'] / (row['revenue_growth'] * 100)
                if pd.notna(row['pe_ratio']) and pd.notna(row['revenue_growth'])
                and row['revenue_growth'] > 0
                else None,
                axis=1
            )

        return df
