"""
Data aggregator that combines data from multiple sources into a unified dataset.
"""

import pandas as pd
from typing import List, Optional

from src.data_ingestion.financial_data import FinancialDataCollector
from src.data_ingestion.sentiment_data import SentimentDataCollector
from src.data_ingestion.market_data import MarketDataCollector
from src.utils.logger import log
from src.utils.helpers import save_dataframe
from config.config import DATA_RAW_PATH


class DataAggregator:
    """
    Orchestrates data collection from multiple sources and combines them.

    This is the main entry point for data ingestion.
    Handles the pipeline: financial → sentiment → market context
    """

    def __init__(self):
        self.financial_collector = FinancialDataCollector()
        self.sentiment_collector = SentimentDataCollector()
        self.market_collector = MarketDataCollector()

    def collect_all_data(
        self,
        tickers: List[str],
        period: str = "1y",
        include_sentiment: bool = True,
        include_market: bool = True,
        save_raw: bool = True
    ) -> pd.DataFrame:
        """
        Collect data from all sources for given companies.

        Args:
            tickers: List of stock tickers
            period: Historical period for financial data
            include_sentiment: Whether to collect sentiment data
            include_market: Whether to add market context
            save_raw: Whether to save raw data to disk

        Returns:
            Combined DataFrame with all data
        """
        log.info(f"Starting data collection for {len(tickers)} companies")

        # 1. Collect financial data (required)
        financial_df = self.financial_collector.collect(tickers, period=period)

        if financial_df.empty:
            log.error("No financial data collected. Aborting.")
            return pd.DataFrame()

        # 2. Collect sentiment data (optional)
        if include_sentiment:
            companies = [
                {'ticker': row['ticker'], 'company_name': row.get('company_name', row['ticker'])}
                for _, row in financial_df.iterrows()
            ]
            sentiment_df = self.sentiment_collector.collect(companies)

            if not sentiment_df.empty:
                financial_df = financial_df.merge(
                    sentiment_df,
                    on='ticker',
                    how='left',
                    suffixes=('', '_sentiment')
                )
                log.info("Merged sentiment data")
            else:
                log.warning("No sentiment data to merge")

        # 3. Add market context (optional)
        if include_market:
            financial_df = self.market_collector.collect(financial_df)

        # Save raw combined data
        if save_raw:
            output_path = DATA_RAW_PATH / f"combined_data_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.parquet"
            save_dataframe(financial_df, output_path, format="parquet")

        log.info(f"Data collection complete: {len(financial_df)} companies, {len(financial_df.columns)} features")

        return financial_df

    def close(self):
        """Clean up resources."""
        self.financial_collector.close()
        self.sentiment_collector.close()
        self.market_collector.close()
