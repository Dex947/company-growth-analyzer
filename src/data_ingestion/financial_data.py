"""
Financial data collector using Yahoo Finance and Alpha Vantage.
Retrieves company financial metrics, stock prices, and fundamental data.
"""

import pandas as pd
import yfinance as yf
import numpy as np
from typing import List, Dict, Any
from datetime import datetime, timedelta
import time

from src.data_ingestion.base_collector import BaseDataCollector
from src.utils.logger import log
from config.config import ALPHA_VANTAGE_API_KEY


class FinancialDataCollector(BaseDataCollector):
    """
    Collect financial data for companies.

    Sources:
    - Yahoo Finance (free, no API key required)
    - Alpha Vantage (requires API key for fundamental data)

    Question: Is market cap a reliable indicator of future success?
    Large caps are stable but may have limited growth potential.
    Small caps are risky but can explode. This collector gathers
    both types of metrics for the model to decide.
    """

    def __init__(self, quarters_back: int = 12):
        super().__init__()
        self.alpha_vantage_api_key = ALPHA_VANTAGE_API_KEY
        self.quarters_back = quarters_back
        self.days_per_quarter = 63

    def collect(self, tickers: List[str], period: str = "1y", temporal: bool = True) -> pd.DataFrame:
        """
        Collect financial data for given company tickers.

        Args:
            tickers: List of stock tickers (e.g., ['AAPL', 'GOOGL'])
            period: Historical period ('1y', '2y', '5y')

        Returns:
            DataFrame with financial metrics per company
        """
        log.info(f"Collecting financial data for {len(tickers)} companies")

        companies_data = []

        for ticker in tickers:
            try:
                data = self._collect_ticker_data(ticker, period)
                if data:
                    companies_data.append(data)
            except Exception as e:
                log.error(f"Failed to collect data for {ticker}: {e}")
                continue

        if not companies_data:
            log.warning("No financial data collected")
            return pd.DataFrame()

        df = pd.DataFrame(companies_data)
        log.info(f"Collected financial data for {len(df)} companies")
        return df

    def _collect_ticker_data(self, ticker: str, period: str) -> Dict[str, Any]:
        """Collect data for a single ticker using yfinance."""
        cache_key = f"financial_{ticker}_{period}"

        # Check cache first
        cached_data = self._get_cached(cache_key)
        if cached_data:
            return cached_data

        log.debug(f"Fetching data for {ticker}")

        # Fetch from yfinance
        stock = yf.Ticker(ticker)
        info = stock.info
        hist = stock.history(period=period)

        if hist.empty:
            log.warning(f"No historical data for {ticker}")
            return None

        # Calculate metrics
        current_price = hist['Close'].iloc[-1]
        price_52w_high = hist['High'].max()
        price_52w_low = hist['Low'].min()
        avg_volume = hist['Volume'].mean()

        # Calculate returns
        returns_1m = self._calculate_return(hist, days=30)
        returns_3m = self._calculate_return(hist, days=90)
        returns_6m = self._calculate_return(hist, days=180)
        returns_1y = self._calculate_return(hist, days=365)

        # Volatility (standard deviation of daily returns)
        daily_returns = hist['Close'].pct_change()
        volatility = daily_returns.std() * (252 ** 0.5)  # Annualized

        data = {
            'ticker': ticker,
            'company_name': info.get('longName', ticker),
            'sector': info.get('sector', 'Unknown'),
            'industry': info.get('industry', 'Unknown'),
            'market_cap': info.get('marketCap', 0),
            'current_price': current_price,
            'price_52w_high': price_52w_high,
            'price_52w_low': price_52w_low,
            'avg_volume': avg_volume,
            'pe_ratio': info.get('trailingPE', None),
            'forward_pe': info.get('forwardPE', None),
            'peg_ratio': info.get('pegRatio', None),
            'price_to_book': info.get('priceToBook', None),
            'debt_to_equity': info.get('debtToEquity', None),
            'revenue': info.get('totalRevenue', 0),
            'revenue_growth': info.get('revenueGrowth', None),
            'profit_margin': info.get('profitMargins', None),
            'operating_margin': info.get('operatingMargins', None),
            'return_on_equity': info.get('returnOnEquity', None),
            'return_on_assets': info.get('returnOnAssets', None),
            'returns_1m': returns_1m,
            'returns_3m': returns_3m,
            'returns_6m': returns_6m,
            'returns_1y': returns_1y,
            'volatility': volatility,
            'beta': info.get('beta', None),
            'employees': info.get('fullTimeEmployees', None),
            'data_collection_date': datetime.now().isoformat()
        }

        # Cache the data
        self._set_cache(cache_key, data)

        return data

    @staticmethod
    def _calculate_return(hist: pd.DataFrame, days: int) -> float:
        """Calculate return over specified number of days."""
        if len(hist) < days:
            return None

        start_price = hist['Close'].iloc[-days]
        end_price = hist['Close'].iloc[-1]

        if start_price == 0:
            return None

        return (end_price - start_price) / start_price
