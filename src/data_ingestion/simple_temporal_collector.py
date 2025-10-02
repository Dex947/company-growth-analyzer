"""
Simplified temporal data collector using only reliable price and basic metrics.

Focuses on what consistently works with yfinance to maximize sample collection.
"""

import pandas as pd
import yfinance as yf
from typing import List, Dict, Any
from datetime import datetime, timedelta
import time
import numpy as np

from src.data_ingestion.base_collector import BaseDataCollector
from src.utils.logger import log


class SimpleTemporalCollector(BaseDataCollector):
    """
    Collect quarterly snapshots using reliable price data and basic info.

    Simpler than full fundamental collector but more robust - ensures we get
    maximum samples for training.
    """

    def __init__(self, quarters_back: int = 12):
        """
        Initialize collector.

        Args:
            quarters_back: Number of quarters to sample (~3 months each)
        """
        super().__init__()
        self.quarters_back = quarters_back
        self.days_per_quarter = 63  # ~3 months of trading days

    def collect(self, tickers: List[str]) -> pd.DataFrame:
        """
        Collect temporal data for tickers.

        Args:
            tickers: List of stock tickers

        Returns:
            DataFrame with (companies × quarters) rows
        """
        log.info(f"Collecting simple temporal data for {len(tickers)} companies")

        all_data = []

        for i, ticker in enumerate(tickers):
            if (i + 1) % 10 == 0:
                log.info(f"Progress: {i+1}/{len(tickers)} companies")

            try:
                ticker_data = self._collect_ticker(ticker)
                all_data.extend(ticker_data)

                # Rate limiting
                if (i + 1) % 10 == 0:
                    time.sleep(2)

            except Exception as e:
                log.error(f"Error collecting {ticker}: {e}")
                continue

        if not all_data:
            log.warning("No data collected")
            return pd.DataFrame()

        df = pd.DataFrame(all_data)
        log.info(f"Collected {len(df)} temporal samples")

        return df

    def _collect_ticker(self, ticker: str) -> List[Dict[str, Any]]:
        """Collect quarterly snapshots for one ticker."""
        stock = yf.Ticker(ticker)

        # Get 5 years of price history
        try:
            hist = stock.history(period="5y")
        except Exception as e:
            log.warning(f"Failed to get history for {ticker}: {e}")
            return []

        if hist.empty or len(hist) < 252:  # Need at least 1 year
            log.warning(f"Insufficient price history for {ticker}")
            return []

        # Get basic info
        try:
            info = stock.info
        except Exception:
            info = {}

        # Sample at quarterly intervals
        samples = []
        total_days = len(hist)

        for i in range(self.quarters_back):
            # Work backwards from most recent data
            end_idx = total_days - (i * self.days_per_quarter)
            start_idx = end_idx - self.days_per_quarter

            if start_idx < 252:  # Need at least 1 year before for calculations
                break

            try:
                snapshot = self._create_snapshot(ticker, hist, end_idx, info)
                if snapshot:
                    samples.append(snapshot)
            except Exception as e:
                log.debug(f"Skipping snapshot for {ticker} at index {end_idx}: {e}")
                continue

        return samples

    def _create_snapshot(
        self,
        ticker: str,
        hist: pd.DataFrame,
        idx: int,
        info: Dict
    ) -> Dict[str, Any]:
        """
        Create a snapshot at a specific point in time.

        Args:
            ticker: Stock ticker
            hist: Price history DataFrame
            idx: Index position in history
            info: Stock info dictionary

        Returns:
            Dictionary of metrics
        """
        # Basic identification
        sample_date = hist.index[idx]
        price = hist['Close'].iloc[idx]
        volume = hist['Volume'].iloc[idx]

        # Returns (backward looking from this point)
        returns_1m = self._calc_return(hist, idx, days=21)
        returns_3m = self._calc_return(hist, idx, days=63)
        returns_6m = self._calc_return(hist, idx, days=126)
        returns_12m = self._calc_return(hist, idx, days=252)

        # Volatility (60-day)
        prices_60d = hist['Close'].iloc[max(0, idx-60):idx+1]
        volatility = prices_60d.pct_change().std() * np.sqrt(252) if len(prices_60d) > 10 else None

        # Moving averages
        ma_50 = hist['Close'].iloc[max(0, idx-50):idx+1].mean() if idx >= 50 else None
        ma_200 = hist['Close'].iloc[max(0, idx-200):idx+1].mean() if idx >= 200 else None

        # Momentum indicators
        rsi = self._calc_rsi(hist['Close'].iloc[max(0, idx-14):idx+1]) if idx >= 14 else None

        # Volume indicators
        avg_volume = hist['Volume'].iloc[max(0, idx-20):idx+1].mean() if idx >= 20 else None
        volume_ratio = volume / avg_volume if avg_volume and avg_volume > 0 else None

        # Market cap (from info, assume relatively stable)
        market_cap = info.get('marketCap')

        # Basic fundamentals (from info - these are current, not historical)
        pe_ratio = info.get('trailingPE')
        pb_ratio = info.get('priceToBook')
        ps_ratio = info.get('priceToSalesTrailing12Months')
        profit_margin = info.get('profitMargins')
        operating_margin = info.get('operatingMargins')
        roe = info.get('returnOnEquity')
        debt_to_equity = info.get('debtToEquity')

        # Revenue and earnings growth (from info)
        revenue_growth = info.get('revenueGrowth')
        earnings_growth = info.get('earningsGrowth')

        return {
            'ticker': ticker,
            'date': sample_date.strftime('%Y-%m-%d'),
            'quarter': f"{sample_date.year}Q{(sample_date.month - 1) // 3 + 1}",

            # Price metrics
            'price': float(price),
            'volume': float(volume) if volume else None,

            # Returns
            'returns_1m': returns_1m,
            'returns_3m': returns_3m,
            'returns_6m': returns_6m,
            'returns_12m': returns_12m,

            # Volatility & risk
            'volatility': volatility,

            # Technical indicators
            'ma_50': ma_50,
            'ma_200': ma_200,
            'rsi': rsi,
            'volume_ratio': volume_ratio,

            # Fundamentals (note: these are current values, not historical)
            'market_cap': market_cap,
            'pe_ratio': pe_ratio,
            'pb_ratio': pb_ratio,
            'ps_ratio': ps_ratio,
            'profit_margin': profit_margin,
            'operating_margin': operating_margin,
            'roe': roe,
            'debt_to_equity': debt_to_equity / 100 if debt_to_equity else None,  # Convert percentage
            'revenue_growth': revenue_growth,
            'earnings_growth': earnings_growth,
        }

    def _calc_return(self, hist: pd.DataFrame, idx: int, days: int) -> float:
        """Calculate return over specified days."""
        try:
            if idx < days:
                return None

            end_price = hist['Close'].iloc[idx]
            start_price = hist['Close'].iloc[idx - days]

            if start_price == 0:
                return None

            return float((end_price - start_price) / start_price)

        except Exception:
            return None

    def _calc_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate Relative Strength Index."""
        try:
            if len(prices) < period + 1:
                return None

            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))

            return float(rsi.iloc[-1])

        except Exception:
            return None
