"""Data ingestion modules for multi-source company data collection."""
from .financial_data import FinancialDataCollector
from .sentiment_data import SentimentDataCollector
from .market_data import MarketDataCollector
from .data_aggregator import DataAggregator

__all__ = [
    "FinancialDataCollector",
    "SentimentDataCollector",
    "MarketDataCollector",
    "DataAggregator"
]
