"""
Sentiment data collector for analyzing public perception from news and social media.
Uses various NLP techniques to gauge company sentiment.
"""

import pandas as pd
from typing import List, Dict, Any
from datetime import datetime, timedelta
import feedparser
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from src.data_ingestion.base_collector import BaseDataCollector
from src.utils.logger import log
from config.config import NEWS_API_KEY, SENTIMENT_WINDOW_DAYS, MIN_NEWS_ARTICLES


class SentimentDataCollector(BaseDataCollector):
    """
    Collect and analyze sentiment data about companies.

    Sources:
    - News RSS feeds
    - NewsAPI (requires API key)

    Critical assumption: Sentiment reflects future performance.
    But does it? Positive news might already be priced in.
    Negative sentiment could precede turnaround. The model
    should learn the actual correlation, not assume it.
    """

    def __init__(self):
        super().__init__()
        self.news_api_key = NEWS_API_KEY
        self.vader_analyzer = SentimentIntensityAnalyzer()

    def collect(self, companies: List[Dict[str, str]]) -> pd.DataFrame:
        """
        Collect sentiment data for companies.

        Args:
            companies: List of dicts with 'ticker' and 'company_name'

        Returns:
            DataFrame with sentiment metrics per company
        """
        log.info(f"Collecting sentiment data for {len(companies)} companies")

        sentiment_data = []

        for company in companies:
            try:
                data = self._collect_company_sentiment(
                    company['ticker'],
                    company.get('company_name', company['ticker'])
                )
                if data:
                    sentiment_data.append(data)
            except Exception as e:
                log.error(f"Failed to collect sentiment for {company['ticker']}: {e}")
                continue

        if not sentiment_data:
            log.warning("No sentiment data collected")
            return pd.DataFrame()

        df = pd.DataFrame(sentiment_data)
        log.info(f"Collected sentiment data for {len(df)} companies")
        return df

    def _collect_company_sentiment(self, ticker: str, company_name: str) -> Dict[str, Any]:
        """Collect sentiment for a single company."""
        cache_key = f"sentiment_{ticker}_{SENTIMENT_WINDOW_DAYS}d"

        # Check cache
        cached_data = self._get_cached(cache_key)
        if cached_data:
            return cached_data

        log.debug(f"Analyzing sentiment for {ticker}")

        # Collect news articles
        articles = self._fetch_news_articles(company_name, ticker)

        if len(articles) < MIN_NEWS_ARTICLES:
            log.warning(f"Insufficient articles for {ticker}: {len(articles)}")
            # Return neutral sentiment if insufficient data
            data = {
                'ticker': ticker,
                'article_count': len(articles),
                'avg_sentiment_textblob': 0.0,
                'avg_sentiment_vader': 0.0,
                'sentiment_variance': 0.0,
                'positive_ratio': 0.5,
                'negative_ratio': 0.5,
                'data_collection_date': datetime.now().isoformat()
            }
        else:
            # Analyze sentiment
            sentiments = [self._analyze_text(article['title'] + ' ' + article.get('description', ''))
                         for article in articles]

            textblob_scores = [s['textblob'] for s in sentiments]
            vader_scores = [s['vader'] for s in sentiments]

            positive_count = sum(1 for s in vader_scores if s > 0.05)
            negative_count = sum(1 for s in vader_scores if s < -0.05)
            total_count = len(vader_scores)

            data = {
                'ticker': ticker,
                'article_count': len(articles),
                'avg_sentiment_textblob': sum(textblob_scores) / len(textblob_scores),
                'avg_sentiment_vader': sum(vader_scores) / len(vader_scores),
                'sentiment_variance': pd.Series(vader_scores).var(),
                'positive_ratio': positive_count / total_count if total_count > 0 else 0.5,
                'negative_ratio': negative_count / total_count if total_count > 0 else 0.5,
                'data_collection_date': datetime.now().isoformat()
            }

        # Cache the data
        self._set_cache(cache_key, data)

        return data

    def _fetch_news_articles(self, company_name: str, ticker: str) -> List[Dict[str, Any]]:
        """
        Fetch news articles for a company.

        Uses multiple sources:
        1. Google News RSS (free)
        2. NewsAPI (if API key available)

        Returns list of articles with title, description, published date.
        """
        articles = []

        # Try Google News RSS feed
        try:
            feed_url = f"https://news.google.com/rss/search?q={company_name}+OR+{ticker}"
            feed = feedparser.parse(feed_url)

            cutoff_date = datetime.now() - timedelta(days=SENTIMENT_WINDOW_DAYS)

            for entry in feed.entries[:50]:  # Limit to recent articles
                try:
                    published = datetime(*entry.published_parsed[:6])
                    if published < cutoff_date:
                        continue

                    articles.append({
                        'title': entry.title,
                        'description': entry.get('summary', ''),
                        'published': published.isoformat(),
                        'source': 'google_news'
                    })
                except Exception:
                    continue

        except Exception as e:
            log.warning(f"Failed to fetch Google News for {ticker}: {e}")

        # Add NewsAPI if key is available
        if self.news_api_key and self.news_api_key != "your_news_api_key_here":
            try:
                newsapi_articles = self._fetch_newsapi(company_name, ticker)
                articles.extend(newsapi_articles)
            except Exception as e:
                log.warning(f"Failed to fetch NewsAPI for {ticker}: {e}")

        return articles

    def _fetch_newsapi(self, company_name: str, ticker: str) -> List[Dict[str, Any]]:
        """Fetch articles from NewsAPI."""
        from_date = (datetime.now() - timedelta(days=SENTIMENT_WINDOW_DAYS)).strftime('%Y-%m-%d')

        url = "https://newsapi.org/v2/everything"
        params = {
            'q': f'{company_name} OR {ticker}',
            'from': from_date,
            'sortBy': 'publishedAt',
            'language': 'en',
            'apiKey': self.news_api_key
        }

        data = self._make_request(url, params=params)

        articles = []
        for article in data.get('articles', []):
            articles.append({
                'title': article.get('title', ''),
                'description': article.get('description', ''),
                'published': article.get('publishedAt', ''),
                'source': 'newsapi'
            })

        return articles

    def _analyze_text(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment of text using multiple methods.

        Returns both TextBlob (simple) and VADER (social media optimized) scores.
        VADER is better for short texts and captures intensity.
        """
        # TextBlob sentiment
        blob = TextBlob(text)
        textblob_score = blob.sentiment.polarity

        # VADER sentiment
        vader_score = self.vader_analyzer.polarity_scores(text)['compound']

        return {
            'textblob': textblob_score,
            'vader': vader_score
        }
