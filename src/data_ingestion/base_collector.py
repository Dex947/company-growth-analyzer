"""
Base class for data collectors.
Provides common functionality for API interaction, caching, and error handling.
"""

import requests
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd

from src.utils.logger import log
from src.utils.helpers import retry_on_failure, save_json, load_json
from config.config import MAX_RETRIES, REQUEST_TIMEOUT, CACHE_EXPIRY_DAYS, DATA_RAW_PATH


class BaseDataCollector(ABC):
    """
    Abstract base class for data collectors.

    Provides:
    - API request handling with retries
    - Response caching to reduce API calls
    - Error logging and handling
    """

    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize the data collector.

        Args:
            cache_dir: Directory for caching responses. Defaults to DATA_RAW_PATH/cache
        """
        self.cache_dir = cache_dir or (DATA_RAW_PATH / "cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()

    @retry_on_failure(max_retries=MAX_RETRIES, delay=2)
    def _make_request(self, url: str, params: Optional[Dict] = None,
                     headers: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Make an HTTP request with retry logic.

        Args:
            url: API endpoint URL
            params: Query parameters
            headers: Request headers

        Returns:
            JSON response as dictionary

        Note: Retries are essential for production - APIs can be flaky.
        But should we retry on 4xx errors? Probably not, those indicate
        bad requests. Current implementation retries all errors.
        """
        try:
            response = self.session.get(
                url,
                params=params,
                headers=headers,
                timeout=REQUEST_TIMEOUT
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            log.error(f"Request failed for {url}: {e}")
            raise

    def _get_cache_path(self, key: str) -> Path:
        """Generate cache file path for a given key."""
        safe_key = key.replace("/", "_").replace(":", "_")
        return self.cache_dir / f"{safe_key}.json"

    def _is_cache_valid(self, cache_path: Path) -> bool:
        """Check if cached data is still valid based on expiry time."""
        if not cache_path.exists():
            return False

        cache_age = datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)
        return cache_age < timedelta(days=CACHE_EXPIRY_DAYS)

    def _get_cached(self, key: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached data if available and valid."""
        cache_path = self._get_cache_path(key)

        if self._is_cache_valid(cache_path):
            try:
                data = load_json(cache_path)
                log.debug(f"Cache hit for {key}")
                return data
            except Exception as e:
                log.warning(f"Failed to load cache for {key}: {e}")
                return None

        return None

    def _set_cache(self, key: str, data: Dict[str, Any]):
        """Store data in cache."""
        cache_path = self._get_cache_path(key)
        try:
            save_json(data, cache_path)
            log.debug(f"Cached data for {key}")
        except Exception as e:
            log.warning(f"Failed to cache data for {key}: {e}")

    @abstractmethod
    def collect(self, *args, **kwargs) -> pd.DataFrame:
        """
        Collect data from the source.

        Must be implemented by subclasses.

        Returns:
            DataFrame with collected data
        """
        pass

    def close(self):
        """Close the HTTP session."""
        self.session.close()
