"""
FMP (Financial Modeling Prep) API Client

Handles data fetching from FMP API with rate limiting and error handling.
"""

import time
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
import requests
import pandas as pd
from collections import deque


class RateLimiter:
    """Token bucket rate limiter for API calls."""

    def __init__(self, max_calls: int, period: int):
        """
        Initialize rate limiter.

        Args:
            max_calls: Maximum number of calls allowed in period
            period: Time period in seconds
        """
        self.max_calls = max_calls
        self.period = period
        self.calls = deque()

    def wait_if_needed(self):
        """Wait if rate limit would be exceeded."""
        now = time.time()

        # Remove calls outside the current period
        while self.calls and self.calls[0] < now - self.period:
            self.calls.popleft()

        # If at limit, wait
        if len(self.calls) >= self.max_calls:
            sleep_time = self.period - (now - self.calls[0])
            if sleep_time > 0:
                print(f"Rate limit reached. Waiting {sleep_time:.2f} seconds...")
                time.sleep(sleep_time)
                # Clean up old calls after waiting
                now = time.time()
                while self.calls and self.calls[0] < now - self.period:
                    self.calls.popleft()

        # Record this call
        self.calls.append(time.time())


class FMPClient:
    """Client for fetching financial data from FMP API."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://financialmodelingprep.com/api/v3",
        rate_limit_calls: int = 250,
        rate_limit_period: int = 60
    ):
        """
        Initialize FMP client.

        Args:
            api_key: FMP API key
            base_url: Base URL for API
            rate_limit_calls: Maximum API calls per period
            rate_limit_period: Rate limit period in seconds
        """
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.rate_limiter = RateLimiter(rate_limit_calls, rate_limit_period)
        self.session = requests.Session()

    def _make_request(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Make an API request with rate limiting.

        Args:
            endpoint: API endpoint
            params: Query parameters

        Returns:
            JSON response data

        Raises:
            requests.RequestException: If request fails
        """
        self.rate_limiter.wait_if_needed()

        if params is None:
            params = {}
        params['apikey'] = self.api_key

        url = f"{self.base_url}/{endpoint.lstrip('/')}"

        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            raise Exception(f"API request failed: {str(e)}")

    def get_historical_prices(
        self,
        symbol: str,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fetch historical daily prices for a symbol.

        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            from_date: Start date in YYYY-MM-DD format
            to_date: End date in YYYY-MM-DD format

        Returns:
            DataFrame with OHLCV data
        """
        # If no dates specified, get last 5 years
        if to_date is None:
            to_date = datetime.now().strftime('%Y-%m-%d')
        if from_date is None:
            from_date = (datetime.now() - timedelta(days=365*5)).strftime('%Y-%m-%d')

        endpoint = f"historical-price-full/{symbol}"
        params = {
            'from': from_date,
            'to': to_date
        }

        data = self._make_request(endpoint, params)

        if 'historical' not in data:
            raise ValueError(f"No data found for symbol {symbol}")

        df = pd.DataFrame(data['historical'])

        # Clean and format data
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)

        # Rename columns to standard format
        df = df.rename(columns={
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume',
            'adjClose': 'Adj Close'
        })

        # Select relevant columns
        columns = ['date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        df = df[[col for col in columns if col in df.columns]]

        return df

    def get_quote(self, symbol: str) -> Dict[str, Any]:
        """
        Get current quote for a symbol.

        Args:
            symbol: Stock symbol

        Returns:
            Quote data
        """
        endpoint = f"quote/{symbol}"
        data = self._make_request(endpoint)
        return data[0] if data else {}

    def get_company_profile(self, symbol: str) -> Dict[str, Any]:
        """
        Get company profile information.

        Args:
            symbol: Stock symbol

        Returns:
            Company profile data
        """
        endpoint = f"profile/{symbol}"
        data = self._make_request(endpoint)
        return data[0] if data else {}

    def search_symbols(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search for symbols.

        Args:
            query: Search query
            limit: Maximum results

        Returns:
            List of matching symbols
        """
        endpoint = "search"
        params = {'query': query, 'limit': limit}
        data = self._make_request(endpoint, params)
        return data if isinstance(data, list) else []
