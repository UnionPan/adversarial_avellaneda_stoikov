"""
Kraken Data Fetcher for Cryptocurrency Historical Data

Fetch OHLCV data directly from Kraken exchange.

Advantages:
- NO API key required for public data
- NO geo-blocking (works globally)
- Real exchange data (not aggregated)
- Very reliable and well-documented
- High rate limits

Kraken API Docs: https://docs.kraken.com/rest/

author: Yunian Pan
email: yp1170@nyu.edu
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from typing import Optional, Dict, List
import time
import warnings


class KrakenFetcher:
    """
    Fetch cryptocurrency historical data from Kraken exchange.

    Public API (no authentication required):
    - Historical OHLCV data
    - Ticker information
    - Trading pairs

    Rate Limits:
    - Very generous for public endpoints
    - No geo-blocking

    Example:
        >>> fetcher = KrakenFetcher()
        >>> # Get Bitcoin data
        >>> btc_data = fetcher.get_ohlcv('XXBTZUSD', interval=1440, days=365)
        >>> print(btc_data.head())
        >>>
        >>> # Get multiple cryptocurrencies
        >>> cryptos = fetcher.get_multiple(['XXBTZUSD', 'XETHZUSD'], days=90)
    """

    BASE_URL = "https://api.kraken.com/0/public"

    # Popular trading pairs (Kraken symbol: description)
    POPULAR_PAIRS = {
        'XXBTZUSD': 'Bitcoin/USD',
        'XETHZUSD': 'Ethereum/USD',
        'XLTCZUSD': 'Litecoin/USD',
        'XXRPZUSD': 'Ripple/USD',
        'ADAUSD': 'Cardano/USD',
        'SOLUSD': 'Solana/USD',
        'DOTUSD': 'Polkadot/USD',
        'MATICUSD': 'Polygon/USD',
        'AVAXUSD': 'Avalanche/USD',
        'LINKUSD': 'Chainlink/USD',
        'ATOMUSD': 'Cosmos/USD',
        'UNIUSD': 'Uniswap/USD',
    }

    # Interval mapping (minutes)
    INTERVALS = {
        1: '1 minute',
        5: '5 minutes',
        15: '15 minutes',
        30: '30 minutes',
        60: '1 hour',
        240: '4 hours',
        1440: '1 day',
        10080: '1 week',
        21600: '15 days',
    }

    def __init__(
        self,
        rate_limit_delay: float = 0.2,
        verbose: bool = True,
    ):
        """
        Initialize Kraken fetcher.

        Args:
            rate_limit_delay: Delay between API calls in seconds (default 0.2s)
            verbose: Print progress messages
        """
        self.rate_limit_delay = rate_limit_delay
        self.verbose = verbose
        self.last_request_time = 0

    def _rate_limit(self):
        """Enforce rate limiting between API calls."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self.last_request_time = time.time()

    def _make_request(self, endpoint: str, params: Dict = None) -> Dict:
        """
        Make API request with error handling.

        Args:
            endpoint: API endpoint (e.g., 'OHLC', 'Ticker')
            params: Query parameters

        Returns:
            JSON response as dictionary

        Raises:
            requests.HTTPError: If request fails
        """
        self._rate_limit()

        url = f"{self.BASE_URL}/{endpoint}"

        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            # Check for Kraken API errors
            if data.get('error') and len(data['error']) > 0:
                raise ValueError(f"Kraken API error: {data['error']}")

            return data['result']

        except requests.exceptions.HTTPError as e:
            if response.status_code == 429:
                warnings.warn("Rate limit exceeded. Waiting 60 seconds...")
                time.sleep(60)
                return self._make_request(endpoint, params)  # Retry
            else:
                raise e

        except requests.exceptions.RequestException as e:
            raise Exception(f"API request failed: {e}")

    def get_ohlcv(
        self,
        pair: str,
        interval: int = 1440,
        days: Optional[int] = 365,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        since: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Get OHLCV (candlestick) data for a trading pair.

        Args:
            pair: Trading pair (e.g., 'XXBTZUSD', 'XETHZUSD')
                 Use Kraken symbols! Find with list_trading_pairs()
            interval: Candlestick interval in minutes (default 1440 = 1 day)
                     Options: 1, 5, 15, 30, 60, 240, 1440, 10080, 21600
            days: Number of days of history (ignored if start_date/since provided)
            start_date: Start date (optional, overrides days)
            end_date: End date (optional, defaults to now)
            since: Start timestamp in seconds (optional; overrides start_date/days)

        Returns:
            DataFrame with columns: Date, Open, High, Low, Close, Volume

        Example:
            >>> # Last 365 days of daily data
            >>> btc = fetcher.get_ohlcv('XXBTZUSD', interval=1440, days=365)
            >>>
            >>> # 4-hour candles for last 30 days
            >>> btc = fetcher.get_ohlcv('XXBTZUSD', interval=240, days=30)
            >>>
            >>> # Specific date range
            >>> btc = fetcher.get_ohlcv('XXBTZUSD', start_date=date(2018, 1, 1), end_date=date.today())
        """
        if self.verbose:
            interval_desc = self.INTERVALS.get(interval, f'{interval} minutes')
            print(f"Fetching {pair} OHLC data ({interval_desc} interval)...")

        # Validate interval
        if interval not in self.INTERVALS:
            raise ValueError(
                f"Invalid interval: {interval}. "
                f"Options: {', '.join(map(str, self.INTERVALS.keys()))}"
            )

        # Determine date range
        if end_date is None:
            end_dt = datetime.now()
        else:
            end_dt = datetime.combine(end_date, datetime.max.time())

        if since is not None:
            start_dt = datetime.fromtimestamp(since)
        elif start_date is not None:
            start_dt = datetime.combine(start_date, datetime.min.time())
            since = int(start_dt.timestamp())
        else:
            start_dt = end_dt - timedelta(days=days)
            since = int(start_dt.timestamp())

        # Fetch data with pagination using the `since` cursor
        endpoint = "OHLC"
        all_candles = []
        cursor = since

        while True:
            params = {
                'pair': pair,
                'interval': interval,
                'since': cursor,
            }

            result = self._make_request(endpoint, params)

            # Kraken returns: {pair_name: [[time, open, high, low, close, vwap, volume, count], ...], 'last': ts}
            pair_keys = [k for k in result.keys() if k != 'last']
            pair_key = pair_keys[0] if pair_keys else None
            if pair_key is None:
                break

            candles = result[pair_key]
            if not candles:
                break

            all_candles.extend(candles)

            # Advance cursor; Kraken's 'last' is the id of the last candle returned
            last_cursor = result.get('last', candles[-1][0])

            # Stop if we reached or passed the desired end date
            if last_cursor >= int(end_dt.timestamp()):
                break

            # Prevent infinite loops
            if last_cursor <= cursor:
                break

            cursor = last_cursor

        # Create DataFrame
        if not all_candles:
            raise ValueError(f"No data returned for {pair}")

        df = pd.DataFrame(all_candles, columns=[
            'time', 'Open', 'High', 'Low', 'Close', 'vwap', 'Volume', 'count'
        ])

        # Convert types
        df['Date'] = pd.to_datetime(df['time'], unit='s')
        df['Open'] = df['Open'].astype(float)
        df['High'] = df['High'].astype(float)
        df['Low'] = df['Low'].astype(float)
        df['Close'] = df['Close'].astype(float)
        df['Volume'] = df['Volume'].astype(float)

        # Select and reorder columns
        df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]

        # Filter to end date window (if provided)
        df = df[(df['Date'] >= start_dt) & (df['Date'] <= end_dt)]
        df = df.sort_values('Date').reset_index(drop=True)

        if self.verbose:
            print(f"  Retrieved {len(df)} candles")
            print(f"  Date range: {df['Date'].min()} to {df['Date'].max()}")

        return df

    def get_ticker(self, pairs: List[str]) -> Dict[str, Dict]:
        """
        Get current ticker information for trading pairs.

        Args:
            pairs: List of trading pairs (e.g., ['XXBTZUSD', 'XETHZUSD'])

        Returns:
            Dictionary mapping pair to ticker info

        Example:
            >>> tickers = fetcher.get_ticker(['XXBTZUSD', 'XETHZUSD'])
            >>> btc_price = float(tickers['XXBTZUSD']['c'][0])
            >>> print(f"BTC: ${btc_price:,.2f}")
        """
        endpoint = "Ticker"
        params = {'pair': ','.join(pairs)}

        result = self._make_request(endpoint, params)
        return result

    def get_current_price(self, pairs: List[str]) -> Dict[str, float]:
        """
        Get current price for trading pairs.

        Args:
            pairs: List of trading pairs

        Returns:
            Dictionary mapping pair to current price

        Example:
            >>> prices = fetcher.get_current_price(['XXBTZUSD', 'XETHZUSD'])
            >>> print(f"BTC: ${prices['XXBTZUSD']:,.2f}")
        """
        tickers = self.get_ticker(pairs)

        prices = {}
        for pair_name, ticker_data in tickers.items():
            # 'c' is the last trade closed array [price, lot volume]
            prices[pair_name] = float(ticker_data['c'][0])

        return prices

    def get_multiple(
        self,
        pairs: List[str],
        interval: int = 1440,
        days: int = 365,
    ) -> Dict[str, pd.DataFrame]:
        """
        Get OHLCV data for multiple trading pairs.

        Args:
            pairs: List of trading pairs
            interval: Candlestick interval in minutes
            days: Number of days of history

        Returns:
            Dictionary mapping pair to DataFrame

        Example:
            >>> cryptos = fetcher.get_multiple(
            ...     ['XXBTZUSD', 'XETHZUSD', 'SOLUSD'],
            ...     interval=1440,
            ...     days=180
            ... )
            >>> btc_data = cryptos['XXBTZUSD']
            >>> eth_data = cryptos['XETHZUSD']
        """
        results = {}

        if self.verbose:
            print(f"Fetching data for {len(pairs)} trading pairs...")

        for pair in pairs:
            try:
                df = self.get_ohlcv(
                    pair,
                    interval=interval,
                    days=days,
                )
                results[pair] = df
            except Exception as e:
                warnings.warn(f"Failed to fetch {pair}: {e}")
                results[pair] = None

        return results

    def list_trading_pairs(self) -> pd.DataFrame:
        """
        List all available trading pairs.

        Returns:
            DataFrame with trading pair information

        Example:
            >>> pairs = fetcher.list_trading_pairs()
            >>> # Filter for USD pairs
            >>> usd_pairs = pairs[pairs['quote'] == 'USD']
            >>> print(usd_pairs)
        """
        endpoint = "AssetPairs"

        result = self._make_request(endpoint)

        # Parse pairs
        pairs_list = []
        for pair_name, pair_info in result.items():
            # Skip .d pairs (dark pool)
            if pair_name.endswith('.d'):
                continue

            pairs_list.append({
                'pair': pair_name,
                'altname': pair_info.get('altname', ''),
                'base': pair_info.get('base', ''),
                'quote': pair_info.get('quote', ''),
                'status': pair_info.get('status', ''),
            })

        return pd.DataFrame(pairs_list)

    def search_pair(self, query: str) -> List[str]:
        """
        Search for trading pairs by base or quote asset.

        Args:
            query: Search term (e.g., 'BTC', 'ETH', 'USD')

        Returns:
            List of matching pair names

        Example:
            >>> # Find all BTC pairs
            >>> btc_pairs = fetcher.search_pair('BTC')
            >>> print(btc_pairs)
        """
        query_upper = query.upper()

        pairs_df = self.list_trading_pairs()

        # Search in pair name, base, and quote
        matches = pairs_df[
            pairs_df['pair'].str.contains(query_upper) |
            pairs_df['altname'].str.contains(query_upper) |
            pairs_df['base'].str.contains(query_upper) |
            pairs_df['quote'].str.contains(query_upper)
        ]['pair'].tolist()

        return matches

    def list_popular_pairs(self) -> pd.DataFrame:
        """
        List popular cryptocurrency trading pairs.

        Returns:
            DataFrame with pair and description

        Example:
            >>> pairs = fetcher.list_popular_pairs()
            >>> print(pairs)
        """
        pairs = []
        for pair, description in self.POPULAR_PAIRS.items():
            pairs.append({
                'pair': pair,
                'description': description,
            })

        return pd.DataFrame(pairs)


def download_bitcoin(
    interval: int = 1440,
    days: int = 365,
) -> pd.DataFrame:
    """
    Convenience function to download Bitcoin data from Kraken.

    Args:
        interval: Data interval in minutes (default 1440 = 1 day)
        days: Number of days (default 365)

    Returns:
        DataFrame with Bitcoin OHLCV data

    Example:
        >>> from calibration.data import download_bitcoin
        >>> btc = download_bitcoin(interval=1440, days=730)  # 2 years
    """
    fetcher = KrakenFetcher()
    return fetcher.get_ohlcv('XXBTZUSD', interval=interval, days=days)


def download_crypto_basket(
    pairs: List[str] = None,
    interval: int = 1440,
    days: int = 365,
) -> Dict[str, pd.DataFrame]:
    """
    Download data for a basket of cryptocurrencies from Kraken.

    Args:
        pairs: List of pairs (default: BTC, ETH, SOL)
        interval: Data interval in minutes
        days: Number of days

    Returns:
        Dictionary mapping pair to DataFrame

    Example:
        >>> basket = download_crypto_basket(
        ...     pairs=['XXBTZUSD', 'XETHZUSD', 'SOLUSD'],
        ...     days=180
        ... )
        >>> btc = basket['XXBTZUSD']
    """
    if pairs is None:
        pairs = ['XXBTZUSD', 'XETHZUSD', 'SOLUSD']

    fetcher = KrakenFetcher()
    return fetcher.get_multiple(pairs, interval=interval, days=days)
