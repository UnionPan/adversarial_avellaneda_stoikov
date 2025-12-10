"""
yfinance data fetcher for options and spot data.

author: Yunian Pan
email: yp1170@nyu.edu
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import date, datetime
from typing import Optional, List
import logging

from .data_provider import DataProvider, OptionChain, OptionQuote, MarketData


logger = logging.getLogger(__name__)


class YFinanceFetcher(DataProvider):
    """
    Data provider using yfinance (Yahoo Finance).
    
    Example:
        fetcher = YFinanceFetcher()
        chain = fetcher.get_option_chain('SPY')
        filtered = chain.filter(min_volume=100, min_open_interest=50)
        smile = filtered.get_slice(filtered.get_expiries()[0])
    """
    
    def __init__(
        self,
        risk_free_rate: float = 0.05,
        dividend_yield: float = 0.0,
    ):
        """
        Initialize fetcher.
        
        Args:
            risk_free_rate: Default risk-free rate for calculations
            dividend_yield: Default dividend yield
        """
        self.risk_free_rate = risk_free_rate
        self.dividend_yield = dividend_yield
        self._cache = {}
    
    def get_spot(self, ticker: str) -> float:
        """
        Get current spot price.
        
        Args:
            ticker: Stock/ETF ticker symbol
            
        Returns:
            Current price
        """
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Try different price fields
        for field in ['regularMarketPrice', 'currentPrice', 'previousClose']:
            if field in info and info[field] is not None:
                return float(info[field])
        
        # Fallback: get from history
        hist = stock.history(period='1d')
        if not hist.empty:
            return float(hist['Close'].iloc[-1])
            
        raise ValueError(f"Could not get spot price for {ticker}")
    
    def get_history(
        self,
        ticker: str,
        start: date,
        end: date,
    ) -> pd.DataFrame:
        """
        Get historical OHLCV data.
        
        Args:
            ticker: Stock/ETF ticker
            start: Start date
            end: End date
            
        Returns:
            DataFrame with OHLCV columns
        """
        stock = yf.Ticker(ticker)
        df = stock.history(start=start, end=end)
        
        if df.empty:
            raise ValueError(f"No history data for {ticker}")
            
        return df
    
    def get_option_chain(
        self,
        ticker: str,
        reference_date: Optional[date] = None,
        expiry: Optional[str] = None,
    ) -> OptionChain:
        """
        Get option chain for a ticker.
        
        Args:
            ticker: Stock/ETF ticker
            reference_date: Reference date (defaults to today)
            expiry: Specific expiry date string (YYYY-MM-DD) or None for all
            
        Returns:
            OptionChain object with all available options
        """
        if reference_date is None:
            reference_date = date.today()
            
        stock = yf.Ticker(ticker)
        spot = self.get_spot(ticker)
        
        # Get available expiries
        try:
            available_expiries = stock.options
        except Exception as e:
            raise ValueError(f"Could not get options for {ticker}: {e}")
        
        if not available_expiries:
            raise ValueError(f"No options available for {ticker}")
        
        options = []
        
        # Process expiries
        expiries_to_fetch = [expiry] if expiry else available_expiries
        
        for exp_str in expiries_to_fetch:
            if exp_str not in available_expiries:
                logger.warning(f"Expiry {exp_str} not available for {ticker}")
                continue
                
            try:
                chain = stock.option_chain(exp_str)
                expiry_date = datetime.strptime(exp_str, '%Y-%m-%d').date()
                
                # Process calls
                for _, row in chain.calls.iterrows():
                    opt = self._row_to_option(row, expiry_date, 'call')
                    if opt:
                        options.append(opt)
                
                # Process puts
                for _, row in chain.puts.iterrows():
                    opt = self._row_to_option(row, expiry_date, 'put')
                    if opt:
                        options.append(opt)
                        
            except Exception as e:
                logger.warning(f"Error fetching {exp_str}: {e}")
                continue
        
        return OptionChain(
            underlying=ticker,
            spot_price=spot,
            reference_date=reference_date,
            risk_free_rate=self.risk_free_rate,
            dividend_yield=self.dividend_yield,
            options=options,
        )
    
    def _row_to_option(
        self,
        row: pd.Series,
        expiry: date,
        option_type: str,
    ) -> Optional[OptionQuote]:
        """Convert yfinance row to OptionQuote."""
        try:
            bid = float(row.get('bid', 0) or 0)
            ask = float(row.get('ask', 0) or 0)
            
            # Skip invalid quotes
            if bid <= 0 or ask <= 0 or ask < bid:
                mid = float(row.get('lastPrice', 0) or 0)
                if mid <= 0:
                    return None
            else:
                mid = (bid + ask) / 2
            
            # Get implied vol (Yahoo sometimes provides this)
            iv = row.get('impliedVolatility')
            if iv is not None and not np.isnan(iv):
                iv = float(iv)
            else:
                iv = None
            
            return OptionQuote(
                strike=float(row['strike']),
                expiry=expiry,
                option_type=option_type,
                bid=bid,
                ask=ask,
                mid=mid,
                last=float(row.get('lastPrice', mid) or mid),
                volume=int(row.get('volume', 0) or 0),
                open_interest=int(row.get('openInterest', 0) or 0),
                implied_volatility=iv,
            )
        except Exception as e:
            logger.debug(f"Error parsing option row: {e}")
            return None
    
    def get_option_expiries(self, ticker: str) -> List[str]:
        """Get available option expiry dates."""
        stock = yf.Ticker(ticker)
        return list(stock.options)


# Convenience function
def fetch_spy_data(days: int = 252) -> MarketData:
    """
    Quick helper to fetch SPY data for testing.
    
    Args:
        days: Number of days of history
        
    Returns:
        MarketData for SPY
    """
    fetcher = YFinanceFetcher()
    return fetcher.get_market_data('SPY', history_days=days)
