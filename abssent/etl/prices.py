"""
Price data loader module for downloading stock prices.

Uses yfinance for free stock price data with error handling, data validation,
and support for bulk downloads with progress tracking.
"""

import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd
import yfinance as yf
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from ..utils.config import get_config
from ..utils.io import save_parquet, load_parquet, ensure_dir
from ..utils.logging import get_logger, LoggingMixin, log_performance

logger = get_logger(__name__)


class PriceLoader(LoggingMixin):
    """
    Stock price data loader using yfinance.
    
    Features:
    - Bulk download with progress tracking
    - Data validation and cleaning
    - Automatic retry on failures
    - Forward-fill missing data option
    - Save to Parquet format for efficiency
    """
    
    def __init__(self, rate_limit: float = 0.1, max_retries: int = 3):
        """
        Initialize the price loader.
        
        Args:
            rate_limit: Rate limit between requests (seconds)
            max_retries: Maximum retry attempts for failed downloads
        """
        self.config = get_config()
        self.rate_limit = rate_limit
        self.max_retries = max_retries
        self._last_request_time = 0.0
    
    def _enforce_rate_limit(self) -> None:
        """Enforce rate limiting between requests."""
        current_time = time.time()
        time_since_last = current_time - self._last_request_time
        
        if time_since_last < self.rate_limit:
            sleep_time = self.rate_limit - time_since_last
            time.sleep(sleep_time)
        
        self._last_request_time = time.time()
    
    @log_performance
    def download_ticker_data(
        self,
        ticker: str,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        interval: str = "1d"
    ) -> Optional[pd.DataFrame]:
        """
        Download price data for a single ticker.
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date for data
            end_date: End date for data
            interval: Data interval (1d, 1wk, 1mo, etc.)
            
        Returns:
            DataFrame with price data or None if failed
        """
        self._enforce_rate_limit()
        
        for attempt in range(self.max_retries):
            try:
                # Create yfinance ticker object
                stock = yf.Ticker(ticker)
                
                # Download data
                data = stock.history(
                    start=start_date,
                    end=end_date,
                    interval=interval,
                    auto_adjust=True,
                    prepost=False,
                    threads=False
                )
                
                if data.empty:
                    self.logger.warning(f"No data found for {ticker}")
                    return None
                
                # Clean and format data
                data = self._clean_price_data(data, ticker)
                
                self.logger.debug(f"Downloaded {len(data)} days of data for {ticker}")
                return data
                
            except Exception as e:
                self.logger.warning(f"Attempt {attempt + 1} failed for {ticker}: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    self.logger.error(f"Failed to download data for {ticker} after {self.max_retries} attempts")
                    return None
    
    def _clean_price_data(self, data: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """
        Clean and validate price data.
        
        Args:
            data: Raw price data from yfinance
            ticker: Ticker symbol for labeling
            
        Returns:
            Cleaned DataFrame
        """
        # Reset index to make Date a column
        data = data.reset_index()
        
        # Add ticker column
        data['ticker'] = ticker
        
        # Rename columns to lowercase
        data.columns = [col.lower().replace(' ', '_') for col in data.columns]
        
        # Ensure we have the required columns
        required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in data.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Remove rows with missing prices
        data = data.dropna(subset=['open', 'high', 'low', 'close'])
        
        # Basic data validation
        # Price should be positive
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            data = data[data[col] > 0]
        
        # High should be >= low
        data = data[data['high'] >= data['low']]
        
        # Volume should be non-negative
        data = data[data['volume'] >= 0]
        
        # Calculate returns
        data['return_1d'] = data['close'].pct_change()
        data['return_1d_fwd'] = data['return_1d'].shift(-1)  # Forward return for prediction
        
        # Calculate volatility (rolling 20-day)
        data['volatility_20d'] = data['return_1d'].rolling(20).std() * (252 ** 0.5)
        
        # Sort by date
        data = data.sort_values('date').reset_index(drop=True)
        
        return data
    
    @log_performance
    def download_multiple_tickers(
        self,
        tickers: List[str],
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        save_individual: bool = True,
        save_combined: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Download price data for multiple tickers.
        
        Args:
            tickers: List of ticker symbols
            start_date: Start date for data
            end_date: End date for data
            save_individual: Whether to save individual ticker files
            save_combined: Whether to save combined file
            
        Returns:
            Dictionary mapping tickers to their price DataFrames
        """
        results = {}
        failed_tickers = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
        ) as progress:
            
            download_task = progress.add_task(
                f"Downloading price data for {len(tickers)} tickers...",
                total=len(tickers)
            )
            
            for ticker in tickers:
                try:
                    data = self.download_ticker_data(ticker, start_date, end_date)
                    
                    if data is not None and not data.empty:
                        results[ticker] = data
                        
                        # Save individual file if requested
                        if save_individual:
                            file_path = (
                                self.config.PRICES_DIR / 
                                f"{ticker}_{pd.to_datetime(start_date).strftime('%Y%m%d')}_"
                                f"{pd.to_datetime(end_date).strftime('%Y%m%d')}.parquet"
                            )
                            save_parquet(data, file_path)
                    else:
                        failed_tickers.append(ticker)
                        
                except Exception as e:
                    self.logger.error(f"Error downloading {ticker}: {e}")
                    failed_tickers.append(ticker)
                
                progress.advance(download_task)
        
        # Save combined file if requested
        if save_combined and results:
            combined_data = pd.concat(results.values(), ignore_index=True)
            combined_path = (
                self.config.PRICES_DIR / 
                f"combined_prices_{pd.to_datetime(start_date).strftime('%Y%m%d')}_"
                f"{pd.to_datetime(end_date).strftime('%Y%m%d')}.parquet"
            )
            save_parquet(combined_data, combined_path)
            self.logger.info(f"Saved combined price data to {combined_path}")
        
        # Report results
        success_count = len(results)
        total_count = len(tickers)
        self.logger.info(
            f"Successfully downloaded {success_count}/{total_count} tickers"
        )
        
        if failed_tickers:
            self.logger.warning(f"Failed to download: {', '.join(failed_tickers)}")
        
        return results
    
    def get_price_data(
        self,
        ticker: str,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        use_cache: bool = True
    ) -> Optional[pd.DataFrame]:
        """
        Get price data for a ticker, using cache if available.
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date for data
            end_date: End date for data
            use_cache: Whether to use cached data if available
            
        Returns:
            DataFrame with price data
        """
        # Check for cached file
        cache_path = (
            self.config.PRICES_DIR / 
            f"{ticker}_{pd.to_datetime(start_date).strftime('%Y%m%d')}_"
            f"{pd.to_datetime(end_date).strftime('%Y%m%d')}.parquet"
        )
        
        if use_cache and cache_path.exists():
            self.logger.debug(f"Loading cached data for {ticker}")
            return load_parquet(cache_path, engine="pandas")
        
        # Download fresh data
        return self.download_ticker_data(ticker, start_date, end_date)
    
    @log_performance
    def update_price_data(
        self,
        tickers: Optional[List[str]] = None,
        days_back: int = 5
    ) -> Dict[str, pd.DataFrame]:
        """
        Update price data with recent data.
        
        Args:
            tickers: List of tickers to update (default: config tickers)
            days_back: Number of days back to update
            
        Returns:
            Dictionary of updated price data
        """
        if tickers is None:
            tickers = self.config.TARGET_TICKERS
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        self.logger.info(f"Updating price data for {len(tickers)} tickers from {start_date.date()}")
        
        return self.download_multiple_tickers(
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            save_individual=True,
            save_combined=True
        )


# Convenience functions
@log_performance
def download_price_data(
    tickers: Union[str, List[str]],
    start_date: Union[str, datetime],
    end_date: Union[str, datetime],
    save_dir: Optional[Path] = None
) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Download price data for one or more tickers.
    
    Args:
        tickers: Single ticker or list of tickers
        start_date: Start date for data
        end_date: End date for data
        save_dir: Directory to save data (default: config.PRICES_DIR)
        
    Returns:
        DataFrame for single ticker or dict of DataFrames for multiple tickers
    """
    # Ensure tickers is a list
    if isinstance(tickers, str):
        tickers = [tickers]
        single_ticker = True
    else:
        single_ticker = False
    
    # Create loader
    loader = PriceLoader()
    
    # Update save directory if specified
    if save_dir:
        loader.config.PRICES_DIR = Path(save_dir)
        ensure_dir(loader.config.PRICES_DIR)
    
    # Download data
    results = loader.download_multiple_tickers(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date
    )
    
    # Return format based on input
    if single_ticker:
        ticker = tickers[0]
        return results.get(ticker)
    else:
        return results


def calculate_returns(
    prices: pd.DataFrame,
    price_col: str = "close",
    periods: List[int] = [1, 5, 20]
) -> pd.DataFrame:
    """
    Calculate returns for various periods.
    
    Args:
        prices: Price DataFrame with date and price columns
        price_col: Name of price column to use
        periods: List of periods (in days) to calculate returns for
        
    Returns:
        DataFrame with additional return columns
    """
    result = prices.copy()
    
    for period in periods:
        return_col = f"return_{period}d"
        fwd_return_col = f"return_{period}d_fwd"
        
        # Historical returns
        result[return_col] = result[price_col].pct_change(periods=period)
        
        # Forward returns (for prediction targets)
        result[fwd_return_col] = result[return_col].shift(-period)
    
    return result


def calculate_technical_indicators(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate common technical indicators.
    
    Args:
        prices: Price DataFrame with OHLCV data
        
    Returns:
        DataFrame with additional indicator columns
    """
    result = prices.copy()
    
    # Moving averages
    for window in [5, 10, 20, 50]:
        result[f"sma_{window}"] = result["close"].rolling(window).mean()
        result[f"close_vs_sma_{window}"] = result["close"] / result[f"sma_{window}"] - 1
    
    # Bollinger Bands
    result["bb_middle"] = result["close"].rolling(20).mean()
    bb_std = result["close"].rolling(20).std()
    result["bb_upper"] = result["bb_middle"] + (bb_std * 2)
    result["bb_lower"] = result["bb_middle"] - (bb_std * 2)
    result["bb_position"] = (result["close"] - result["bb_lower"]) / (result["bb_upper"] - result["bb_lower"])
    
    # RSI (Relative Strength Index)
    delta = result["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    result["rsi"] = 100 - (100 / (1 + rs))
    
    # Volume indicators
    result["volume_sma_20"] = result["volume"].rolling(20).mean()
    result["volume_ratio"] = result["volume"] / result["volume_sma_20"]
    
    return result 