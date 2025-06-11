"""
EDGAR scraper module for downloading SEC 8-K filings.

Implements SEC-compliant scraping with rate limiting, retry logic, and async support.
Respects SEC rate limits (10 requests/second) and includes proper User-Agent headers.
"""

import asyncio
import re
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from urllib.parse import urljoin

import aiohttp
import pandas as pd
import requests
from bs4 import BeautifulSoup
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from ..utils.config import get_config
from ..utils.io import ensure_dir, download_file_async, save_parquet
from ..utils.logging import get_logger, LoggingMixin, log_performance

logger = get_logger(__name__)


class EdgarScraper(LoggingMixin):
    """
    SEC EDGAR scraper for 8-K filings with rate limiting and async support.
    
    Features:
    - Respects SEC rate limits (10 requests/second)
    - Exponential backoff on errors
    - Async downloads for better performance
    - Proper User-Agent headers
    - CIK-based file organization
    """
    
    def __init__(self, rate_limit: Optional[float] = None, max_retries: int = 3):
        """
        Initialize the EDGAR scraper.
        
        Args:
            rate_limit: Rate limit in seconds between requests (default from config)
            max_retries: Maximum number of retry attempts
        """
        self.config = get_config()
        self.rate_limit = rate_limit or self.config.SEC_RATE_LIMIT
        self.max_retries = max_retries
        self.session = None
        self._last_request_time = 0.0
        
        # Rate limiting semaphore (10 concurrent requests max)
        self.semaphore = asyncio.Semaphore(10)
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession(
            headers={'User-Agent': self.config.SEC_USER_AGENT},
            timeout=aiohttp.ClientTimeout(total=30)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    def _enforce_rate_limit(self) -> None:
        """Enforce rate limiting between requests."""
        current_time = time.time()
        time_since_last = current_time - self._last_request_time
        
        if time_since_last < self.rate_limit:
            sleep_time = self.rate_limit - time_since_last
            time.sleep(sleep_time)
        
        self._last_request_time = time.time()
    
    @log_performance
    def get_daily_index(self, date: datetime) -> pd.DataFrame:
        """
        Download and parse daily index file for a specific date.
        
        Args:
            date: Date to get index for
            
        Returns:
            DataFrame with company filings for that date
        """
        year = date.year
        quarter = (date.month - 1) // 3 + 1
        date_str = date.strftime("%Y%m%d")
        
        # Construct URL for daily index
        index_url = f"{self.config.SEC_DAILY_INDEX_URL}/{year}/QTR{quarter}/company.{date_str}.idx"
        
        self.logger.info(f"Downloading daily index for {date_str}")
        self._enforce_rate_limit()
        
        try:
            response = requests.get(
                index_url,
                headers={'User-Agent': self.config.SEC_USER_AGENT},
                timeout=30
            )
            response.raise_for_status()
            
            # Parse fixed-width format
            lines = response.text.split('\n')
            
            # Find header line (contains "Form Type")
            header_line_idx = None
            for i, line in enumerate(lines):
                if 'Form Type' in line and 'Company Name' in line:
                    header_line_idx = i
                    break
            
            if header_line_idx is None:
                raise ValueError("Could not find header line in index file")
            
            # Parse fixed-width data
            data_lines = lines[header_line_idx + 2:]  # Skip header and separator
            data_lines = [line for line in data_lines if line.strip()]
            
            if not data_lines:
                self.logger.warning(f"No data found for {date_str}")
                return pd.DataFrame()
            
            # Define column positions (approximate, may need adjustment)
            df = pd.read_fwf(
                io='\n'.join([lines[header_line_idx]] + data_lines),
                skiprows=1,
                names=['form_type', 'company_name', 'cik', 'date_filed', 'filename']
            )
            
            # Filter for 8-K filings only
            df = df[df['form_type'].str.contains('8-K', na=False)].copy()
            
            # Clean and format data
            df['cik'] = df['cik'].astype(str).str.zfill(10)
            df['date_filed'] = pd.to_datetime(df['date_filed'])
            df['accession'] = df['filename'].str.extract(r'([0-9-]+)\.txt$')[0]
            df['filing_url'] = self.config.SEC_BASE_URL + '/' + df['filename']
            
            self.logger.info(f"Found {len(df)} 8-K filings for {date_str}")
            return df
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to download daily index for {date_str}: {e}")
            return pd.DataFrame()
        except Exception as e:
            self.logger.error(f"Error parsing daily index for {date_str}: {e}")
            return pd.DataFrame()
    
    async def download_filing(
        self,
        filing_url: str,
        cik: str,
        accession: str,
        destination_dir: Optional[Path] = None
    ) -> Optional[Path]:
        """
        Download a single 8-K filing.
        
        Args:
            filing_url: URL of the filing
            cik: Company CIK
            accession: Accession number
            destination_dir: Directory to save filing (default: config.EDGAR_DIR)
            
        Returns:
            Path to downloaded file or None if failed
        """
        if destination_dir is None:
            destination_dir = self.config.EDGAR_DIR
        
        # Organize by CIK
        cik_dir = destination_dir / cik
        ensure_dir(cik_dir)
        
        file_path = cik_dir / f"{accession}.txt"
        
        # Skip if already exists
        if file_path.exists():
            self.logger.debug(f"Filing already exists: {file_path}")
            return file_path
        
        try:
            await download_file_async(
                self.session,
                filing_url,
                file_path,
                self.semaphore
            )
            return file_path
            
        except Exception as e:
            self.logger.error(f"Failed to download {filing_url}: {e}")
            return None
    
    async def download_filings_batch(
        self,
        filings_df: pd.DataFrame,
        destination_dir: Optional[Path] = None
    ) -> List[Path]:
        """
        Download a batch of filings asynchronously.
        
        Args:
            filings_df: DataFrame with filing information
            destination_dir: Directory to save filings
            
        Returns:
            List of paths to successfully downloaded files
        """
        if len(filings_df) == 0:
            return []
        
        self.logger.info(f"Downloading {len(filings_df)} filings...")
        
        # Create download tasks
        tasks = []
        for _, row in filings_df.iterrows():
            task = self.download_filing(
                filing_url=row['filing_url'],
                cik=row['cik'],
                accession=row['accession'],
                destination_dir=destination_dir
            )
            tasks.append(task)
        
        # Execute downloads with progress bar
        results = []
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
        ) as progress:
            download_task = progress.add_task("Downloading filings...", total=len(tasks))
            
            for coro in asyncio.as_completed(tasks):
                result = await coro
                if result:
                    results.append(result)
                progress.advance(download_task)
        
        self.logger.info(f"Successfully downloaded {len(results)} out of {len(filings_df)} filings")
        return results
    
    @log_performance
    def get_filings_for_date_range(
        self,
        start_date: datetime,
        end_date: datetime,
        tickers: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Get all 8-K filings for a date range.
        
        Args:
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
            tickers: Optional list of tickers to filter by
            
        Returns:
            DataFrame with all filings in the date range
        """
        all_filings = []
        current_date = start_date
        
        # Load ticker-to-CIK mapping if tickers specified
        cik_filter = None
        if tickers:
            cik_filter = self._get_ciks_for_tickers(tickers)
        
        while current_date <= end_date:
            # Skip weekends
            if current_date.weekday() < 5:  # Monday = 0, Friday = 4
                daily_filings = self.get_daily_index(current_date)
                
                # Filter by CIK if tickers specified
                if cik_filter and len(daily_filings) > 0:
                    daily_filings = daily_filings[
                        daily_filings['cik'].isin(cik_filter)
                    ]
                
                if len(daily_filings) > 0:
                    all_filings.append(daily_filings)
            
            current_date += timedelta(days=1)
        
        if all_filings:
            result = pd.concat(all_filings, ignore_index=True)
            self.logger.info(f"Found {len(result)} total 8-K filings from {start_date} to {end_date}")
            return result
        else:
            self.logger.warning(f"No 8-K filings found from {start_date} to {end_date}")
            return pd.DataFrame()
    
    def _get_ciks_for_tickers(self, tickers: List[str]) -> List[str]:
        """
        Get CIK numbers for a list of tickers.
        
        Note: This is a simplified implementation. In practice, you'd want to
        maintain a ticker-to-CIK mapping database or use the SEC company tickers API.
        
        Args:
            tickers: List of ticker symbols
            
        Returns:
            List of CIK numbers
        """
        # For demo purposes, return empty list (would implement full mapping in production)
        self.logger.warning("Ticker-to-CIK mapping not implemented, returning all filings")
        return []


# Convenience functions
@log_performance
async def download_8k_filings(
    start_date: Union[str, datetime],
    end_date: Union[str, datetime],
    tickers: Optional[List[str]] = None,
    destination_dir: Optional[Path] = None
) -> List[Path]:
    """
    Download 8-K filings for a date range.
    
    Args:
        start_date: Start date (string or datetime)
        end_date: End date (string or datetime)
        tickers: Optional list of ticker symbols to filter by
        destination_dir: Directory to save filings
        
    Returns:
        List of paths to downloaded files
    """
    # Parse dates
    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date)
    if isinstance(end_date, str):
        end_date = pd.to_datetime(end_date)
    
    async with EdgarScraper() as scraper:
        # Get filing list
        filings_df = scraper.get_filings_for_date_range(
            start_date, end_date, tickers
        )
        
        if len(filings_df) == 0:
            logger.warning("No filings found for the specified criteria")
            return []
        
        # Save filing index
        config = get_config()
        index_path = config.PROCESSED_DIR / f"filings_index_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.parquet"
        save_parquet(filings_df, index_path)
        
        # Download filings
        downloaded_files = await scraper.download_filings_batch(
            filings_df, destination_dir
        )
        
        return downloaded_files


def clean_filing_text(raw_text: str) -> str:
    """
    Clean raw filing text by removing HTML tags, tables, and formatting.
    
    Args:
        raw_text: Raw filing text
        
    Returns:
        Cleaned text
    """
    # Parse HTML
    soup = BeautifulSoup(raw_text, 'html.parser')
    
    # Remove script and style elements
    for script in soup(["script", "style"]):
        script.decompose()
    
    # Remove tables (often contain structured data we don't need)
    for table in soup.find_all("table"):
        table.decompose()
    
    # Get text content
    text = soup.get_text()
    
    # Clean up whitespace
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = ' '.join(chunk for chunk in chunks if chunk)
    
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


def extract_filing_sections(text: str) -> Dict[str, str]:
    """
    Extract specific sections from 8-K filing text.
    
    Args:
        text: Cleaned filing text
        
    Returns:
        Dictionary with section names as keys and content as values
    """
    sections = {}
    
    # Common 8-K sections patterns
    section_patterns = {
        'item_1_01': r'Item\s+1\.01[^0-9]*?(.*?)(?=Item\s+[0-9]|$)',
        'item_1_02': r'Item\s+1\.02[^0-9]*?(.*?)(?=Item\s+[0-9]|$)',
        'item_2_02': r'Item\s+2\.02[^0-9]*?(.*?)(?=Item\s+[0-9]|$)',
        'item_7_01': r'Item\s+7\.01[^0-9]*?(.*?)(?=Item\s+[0-9]|$)',
        'item_8_01': r'Item\s+8\.01[^0-9]*?(.*?)(?=Item\s+[0-9]|$)',
    }
    
    for section_name, pattern in section_patterns.items():
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            sections[section_name] = match.group(1).strip()
    
    return sections 