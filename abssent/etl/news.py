"""
News collector module for gathering financial news headlines.

Collects news from free sources like Yahoo Finance RSS feeds with
deduplication and ticker association.
"""

import hashlib
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Union
from urllib.parse import urljoin, urlparse

import feedparser
import pandas as pd
import requests
from bs4 import BeautifulSoup

from ..utils.config import get_config
from ..utils.io import save_parquet, load_parquet
from ..utils.logging import get_logger, LoggingMixin, log_performance

logger = get_logger(__name__)


class NewsCollector(LoggingMixin):
    """
    Financial news collector from free sources.
    
    Features:
    - RSS feed parsing from Yahoo Finance and other sources
    - Deduplication by URL hash
    - Ticker association
    - Rate limiting
    - Data persistence
    """
    
    def __init__(self, rate_limit: float = 1.0):
        """
        Initialize the news collector.
        
        Args:
            rate_limit: Rate limit between requests (seconds)
        """
        self.config = get_config()
        self.rate_limit = rate_limit
        self._last_request_time = 0.0
        self._seen_urls: Set[str] = set()
        
        # Free news sources
        self.news_sources = {
            'yahoo_finance_general': 'https://feeds.finance.yahoo.com/rss/2.0/headline',
            'marketwatch_breaking': 'https://feeds.marketwatch.com/marketwatch/marketpulse/',
            'seeking_alpha_market': 'https://seekingalpha.com/market_currents.xml',
        }
    
    def _enforce_rate_limit(self) -> None:
        """Enforce rate limiting between requests."""
        current_time = time.time()
        time_since_last = current_time - self._last_request_time
        
        if time_since_last < self.rate_limit:
            sleep_time = self.rate_limit - time_since_last
            time.sleep(sleep_time)
        
        self._last_request_time = time.time()
    
    def _get_url_hash(self, url: str) -> str:
        """Generate a hash for URL deduplication."""
        return hashlib.md5(url.encode()).hexdigest()
    
    @log_performance
    def collect_rss_feed(self, feed_url: str, source_name: str) -> List[Dict]:
        """
        Collect news from an RSS feed.
        
        Args:
            feed_url: RSS feed URL
            source_name: Name identifier for the source
            
        Returns:
            List of news article dictionaries
        """
        self._enforce_rate_limit()
        
        try:
            # Parse RSS feed
            feed = feedparser.parse(feed_url)
            
            if feed.bozo:
                self.logger.warning(f"RSS feed {feed_url} has parsing issues")
            
            articles = []
            
            for entry in feed.entries:
                # Extract article information
                article = {
                    'title': getattr(entry, 'title', ''),
                    'summary': getattr(entry, 'summary', ''),
                    'link': getattr(entry, 'link', ''),
                    'published': getattr(entry, 'published', ''),
                    'source': source_name,
                    'collected_at': datetime.now().isoformat()
                }
                
                # Parse publication date
                if hasattr(entry, 'published_parsed') and entry.published_parsed:
                    article['published_date'] = datetime(*entry.published_parsed[:6])
                else:
                    # Fallback to current time if no date available
                    article['published_date'] = datetime.now()
                
                # Generate URL hash for deduplication
                url_hash = self._get_url_hash(article['link'])
                article['url_hash'] = url_hash
                
                # Skip if we've seen this URL before
                if url_hash in self._seen_urls:
                    continue
                
                self._seen_urls.add(url_hash)
                articles.append(article)
            
            self.logger.info(f"Collected {len(articles)} new articles from {source_name}")
            return articles
            
        except Exception as e:
            self.logger.error(f"Failed to collect from {feed_url}: {e}")
            return []
    
    def extract_ticker_mentions(self, text: str) -> List[str]:
        """
        Extract ticker mentions from text.
        
        Args:
            text: Text to search for ticker mentions
            
        Returns:
            List of found ticker symbols
        """
        import re
        
        # Simple pattern for ticker symbols (1-5 uppercase letters)
        # This is a basic implementation - could be enhanced with a ticker database
        ticker_pattern = r'\b[A-Z]{1,5}\b'
        
        # Find all potential tickers
        potential_tickers = re.findall(ticker_pattern, text)
        
        # Filter against known tickers from config
        valid_tickers = [
            ticker for ticker in potential_tickers 
            if ticker in self.config.TARGET_TICKERS
        ]
        
        return list(set(valid_tickers))  # Remove duplicates
    
    @log_performance
    def collect_ticker_news(self, ticker: str, days_back: int = 7) -> List[Dict]:
        """
        Collect news specifically mentioning a ticker.
        
        Args:
            ticker: Stock ticker symbol
            days_back: Number of days back to search
            
        Returns:
            List of relevant news articles
        """
        # Yahoo Finance ticker-specific RSS (when available)
        ticker_rss_url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}"
        
        articles = self.collect_rss_feed(ticker_rss_url, f"yahoo_{ticker}")
        
        # Filter by date
        cutoff_date = datetime.now() - timedelta(days=days_back)
        recent_articles = [
            article for article in articles
            if article['published_date'] >= cutoff_date
        ]
        
        return recent_articles
    
    @log_performance
    def collect_general_news(self, max_articles_per_source: int = 50) -> List[Dict]:
        """
        Collect general financial news from all sources.
        
        Args:
            max_articles_per_source: Maximum articles to collect per source
            
        Returns:
            List of all collected articles
        """
        all_articles = []
        
        for source_name, feed_url in self.news_sources.items():
            try:
                articles = self.collect_rss_feed(feed_url, source_name)
                
                # Limit articles per source
                if len(articles) > max_articles_per_source:
                    articles = articles[:max_articles_per_source]
                
                all_articles.extend(articles)
                
            except Exception as e:
                self.logger.error(f"Failed to collect from {source_name}: {e}")
                continue
        
        return all_articles
    
    def associate_tickers(self, articles: List[Dict]) -> List[Dict]:
        """
        Associate articles with relevant tickers.
        
        Args:
            articles: List of article dictionaries
            
        Returns:
            Articles with ticker associations added
        """
        for article in articles:
            # Combine title and summary for ticker extraction
            text = f"{article.get('title', '')} {article.get('summary', '')}"
            
            # Extract ticker mentions
            tickers = self.extract_ticker_mentions(text)
            article['tickers'] = tickers
            article['ticker_count'] = len(tickers)
        
        return articles
    
    @log_performance
    def save_news_data(self, articles: List[Dict], filename: Optional[str] = None) -> Path:
        """
        Save collected news data to Parquet format.
        
        Args:
            articles: List of article dictionaries
            filename: Optional filename (default: auto-generated)
            
        Returns:
            Path to saved file
        """
        if not articles:
            raise ValueError("No articles to save")
        
        # Convert to DataFrame
        df = pd.DataFrame(articles)
        
        # Generate filename if not provided
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"news_articles_{timestamp}.parquet"
        
        file_path = self.config.NEWS_DIR / filename
        save_parquet(df, file_path)
        
        self.logger.info(f"Saved {len(articles)} articles to {file_path}")
        return file_path
    
    def load_news_data(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """
        Load news data from Parquet file.
        
        Args:
            file_path: Path to news data file
            
        Returns:
            DataFrame with news data
        """
        return load_parquet(file_path, engine="pandas")


# Convenience functions
@log_performance
def collect_news_headlines(
    tickers: Optional[List[str]] = None,
    days_back: int = 7,
    include_general: bool = True,
    save_data: bool = True
) -> List[Dict]:
    """
    Collect news headlines for specified tickers.
    
    Args:
        tickers: List of ticker symbols (default: config tickers)
        days_back: Number of days back to collect
        include_general: Whether to include general financial news
        save_data: Whether to save collected data
        
    Returns:
        List of collected articles
    """
    collector = NewsCollector()
    all_articles = []
    
    # Collect ticker-specific news
    if tickers:
        for ticker in tickers:
            ticker_articles = collector.collect_ticker_news(ticker, days_back)
            all_articles.extend(ticker_articles)
    
    # Collect general financial news
    if include_general:
        general_articles = collector.collect_general_news()
        all_articles.extend(general_articles)
    
    # Associate tickers with all articles
    all_articles = collector.associate_tickers(all_articles)
    
    # Filter by date if specified
    if days_back > 0:
        cutoff_date = datetime.now() - timedelta(days=days_back)
        all_articles = [
            article for article in all_articles
            if article['published_date'] >= cutoff_date
        ]
    
    # Save data if requested
    if save_data and all_articles:
        collector.save_news_data(all_articles)
    
    logger.info(f"Collected {len(all_articles)} total articles")
    return all_articles


def filter_news_by_sentiment_keywords(
    articles: List[Dict],
    positive_keywords: List[str] = None,
    negative_keywords: List[str] = None
) -> Dict[str, List[Dict]]:
    """
    Filter news articles by sentiment-indicating keywords.
    
    Args:
        articles: List of article dictionaries
        positive_keywords: List of positive sentiment keywords
        negative_keywords: List of negative sentiment keywords
        
    Returns:
        Dictionary with 'positive', 'negative', and 'neutral' article lists
    """
    if positive_keywords is None:
        positive_keywords = [
            'gains', 'up', 'rises', 'surge', 'rally', 'bullish', 'positive',
            'growth', 'beat', 'exceeds', 'outperforms', 'upgrade', 'buy'
        ]
    
    if negative_keywords is None:
        negative_keywords = [
            'falls', 'down', 'drops', 'decline', 'crash', 'bearish', 'negative',
            'loss', 'miss', 'disappoints', 'underperforms', 'downgrade', 'sell'
        ]
    
    categorized = {'positive': [], 'negative': [], 'neutral': []}
    
    for article in articles:
        text = f"{article.get('title', '')} {article.get('summary', '')}".lower()
        
        positive_count = sum(1 for keyword in positive_keywords if keyword in text)
        negative_count = sum(1 for keyword in negative_keywords if keyword in text)
        
        if positive_count > negative_count:
            categorized['positive'].append(article)
        elif negative_count > positive_count:
            categorized['negative'].append(article)
        else:
            categorized['neutral'].append(article)
    
    return categorized 