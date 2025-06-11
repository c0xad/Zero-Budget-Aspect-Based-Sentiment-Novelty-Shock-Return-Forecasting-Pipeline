"""
Novelty detection module for financial text analysis.

Implements novelty-shock computation using rolling window embeddings
and cosine similarity for detecting unusual content in filings.
"""

import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import duckdb
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from ..utils.config import get_config
from ..utils.io import save_parquet, load_parquet, save_pickle, load_pickle
from ..utils.logging import get_logger, LoggingMixin, log_performance

logger = get_logger(__name__)


class NoveltyDetector(LoggingMixin):
    """
    Financial text novelty detector using rolling window embeddings.
    
    Features:
    - Rolling window embedding computation
    - Cosine similarity-based novelty scoring
    - Firm-specific novelty detection
    - Persistent storage of historical embeddings
    - Configurable window sizes and thresholds
    """
    
    def __init__(
        self,
        embedding_model: Optional[str] = None,
        window_days: Optional[int] = None,
        min_history_days: int = 5,
        cache_embeddings: bool = True,
        use_database: bool = True
    ):
        """
        Initialize the novelty detector.
        
        Args:
            embedding_model: Sentence transformer model name
            window_days: Rolling window size in days
            min_history_days: Minimum history required for novelty computation
            cache_embeddings: Whether to cache computed embeddings
            use_database: Whether to use DuckDB for persistence
        """
        self.config = get_config()
        self.embedding_model_name = embedding_model or self.config.EMBEDDING_MODEL
        self.window_days = window_days or self.config.NOVELTY_WINDOW_DAYS
        self.min_history_days = min_history_days
        self.cache_embeddings = cache_embeddings
        self.use_database = use_database
        
        # Initialize components
        self.embedding_model = None
        self._embedding_cache = {}
        self._firm_history = {}  # In-memory cache for recent embeddings
        self.db_connection = None
        
        # Load model and setup storage
        self._load_embedding_model()
        if self.use_database:
            self._setup_database()
    
    def _load_embedding_model(self) -> None:
        """Load the sentence embedding model."""
        try:
            self.logger.info(f"Loading embedding model: {self.embedding_model_name}")
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            self.logger.info("Embedding model loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load embedding model: {e}")
            raise
    
    def _setup_database(self) -> None:
        """Setup DuckDB for storing embeddings and novelty scores."""
        try:
            self.db_connection = duckdb.connect(self.config.DUCKDB_PATH)
            
            # Create tables for embeddings and novelty scores
            self.db_connection.execute("""
                CREATE TABLE IF NOT EXISTS firm_embeddings (
                    ticker VARCHAR,
                    date DATE,
                    embedding FLOAT[],
                    text_length INTEGER,
                    num_paragraphs INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (ticker, date)
                )
            """)
            
            self.db_connection.execute("""
                CREATE TABLE IF NOT EXISTS novelty_scores (
                    ticker VARCHAR,
                    date DATE,
                    novelty_score FLOAT,
                    rolling_mean_embedding FLOAT[],
                    window_size INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (ticker, date)
                )
            """)
            
            self.logger.info("Database setup completed")
        except Exception as e:
            self.logger.error(f"Failed to setup database: {e}")
            self.use_database = False
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text with caching."""
        if self.cache_embeddings and text in self._embedding_cache:
            return self._embedding_cache[text]
        
        embedding = self.embedding_model.encode(text)
        
        if self.cache_embeddings:
            self._embedding_cache[text] = embedding
        
        return embedding
    
    def _aggregate_text_embeddings(
        self, 
        texts: List[str],
        aggregation: str = "mean"
    ) -> np.ndarray:
        """
        Aggregate multiple text embeddings into a single embedding.
        
        Args:
            texts: List of text strings
            aggregation: Aggregation method ('mean', 'weighted_mean', 'max_pool')
            
        Returns:
            Aggregated embedding
        """
        if not texts:
            # Return zero embedding if no texts
            sample_embedding = self.embedding_model.encode("")
            return np.zeros_like(sample_embedding)
        
        # Get embeddings for all texts
        embeddings = []
        for text in texts:
            if text and text.strip():
                embedding = self._get_embedding(text)
                embeddings.append(embedding)
        
        if not embeddings:
            # Return zero embedding if no valid texts
            sample_embedding = self.embedding_model.encode("")
            return np.zeros_like(sample_embedding)
        
        embeddings = np.array(embeddings)
        
        if aggregation == "mean":
            return np.mean(embeddings, axis=0)
        elif aggregation == "weighted_mean":
            # Weight by text length
            weights = [len(text) for text in texts if text and text.strip()]
            if len(weights) != len(embeddings):
                weights = [1.0] * len(embeddings)
            weights = np.array(weights) / sum(weights)
            return np.average(embeddings, axis=0, weights=weights)
        elif aggregation == "max_pool":
            return np.max(embeddings, axis=0)
        else:
            raise ValueError(f"Unsupported aggregation method: {aggregation}")
    
    def _get_historical_embeddings(
        self,
        ticker: str,
        date: datetime,
        window_days: Optional[int] = None
    ) -> List[Tuple[datetime, np.ndarray]]:
        """
        Get historical embeddings for a firm within the specified window.
        
        Args:
            ticker: Stock ticker symbol
            date: Reference date
            window_days: Window size in days (defaults to self.window_days)
            
        Returns:
            List of (date, embedding) tuples
        """
        if window_days is None:
            window_days = self.window_days
        
        start_date = date - timedelta(days=window_days)
        
        # First check in-memory cache
        if ticker in self._firm_history:
            cached_embeddings = [
                (hist_date, embedding)
                for hist_date, embedding in self._firm_history[ticker]
                if start_date <= hist_date < date
            ]
            if len(cached_embeddings) >= self.min_history_days:
                return cached_embeddings
        
        # Query database if available
        if self.use_database and self.db_connection:
            try:
                result = self.db_connection.execute("""
                    SELECT date, embedding
                    FROM firm_embeddings
                    WHERE ticker = ? AND date >= ? AND date < ?
                    ORDER BY date
                """, [ticker, start_date.date(), date.date()]).fetchall()
                
                historical_embeddings = []
                for row in result:
                    hist_date = datetime.strptime(str(row[0]), "%Y-%m-%d")
                    embedding = np.array(row[1])
                    historical_embeddings.append((hist_date, embedding))
                
                return historical_embeddings
                
            except Exception as e:
                self.logger.warning(f"Failed to query historical embeddings: {e}")
        
        return []
    
    def _store_embedding(
        self,
        ticker: str,
        date: datetime,
        embedding: np.ndarray,
        text_length: int = 0,
        num_paragraphs: int = 0
    ) -> None:
        """Store embedding for future use."""
        # Store in memory cache
        if ticker not in self._firm_history:
            self._firm_history[ticker] = []
        
        self._firm_history[ticker].append((date, embedding))
        
        # Keep only recent embeddings in memory (last 60 days)
        cutoff_date = date - timedelta(days=60)
        self._firm_history[ticker] = [
            (hist_date, emb) for hist_date, emb in self._firm_history[ticker]
            if hist_date >= cutoff_date
        ]
        
        # Store in database
        if self.use_database and self.db_connection:
            try:
                self.db_connection.execute("""
                    INSERT OR REPLACE INTO firm_embeddings 
                    (ticker, date, embedding, text_length, num_paragraphs)
                    VALUES (?, ?, ?, ?, ?)
                """, [ticker, date.date(), embedding.tolist(), text_length, num_paragraphs])
                
            except Exception as e:
                self.logger.warning(f"Failed to store embedding in database: {e}")
    
    def _store_novelty_score(
        self,
        ticker: str,
        date: datetime,
        novelty_score: float,
        rolling_mean_embedding: np.ndarray,
        window_size: int
    ) -> None:
        """Store novelty score for future reference."""
        if self.use_database and self.db_connection:
            try:
                self.db_connection.execute("""
                    INSERT OR REPLACE INTO novelty_scores
                    (ticker, date, novelty_score, rolling_mean_embedding, window_size)
                    VALUES (?, ?, ?, ?, ?)
                """, [
                    ticker, 
                    date.date(), 
                    novelty_score, 
                    rolling_mean_embedding.tolist(), 
                    window_size
                ])
                
            except Exception as e:
                self.logger.warning(f"Failed to store novelty score in database: {e}")
    
    @log_performance
    def compute_novelty(
        self,
        texts: List[str],
        ticker: str,
        date: datetime,
        aggregation: str = "mean"
    ) -> float:
        """
        Compute novelty score for texts on a given date.
        
        Args:
            texts: List of text paragraphs
            ticker: Stock ticker symbol
            date: Date of the texts
            aggregation: Method to aggregate text embeddings
            
        Returns:
            Novelty score (0-1, where 1 is most novel)
        """
        if not texts or not any(text.strip() for text in texts):
            return 0.0
        
        # Aggregate current texts into single embedding
        current_embedding = self._aggregate_text_embeddings(texts, aggregation)
        
        # Store current embedding
        text_length = sum(len(text) for text in texts)
        self._store_embedding(ticker, date, current_embedding, text_length, len(texts))
        
        # Get historical embeddings
        historical_embeddings = self._get_historical_embeddings(ticker, date)
        
        if len(historical_embeddings) < self.min_history_days:
            # Not enough history - return moderate novelty
            novelty_score = 0.5
            self.logger.debug(
                f"Insufficient history for {ticker} on {date}: "
                f"{len(historical_embeddings)} < {self.min_history_days}"
            )
        else:
            # Compute rolling mean of historical embeddings
            hist_embeddings = np.array([emb for _, emb in historical_embeddings])
            rolling_mean_embedding = np.mean(hist_embeddings, axis=0)
            
            # Compute cosine similarity
            similarity = cosine_similarity(
                current_embedding.reshape(1, -1),
                rolling_mean_embedding.reshape(1, -1)
            )[0, 0]
            
            # Convert similarity to novelty (1 - similarity)
            novelty_score = 1.0 - max(0.0, similarity)
            
            # Store novelty score
            self._store_novelty_score(
                ticker, date, novelty_score, rolling_mean_embedding, len(historical_embeddings)
            )
        
        return novelty_score
    
    @log_performance
    def compute_novelty_batch(
        self,
        data: List[Dict],
        text_column: str = "text",
        ticker_column: str = "ticker",
        date_column: str = "date",
        aggregation: str = "mean"
    ) -> List[float]:
        """
        Compute novelty scores for multiple firm-date observations.
        
        Args:
            data: List of dictionaries with text, ticker, and date info
            text_column: Column name for text data
            ticker_column: Column name for ticker
            date_column: Column name for date
            aggregation: Method to aggregate text embeddings
            
        Returns:
            List of novelty scores
        """
        novelty_scores = []
        
        for item in data:
            texts = item.get(text_column, [])
            if isinstance(texts, str):
                texts = [texts]
            
            ticker = item.get(ticker_column, "")
            date = item.get(date_column)
            
            if isinstance(date, str):
                date = datetime.strptime(date, "%Y-%m-%d")
            
            novelty_score = self.compute_novelty(texts, ticker, date, aggregation)
            novelty_scores.append(novelty_score)
        
        return novelty_scores
    
    @log_performance
    def compute_novelty_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str = "text",
        ticker_column: str = "ticker", 
        date_column: str = "date",
        group_by_firm_date: bool = True,
        aggregation: str = "mean"
    ) -> pd.DataFrame:
        """
        Compute novelty scores for a DataFrame.
        
        Args:
            df: Input DataFrame
            text_column: Column name for text data
            ticker_column: Column name for ticker
            date_column: Column name for date
            group_by_firm_date: Whether to group texts by firm-date
            aggregation: Method to aggregate text embeddings
            
        Returns:
            DataFrame with added novelty_score column
        """
        result_df = df.copy()
        
        if group_by_firm_date:
            # Group by ticker and date, aggregate texts
            grouped_data = []
            
            for (ticker, date), group in df.groupby([ticker_column, date_column]):
                texts = group[text_column].tolist()
                
                # Convert date if needed
                if isinstance(date, str):
                    date = datetime.strptime(date, "%Y-%m-%d")
                
                novelty_score = self.compute_novelty(texts, ticker, date, aggregation)
                
                # Add novelty score to all rows in this group
                group_indices = group.index
                for idx in group_indices:
                    grouped_data.append((idx, novelty_score))
            
            # Add novelty scores to result DataFrame
            for idx, novelty_score in grouped_data:
                result_df.loc[idx, 'novelty_score'] = novelty_score
        
        else:
            # Compute novelty for each row individually
            novelty_scores = []
            
            for _, row in df.iterrows():
                text = row[text_column]
                ticker = row[ticker_column]
                date = row[date_column]
                
                if isinstance(date, str):
                    date = datetime.strptime(date, "%Y-%m-%d")
                
                novelty_score = self.compute_novelty([text], ticker, date, aggregation)
                novelty_scores.append(novelty_score)
            
            result_df['novelty_score'] = novelty_scores
        
        return result_df
    
    def get_novelty_statistics(
        self,
        ticker: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict:
        """Get statistics about computed novelty scores."""
        if not self.use_database or not self.db_connection:
            return {"error": "Database not available"}
        
        try:
            query = "SELECT * FROM novelty_scores WHERE 1=1"
            params = []
            
            if ticker:
                query += " AND ticker = ?"
                params.append(ticker)
            
            if start_date:
                query += " AND date >= ?"
                params.append(start_date.date())
            
            if end_date:
                query += " AND date <= ?"
                params.append(end_date.date())
            
            df = self.db_connection.execute(query, params).df()
            
            if df.empty:
                return {"count": 0}
            
            stats = {
                "count": len(df),
                "mean_novelty": df['novelty_score'].mean(),
                "std_novelty": df['novelty_score'].std(),
                "min_novelty": df['novelty_score'].min(),
                "max_novelty": df['novelty_score'].max(),
                "unique_tickers": df['ticker'].nunique(),
                "date_range": {
                    "start": df['date'].min(),
                    "end": df['date'].max()
                }
            }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to get novelty statistics: {e}")
            return {"error": str(e)}
    
    def save_cache(self, cache_file: Optional[Path] = None) -> None:
        """Save embedding cache to disk."""
        if not self.cache_embeddings or not self._embedding_cache:
            return
        
        if cache_file is None:
            cache_file = self.config.MODELS_DIR / "novelty_embedding_cache.pkl"
        
        save_pickle(self._embedding_cache, cache_file)
        self.logger.info(f"Saved embedding cache with {len(self._embedding_cache)} entries")
    
    def load_cache(self, cache_file: Optional[Path] = None) -> None:
        """Load embedding cache from disk."""
        if not self.cache_embeddings:
            return
        
        if cache_file is None:
            cache_file = self.config.MODELS_DIR / "novelty_embedding_cache.pkl"
        
        if cache_file.exists():
            self._embedding_cache = load_pickle(cache_file)
            self.logger.info(f"Loaded embedding cache with {len(self._embedding_cache)} entries")
    
    def close(self) -> None:
        """Close database connection."""
        if self.db_connection:
            self.db_connection.close()


# Convenience functions
@log_performance
def calculate_novelty(
    texts: Union[str, List[str]],
    ticker: str,
    date: Union[str, datetime],
    embedding_model: Optional[str] = None,
    window_days: Optional[int] = None
) -> float:
    """
    Calculate novelty score for text(s) using default detector.
    
    Args:
        texts: Single text or list of texts
        ticker: Stock ticker symbol
        date: Date of the texts
        embedding_model: Embedding model name
        window_days: Rolling window size in days
        
    Returns:
        Novelty score
    """
    # Initialize detector
    detector = NoveltyDetector(
        embedding_model=embedding_model,
        window_days=window_days
    )
    
    # Handle date conversion
    if isinstance(date, str):
        date = datetime.strptime(date, "%Y-%m-%d")
    
    # Handle single text vs list
    if isinstance(texts, str):
        texts = [texts]
    
    return detector.compute_novelty(texts, ticker, date)


def novelty_time_series(
    df: pd.DataFrame,
    ticker: str,
    date_column: str = "date",
    novelty_column: str = "novelty_score",
    window: Optional[int] = None
) -> pd.DataFrame:
    """
    Create a time series of novelty scores for a specific ticker.
    
    Args:
        df: DataFrame with novelty data
        ticker: Stock ticker to filter
        date_column: Name of date column
        novelty_column: Name of novelty score column
        window: Optional rolling window for smoothing
        
    Returns:
        Time series DataFrame
    """
    # Filter for specific ticker
    ticker_df = df[df['ticker'] == ticker].copy()
    
    # Sort by date
    ticker_df = ticker_df.sort_values(date_column).reset_index(drop=True)
    
    # Apply rolling window if specified
    if window:
        ticker_df[f"{novelty_column}_rolling"] = (
            ticker_df[novelty_column].rolling(window).mean()
        )
    
    return ticker_df[[date_column, novelty_column] + 
                    ([f"{novelty_column}_rolling"] if window else [])]


def cross_sectional_novelty_analysis(
    df: pd.DataFrame,
    date: Union[str, datetime],
    novelty_column: str = "novelty_score",
    ticker_column: str = "ticker"
) -> pd.DataFrame:
    """
    Analyze novelty scores across firms for a specific date.
    
    Args:
        df: DataFrame with novelty data
        date: Specific date to analyze
        novelty_column: Name of novelty score column
        ticker_column: Name of ticker column
        
    Returns:
        Cross-sectional analysis DataFrame
    """
    # Filter for specific date
    if isinstance(date, str):
        date_df = df[df['date'] == date].copy()
    else:
        date_df = df[df['date'] == date.strftime("%Y-%m-%d")].copy()
    
    if date_df.empty:
        return pd.DataFrame()
    
    # Compute statistics
    stats = {
        'mean_novelty': date_df[novelty_column].mean(),
        'median_novelty': date_df[novelty_column].median(),
        'std_novelty': date_df[novelty_column].std(),
        'min_novelty': date_df[novelty_column].min(),
        'max_novelty': date_df[novelty_column].max(),
        'count': len(date_df)
    }
    
    # Add percentile ranks
    date_df['novelty_percentile'] = (
        date_df[novelty_column].rank(pct=True) * 100
    )
    
    # Sort by novelty score (descending)
    result_df = date_df.sort_values(novelty_column, ascending=False)
    
    return result_df 