"""
Sentiment analysis module using FinBERT for financial text analysis.

Implements financial sentiment analysis with batch processing, caching,
and support for both CPU and GPU inference.
"""

import gc
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline
)

from ..utils.config import get_config
from ..utils.io import save_parquet, load_parquet, save_pickle, load_pickle
from ..utils.logging import get_logger, LoggingMixin, log_performance

logger = get_logger(__name__)


class SentimentAnalyzer(LoggingMixin):
    """
    Financial sentiment analyzer using FinBERT.
    
    Features:
    - Pre-trained FinBERT model for financial sentiment
    - Batch processing for efficiency
    - CPU/GPU support
    - Result caching
    - Confidence scoring
    """
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        batch_size: Optional[int] = None,
        use_gpu: bool = True,
        cache_results: bool = True
    ):
        """
        Initialize the sentiment analyzer.
        
        Args:
            model_name: Model name (default: from config)
            batch_size: Batch size for inference (default: from config)
            use_gpu: Whether to use GPU if available
            cache_results: Whether to cache results for repeated inference
        """
        self.config = get_config()
        self.model_name = model_name or self.config.SENTIMENT_MODEL
        self.batch_size = batch_size or self.config.BATCH_SIZE
        self.cache_results = cache_results
        
        # Device setup
        self.device = torch.device(
            "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        )
        self.logger.info(f"Using device: {self.device}")
        
        # Model components
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        self._cache = {}
        
        # Load model
        self._load_model()
    
    def _load_model(self) -> None:
        """Load the FinBERT model and tokenizer."""
        try:
            self.logger.info(f"Loading FinBERT model: {self.model_name}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Load model
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name
            )
            self.model.to(self.device)
            self.model.eval()
            
            # Create pipeline for easier inference
            self.pipeline = pipeline(
                "sentiment-analysis",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device.type == "cuda" else -1,
                return_all_scores=True
            )
            
            self.logger.info("FinBERT model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load FinBERT model: {e}")
            raise
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        import hashlib
        return hashlib.md5(text.encode()).hexdigest()
    
    def _clean_text(self, text: str) -> str:
        """Clean and prepare text for sentiment analysis."""
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        # Truncate if too long (FinBERT has 512 token limit)
        max_length = self.config.MAX_SEQUENCE_LENGTH
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        
        if len(tokens) > max_length - 2:  # Account for [CLS] and [SEP]
            tokens = tokens[:max_length - 2]
            text = self.tokenizer.decode(tokens, skip_special_tokens=True)
        
        return text
    
    @log_performance
    def analyze_text(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment of a single text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment scores and label
        """
        # Check cache
        if self.cache_results:
            cache_key = self._get_cache_key(text)
            if cache_key in self._cache:
                return self._cache[cache_key]
        
        # Clean text
        clean_text = self._clean_text(text)
        
        if not clean_text.strip():
            return {
                "label": "neutral",
                "positive": 0.33,
                "negative": 0.33,
                "neutral": 0.34,
                "compound": 0.0
            }
        
        try:
            # Get predictions
            results = self.pipeline(clean_text)
            
            # Parse results (FinBERT typically returns positive, negative, neutral)
            scores = {result["label"].lower(): result["score"] for result in results[0]}
            
            # Determine primary label
            primary_label = max(scores.keys(), key=lambda k: scores[k])
            
            # Calculate compound score (positive - negative)
            compound = scores.get("positive", 0) - scores.get("negative", 0)
            
            result = {
                "label": primary_label,
                "compound": compound,
                **scores
            }
            
            # Cache result
            if self.cache_results:
                self._cache[cache_key] = result
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error analyzing sentiment for text: {e}")
            return {
                "label": "neutral",
                "positive": 0.33,
                "negative": 0.33,
                "neutral": 0.34,
                "compound": 0.0
            }
    
    @log_performance
    def analyze_batch(self, texts: List[str]) -> List[Dict[str, float]]:
        """
        Analyze sentiment of multiple texts in batches.
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            List of sentiment dictionaries
        """
        if not texts:
            return []
        
        results = []
        
        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            batch_results = []
            
            for text in batch_texts:
                result = self.analyze_text(text)
                batch_results.append(result)
            
            results.extend(batch_results)
            
            # Memory cleanup
            if i % (self.batch_size * 5) == 0:
                gc.collect()
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
        
        return results
    
    @log_performance
    def analyze_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str,
        output_prefix: str = "sentiment_"
    ) -> pd.DataFrame:
        """
        Analyze sentiment for a DataFrame column.
        
        Args:
            df: Input DataFrame
            text_column: Name of column containing text
            output_prefix: Prefix for output columns
            
        Returns:
            DataFrame with added sentiment columns
        """
        # Extract texts
        texts = df[text_column].fillna("").astype(str).tolist()
        
        # Analyze sentiments
        sentiment_results = self.analyze_batch(texts)
        
        # Convert to DataFrame
        sentiment_df = pd.DataFrame(sentiment_results)
        
        # Add prefix to column names
        sentiment_df.columns = [f"{output_prefix}{col}" for col in sentiment_df.columns]
        
        # Combine with original DataFrame
        result_df = pd.concat([df.reset_index(drop=True), sentiment_df], axis=1)
        
        return result_df
    
    def save_cache(self, cache_file: Optional[Path] = None) -> None:
        """Save sentiment cache to disk."""
        if not self.cache_results or not self._cache:
            return
        
        if cache_file is None:
            cache_file = self.config.MODELS_DIR / "sentiment_cache.pkl"
        
        save_pickle(self._cache, cache_file)
        self.logger.info(f"Saved sentiment cache with {len(self._cache)} entries")
    
    def load_cache(self, cache_file: Optional[Path] = None) -> None:
        """Load sentiment cache from disk."""
        if not self.cache_results:
            return
        
        if cache_file is None:
            cache_file = self.config.MODELS_DIR / "sentiment_cache.pkl"
        
        if cache_file.exists():
            self._cache = load_pickle(cache_file)
            self.logger.info(f"Loaded sentiment cache with {len(self._cache)} entries")
    
    def clear_cache(self) -> None:
        """Clear the sentiment cache."""
        self._cache.clear()
        self.logger.info("Sentiment cache cleared")


# Convenience functions
@log_performance
def analyze_sentiment(
    texts: Union[str, List[str]],
    model_name: Optional[str] = None,
    batch_size: Optional[int] = None,
    use_gpu: bool = True
) -> Union[Dict[str, float], List[Dict[str, float]]]:
    """
    Analyze sentiment for text(s) using FinBERT.
    
    Args:
        texts: Single text or list of texts
        model_name: Model name (default: from config)
        batch_size: Batch size for processing
        use_gpu: Whether to use GPU if available
        
    Returns:
        Sentiment results (single dict or list of dicts)
    """
    # Initialize analyzer
    analyzer = SentimentAnalyzer(
        model_name=model_name,
        batch_size=batch_size,
        use_gpu=use_gpu
    )
    
    # Handle single text vs list
    if isinstance(texts, str):
        return analyzer.analyze_text(texts)
    else:
        return analyzer.analyze_batch(texts)


def aggregate_sentiment_scores(
    sentiment_scores: List[Dict[str, float]],
    weights: Optional[List[float]] = None
) -> Dict[str, float]:
    """
    Aggregate multiple sentiment scores into a single score.
    
    Args:
        sentiment_scores: List of sentiment dictionaries
        weights: Optional weights for aggregation
        
    Returns:
        Aggregated sentiment scores
    """
    if not sentiment_scores:
        return {
            "label": "neutral",
            "positive": 0.33,
            "negative": 0.33,
            "neutral": 0.34,
            "compound": 0.0
        }
    
    if weights is None:
        weights = [1.0] * len(sentiment_scores)
    
    if len(weights) != len(sentiment_scores):
        raise ValueError("Number of weights must match number of sentiment scores")
    
    # Normalize weights
    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]
    
    # Aggregate scores
    aggregated = {}
    
    # Get all possible keys
    all_keys = set()
    for score in sentiment_scores:
        all_keys.update(score.keys())
    
    # Calculate weighted averages
    for key in all_keys:
        if key == "label":
            continue  # Handle label separately
        
        weighted_sum = sum(
            score.get(key, 0) * weight
            for score, weight in zip(sentiment_scores, weights)
        )
        aggregated[key] = weighted_sum
    
    # Determine aggregated label
    if "positive" in aggregated and "negative" in aggregated and "neutral" in aggregated:
        max_score = max(
            aggregated.get("positive", 0),
            aggregated.get("negative", 0),
            aggregated.get("neutral", 0)
        )
        
        if aggregated.get("positive", 0) == max_score:
            aggregated["label"] = "positive"
        elif aggregated.get("negative", 0) == max_score:
            aggregated["label"] = "negative"
        else:
            aggregated["label"] = "neutral"
    else:
        aggregated["label"] = "neutral"
    
    return aggregated


def sentiment_time_series(
    df: pd.DataFrame,
    date_column: str,
    sentiment_column: str,
    aggregation: str = "mean",
    window: Optional[int] = None
) -> pd.DataFrame:
    """
    Create a time series of sentiment scores.
    
    Args:
        df: DataFrame with sentiment data
        date_column: Name of date column
        sentiment_column: Name of sentiment score column
        aggregation: Aggregation method ('mean', 'median', 'sum')
        window: Optional rolling window size
        
    Returns:
        Time series DataFrame
    """
    # Group by date and aggregate
    if aggregation == "mean":
        ts_df = df.groupby(date_column)[sentiment_column].mean().reset_index()
    elif aggregation == "median":
        ts_df = df.groupby(date_column)[sentiment_column].median().reset_index()
    elif aggregation == "sum":
        ts_df = df.groupby(date_column)[sentiment_column].sum().reset_index()
    else:
        raise ValueError(f"Unsupported aggregation: {aggregation}")
    
    # Sort by date
    ts_df = ts_df.sort_values(date_column).reset_index(drop=True)
    
    # Apply rolling window if specified
    if window:
        ts_df[f"{sentiment_column}_rolling"] = (
            ts_df[sentiment_column].rolling(window).mean()
        )
    
    return ts_df 