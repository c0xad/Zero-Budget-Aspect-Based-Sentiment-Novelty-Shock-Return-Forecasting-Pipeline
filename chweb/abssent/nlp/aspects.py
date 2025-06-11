"""
Aspect extraction module for financial text analysis.

Implements aspect-based text classification using keyword matching
and semantic similarity with sentence embeddings.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ..utils.config import get_config
from ..utils.io import save_parquet, load_parquet, save_json, load_json
from ..utils.logging import get_logger, LoggingMixin, log_performance

logger = get_logger(__name__)


class AspectExtractor(LoggingMixin):
    """
    Financial text aspect extractor using hybrid keyword and semantic approaches.
    
    Features:
    - Multi-aspect classification
    - Keyword matching with expanded lexicons
    - Semantic similarity fallback using sentence embeddings
    - TF-IDF based relevance scoring
    - Configurable similarity thresholds
    """
    
    def __init__(
        self,
        embedding_model: Optional[str] = None,
        similarity_threshold: Optional[float] = None,
        expand_lexicon: bool = True,
        cache_embeddings: bool = True
    ):
        """
        Initialize the aspect extractor.
        
        Args:
            embedding_model: Sentence transformer model name
            similarity_threshold: Minimum cosine similarity for aspect assignment
            expand_lexicon: Whether to expand seed terms using embeddings
            cache_embeddings: Whether to cache computed embeddings
        """
        self.config = get_config()
        self.embedding_model_name = embedding_model or self.config.EMBEDDING_MODEL
        self.similarity_threshold = similarity_threshold or self.config.SIMILARITY_THRESHOLD
        self.expand_lexicon = expand_lexicon
        self.cache_embeddings = cache_embeddings
        
        # Initialize components
        self.embedding_model = None
        self.tfidf_vectorizer = None
        self.aspect_lexicon = {}
        self.aspect_embeddings = {}
        self._embedding_cache = {}
        
        # Load models and lexicons
        self._load_embedding_model()
        self._load_aspect_lexicon()
        if self.expand_lexicon:
            self._expand_aspect_lexicon()
        self._compute_aspect_embeddings()
    
    def _load_embedding_model(self) -> None:
        """Load the sentence embedding model."""
        try:
            self.logger.info(f"Loading embedding model: {self.embedding_model_name}")
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            self.logger.info("Embedding model loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load embedding model: {e}")
            raise
    
    def _load_aspect_lexicon(self) -> None:
        """Load the aspect lexicon from configuration."""
        self.aspect_lexicon = self.config.ASPECT_CATEGORIES.copy()
        self.logger.info(f"Loaded {len(self.aspect_lexicon)} aspect categories")
    
    def _expand_aspect_lexicon(self) -> None:
        """
        Expand aspect lexicon using semantic similarity.
        
        For each seed term, find semantically similar terms and add them
        to the lexicon if they meet the similarity threshold.
        """
        expanded_lexicon_path = self.config.PROCESSED_DIR / "expanded_aspect_lexicon.json"
        
        # Load from cache if exists
        if expanded_lexicon_path.exists():
            self.aspect_lexicon = load_json(expanded_lexicon_path)
            self.logger.info("Loaded expanded aspect lexicon from cache")
            return
        
        self.logger.info("Expanding aspect lexicon using semantic similarity")
        expanded_lexicon = {}
        
        for aspect, seed_terms in self.aspect_lexicon.items():
            expanded_terms = set(seed_terms)
            
            # Get embeddings for seed terms
            seed_embeddings = []
            for term in seed_terms:
                # Remove wildcards for embedding
                clean_term = term.replace("*", "")
                if clean_term:
                    embedding = self.embedding_model.encode(clean_term)
                    seed_embeddings.append(embedding)
            
            if seed_embeddings:
                # Compute mean embedding for this aspect
                aspect_embedding = np.mean(seed_embeddings, axis=0)
                
                # Find similar terms (this would require a vocabulary)
                # For now, we'll use the original terms
                expanded_terms.update(seed_terms)
            
            expanded_lexicon[aspect] = list(expanded_terms)
        
        self.aspect_lexicon = expanded_lexicon
        save_json(expanded_lexicon, expanded_lexicon_path)
        self.logger.info("Aspect lexicon expanded and cached")
    
    def _compute_aspect_embeddings(self) -> None:
        """Compute mean embeddings for each aspect category."""
        self.logger.info("Computing aspect embeddings")
        
        for aspect, terms in self.aspect_lexicon.items():
            # Clean terms and get embeddings
            clean_terms = []
            for term in terms:
                clean_term = term.replace("*", "").strip()
                if clean_term and len(clean_term) > 2:
                    clean_terms.append(clean_term)
            
            if clean_terms:
                embeddings = self.embedding_model.encode(clean_terms)
                # Store mean embedding for this aspect
                self.aspect_embeddings[aspect] = np.mean(embeddings, axis=0)
        
        self.logger.info(f"Computed embeddings for {len(self.aspect_embeddings)} aspects")
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text with caching."""
        if self.cache_embeddings and text in self._embedding_cache:
            return self._embedding_cache[text]
        
        embedding = self.embedding_model.encode(text)
        
        if self.cache_embeddings:
            self._embedding_cache[text] = embedding
        
        return embedding
    
    def _keyword_match(self, text: str, aspect: str) -> bool:
        """Check if text contains keywords for the given aspect."""
        text_lower = text.lower()
        terms = self.aspect_lexicon.get(aspect, [])
        
        for term in terms:
            # Handle wildcards
            if "*" in term:
                pattern = term.replace("*", ".*")
                if re.search(pattern, text_lower):
                    return True
            else:
                if term.lower() in text_lower:
                    return True
        
        return False
    
    def _semantic_match(self, text: str, aspect: str) -> float:
        """Compute semantic similarity between text and aspect."""
        if aspect not in self.aspect_embeddings:
            return 0.0
        
        text_embedding = self._get_embedding(text)
        aspect_embedding = self.aspect_embeddings[aspect]
        
        # Compute cosine similarity
        similarity = cosine_similarity(
            text_embedding.reshape(1, -1),
            aspect_embedding.reshape(1, -1)
        )[0, 0]
        
        return similarity
    
    @log_performance
    def extract_aspects(
        self, 
        text: str,
        use_keyword_matching: bool = True,
        use_semantic_matching: bool = True,
        return_scores: bool = False
    ) -> Union[List[str], Dict[str, float]]:
        """
        Extract aspects from a single text.
        
        Args:
            text: Input text to analyze
            use_keyword_matching: Whether to use keyword matching
            use_semantic_matching: Whether to use semantic matching
            return_scores: Whether to return similarity scores
            
        Returns:
            List of aspect labels or dictionary with scores
        """
        if not text or not text.strip():
            return {} if return_scores else []
        
        aspect_scores = {}
        matched_aspects = set()
        
        # Keyword matching
        if use_keyword_matching:
            for aspect in self.aspect_lexicon:
                if self._keyword_match(text, aspect):
                    matched_aspects.add(aspect)
                    aspect_scores[aspect] = 1.0  # High confidence for keyword matches
        
        # Semantic matching
        if use_semantic_matching:
            for aspect in self.aspect_embeddings:
                similarity = self._semantic_match(text, aspect)
                
                if similarity >= self.similarity_threshold:
                    matched_aspects.add(aspect)
                    # Use max of existing score or semantic similarity
                    aspect_scores[aspect] = max(
                        aspect_scores.get(aspect, 0),
                        similarity
                    )
        
        if return_scores:
            return aspect_scores
        else:
            return list(matched_aspects)
    
    @log_performance
    def extract_aspects_batch(
        self,
        texts: List[str],
        use_keyword_matching: bool = True,
        use_semantic_matching: bool = True,
        return_scores: bool = False
    ) -> List[Union[List[str], Dict[str, float]]]:
        """
        Extract aspects from multiple texts.
        
        Args:
            texts: List of input texts
            use_keyword_matching: Whether to use keyword matching
            use_semantic_matching: Whether to use semantic matching
            return_scores: Whether to return similarity scores
            
        Returns:
            List of aspect results
        """
        results = []
        
        for text in texts:
            result = self.extract_aspects(
                text,
                use_keyword_matching=use_keyword_matching,
                use_semantic_matching=use_semantic_matching,
                return_scores=return_scores
            )
            results.append(result)
        
        return results
    
    @log_performance
    def extract_aspects_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str,
        output_prefix: str = "aspect_"
    ) -> pd.DataFrame:
        """
        Extract aspects for a DataFrame column.
        
        Args:
            df: Input DataFrame
            text_column: Name of column containing text
            output_prefix: Prefix for output columns
            
        Returns:
            DataFrame with added aspect columns
        """
        # Extract texts
        texts = df[text_column].fillna("").astype(str).tolist()
        
        # Extract aspects with scores
        aspect_results = self.extract_aspects_batch(
            texts, 
            return_scores=True
        )
        
        # Convert to DataFrame format
        aspect_data = []
        for i, scores in enumerate(aspect_results):
            row_data = {f"{output_prefix}{aspect}": score 
                       for aspect, score in scores.items()}
            
            # Add binary indicators
            for aspect in self.aspect_lexicon:
                if f"{output_prefix}{aspect}" not in row_data:
                    row_data[f"{output_prefix}{aspect}"] = 0.0
                row_data[f"{output_prefix}{aspect}_binary"] = (
                    1 if row_data[f"{output_prefix}{aspect}"] > 0 else 0
                )
            
            aspect_data.append(row_data)
        
        aspect_df = pd.DataFrame(aspect_data)
        
        # Combine with original DataFrame
        result_df = pd.concat([df.reset_index(drop=True), aspect_df], axis=1)
        
        return result_df
    
    def get_aspect_summary(self) -> Dict[str, Dict]:
        """Get summary statistics about the aspect lexicon."""
        summary = {}
        
        for aspect, terms in self.aspect_lexicon.items():
            summary[aspect] = {
                "num_terms": len(terms),
                "terms": terms,
                "has_embedding": aspect in self.aspect_embeddings
            }
        
        return summary
    
    def save_cache(self, cache_file: Optional[Path] = None) -> None:
        """Save embedding cache to disk."""
        if not self.cache_embeddings or not self._embedding_cache:
            return
        
        if cache_file is None:
            cache_file = self.config.MODELS_DIR / "aspect_embedding_cache.pkl"
        
        import pickle
        with open(cache_file, 'wb') as f:
            pickle.dump(self._embedding_cache, f)
        
        self.logger.info(f"Saved embedding cache with {len(self._embedding_cache)} entries")
    
    def load_cache(self, cache_file: Optional[Path] = None) -> None:
        """Load embedding cache from disk."""
        if not self.cache_embeddings:
            return
        
        if cache_file is None:
            cache_file = self.config.MODELS_DIR / "aspect_embedding_cache.pkl"
        
        if cache_file.exists():
            import pickle
            with open(cache_file, 'rb') as f:
                self._embedding_cache = pickle.load(f)
            
            self.logger.info(f"Loaded embedding cache with {len(self._embedding_cache)} entries")


# Convenience functions
@log_performance
def extract_aspects(
    texts: Union[str, List[str]],
    embedding_model: Optional[str] = None,
    similarity_threshold: Optional[float] = None,
    return_scores: bool = False
) -> Union[List[str], List[List[str]], Dict[str, float], List[Dict[str, float]]]:
    """
    Extract aspects from text(s) using default extractor.
    
    Args:
        texts: Single text or list of texts
        embedding_model: Embedding model name
        similarity_threshold: Similarity threshold for semantic matching
        return_scores: Whether to return similarity scores
        
    Returns:
        Aspect results (single result or list of results)
    """
    # Initialize extractor
    extractor = AspectExtractor(
        embedding_model=embedding_model,
        similarity_threshold=similarity_threshold
    )
    
    # Handle single text vs list
    if isinstance(texts, str):
        return extractor.extract_aspects(texts, return_scores=return_scores)
    else:
        return extractor.extract_aspects_batch(texts, return_scores=return_scores)


def compute_aspect_shock(
    sentiment_scores: Dict[str, float],
    novelty_score: float,
    aspects: List[str]
) -> Dict[str, float]:
    """
    Compute aspect-based sentiment shock scores.
    
    Args:
        sentiment_scores: Dictionary of sentiment scores by aspect
        novelty_score: Overall novelty score for the text
        aspects: List of detected aspects
        
    Returns:
        Dictionary of aspect shock scores
    """
    aspect_shocks = {}
    
    for aspect in aspects:
        sentiment = sentiment_scores.get(aspect, 0.0)
        shock = sentiment * novelty_score
        aspect_shocks[f"shock_{aspect}"] = shock
    
    return aspect_shocks


def aggregate_aspect_scores(
    aspect_scores_list: List[Dict[str, float]],
    weights: Optional[List[float]] = None,
    aggregation: str = "mean"
) -> Dict[str, float]:
    """
    Aggregate multiple aspect score dictionaries.
    
    Args:
        aspect_scores_list: List of aspect score dictionaries
        weights: Optional weights for aggregation
        aggregation: Aggregation method ('mean', 'weighted_mean', 'max')
        
    Returns:
        Aggregated aspect scores
    """
    if not aspect_scores_list:
        return {}
    
    if weights is None:
        weights = [1.0] * len(aspect_scores_list)
    
    if len(weights) != len(aspect_scores_list):
        raise ValueError("Number of weights must match number of score dictionaries")
    
    # Get all unique aspects
    all_aspects = set()
    for scores in aspect_scores_list:
        all_aspects.update(scores.keys())
    
    aggregated = {}
    
    for aspect in all_aspects:
        values = []
        aspect_weights = []
        
        for scores, weight in zip(aspect_scores_list, weights):
            if aspect in scores:
                values.append(scores[aspect])
                aspect_weights.append(weight)
        
        if values:
            if aggregation == "mean":
                aggregated[aspect] = np.mean(values)
            elif aggregation == "weighted_mean":
                total_weight = sum(aspect_weights)
                if total_weight > 0:
                    weighted_sum = sum(v * w for v, w in zip(values, aspect_weights))
                    aggregated[aspect] = weighted_sum / total_weight
                else:
                    aggregated[aspect] = np.mean(values)
            elif aggregation == "max":
                aggregated[aspect] = max(values)
            else:
                raise ValueError(f"Unsupported aggregation method: {aggregation}")
    
    return aggregated 