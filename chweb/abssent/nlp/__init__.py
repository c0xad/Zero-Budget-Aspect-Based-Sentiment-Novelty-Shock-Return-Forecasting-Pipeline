"""
NLP (Natural Language Processing) module for ABSSENT package.

Contains sentiment analysis, aspect extraction, and novelty detection components.
"""

from .sentiment import SentimentAnalyzer, analyze_sentiment
from .aspects import AspectExtractor, extract_aspects
from .novelty import NoveltyDetector, calculate_novelty

__all__ = [
    "SentimentAnalyzer",
    "analyze_sentiment",
    "AspectExtractor", 
    "extract_aspects",
    "NoveltyDetector",
    "calculate_novelty",
] 