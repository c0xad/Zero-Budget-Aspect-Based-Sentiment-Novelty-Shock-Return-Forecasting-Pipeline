"""
ABSSENT: Aspect-Based Sentiment & Novelty-Shock Return Forecasting Pipeline

A zero-budget, fully reproducible research & production pipeline that forecasts 
next-day equity returns using aspect-based sentiment analysis from SEC 8-K filings.
"""

__version__ = "0.1.0"
__author__ = "Research Team"
__email__ = "research@abssent.ai"

from .etl import edgar, prices, news
from .nlp import sentiment, aspects, novelty
from .utils import config, logging, io

__all__ = [
    "edgar",
    "prices", 
    "news",
    "sentiment",
    "aspects",
    "novelty",
    "config",
    "logging",
    "io",
] 