"""
Configuration module for ABSSENT package.

Contains all settings, paths, and constants used throughout the project.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class Config:
    """Main configuration class for ABSSENT project."""
    
    # Project paths
    PROJECT_ROOT: Path = Path(__file__).parent.parent.parent
    RAW_DATA_DIR: Path = PROJECT_ROOT / "raw-data"
    PROCESSED_DIR: Path = PROJECT_ROOT / "processed" 
    FEATURES_DIR: Path = PROJECT_ROOT / "features"
    MODELS_DIR: Path = PROJECT_ROOT / "models"
    REPORTS_DIR: Path = PROJECT_ROOT / "reports"
    
    # Data source paths
    EDGAR_DIR: Path = RAW_DATA_DIR / "edgar"
    NEWS_DIR: Path = RAW_DATA_DIR / "news"
    PRICES_DIR: Path = RAW_DATA_DIR / "prices"
    
    # SEC EDGAR settings
    SEC_BASE_URL: str = "https://www.sec.gov/Archives"
    SEC_DAILY_INDEX_URL: str = "https://www.sec.gov/Archives/edgar/daily-index"
    SEC_RATE_LIMIT: float = 0.1  # 10 requests per second
    SEC_USER_AGENT: str = "abssent-research-bot/1.0 (+https://github.com/abssent/abssent)"
    
    # Target tickers (S&P 1500 subset for demo)
    TARGET_TICKERS: List[str] = None
    
    # Aspect categories and seed terms
    ASPECT_CATEGORIES: Dict[str, List[str]] = None
    
    # Model settings
    SENTIMENT_MODEL: str = "ProsusAI/finbert"
    EMBEDDING_MODEL: str = "all-mpnet-base-v2"
    MAX_SEQUENCE_LENGTH: int = 512
    BATCH_SIZE: int = 32
    
    # Feature engineering
    NOVELTY_WINDOW_DAYS: int = 30
    MIN_PARAGRAPH_LENGTH: int = 50
    SIMILARITY_THRESHOLD: float = 0.4
    
    # Date ranges
    LOOKBACK_YEARS: int = 10
    TRAIN_TEST_SPLIT: float = 0.7
    
    # Database settings
    DUCKDB_PATH: str = str(PROJECT_ROOT / "data.duckdb")
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # API settings
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8080
    
    def __post_init__(self) -> None:
        """Ensure all directories exist and set default values for mutable fields."""
        # Set default values for mutable fields
        if self.TARGET_TICKERS is None:
            self.TARGET_TICKERS = [
                "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "NFLX",
                "JPM", "JNJ", "V", "PG", "UNH", "HD", "DIS", "MA", "PYPL", "ADBE",
                "CRM", "INTC", "CMCSA", "VZ", "KO", "PFE", "PEP", "WMT", "BAC", "XOM"
            ]
        
        if self.ASPECT_CATEGORIES is None:
            self.ASPECT_CATEGORIES = {
                "liquidity": [
                    "liquid*", "cash flow", "working capital", "debt", "credit facility",
                    "loan", "borrowing", "financing", "capital structure"
                ],
                "guidance": [
                    "guidance", "forecast", "outlook", "projection", "estimate",
                    "expect*", "anticipate", "target", "goal"
                ],
                "management_change": [
                    "resignation", "appoint*", "retire*", "successor", "interim",
                    "chief executive", "president", "director", "board"
                ],
                "risk": [
                    "risk factor*", "uncertainty", "litigation", "regulatory",
                    "compliance", "investigation", "material weakness"
                ],
                "operations": [
                    "acquisition", "merger", "divestiture", "restructur*", "consolidat*",
                    "expansion", "capacity", "facility", "manufacturing"
                ],
                "financial_performance": [
                    "revenue", "earnings", "profit", "margin", "cost", "expense",
                    "impairment", "write-down", "charge", "gain", "loss"
                ]
            }
        
        # Ensure all directories exist
        for path_attr in ["RAW_DATA_DIR", "PROCESSED_DIR", "FEATURES_DIR", 
                         "MODELS_DIR", "REPORTS_DIR", "EDGAR_DIR", "NEWS_DIR", "PRICES_DIR"]:
            path = getattr(self, path_attr)
            path.mkdir(parents=True, exist_ok=True)


# Global config instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = Config()
    return _config


def set_config(config: Config) -> None:
    """Set the global configuration instance."""
    global _config
    _config = config


# Environment variable overrides
def load_config_from_env() -> Config:
    """Load configuration with environment variable overrides."""
    config = Config()
    
    # Override with environment variables if present
    if "ABSSENT_DATA_DIR" in os.environ:
        config.RAW_DATA_DIR = Path(os.environ["ABSSENT_DATA_DIR"])
    
    if "ABSSENT_LOG_LEVEL" in os.environ:
        config.LOG_LEVEL = os.environ["ABSSENT_LOG_LEVEL"]
    
    if "ABSSENT_SEC_RATE_LIMIT" in os.environ:
        config.SEC_RATE_LIMIT = float(os.environ["ABSSENT_SEC_RATE_LIMIT"])
    
    return config 