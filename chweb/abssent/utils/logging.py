"""
Logging utilities for ABSSENT package.

Provides structured logging with rich formatting and proper configuration.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler
from rich.traceback import install

from .config import get_config


def setup_logging(
    level: Optional[str] = None,
    log_file: Optional[Path] = None,
    rich_tracebacks: bool = True
) -> None:
    """
    Set up logging configuration for the ABSSENT project.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path to write logs to
        rich_tracebacks: Whether to use rich tracebacks
    """
    config = get_config()
    
    if level is None:
        level = config.LOG_LEVEL
    
    # Install rich tracebacks for better error formatting
    if rich_tracebacks:
        install(show_locals=True)
    
    # Create console for rich logging
    console = Console(stderr=True)
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=config.LOG_FORMAT,
        handlers=[
            RichHandler(
                console=console,
                show_time=True,
                show_level=True,
                show_path=True,
                markup=True,
                rich_tracebacks=rich_tracebacks,
                tracebacks_show_locals=True
            )
        ]
    )
    
    # Add file handler if specified
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter(config.LOG_FORMAT)
        )
        logging.getLogger().addHandler(file_handler)
    
    # Suppress noisy third-party loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("aiohttp").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)


class LoggingMixin:
    """Mixin class to add logging capabilities to any class."""
    
    @property
    def logger(self) -> logging.Logger:
        """Get logger for this class."""
        return get_logger(self.__class__.__module__ + "." + self.__class__.__name__)


def log_function_call(func):
    """
    Decorator to log function calls with arguments and return values.
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function
    """
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__ + "." + func.__name__)
        
        # Log function entry
        args_str = ", ".join([repr(arg) for arg in args])
        kwargs_str = ", ".join([f"{k}={repr(v)}" for k, v in kwargs.items()])
        all_args = ", ".join(filter(None, [args_str, kwargs_str]))
        
        logger.debug(f"Calling {func.__name__}({all_args})")
        
        try:
            result = func(*args, **kwargs)
            logger.debug(f"{func.__name__} returned: {repr(result)}")
            return result
        except Exception as e:
            logger.error(f"{func.__name__} raised {type(e).__name__}: {e}")
            raise
    
    return wrapper


def log_performance(func):
    """
    Decorator to log function performance metrics.
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function
    """
    import time
    
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__ + "." + func.__name__)
        
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            logger.info(f"{func.__name__} completed in {duration:.2f}s")
            return result
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"{func.__name__} failed after {duration:.2f}s: {e}")
            raise
    
    return wrapper 