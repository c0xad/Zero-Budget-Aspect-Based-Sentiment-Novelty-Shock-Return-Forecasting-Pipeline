"""
I/O utilities for ABSSENT package.

Provides file operations, downloads, and data persistence utilities.
"""

import asyncio
import hashlib
import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Union

import aiohttp
import pandas as pd
import polars as pl
import requests
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from .config import get_config
from .logging import get_logger, log_performance

logger = get_logger(__name__)


def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Ensure directory exists, creating it if necessary.
    
    Args:
        path: Directory path
        
    Returns:
        Path object for the directory
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_file_hash(file_path: Union[str, Path]) -> str:
    """
    Calculate SHA256 hash of a file.
    
    Args:
        file_path: Path to file
        
    Returns:
        Hexadecimal hash string
    """
    hasher = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


@log_performance
def download_file(
    url: str,
    destination: Union[str, Path],
    chunk_size: int = 8192,
    timeout: int = 30,
    user_agent: Optional[str] = None
) -> Path:
    """
    Download a file from URL to destination with progress bar.
    
    Args:
        url: URL to download from
        destination: Local file path to save to
        chunk_size: Size of chunks to download
        timeout: Request timeout in seconds
        user_agent: Custom user agent string
        
    Returns:
        Path to downloaded file
    """
    config = get_config()
    destination = Path(destination)
    ensure_dir(destination.parent)
    
    headers = {}
    if user_agent:
        headers['User-Agent'] = user_agent
    else:
        headers['User-Agent'] = config.SEC_USER_AGENT
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
    ) as progress:
        
        response = requests.get(url, headers=headers, stream=True, timeout=timeout)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        task = progress.add_task(f"Downloading {destination.name}", total=total_size)
        
        with open(destination, 'wb') as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    progress.advance(task, len(chunk))
    
    logger.info(f"Downloaded {url} to {destination}")
    return destination


async def download_file_async(
    session: aiohttp.ClientSession,
    url: str,
    destination: Union[str, Path],
    semaphore: Optional[asyncio.Semaphore] = None
) -> Path:
    """
    Async download of a file with rate limiting.
    
    Args:
        session: aiohttp client session
        url: URL to download from
        destination: Local file path to save to
        semaphore: Optional semaphore for rate limiting
        
    Returns:
        Path to downloaded file
    """
    destination = Path(destination)
    ensure_dir(destination.parent)
    
    if semaphore:
        await semaphore.acquire()
    
    try:
        async with session.get(url) as response:
            response.raise_for_status()
            
            with open(destination, 'wb') as f:
                async for chunk in response.content.iter_chunked(8192):
                    f.write(chunk)
            
            logger.debug(f"Downloaded {url} to {destination}")
            return destination
    finally:
        if semaphore:
            semaphore.release()


@log_performance
def save_parquet(
    data: Union[pd.DataFrame, pl.DataFrame],
    file_path: Union[str, Path],
    compression: str = "snappy"
) -> Path:
    """
    Save DataFrame to Parquet format.
    
    Args:
        data: DataFrame to save
        file_path: Output file path
        compression: Compression algorithm
        
    Returns:
        Path to saved file
    """
    file_path = Path(file_path)
    ensure_dir(file_path.parent)
    
    if isinstance(data, pd.DataFrame):
        data.to_parquet(file_path, compression=compression, index=False)
    elif isinstance(data, pl.DataFrame):
        data.write_parquet(file_path, compression=compression)
    else:
        raise ValueError(f"Unsupported data type: {type(data)}")
    
    logger.info(f"Saved {len(data)} rows to {file_path}")
    return file_path


@log_performance
def load_parquet(
    file_path: Union[str, Path],
    engine: str = "polars"
) -> Union[pd.DataFrame, pl.DataFrame]:
    """
    Load DataFrame from Parquet format.
    
    Args:
        file_path: Input file path
        engine: DataFrame engine ("pandas" or "polars")
        
    Returns:
        Loaded DataFrame
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    if engine == "pandas":
        data = pd.read_parquet(file_path)
    elif engine == "polars":
        data = pl.read_parquet(file_path)
    else:
        raise ValueError(f"Unsupported engine: {engine}")
    
    logger.info(f"Loaded {len(data)} rows from {file_path}")
    return data


def save_json(data: Dict[str, Any], file_path: Union[str, Path]) -> Path:
    """
    Save dictionary to JSON format.
    
    Args:
        data: Dictionary to save
        file_path: Output file path
        
    Returns:
        Path to saved file
    """
    import json
    
    file_path = Path(file_path)
    ensure_dir(file_path.parent)
    
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    logger.info(f"Saved JSON to {file_path}")
    return file_path


def load_json(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load dictionary from JSON format.
    
    Args:
        file_path: Input file path
        
    Returns:
        Loaded dictionary
    """
    import json
    
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    logger.info(f"Loaded JSON from {file_path}")
    return data


def save_pickle(data: Any, file_path: Union[str, Path]) -> Path:
    """
    Save object to pickle format.
    
    Args:
        data: Object to save
        file_path: Output file path
        
    Returns:
        Path to saved file
    """
    import pickle
    
    file_path = Path(file_path)
    ensure_dir(file_path.parent)
    
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)
    
    logger.info(f"Saved pickle to {file_path}")
    return file_path


def load_pickle(file_path: Union[str, Path]) -> Any:
    """
    Load object from pickle format.
    
    Args:
        file_path: Input file path
        
    Returns:
        Loaded object
    """
    import pickle
    
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    logger.info(f"Loaded pickle from {file_path}")
    return data


def backup_file(file_path: Union[str, Path], backup_dir: Optional[Union[str, Path]] = None) -> Path:
    """
    Create a backup copy of a file.
    
    Args:
        file_path: File to backup
        backup_dir: Directory to store backup (default: same directory with .backup suffix)
        
    Returns:
        Path to backup file
    """
    file_path = Path(file_path)
    
    if backup_dir is None:
        backup_path = file_path.with_suffix(file_path.suffix + '.backup')
    else:
        backup_dir = Path(backup_dir)
        ensure_dir(backup_dir)
        backup_path = backup_dir / file_path.name
    
    shutil.copy2(file_path, backup_path)
    logger.info(f"Backed up {file_path} to {backup_path}")
    return backup_path 