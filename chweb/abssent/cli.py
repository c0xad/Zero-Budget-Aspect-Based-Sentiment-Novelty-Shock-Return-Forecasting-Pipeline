"""
Command Line Interface for ABSSENT package.

Provides easy-to-use commands for data collection, analysis, and prediction.
"""

import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.progress import track

from .etl import download_8k_filings, download_price_data, collect_news_headlines
from .nlp import analyze_sentiment
from .utils.config import get_config
from .utils.logging import setup_logging, get_logger

# Create typer app
app = typer.Typer(
    name="abssent",
    help="Zero-Budget Aspect-Based Sentiment & Novelty-Shock Return Forecasting Pipeline",
    add_completion=False
)

console = Console()


@app.callback()
def main(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
    log_file: Optional[Path] = typer.Option(None, "--log-file", help="Log file path")
):
    """ABSSENT: Aspect-Based Sentiment Analysis for Financial Forecasting."""
    # Setup logging
    log_level = "DEBUG" if verbose else "INFO"
    setup_logging(level=log_level, log_file=log_file)


@app.command()
def download(
    tickers: List[str] = typer.Option(None, "--tickers", "-t", help="Stock tickers to download"),
    days: int = typer.Option(30, "--days", "-d", help="Number of days back to download"),
    include_filings: bool = typer.Option(True, "--filings/--no-filings", help="Download 8-K filings"),
    include_prices: bool = typer.Option(True, "--prices/--no-prices", help="Download price data"),
    include_news: bool = typer.Option(True, "--news/--no-news", help="Download news data"),
):
    """Download financial data for analysis."""
    logger = get_logger(__name__)
    config = get_config()
    
    # Use default tickers if none specified
    if not tickers:
        tickers = config.TARGET_TICKERS[:10]  # Limit to first 10 for demo
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    console.print(f"📊 Downloading data for {len(tickers)} tickers from {start_date.date()} to {end_date.date()}")
    
    try:
        # Download 8-K filings
        if include_filings:
            console.print("📄 Downloading 8-K filings...")
            filings = asyncio.run(download_8k_filings(
                start_date=start_date,
                end_date=end_date,
                tickers=tickers
            ))
            console.print(f"✅ Downloaded {len(filings)} filings")
        
        # Download price data
        if include_prices:
            console.print("💰 Downloading price data...")
            prices = download_price_data(
                tickers=tickers,
                start_date=start_date,
                end_date=end_date
            )
            console.print(f"✅ Downloaded price data for {len(prices)} tickers")
        
        # Download news data
        if include_news:
            console.print("📰 Downloading news data...")
            news = collect_news_headlines(
                tickers=tickers,
                days_back=days,
                include_general=True
            )
            console.print(f"✅ Downloaded {len(news)} news articles")
        
        console.print("🎉 Data download completed successfully!")
        
    except Exception as e:
        console.print(f"❌ Error during download: {e}")
        raise typer.Exit(code=1)


@app.command()
def analyze(
    input_dir: Path = typer.Option("raw-data/edgar", "--input-dir", "-i", help="Input directory with text files"),
    output: Path = typer.Option("features/sentiment.parquet", "--output", "-o", help="Output file path"),
    model: str = typer.Option("ProsusAI/finbert", "--model", "-m", help="Sentiment analysis model"),
    batch_size: int = typer.Option(32, "--batch-size", "-b", help="Batch size for processing"),
):
    """Analyze sentiment of financial texts."""
    logger = get_logger(__name__)
    
    if not input_dir.exists():
        console.print(f"❌ Input directory does not exist: {input_dir}")
        raise typer.Exit(code=1)
    
    console.print(f"🔍 Analyzing sentiment in {input_dir}")
    
    try:
        # Find all text files
        text_files = list(input_dir.rglob("*.txt"))
        
        if not text_files:
            console.print(f"❌ No text files found in {input_dir}")
            raise typer.Exit(code=1)
        
        console.print(f"📝 Found {len(text_files)} text files")
        
        # Process files
        results = []
        for file_path in track(text_files, description="Processing files..."):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Analyze sentiment
                sentiment = analyze_sentiment(content, model_name=model)
                
                result = {
                    'file_path': str(file_path),
                    'file_name': file_path.name,
                    **sentiment
                }
                results.append(result)
                
            except Exception as e:
                logger.warning(f"Failed to process {file_path}: {e}")
                continue
        
        # Save results
        if results:
            import pandas as pd
            from .utils.io import save_parquet
            
            df = pd.DataFrame(results)
            save_parquet(df, output)
            console.print(f"✅ Saved {len(results)} sentiment analyses to {output}")
        else:
            console.print("❌ No files were successfully processed")
            raise typer.Exit(code=1)
        
    except Exception as e:
        console.print(f"❌ Error during analysis: {e}")
        raise typer.Exit(code=1)


@app.command()
def predict(
    date: str = typer.Option(..., "--date", "-d", help="Prediction date (YYYY-MM-DD)"),
    tickers: List[str] = typer.Option(..., "--tickers", "-t", help="Stock tickers to predict"),
    output: Path = typer.Option("predictions.csv", "--output", "-o", help="Output file path"),
    model_dir: Path = typer.Option("models", "--model-dir", "-m", help="Directory with trained models"),
):
    """Generate return predictions for specified tickers and date."""
    logger = get_logger(__name__)
    
    try:
        pred_date = datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        console.print("❌ Invalid date format. Use YYYY-MM-DD")
        raise typer.Exit(code=1)
    
    console.print(f"🔮 Generating predictions for {tickers} on {pred_date.date()}")
    
    try:
        # This is a placeholder - in the full implementation, this would:
        # 1. Load trained models from model_dir
        # 2. Collect recent data for the specified tickers
        # 3. Generate features using the NLP pipeline
        # 4. Make predictions using the trained models
        # 5. Save results to output file
        
        import pandas as pd
        import numpy as np
        
        # Generate dummy predictions for demonstration
        predictions = []
        for ticker in tickers:
            pred = {
                'date': pred_date.date(),
                'ticker': ticker,
                'predicted_return': np.random.normal(0.001, 0.02),  # Random return ~0.1% ± 2%
                'confidence': np.random.uniform(0.3, 0.8),
                'sentiment_score': np.random.uniform(-0.5, 0.5),
                'novelty_score': np.random.uniform(0.0, 1.0)
            }
            predictions.append(pred)
        
        # Save predictions
        df = pd.DataFrame(predictions)
        df.to_csv(output, index=False)
        
        console.print(f"✅ Saved predictions for {len(tickers)} tickers to {output}")
        
        # Display results
        console.print("\n📈 Predictions:")
        for _, row in df.iterrows():
            direction = "📈" if row['predicted_return'] > 0 else "📉"
            console.print(
                f"{direction} {row['ticker']}: {row['predicted_return']:.3f} "
                f"(confidence: {row['confidence']:.2f})"
            )
        
    except Exception as e:
        console.print(f"❌ Error during prediction: {e}")
        raise typer.Exit(code=1)


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host", help="Host address"),
    port: int = typer.Option(8080, "--port", "-p", help="Port number"),
    reload: bool = typer.Option(False, "--reload", help="Enable auto-reload for development"),
):
    """Start the FastAPI prediction server."""
    try:
        import uvicorn
        from .api import app as api_app
        
        console.print(f"🚀 Starting ABSSENT API server on {host}:{port}")
        console.print("📋 Available endpoints:")
        console.print("  • GET  /health        - Health check")
        console.print("  • POST /predict       - Generate predictions")
        console.print("  • GET  /docs          - API documentation")
        
        uvicorn.run(
            "abssent.api:app",
            host=host,
            port=port,
            reload=reload
        )
        
    except ImportError:
        console.print("❌ FastAPI/uvicorn not installed. Install with: pip install fastapi uvicorn")
        raise typer.Exit(code=1)
    except Exception as e:
        console.print(f"❌ Error starting server: {e}")
        raise typer.Exit(code=1)


@app.command()
def demo(
    quick: bool = typer.Option(False, "--quick", help="Run quick demo with sample data")
):
    """Run a demonstration of the ABSSENT pipeline."""
    console.print("🎭 Running ABSSENT Demo")
    
    if quick:
        # Quick demo with sample text
        sample_text = """
        The company reported strong quarterly earnings, exceeding analyst expectations.
        Management provided positive guidance for the next quarter, citing improved
        liquidity and successful debt refinancing. The acquisition of the competitor
        is expected to drive significant growth.
        """
        
        console.print("📝 Analyzing sample financial text...")
        sentiment = analyze_sentiment(sample_text)
        
        console.print(f"✅ Sentiment Analysis Results:")
        console.print(f"   • Label: {sentiment['label']}")
        console.print(f"   • Compound Score: {sentiment['compound']:.3f}")
        console.print(f"   • Positive: {sentiment.get('positive', 0):.3f}")
        console.print(f"   • Negative: {sentiment.get('negative', 0):.3f}")
        console.print(f"   • Neutral: {sentiment.get('neutral', 0):.3f}")
        
    else:
        # Full demo - download sample data and process
        console.print("📊 Running full pipeline demo...")
        
        # Use a small subset of tickers for demo
        demo_tickers = ["AAPL", "MSFT"]
        
        # Download recent data (last 7 days)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        
        try:
            console.print("📄 Downloading sample 8-K filings...")
            filings = asyncio.run(download_8k_filings(
                start_date=start_date,
                end_date=end_date,
                tickers=demo_tickers
            ))
            
            console.print("💰 Downloading price data...")
            prices = download_price_data(
                tickers=demo_tickers,
                start_date=start_date,
                end_date=end_date
            )
            
            console.print("🎉 Demo completed successfully!")
            console.print(f"   • Downloaded {len(filings)} filings")
            console.print(f"   • Downloaded price data for {len(prices)} tickers")
            
        except Exception as e:
            console.print(f"❌ Demo failed: {e}")
            console.print("💡 Try running with --quick flag for a simple demo")


if __name__ == "__main__":
    app() 