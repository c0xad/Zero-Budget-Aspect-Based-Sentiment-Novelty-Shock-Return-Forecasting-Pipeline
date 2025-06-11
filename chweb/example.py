#!/usr/bin/env python3
"""
ABSSENT Pipeline Example

This script demonstrates the core functionality of the ABSSENT package:
- Sentiment analysis of financial texts
- Price data downloading
- Basic feature engineering
- Simple prediction workflow

Run this script to see the pipeline in action!
"""

import asyncio
import os
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import numpy as np

# Import ABSSENT modules
from abssent.nlp import analyze_sentiment
from abssent.etl import download_price_data
from abssent.utils.config import get_config
from abssent.utils.logging import setup_logging, get_logger

# Setup logging
setup_logging(level="INFO")
logger = get_logger(__name__)


def demo_sentiment_analysis():
    """Demonstrate sentiment analysis on sample financial texts."""
    print("\n🔍 Sentiment Analysis Demo")
    print("=" * 50)
    
    # Sample financial texts with different sentiments
    sample_texts = [
        {
            "text": "The company reported record quarterly earnings, significantly beating analyst expectations. Strong demand across all product lines contributed to the outstanding performance.",
            "expected": "positive"
        },
        {
            "text": "Due to ongoing supply chain disruptions and rising costs, the company has lowered its full-year guidance. Management expects challenging conditions to persist.",
            "expected": "negative"
        },
        {
            "text": "The company announced a strategic acquisition to expand its market presence. The transaction is expected to close in Q3.",
            "expected": "neutral/positive"
        },
        {
            "text": "Regulatory investigations have resulted in significant legal fees and potential fines. The company is cooperating fully with authorities.",
            "expected": "negative"
        },
        {
            "text": "The new CEO outlined a comprehensive turnaround plan focusing on operational efficiency and debt reduction. Initial market response has been cautiously optimistic.",
            "expected": "neutral/positive"
        }
    ]
    
    results = []
    
    for i, sample in enumerate(sample_texts, 1):
        print(f"\n📝 Sample {i}: {sample['expected'].upper()}")
        print(f"Text: {sample['text'][:100]}...")
        
        try:
            # Analyze sentiment
            sentiment = analyze_sentiment(sample['text'])
            
            print(f"✅ Results:")
            print(f"   • Label: {sentiment['label'].upper()}")
            print(f"   • Compound Score: {sentiment['compound']:+.3f}")
            print(f"   • Confidence Scores:")
            print(f"     - Positive: {sentiment.get('positive', 0):.3f}")
            print(f"     - Negative: {sentiment.get('negative', 0):.3f}")
            print(f"     - Neutral: {sentiment.get('neutral', 0):.3f}")
            
            results.append({
                'sample_id': i,
                'expected': sample['expected'],
                'predicted': sentiment['label'],
                'compound': sentiment['compound'],
                'text_preview': sample['text'][:100] + "...",
                **sentiment
            })
            
        except Exception as e:
            print(f"❌ Error analyzing sentiment: {e}")
            continue
    
    # Summary
    if results:
        df = pd.DataFrame(results)
        print(f"\n📊 Summary:")
        print(f"   • Processed {len(results)} texts")
        print(f"   • Positive sentiment: {len(df[df['predicted'] == 'positive'])}")
        print(f"   • Negative sentiment: {len(df[df['predicted'] == 'negative'])}")
        print(f"   • Neutral sentiment: {len(df[df['predicted'] == 'neutral'])}")
        print(f"   • Average compound score: {df['compound'].mean():+.3f}")
        
        return df
    
    return None


def demo_price_data():
    """Demonstrate price data downloading and processing."""
    print("\n💰 Price Data Demo")
    print("=" * 50)
    
    # Use a small set of tickers for demo
    tickers = ["AAPL", "MSFT", "GOOGL"]
    
    # Download last 30 days of data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    print(f"📊 Downloading price data for {tickers}")
    print(f"   • Period: {start_date.date()} to {end_date.date()}")
    
    try:
        # Download price data
        price_data = download_price_data(
            tickers=tickers,
            start_date=start_date,
            end_date=end_date
        )
        
        if price_data:
            for ticker, data in price_data.items():
                if data is not None and not data.empty:
                    latest_price = data['close'].iloc[-1]
                    price_change = data['return_1d'].iloc[-1]
                    volatility = data['volatility_20d'].iloc[-1]
                    
                    print(f"✅ {ticker}:")
                    print(f"   • Latest Price: ${latest_price:.2f}")
                    print(f"   • 1-Day Return: {price_change:+.2%}")
                    print(f"   • 20-Day Volatility: {volatility:.1%}")
                    print(f"   • Data Points: {len(data)}")
                else:
                    print(f"❌ {ticker}: No data available")
            
            # Create combined DataFrame
            combined_data = pd.concat(price_data.values(), ignore_index=True)
            print(f"\n📈 Combined Dataset:")
            print(f"   • Total rows: {len(combined_data)}")
            print(f"   • Date range: {combined_data['date'].min()} to {combined_data['date'].max()}")
            print(f"   • Columns: {list(combined_data.columns)}")
            
            return combined_data
        else:
            print("❌ No price data downloaded")
            
    except Exception as e:
        print(f"❌ Error downloading price data: {e}")
        print("💡 This might be due to network issues or yfinance limitations")
    
    return None


def demo_feature_engineering():
    """Demonstrate simple feature engineering combining sentiment and price data."""
    print("\n🔧 Feature Engineering Demo")
    print("=" * 50)
    
    # Create sample feature data (simulated)
    np.random.seed(42)  # For reproducible results
    
    dates = pd.date_range(start='2024-01-01', end='2024-01-30', freq='D')
    tickers = ["AAPL", "MSFT", "GOOGL"]
    
    features = []
    
    for ticker in tickers:
        for date in dates:
            # Simulate sentiment features
            sentiment_score = np.random.normal(0, 0.1)  # Neutral with some variation
            novelty_score = np.random.exponential(0.3)  # Mostly low novelty with occasional spikes
            
            # Simulate aspect-based features
            guidance_sentiment = np.random.normal(sentiment_score, 0.05)
            liquidity_sentiment = np.random.normal(sentiment_score, 0.05)
            risk_sentiment = np.random.normal(sentiment_score - 0.1, 0.05)  # Slightly more negative
            
            # Calculate aspect shocks (sentiment * novelty)
            guidance_shock = guidance_sentiment * novelty_score
            liquidity_shock = liquidity_sentiment * novelty_score
            risk_shock = risk_sentiment * novelty_score
            
            # Simulate forward return (target variable)
            # Positive correlation with positive sentiment shocks
            base_return = np.random.normal(0.001, 0.015)  # ~0.1% daily return with 1.5% volatility
            sentiment_effect = (guidance_shock * 0.02 + liquidity_shock * 0.01 - risk_shock * 0.015)
            forward_return = base_return + sentiment_effect + np.random.normal(0, 0.01)
            
            feature_row = {
                'date': date,
                'ticker': ticker,
                'sentiment_score': sentiment_score,
                'novelty_score': novelty_score,
                'guidance_sentiment': guidance_sentiment,
                'liquidity_sentiment': liquidity_sentiment,
                'risk_sentiment': risk_sentiment,
                'guidance_shock': guidance_shock,
                'liquidity_shock': liquidity_shock,
                'risk_shock': risk_shock,
                'forward_return': forward_return
            }
            
            features.append(feature_row)
    
    df = pd.DataFrame(features)
    
    print(f"📊 Generated Feature Dataset:")
    print(f"   • Shape: {df.shape}")
    print(f"   • Tickers: {df['ticker'].unique()}")
    print(f"   • Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    
    # Feature statistics
    print(f"\n📈 Feature Statistics:")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    stats = df[numeric_cols].describe()
    
    for col in ['sentiment_score', 'novelty_score', 'guidance_shock', 'forward_return']:
        if col in stats.columns:
            print(f"   • {col}:")
            print(f"     - Mean: {stats.loc['mean', col]:+.4f}")
            print(f"     - Std:  {stats.loc['std', col]:.4f}")
            print(f"     - Range: [{stats.loc['min', col]:+.4f}, {stats.loc['max', col]:+.4f}]")
    
    # Simple correlation analysis
    print(f"\n🔗 Correlations with Forward Return:")
    correlations = df.corr()['forward_return'].sort_values(key=abs, ascending=False)
    
    for feature, corr in correlations.items():
        if feature != 'forward_return' and abs(corr) > 0.05:
            print(f"   • {feature}: {corr:+.3f}")
    
    return df


def demo_simple_prediction():
    """Demonstrate a simple prediction workflow."""
    print("\n🔮 Simple Prediction Demo")
    print("=" * 50)
    
    # Generate sample data
    features_df = demo_feature_engineering()
    
    # Simple linear model using scikit-learn
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score
    
    # Prepare features and target
    feature_cols = ['sentiment_score', 'novelty_score', 'guidance_shock', 'liquidity_shock', 'risk_shock']
    X = features_df[feature_cols]
    y = features_df['forward_return']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Evaluate
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"📊 Model Performance:")
    print(f"   • Training samples: {len(X_train)}")
    print(f"   • Test samples: {len(X_test)}")
    print(f"   • R² Score: {r2:.3f}")
    print(f"   • RMSE: {np.sqrt(mse):.4f}")
    print(f"   • Mean Absolute Error: {np.mean(np.abs(y_test - y_pred)):.4f}")
    
    # Feature importance
    print(f"\n🎯 Feature Coefficients:")
    for feature, coef in zip(feature_cols, model.coef_):
        print(f"   • {feature}: {coef:+.4f}")
    
    # Sample predictions
    print(f"\n🔍 Sample Predictions:")
    sample_indices = np.random.choice(len(X_test), 5, replace=False)
    
    for i, idx in enumerate(sample_indices):
        actual = y_test.iloc[idx]
        predicted = y_pred[idx]
        error = abs(actual - predicted)
        
        print(f"   {i+1}. Actual: {actual:+.3f}, Predicted: {predicted:+.3f}, Error: {error:.3f}")
    
    return model, X_test, y_test, y_pred


def main():
    """Run all demonstrations."""
    print("🎭 ABSSENT Pipeline Demonstration")
    print("=" * 60)
    print("This example showcases the core functionality of the ABSSENT package.")
    print("Note: Some features use simulated data for demonstration purposes.")
    
    try:
        # 1. Sentiment Analysis Demo
        sentiment_results = demo_sentiment_analysis()
        
        # 2. Price Data Demo (may not work without internet/API access)
        price_results = demo_price_data()
        
        # 3. Feature Engineering Demo
        feature_results = demo_feature_engineering()
        
        # 4. Simple Prediction Demo
        model, X_test, y_test, y_pred = demo_simple_prediction()
        
        print("\n🎉 Demo Completed Successfully!")
        print("\n📋 Next Steps:")
        print("   1. Install the package: pip install -e .")
        print("   2. Try the CLI: python -m abssent.cli demo --quick")
        print("   3. Download real data: python -m abssent.cli download --tickers AAPL MSFT")
        print("   4. Explore the full documentation and examples")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        print("💡 This might be due to missing dependencies or network issues.")
        print("   Try installing all requirements: pip install -r requirements.txt")


if __name__ == "__main__":
    main() 