"""
Data Loader and Preprocessor

Handles loading, cleaning, and splitting data for training/validation/testing.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional
from datetime import datetime, timedelta
import yaml


class DataLoader:
    """Load and preprocess financial data for backtesting."""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize data loader.

        Args:
            config_path: Path to configuration file
        """
        self.config = {}
        if config_path:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and validate data.

        Args:
            df: Raw price data

        Returns:
            Cleaned DataFrame
        """
        # Make a copy to avoid modifying original
        df = df.copy()

        # Remove duplicates
        df = df.drop_duplicates(subset=['date'])

        # Sort by date
        df = df.sort_values('date').reset_index(drop=True)

        # Check for missing values
        if df.isnull().any().any():
            print("Warning: Missing values detected. Forward filling...")
            df = df.fillna(method='ffill')

        # Validate OHLC relationship
        invalid_rows = (
            (df['High'] < df['Low']) |
            (df['High'] < df['Open']) |
            (df['High'] < df['Close']) |
            (df['Low'] > df['Open']) |
            (df['Low'] > df['Close'])
        )

        if invalid_rows.any():
            print(f"Warning: {invalid_rows.sum()} rows with invalid OHLC data. Removing...")
            df = df[~invalid_rows]

        # Check for zero or negative prices
        price_cols = ['Open', 'High', 'Low', 'Close']
        for col in price_cols:
            if col in df.columns:
                invalid = df[col] <= 0
                if invalid.any():
                    print(f"Warning: {invalid.sum()} rows with invalid {col} prices. Removing...")
                    df = df[~invalid]

        return df.reset_index(drop=True)

    def train_val_test_split(
        self,
        df: pd.DataFrame,
        train_ratio: float = 0.50,
        val_ratio: float = 0.25,
        test_ratio: float = 0.25
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train/validation/test sets (Anti-Overfitting Protection #1).

        Args:
            df: DataFrame with date column
            train_ratio: Proportion for training
            val_ratio: Proportion for validation
            test_ratio: Proportion for testing

        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        # Validate ratios
        total = train_ratio + val_ratio + test_ratio
        if not np.isclose(total, 1.0):
            raise ValueError(f"Ratios must sum to 1.0, got {total}")

        n = len(df)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)

        train_df = df.iloc[:train_end].copy()
        val_df = df.iloc[train_end:val_end].copy()
        test_df = df.iloc[val_end:].copy()

        print(f"\nData Split Summary:")
        print(f"Total rows: {n}")
        print(f"Train: {len(train_df)} rows ({len(train_df)/n*100:.1f}%) - "
              f"{train_df['date'].min()} to {train_df['date'].max()}")
        print(f"Validation: {len(val_df)} rows ({len(val_df)/n*100:.1f}%) - "
              f"{val_df['date'].min()} to {val_df['date'].max()}")
        print(f"Test: {len(test_df)} rows ({len(test_df)/n*100:.1f}%) - "
              f"{test_df['date'].min()} to {test_df['date'].max()}")

        return train_df, val_df, test_df

    def create_walk_forward_splits(
        self,
        df: pd.DataFrame,
        train_months: int = 12,
        test_months: int = 3,
        step_months: int = 3
    ) -> list[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Create walk-forward analysis splits (Anti-Overfitting Protection #2).

        This creates multiple train/test splits by walking forward through time,
        ensuring the strategy works across different market regimes.

        Args:
            df: DataFrame with date column
            train_months: Months of data for training
            test_months: Months of data for testing
            step_months: Months to step forward each iteration

        Returns:
            List of (train_df, test_df) tuples
        """
        df = df.sort_values('date').reset_index(drop=True)
        splits = []

        start_date = df['date'].min()
        end_date = df['date'].max()

        current_date = start_date

        while current_date < end_date:
            # Define train period
            train_end = current_date + pd.DateOffset(months=train_months)

            # Define test period
            test_start = train_end
            test_end = test_start + pd.DateOffset(months=test_months)

            # Extract data
            train_mask = (df['date'] >= current_date) & (df['date'] < train_end)
            test_mask = (df['date'] >= test_start) & (df['date'] < test_end)

            train_df = df[train_mask].copy()
            test_df = df[test_mask].copy()

            # Only add if both have sufficient data
            if len(train_df) > 30 and len(test_df) > 10:
                splits.append((train_df, test_df))

            # Step forward
            current_date += pd.DateOffset(months=step_months)

            # Stop if test period exceeds available data
            if test_end >= end_date:
                break

        print(f"\nWalk-Forward Analysis: Created {len(splits)} splits")
        for i, (train, test) in enumerate(splits):
            print(f"Split {i+1}: Train {train['date'].min()} to {train['date'].max()} | "
                  f"Test {test['date'].min()} to {test['date'].max()}")

        return splits

    def add_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add return columns to DataFrame.

        Args:
            df: DataFrame with Close prices

        Returns:
            DataFrame with added return columns
        """
        df = df.copy()

        # Simple returns
        df['returns'] = df['Close'].pct_change()

        # Log returns
        df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))

        return df

    def get_date_range_data(
        self,
        df: pd.DataFrame,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Get data for a specific date range.

        Args:
            df: DataFrame with date column
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            Filtered DataFrame
        """
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)

        mask = (df['date'] >= start) & (df['date'] <= end)
        return df[mask].copy()
