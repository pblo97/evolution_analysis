"""
Technical Indicators Implementation

Based on academic evidence and best practices from literature.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
from .base import Indicator


class MomentumIndicator(Indicator):
    """
    Time-Series Momentum (Tier 1 Evidence)

    Based on: Moskowitz, Ooi & Pedersen (2012)
    "Time Series Momentum"

    Strong empirical evidence across asset classes.
    """

    def __init__(self, lookback: int = 50):
        """
        Initialize momentum indicator.

        Args:
            lookback: Lookback period in days (20-200 typical)
        """
        super().__init__('Momentum', {'lookback': lookback})
        self.lookback = lookback

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate momentum as price change over lookback period."""
        df = df.copy()
        df['momentum'] = df['Close'].pct_change(self.lookback)
        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate signals based on momentum.

        Buy when momentum > 0 (price higher than N days ago)
        Sell when momentum < 0 (price lower than N days ago)
        """
        signals = pd.Series(0, index=df.index)
        signals[df['momentum'] > 0] = 1   # Buy signal
        signals[df['momentum'] < 0] = -1  # Sell signal
        return signals


class RSIIndicator(Indicator):
    """
    Relative Strength Index (Tier 3 Evidence - Mixed)

    Classical overbought/oversold indicator.
    Evidence is mixed, but widely used in practice.
    """

    def __init__(self, period: int = 14, oversold: float = 30, overbought: float = 70):
        """
        Initialize RSI indicator.

        Args:
            period: RSI period (typically 14)
            oversold: Oversold threshold (typically 30)
            overbought: Overbought threshold (typically 70)
        """
        super().__init__('RSI', {
            'period': period,
            'oversold': oversold,
            'overbought': overbought
        })
        self.period = period
        self.oversold = oversold
        self.overbought = overbought

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate RSI."""
        df = df.copy()

        # Calculate price changes
        delta = df['Close'].diff()

        # Separate gains and losses
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)

        # Calculate average gains and losses
        avg_gains = gains.rolling(window=self.period, min_periods=self.period).mean()
        avg_losses = losses.rolling(window=self.period, min_periods=self.period).mean()

        # Calculate RS and RSI
        rs = avg_gains / avg_losses
        df['rsi'] = 100 - (100 / (1 + rs))

        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate signals based on RSI.

        Buy when RSI < oversold (oversold condition)
        Sell when RSI > overbought (overbought condition)
        """
        signals = pd.Series(0, index=df.index)
        signals[df['rsi'] < self.oversold] = 1   # Buy on oversold
        signals[df['rsi'] > self.overbought] = -1  # Sell on overbought
        return signals


class MACDIndicator(Indicator):
    """
    Moving Average Convergence Divergence (Tier 3 Evidence - Mixed)

    Momentum and trend-following indicator.
    Lagging indicator with mixed empirical evidence.
    """

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        """
        Initialize MACD indicator.

        Args:
            fast: Fast EMA period
            slow: Slow EMA period
            signal: Signal line EMA period
        """
        super().__init__('MACD', {
            'fast': fast,
            'slow': slow,
            'signal': signal
        })
        self.fast = fast
        self.slow = slow
        self.signal = signal

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate MACD."""
        df = df.copy()

        # Calculate EMAs
        ema_fast = df['Close'].ewm(span=self.fast, adjust=False).mean()
        ema_slow = df['Close'].ewm(span=self.slow, adjust=False).mean()

        # MACD line
        df['macd'] = ema_fast - ema_slow

        # Signal line
        df['macd_signal'] = df['macd'].ewm(span=self.signal, adjust=False).mean()

        # MACD histogram
        df['macd_hist'] = df['macd'] - df['macd_signal']

        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate signals based on MACD crossover.

        Buy when MACD crosses above signal line
        Sell when MACD crosses below signal line
        """
        signals = pd.Series(0, index=df.index)

        # Crossover detection
        macd_above = df['macd'] > df['macd_signal']
        macd_above_prev = df['macd'].shift(1) > df['macd_signal'].shift(1)

        # Buy on bullish crossover
        signals[macd_above & ~macd_above_prev] = 1

        # Sell on bearish crossover
        signals[~macd_above & macd_above_prev] = -1

        return signals


class BollingerBandsIndicator(Indicator):
    """
    Bollinger Bands (Tier 2 Evidence - Moderate)

    Volatility-based indicator for mean reversion.
    Works better in ranging markets.
    """

    def __init__(self, period: int = 20, std_dev: float = 2.0):
        """
        Initialize Bollinger Bands.

        Args:
            period: Moving average period
            std_dev: Number of standard deviations
        """
        super().__init__('BollingerBands', {
            'period': period,
            'std_dev': std_dev
        })
        self.period = period
        self.std_dev = std_dev

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Bollinger Bands."""
        df = df.copy()

        # Middle band (SMA)
        df['bb_middle'] = df['Close'].rolling(window=self.period).mean()

        # Standard deviation
        rolling_std = df['Close'].rolling(window=self.period).std()

        # Upper and lower bands
        df['bb_upper'] = df['bb_middle'] + (self.std_dev * rolling_std)
        df['bb_lower'] = df['bb_middle'] - (self.std_dev * rolling_std)

        # Band width (volatility measure)
        df['bb_width'] = df['bb_upper'] - df['bb_lower']

        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate signals based on Bollinger Bands.

        Buy when price touches lower band (oversold)
        Sell when price touches upper band (overbought)
        """
        signals = pd.Series(0, index=df.index)

        # Buy when price at or below lower band
        signals[df['Close'] <= df['bb_lower']] = 1

        # Sell when price at or above upper band
        signals[df['Close'] >= df['bb_upper']] = -1

        return signals


class ATRIndicator(Indicator):
    """
    Average True Range (Tier 2 Evidence - Strong for Risk Management)

    Volatility indicator primarily used for position sizing and stop losses.
    Well-established for risk management rather than signal generation.
    """

    def __init__(self, period: int = 14, multiplier: float = 2.0):
        """
        Initialize ATR indicator.

        Args:
            period: ATR period
            multiplier: ATR multiplier for stops
        """
        super().__init__('ATR', {
            'period': period,
            'multiplier': multiplier
        })
        self.period = period
        self.multiplier = multiplier

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate ATR."""
        df = df.copy()

        # True Range components
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())

        # True Range
        df['tr'] = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

        # Average True Range
        df['atr'] = df['tr'].rolling(window=self.period).mean()

        # ATR-based stop distance
        df['atr_stop_distance'] = df['atr'] * self.multiplier

        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        ATR doesn't generate entry signals directly.
        Used primarily for position sizing and stop losses.

        Returns neutral signals (0).
        """
        return pd.Series(0, index=df.index)


class EMACrossoverIndicator(Indicator):
    """
    EMA Crossover (Tier 2 Evidence - Moderate)

    Trend-following strategy using exponential moving averages.
    Works in trending markets, fails in ranging markets.
    """

    def __init__(self, fast: int = 50, slow: int = 200):
        """
        Initialize EMA crossover indicator.

        Args:
            fast: Fast EMA period
            slow: Slow EMA period
        """
        super().__init__('EMACrossover', {
            'fast': fast,
            'slow': slow
        })
        self.fast = fast
        self.slow = slow

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate EMAs."""
        df = df.copy()

        df['ema_fast'] = df['Close'].ewm(span=self.fast, adjust=False).mean()
        df['ema_slow'] = df['Close'].ewm(span=self.slow, adjust=False).mean()

        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate signals based on EMA crossover.

        Buy when fast EMA crosses above slow EMA (golden cross)
        Sell when fast EMA crosses below slow EMA (death cross)
        """
        signals = pd.Series(0, index=df.index)

        # Crossover detection
        fast_above = df['ema_fast'] > df['ema_slow']
        fast_above_prev = df['ema_fast'].shift(1) > df['ema_slow'].shift(1)

        # Buy on golden cross
        signals[fast_above & ~fast_above_prev] = 1

        # Sell on death cross
        signals[~fast_above & fast_above_prev] = -1

        return signals
