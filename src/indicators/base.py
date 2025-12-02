"""
Base classes for technical indicators.
"""

from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Any, Tuple


class Indicator(ABC):
    """Base class for all technical indicators."""

    def __init__(self, name: str, parameters: Dict[str, Any]):
        """
        Initialize indicator.

        Args:
            name: Indicator name
            parameters: Dictionary of indicator parameters
        """
        self.name = name
        self.parameters = parameters

    @abstractmethod
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate indicator values and add to DataFrame.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with indicator columns added
        """
        pass

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate buy/sell signals based on indicator.

        Args:
            df: DataFrame with indicator values

        Returns:
            Series with signals: 1 (buy), -1 (sell), 0 (hold)
        """
        pass

    def get_complexity(self) -> int:
        """
        Get indicator complexity (number of parameters).

        Used for complexity penalty in fitness function.

        Returns:
            Number of tunable parameters
        """
        return len(self.parameters)

    def __repr__(self) -> str:
        """String representation."""
        params_str = ', '.join(f"{k}={v}" for k, v in self.parameters.items())
        return f"{self.name}({params_str})"
