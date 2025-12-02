"""
Backtest Engine

Runs backtests for trading strategies with realistic cost modeling.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from .position import PositionManager
from .metrics import PerformanceMetrics
from ..indicators.base import Indicator


class BacktestEngine:
    """Engine for backtesting trading strategies."""

    def __init__(
        self,
        initial_capital: float = 100000,
        commission: float = 0.001,
        slippage: float = 0.0005
    ):
        """
        Initialize backtest engine.

        Args:
            initial_capital: Starting capital
            commission: Commission rate (0.001 = 0.1%)
            slippage: Slippage rate (0.0005 = 0.05%)
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage

    def run(
        self,
        data: pd.DataFrame,
        entry_indicator: Indicator,
        exit_indicator: Optional[Indicator] = None,
        position_size: float = 1.0,
        use_stop_loss: bool = False,
        stop_loss_pct: float = 2.0
    ) -> Dict[str, Any]:
        """
        Run a backtest on historical data.

        Args:
            data: DataFrame with OHLCV data
            entry_indicator: Indicator for entry signals
            exit_indicator: Optional separate indicator for exits (if None, uses entry_indicator)
            position_size: Fraction of capital to use per trade
            use_stop_loss: Whether to use stop losses
            stop_loss_pct: Stop loss percentage

        Returns:
            Dictionary with backtest results and metrics
        """
        # Validate data
        if data.empty or len(data) < 100:
            return self._empty_result("Insufficient data")

        # Use exit indicator or default to entry indicator
        if exit_indicator is None:
            exit_indicator = entry_indicator

        # Calculate indicators
        try:
            data = entry_indicator.calculate(data)
            if exit_indicator != entry_indicator:
                data = exit_indicator.calculate(data)
        except Exception as e:
            return self._empty_result(f"Indicator calculation failed: {str(e)}")

        # Generate signals
        try:
            entry_signals = entry_indicator.generate_signals(data)
            exit_signals = exit_indicator.generate_signals(data)
        except Exception as e:
            return self._empty_result(f"Signal generation failed: {str(e)}")

        # Initialize position manager
        pm = PositionManager(
            initial_capital=self.initial_capital,
            commission=self.commission,
            slippage=self.slippage
        )

        # Run backtest
        for idx in range(len(data)):
            row = data.iloc[idx]
            date = row['date']
            price = row['Close']

            # Update equity curve
            pm.update_equity(date, price)

            # Check if we have a position
            if not pm.is_flat:
                position = pm.current_position

                # Check stop loss
                if use_stop_loss and position.should_stop_loss(price):
                    pm.close_position(date, price, reason='stop_loss')
                    continue

                # Check exit signal
                if exit_signals.iloc[idx] == -position.direction:
                    pm.close_position(date, price, reason='signal')
                    continue

            # Check entry signal (only if flat)
            else:
                signal = entry_signals.iloc[idx]

                if signal != 0:
                    # Calculate stop loss if enabled
                    stop_loss_price = None
                    if use_stop_loss:
                        if signal == 1:  # Long
                            stop_loss_price = price * (1 - stop_loss_pct / 100)
                        else:  # Short
                            stop_loss_price = price * (1 + stop_loss_pct / 100)

                    # Open position
                    pm.open_position(
                        symbol=data.iloc[0].get('symbol', 'UNKNOWN'),
                        date=date,
                        price=price,
                        direction=signal,
                        position_size=position_size,
                        stop_loss=stop_loss_price
                    )

        # Close any remaining position
        if not pm.is_flat:
            last_row = data.iloc[-1]
            pm.close_position(last_row['date'], last_row['Close'], reason='end')

        # Calculate metrics
        equity_curve = pm.get_equity_curve()
        trades = pm.get_trades()
        metrics = PerformanceMetrics.calculate_all(equity_curve, trades)

        return {
            'metrics': metrics,
            'equity_curve': equity_curve,
            'trades': trades,
            'entry_indicator': str(entry_indicator),
            'exit_indicator': str(exit_indicator),
            'success': True,
            'error': None
        }

    def _empty_result(self, error_msg: str) -> Dict[str, Any]:
        """Return empty result with error message."""
        return {
            'metrics': PerformanceMetrics._empty_metrics(),
            'equity_curve': pd.DataFrame(),
            'trades': pd.DataFrame(),
            'entry_indicator': None,
            'exit_indicator': None,
            'success': False,
            'error': error_msg
        }

    def run_multiple_strategies(
        self,
        data: pd.DataFrame,
        strategies: List[Dict[str, Any]],
        verbose: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Run multiple strategies and compare results.

        Args:
            data: DataFrame with OHLCV data
            strategies: List of strategy configurations
            verbose: Print progress

        Returns:
            List of results for each strategy
        """
        results = []

        for i, strategy_config in enumerate(strategies):
            if verbose:
                print(f"Running strategy {i+1}/{len(strategies)}...")

            result = self.run(data, **strategy_config)
            results.append(result)

        return results
