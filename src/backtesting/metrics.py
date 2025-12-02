"""
Performance Metrics Calculation

Calculates key metrics for evaluating trading strategies.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any


class PerformanceMetrics:
    """Calculate trading strategy performance metrics."""

    @staticmethod
    def calculate_all(
        equity_curve: pd.DataFrame,
        trades: pd.DataFrame,
        risk_free_rate: float = 0.02
    ) -> Dict[str, Any]:
        """
        Calculate all performance metrics.

        Args:
            equity_curve: DataFrame with date and equity columns
            trades: DataFrame with trade records
            risk_free_rate: Annual risk-free rate for Sharpe calculation

        Returns:
            Dictionary of performance metrics
        """
        if equity_curve.empty or len(equity_curve) < 2:
            return PerformanceMetrics._empty_metrics()

        metrics = {}

        # Basic metrics
        initial_equity = equity_curve['equity'].iloc[0]
        final_equity = equity_curve['equity'].iloc[-1]

        metrics['initial_capital'] = initial_equity
        metrics['final_equity'] = final_equity
        metrics['total_return'] = ((final_equity / initial_equity) - 1) * 100
        metrics['total_return_abs'] = final_equity - initial_equity

        # Trade statistics
        if not trades.empty:
            metrics.update(PerformanceMetrics._calculate_trade_metrics(trades))
        else:
            metrics.update({
                'num_trades': 0,
                'win_rate': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'profit_factor': 0,
                'avg_trade_return': 0
            })

        # Risk metrics
        metrics.update(PerformanceMetrics._calculate_risk_metrics(equity_curve, risk_free_rate))

        return metrics

    @staticmethod
    def _calculate_trade_metrics(trades: pd.DataFrame) -> Dict[str, float]:
        """Calculate trade-based metrics."""
        metrics = {}

        metrics['num_trades'] = len(trades)

        # Win/Loss statistics
        winning_trades = trades[trades['net_pnl'] > 0]
        losing_trades = trades[trades['net_pnl'] < 0]

        metrics['num_winners'] = len(winning_trades)
        metrics['num_losers'] = len(losing_trades)
        metrics['win_rate'] = (len(winning_trades) / len(trades)) * 100 if len(trades) > 0 else 0

        # Average win/loss
        metrics['avg_win'] = winning_trades['net_pnl'].mean() if len(winning_trades) > 0 else 0
        metrics['avg_loss'] = losing_trades['net_pnl'].mean() if len(losing_trades) > 0 else 0

        # Profit factor (gross profit / gross loss)
        gross_profit = winning_trades['net_pnl'].sum() if len(winning_trades) > 0 else 0
        gross_loss = abs(losing_trades['net_pnl'].sum()) if len(losing_trades) > 0 else 0
        metrics['profit_factor'] = gross_profit / gross_loss if gross_loss > 0 else 0

        # Average trade
        metrics['avg_trade_return'] = trades['return_pct'].mean()
        metrics['best_trade'] = trades['net_pnl'].max()
        metrics['worst_trade'] = trades['net_pnl'].min()

        # Trading costs
        metrics['total_costs'] = trades['cost'].sum()

        return metrics

    @staticmethod
    def _calculate_risk_metrics(equity_curve: pd.DataFrame, risk_free_rate: float) -> Dict[str, float]:
        """Calculate risk-adjusted metrics."""
        metrics = {}

        # Calculate returns
        equity_curve = equity_curve.copy()
        equity_curve['returns'] = equity_curve['equity'].pct_change()

        # Drop NaN values
        returns = equity_curve['returns'].dropna()

        if len(returns) == 0:
            return {
                'sharpe_ratio': 0,
                'sortino_ratio': 0,
                'max_drawdown': 0,
                'max_drawdown_pct': 0,
                'calmar_ratio': 0,
                'volatility': 0
            }

        # Volatility (annualized)
        # Assuming daily data, multiply by sqrt(252)
        metrics['volatility'] = returns.std() * np.sqrt(252) * 100

        # Sharpe Ratio (annualized)
        # (Return - Risk Free Rate) / Volatility
        mean_return = returns.mean() * 252  # Annualized
        std_return = returns.std() * np.sqrt(252)
        metrics['sharpe_ratio'] = (mean_return - risk_free_rate) / std_return if std_return > 0 else 0

        # Sortino Ratio (uses downside deviation)
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252)
        metrics['sortino_ratio'] = (mean_return - risk_free_rate) / downside_std if downside_std > 0 else 0

        # Maximum Drawdown
        equity = equity_curve['equity']
        running_max = equity.expanding().max()
        drawdown = (equity - running_max) / running_max * 100
        metrics['max_drawdown_pct'] = drawdown.min()
        metrics['max_drawdown'] = (equity - running_max).min()

        # Calmar Ratio (Annual Return / Max Drawdown)
        total_return_pct = ((equity.iloc[-1] / equity.iloc[0]) - 1) * 100
        metrics['calmar_ratio'] = abs(total_return_pct / metrics['max_drawdown_pct']) \
            if metrics['max_drawdown_pct'] < 0 else 0

        # Recovery time (days in drawdown)
        in_drawdown = drawdown < -1  # More than 1% drawdown
        metrics['days_in_drawdown'] = in_drawdown.sum()
        metrics['pct_time_in_drawdown'] = (in_drawdown.sum() / len(equity)) * 100

        return metrics

    @staticmethod
    def _empty_metrics() -> Dict[str, Any]:
        """Return empty metrics dictionary."""
        return {
            'initial_capital': 0,
            'final_equity': 0,
            'total_return': 0,
            'total_return_abs': 0,
            'num_trades': 0,
            'win_rate': 0,
            'profit_factor': 0,
            'sharpe_ratio': 0,
            'max_drawdown_pct': 0,
            'avg_trade_return': 0
        }

    @staticmethod
    def print_metrics(metrics: Dict[str, Any], title: str = "Performance Metrics"):
        """Pretty print metrics."""
        print(f"\n{'='*60}")
        print(f"{title:^60}")
        print(f"{'='*60}")

        print(f"\n📊 RETURNS")
        print(f"  Initial Capital:    ${metrics['initial_capital']:,.2f}")
        print(f"  Final Equity:       ${metrics['final_equity']:,.2f}")
        print(f"  Total Return:       {metrics['total_return']:,.2f}%")
        print(f"  Absolute Profit:    ${metrics['total_return_abs']:,.2f}")

        print(f"\n📈 TRADE STATISTICS")
        print(f"  Total Trades:       {metrics['num_trades']}")
        print(f"  Win Rate:           {metrics['win_rate']:.2f}%")
        print(f"  Profit Factor:      {metrics['profit_factor']:.2f}")
        if metrics['num_trades'] > 0:
            print(f"  Avg Win:            ${metrics.get('avg_win', 0):,.2f}")
            print(f"  Avg Loss:           ${metrics.get('avg_loss', 0):,.2f}")
            print(f"  Best Trade:         ${metrics.get('best_trade', 0):,.2f}")
            print(f"  Worst Trade:        ${metrics.get('worst_trade', 0):,.2f}")

        print(f"\n⚠️  RISK METRICS")
        print(f"  Sharpe Ratio:       {metrics['sharpe_ratio']:.2f}")
        print(f"  Sortino Ratio:      {metrics.get('sortino_ratio', 0):.2f}")
        print(f"  Max Drawdown:       {metrics['max_drawdown_pct']:.2f}%")
        print(f"  Calmar Ratio:       {metrics.get('calmar_ratio', 0):.2f}")
        print(f"  Volatility:         {metrics.get('volatility', 0):.2f}%")

        print(f"\n{'='*60}\n")
