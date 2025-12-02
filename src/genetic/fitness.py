"""
Fitness Function with Anti-Overfitting Protections

Multi-objective fitness function with complexity penalty.
"""

import pandas as pd
from typing import Dict, Any
from ..backtesting.engine import BacktestEngine
from .chromosome import StrategyChromosome


class FitnessFunction:
    """
    Multi-objective fitness function for trading strategies.

    Protections:
    1. Complexity penalty - Penalizes strategies with too many parameters
    2. Minimum trade threshold - Rejects strategies with too few trades
    3. Multi-objective - Balances return, risk, and robustness
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize fitness function.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.fitness_config = config.get('fitness', {})
        self.backtesting_config = config.get('backtesting', {})

        # Fitness weights (should sum to 1.0)
        self.sharpe_weight = self.fitness_config.get('sharpe_weight', 0.30)
        self.return_weight = self.fitness_config.get('total_return_weight', 0.30)
        self.drawdown_weight = self.fitness_config.get('max_drawdown_weight', 0.20)
        self.winrate_weight = self.fitness_config.get('win_rate_weight', 0.10)
        self.profit_factor_weight = self.fitness_config.get('profit_factor_weight', 0.10)

        # Complexity penalty (Anti-Overfitting Protection #3)
        self.complexity_penalty = config.get('genetic', {}).get('complexity_penalty', 0.05)

        # Minimum requirements
        self.min_trades = self.fitness_config.get('min_trades', 30)
        self.min_sharpe = self.fitness_config.get('min_sharpe', 0.5)

        # Backtest engine
        self.backtest_engine = BacktestEngine(
            initial_capital=self.backtesting_config.get('initial_capital', 100000),
            commission=self.backtesting_config.get('commission', 0.001),
            slippage=self.backtesting_config.get('slippage', 0.0005)
        )

    def evaluate(
        self,
        chromosome: StrategyChromosome,
        data: pd.DataFrame
    ) -> float:
        """
        Evaluate fitness of a strategy chromosome.

        Args:
            chromosome: Strategy to evaluate
            data: Historical price data for backtesting

        Returns:
            Fitness score (higher is better)
        """
        try:
            # Get indicators from chromosome
            entry_indicator = chromosome.get_entry_indicator()
            exit_indicator = chromosome.get_exit_indicator()

            # Run backtest
            result = self.backtest_engine.run(
                data=data.copy(),
                entry_indicator=entry_indicator,
                exit_indicator=exit_indicator,
                position_size=chromosome.genes['position_size'],
                use_stop_loss=chromosome.genes['use_stop_loss'],
                stop_loss_pct=chromosome.genes['stop_loss_pct']
            )

            if not result['success']:
                return 0.0

            metrics = result['metrics']

            # Check minimum requirements
            if metrics['num_trades'] < self.min_trades:
                # Not enough trades - severely penalize
                return 0.0

            # Calculate normalized components (0-1 scale)
            sharpe_score = self._normalize_sharpe(metrics['sharpe_ratio'])
            return_score = self._normalize_return(metrics['total_return'])
            drawdown_score = self._normalize_drawdown(metrics['max_drawdown_pct'])
            winrate_score = metrics['win_rate'] / 100  # Already 0-100
            profit_factor_score = self._normalize_profit_factor(metrics['profit_factor'])

            # Calculate weighted fitness
            raw_fitness = (
                self.sharpe_weight * sharpe_score +
                self.return_weight * return_score +
                self.drawdown_weight * drawdown_score +
                self.winrate_weight * winrate_score +
                self.profit_factor_weight * profit_factor_score
            )

            # Apply complexity penalty (Anti-Overfitting Protection #3)
            complexity = chromosome.get_complexity()
            complexity_penalty = self.complexity_penalty * complexity

            final_fitness = max(0, raw_fitness - complexity_penalty)

            return final_fitness

        except Exception as e:
            # If any error occurs, return 0 fitness
            print(f"Error evaluating chromosome: {str(e)}")
            return 0.0

    def _normalize_sharpe(self, sharpe: float) -> float:
        """
        Normalize Sharpe ratio to 0-1 scale.

        Sharpe > 3 is excellent, Sharpe < 0 is terrible.
        """
        if sharpe < 0:
            return 0.0
        elif sharpe > 3.0:
            return 1.0
        else:
            return sharpe / 3.0

    def _normalize_return(self, total_return_pct: float) -> float:
        """
        Normalize total return to 0-1 scale.

        Assuming 50% total return is excellent, 0% is neutral.
        Negative returns score 0.
        """
        if total_return_pct < 0:
            return 0.0
        elif total_return_pct > 50:
            return 1.0
        else:
            return total_return_pct / 50.0

    def _normalize_drawdown(self, max_drawdown_pct: float) -> float:
        """
        Normalize max drawdown to 0-1 scale.

        Lower drawdown is better. -20% or worse scores 0, 0% scores 1.
        """
        if max_drawdown_pct >= 0:
            return 1.0
        elif max_drawdown_pct <= -20:
            return 0.0
        else:
            # Convert to positive and normalize
            return 1.0 + (max_drawdown_pct / 20.0)

    def _normalize_profit_factor(self, profit_factor: float) -> float:
        """
        Normalize profit factor to 0-1 scale.

        PF > 2.0 is excellent, PF < 1.0 is losing.
        """
        if profit_factor < 1.0:
            return 0.0
        elif profit_factor > 2.0:
            return 1.0
        else:
            return (profit_factor - 1.0) / 1.0

    def evaluate_detailed(
        self,
        chromosome: StrategyChromosome,
        data: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Evaluate fitness with detailed breakdown.

        Args:
            chromosome: Strategy to evaluate
            data: Historical price data

        Returns:
            Dictionary with fitness components and metrics
        """
        try:
            entry_indicator = chromosome.get_entry_indicator()
            exit_indicator = chromosome.get_exit_indicator()

            result = self.backtest_engine.run(
                data=data.copy(),
                entry_indicator=entry_indicator,
                exit_indicator=exit_indicator,
                position_size=chromosome.genes['position_size'],
                use_stop_loss=chromosome.genes['use_stop_loss'],
                stop_loss_pct=chromosome.genes['stop_loss_pct']
            )

            if not result['success']:
                return {
                    'fitness': 0.0,
                    'error': result['error'],
                    'metrics': {}
                }

            metrics = result['metrics']

            # Check requirements
            meets_requirements = metrics['num_trades'] >= self.min_trades

            # Calculate components
            sharpe_score = self._normalize_sharpe(metrics['sharpe_ratio'])
            return_score = self._normalize_return(metrics['total_return'])
            drawdown_score = self._normalize_drawdown(metrics['max_drawdown_pct'])
            winrate_score = metrics['win_rate'] / 100
            profit_factor_score = self._normalize_profit_factor(metrics['profit_factor'])

            raw_fitness = (
                self.sharpe_weight * sharpe_score +
                self.return_weight * return_score +
                self.drawdown_weight * drawdown_score +
                self.winrate_weight * winrate_score +
                self.profit_factor_weight * profit_factor_score
            )

            complexity = chromosome.get_complexity()
            complexity_penalty_value = self.complexity_penalty * complexity
            final_fitness = max(0, raw_fitness - complexity_penalty_value) if meets_requirements else 0.0

            return {
                'fitness': final_fitness,
                'raw_fitness': raw_fitness,
                'complexity': complexity,
                'complexity_penalty': complexity_penalty_value,
                'meets_requirements': meets_requirements,
                'components': {
                    'sharpe_score': sharpe_score,
                    'return_score': return_score,
                    'drawdown_score': drawdown_score,
                    'winrate_score': winrate_score,
                    'profit_factor_score': profit_factor_score
                },
                'metrics': metrics,
                'backtest_result': result
            }

        except Exception as e:
            return {
                'fitness': 0.0,
                'error': str(e),
                'metrics': {}
            }
