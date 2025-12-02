"""
Strategy Validator

Comprehensive validation to detect overfitting.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List
from ..genetic.chromosome import StrategyChromosome
from ..genetic.fitness import FitnessFunction
from ..data.data_loader import DataLoader


class StrategyValidator:
    """
    Validates strategies across multiple data splits to detect overfitting.

    Implements:
    - Train/Validation/Test evaluation
    - Walk-forward analysis
    - Degradation metrics
    - Consistency checks
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize validator.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.fitness_function = FitnessFunction(config)
        self.data_loader = DataLoader()

    def validate_full(
        self,
        strategy: StrategyChromosome,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        test_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Full validation across train/val/test sets.

        Args:
            strategy: Strategy to validate
            train_data: Training data
            val_data: Validation data
            test_data: Test data

        Returns:
            Validation report
        """
        # Evaluate on each dataset
        train_result = self.fitness_function.evaluate_detailed(strategy, train_data)
        val_result = self.fitness_function.evaluate_detailed(strategy, val_data)
        test_result = self.fitness_function.evaluate_detailed(strategy, test_data)

        # Calculate degradation
        train_fitness = train_result['fitness']
        val_fitness = val_result['fitness']
        test_fitness = test_result['fitness']

        train_to_val_degradation = self._calculate_degradation(train_fitness, val_fitness)
        val_to_test_degradation = self._calculate_degradation(val_fitness, test_fitness)
        train_to_test_degradation = self._calculate_degradation(train_fitness, test_fitness)

        # Overfitting flags
        is_overfit = (
            train_to_val_degradation > 50 or  # More than 50% degradation
            test_fitness < 0.1 or  # Very poor test performance
            (train_fitness > 0.5 and test_fitness < 0.2)  # Large gap
        )

        return {
            'strategy': strategy,
            'train': train_result,
            'validation': val_result,
            'test': test_result,
            'degradation': {
                'train_to_val': train_to_val_degradation,
                'val_to_test': val_to_test_degradation,
                'train_to_test': train_to_test_degradation
            },
            'is_overfit': is_overfit,
            'summary': self._create_summary(train_result, val_result, test_result)
        }

    def walk_forward_validation(
        self,
        strategy: StrategyChromosome,
        data: pd.DataFrame,
        train_months: int = 12,
        test_months: int = 3,
        step_months: int = 3
    ) -> Dict[str, Any]:
        """
        Walk-forward analysis (Anti-Overfitting Protection #2).

        Args:
            strategy: Strategy to validate
            data: Full dataset
            train_months: Training window size
            test_months: Testing window size
            step_months: Step size

        Returns:
            Walk-forward results
        """
        # Create splits
        splits = self.data_loader.create_walk_forward_splits(
            data,
            train_months,
            test_months,
            step_months
        )

        results = []

        for i, (train_df, test_df) in enumerate(splits):
            train_result = self.fitness_function.evaluate_detailed(strategy, train_df)
            test_result = self.fitness_function.evaluate_detailed(strategy, test_df)

            degradation = self._calculate_degradation(
                train_result['fitness'],
                test_result['fitness']
            )

            results.append({
                'period': i + 1,
                'train_start': train_df['date'].min(),
                'train_end': train_df['date'].max(),
                'test_start': test_df['date'].min(),
                'test_end': test_df['date'].max(),
                'train_fitness': train_result['fitness'],
                'test_fitness': test_result['fitness'],
                'degradation': degradation,
                'train_metrics': train_result['metrics'],
                'test_metrics': test_result['metrics']
            })

        # Calculate consistency metrics
        test_fitnesses = [r['test_fitness'] for r in results]
        consistency = {
            'mean_test_fitness': np.mean(test_fitnesses),
            'std_test_fitness': np.std(test_fitnesses),
            'min_test_fitness': np.min(test_fitnesses),
            'max_test_fitness': np.max(test_fitnesses),
            'consistency_score': 1.0 - (np.std(test_fitnesses) / (np.mean(test_fitnesses) + 1e-6))
        }

        # Check if consistent across periods
        is_consistent = (
            consistency['std_test_fitness'] < 0.2 and
            consistency['min_test_fitness'] > 0.1
        )

        return {
            'strategy': strategy,
            'periods': results,
            'consistency': consistency,
            'is_consistent': is_consistent,
            'num_periods': len(results)
        }

    def compare_strategies(
        self,
        strategies: List[StrategyChromosome],
        data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Compare multiple strategies on the same data.

        Args:
            strategies: List of strategies
            data: Evaluation data

        Returns:
            Comparison DataFrame
        """
        results = []

        for i, strategy in enumerate(strategies):
            result = self.fitness_function.evaluate_detailed(strategy, data)

            results.append({
                'rank': i + 1,
                'fitness': result['fitness'],
                'complexity': result['complexity'],
                'sharpe': result['metrics'].get('sharpe_ratio', 0),
                'return_pct': result['metrics'].get('total_return', 0),
                'max_dd_pct': result['metrics'].get('max_drawdown_pct', 0),
                'win_rate': result['metrics'].get('win_rate', 0),
                'num_trades': result['metrics'].get('num_trades', 0),
                'profit_factor': result['metrics'].get('profit_factor', 0),
                'entry_indicator': strategy.genes['entry_indicator_type'],
                'exit_indicator': strategy.genes['exit_indicator_type']
            })

        df = pd.DataFrame(results)
        df = df.sort_values('fitness', ascending=False).reset_index(drop=True)
        df['rank'] = df.index + 1

        return df

    def _calculate_degradation(self, train_fitness: float, test_fitness: float) -> float:
        """Calculate fitness degradation percentage."""
        if train_fitness == 0:
            return 0.0
        return ((train_fitness - test_fitness) / train_fitness) * 100

    def _create_summary(
        self,
        train_result: Dict,
        val_result: Dict,
        test_result: Dict
    ) -> Dict[str, Any]:
        """Create summary statistics."""
        return {
            'train_sharpe': train_result['metrics'].get('sharpe_ratio', 0),
            'val_sharpe': val_result['metrics'].get('sharpe_ratio', 0),
            'test_sharpe': test_result['metrics'].get('sharpe_ratio', 0),
            'train_return': train_result['metrics'].get('total_return', 0),
            'val_return': val_result['metrics'].get('total_return', 0),
            'test_return': test_result['metrics'].get('total_return', 0),
            'avg_sharpe': np.mean([
                train_result['metrics'].get('sharpe_ratio', 0),
                val_result['metrics'].get('sharpe_ratio', 0),
                test_result['metrics'].get('sharpe_ratio', 0)
            ])
        }

    def print_validation_report(self, validation_result: Dict[str, Any]):
        """Print formatted validation report."""
        print(f"\n{'='*70}")
        print(f"{'STRATEGY VALIDATION REPORT':^70}")
        print(f"{'='*70}")

        # Strategy info
        strategy = validation_result['strategy']
        print(f"\n📝 Strategy Configuration:")
        print(f"  Entry:  {strategy.genes['entry_indicator_type']} {strategy.genes['entry_params']}")
        print(f"  Exit:   {strategy.genes['exit_indicator_type']} {strategy.genes['exit_params']}")
        print(f"  Complexity: {strategy.get_complexity()}")

        # Performance across splits
        print(f"\n📊 Performance Across Data Splits:")
        print(f"{'Split':<15} {'Fitness':<12} {'Sharpe':<12} {'Return %':<12} {'Trades':<10}")
        print("-" * 70)

        for split_name in ['train', 'validation', 'test']:
            result = validation_result[split_name]
            metrics = result['metrics']
            print(f"{split_name.capitalize():<15} "
                  f"{result['fitness']:<12.4f} "
                  f"{metrics.get('sharpe_ratio', 0):<12.2f} "
                  f"{metrics.get('total_return', 0):<12.2f} "
                  f"{metrics.get('num_trades', 0):<10}")

        # Degradation
        print(f"\n⚠️  Fitness Degradation:")
        deg = validation_result['degradation']
        print(f"  Train → Validation:  {deg['train_to_val']:.1f}%")
        print(f"  Validation → Test:   {deg['val_to_test']:.1f}%")
        print(f"  Train → Test:        {deg['train_to_test']:.1f}%")

        # Overfitting assessment
        print(f"\n🔍 Overfitting Assessment:")
        if validation_result['is_overfit']:
            print(f"  ❌ WARNING: Potential overfitting detected!")
            print(f"     - High degradation or poor test performance")
            print(f"     - Strategy may not generalize well")
        else:
            print(f"  ✅ Strategy appears robust")
            print(f"     - Reasonable degradation levels")
            print(f"     - Consistent performance across splits")

        print(f"\n{'='*70}\n")
