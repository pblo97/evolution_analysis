"""
Chromosome Representation for Trading Strategies

Encodes trading strategies as genes that can be evolved.
"""

import random
from typing import Dict, Any, List
from dataclasses import dataclass
from ..indicators.technical import (
    MomentumIndicator,
    RSIIndicator,
    MACDIndicator,
    BollingerBandsIndicator,
    ATRIndicator,
    EMACrossoverIndicator
)


@dataclass
class Chromosome:
    """Base class for chromosomes."""

    genes: Dict[str, Any]
    fitness: float = 0.0

    def __repr__(self) -> str:
        return f"Chromosome(fitness={self.fitness:.3f}, genes={self.genes})"


class StrategyChromosome(Chromosome):
    """
    Chromosome representing a trading strategy.

    Gene Structure:
        - entry_indicator_type: Type of entry indicator
        - entry_params: Parameters for entry indicator
        - exit_indicator_type: Type of exit indicator
        - exit_params: Parameters for exit indicator
        - position_size: Fraction of capital to use
        - use_stop_loss: Whether to use stop loss
        - stop_loss_pct: Stop loss percentage
    """

    INDICATOR_TYPES = [
        'momentum',
        'rsi',
        'macd',
        'bollinger',
        'ema_cross'
    ]

    def __init__(self, genes: Dict[str, Any]):
        """Initialize strategy chromosome."""
        super().__init__(genes)
        self.validate_genes()

    def validate_genes(self):
        """Validate gene structure."""
        required_keys = [
            'entry_indicator_type',
            'entry_params',
            'exit_indicator_type',
            'exit_params',
            'position_size',
            'use_stop_loss',
            'stop_loss_pct'
        ]

        for key in required_keys:
            if key not in self.genes:
                raise ValueError(f"Missing required gene: {key}")

    @classmethod
    def random(cls, config: Dict[str, Any]) -> 'StrategyChromosome':
        """
        Create a random strategy chromosome based on configuration.

        Args:
            config: Configuration dictionary with parameter ranges

        Returns:
            Random strategy chromosome
        """
        strategy_config = config.get('strategy', {})
        allowed_indicators = strategy_config.get('indicators', cls.INDICATOR_TYPES)
        param_ranges = strategy_config.get('parameters', {})

        # Random entry indicator
        entry_type = random.choice(allowed_indicators)
        entry_params = cls._random_params(entry_type, param_ranges)

        # Random exit indicator (can be different or same)
        exit_type = random.choice(allowed_indicators)
        exit_params = cls._random_params(exit_type, param_ranges)

        # Random trading parameters
        genes = {
            'entry_indicator_type': entry_type,
            'entry_params': entry_params,
            'exit_indicator_type': exit_type,
            'exit_params': exit_params,
            'position_size': random.uniform(0.8, 1.0),  # 80-100% of capital
            'use_stop_loss': random.choice([True, False]),
            'stop_loss_pct': random.uniform(1.0, 5.0)  # 1-5% stop loss
        }

        return cls(genes)

    @staticmethod
    def _random_params(indicator_type: str, param_ranges: Dict[str, Any]) -> Dict[str, Any]:
        """Generate random parameters for an indicator type."""
        ranges = param_ranges.get(indicator_type, {})

        if indicator_type == 'momentum':
            return {
                'lookback': random.randint(
                    ranges.get('lookback_min', 20),
                    ranges.get('lookback_max', 200)
                )
            }

        elif indicator_type == 'rsi':
            return {
                'period': random.randint(
                    ranges.get('period_min', 10),
                    ranges.get('period_max', 20)
                ),
                'oversold': random.uniform(
                    ranges.get('oversold_min', 25),
                    ranges.get('oversold_max', 35)
                ),
                'overbought': random.uniform(
                    ranges.get('overbought_min', 65),
                    ranges.get('overbought_max', 75)
                )
            }

        elif indicator_type == 'macd':
            fast = random.randint(ranges.get('fast_min', 8), ranges.get('fast_max', 16))
            slow = random.randint(ranges.get('slow_min', 20), ranges.get('slow_max', 30))
            return {
                'fast': fast,
                'slow': max(slow, fast + 5),  # Ensure slow > fast
                'signal': random.randint(
                    ranges.get('signal_min', 7),
                    ranges.get('signal_max', 11)
                )
            }

        elif indicator_type == 'bollinger':
            return {
                'period': random.randint(
                    ranges.get('period_min', 15),
                    ranges.get('period_max', 25)
                ),
                'std_dev': random.uniform(
                    ranges.get('std_min', 1.5),
                    ranges.get('std_max', 2.5)
                )
            }

        elif indicator_type == 'atr':
            return {
                'period': random.randint(
                    ranges.get('period_min', 10),
                    ranges.get('period_max', 20)
                ),
                'multiplier': random.uniform(
                    ranges.get('multiplier_min', 1.5),
                    ranges.get('multiplier_max', 3.0)
                )
            }

        elif indicator_type == 'ema_cross':
            fast = random.randint(ranges.get('fast_min', 20), ranges.get('fast_max', 50))
            slow = random.randint(ranges.get('slow_min', 100), ranges.get('slow_max', 200))
            return {
                'fast': fast,
                'slow': max(slow, fast + 20)  # Ensure slow > fast significantly
            }

        else:
            return {}

    def get_entry_indicator(self):
        """Create entry indicator instance from genes."""
        return self._create_indicator(
            self.genes['entry_indicator_type'],
            self.genes['entry_params']
        )

    def get_exit_indicator(self):
        """Create exit indicator instance from genes."""
        return self._create_indicator(
            self.genes['exit_indicator_type'],
            self.genes['exit_params']
        )

    @staticmethod
    def _create_indicator(indicator_type: str, params: Dict[str, Any]):
        """Create indicator instance from type and parameters."""
        if indicator_type == 'momentum':
            return MomentumIndicator(**params)
        elif indicator_type == 'rsi':
            return RSIIndicator(**params)
        elif indicator_type == 'macd':
            return MACDIndicator(**params)
        elif indicator_type == 'bollinger':
            return BollingerBandsIndicator(**params)
        elif indicator_type == 'atr':
            return ATRIndicator(**params)
        elif indicator_type == 'ema_cross':
            return EMACrossoverIndicator(**params)
        else:
            raise ValueError(f"Unknown indicator type: {indicator_type}")

    def get_complexity(self) -> int:
        """
        Calculate strategy complexity (for penalty).

        Complexity = number of indicators + total parameters
        """
        complexity = 0

        # Count indicators (entry + exit, but if same type, count as 1.5)
        if self.genes['entry_indicator_type'] == self.genes['exit_indicator_type']:
            complexity += 1.5
        else:
            complexity += 2

        # Count parameters
        complexity += len(self.genes['entry_params'])
        complexity += len(self.genes['exit_params'])

        return complexity

    def mutate(self, mutation_rate: float, config: Dict[str, Any]):
        """
        Mutate chromosome genes.

        Args:
            mutation_rate: Probability of mutation per gene
            config: Configuration for parameter ranges
        """
        # Mutate entry indicator
        if random.random() < mutation_rate:
            self.genes['entry_indicator_type'] = random.choice(self.INDICATOR_TYPES)
            self.genes['entry_params'] = self._random_params(
                self.genes['entry_indicator_type'],
                config.get('strategy', {}).get('parameters', {})
            )

        # Mutate exit indicator
        if random.random() < mutation_rate:
            self.genes['exit_indicator_type'] = random.choice(self.INDICATOR_TYPES)
            self.genes['exit_params'] = self._random_params(
                self.genes['exit_indicator_type'],
                config.get('strategy', {}).get('parameters', {})
            )

        # Mutate position size
        if random.random() < mutation_rate:
            self.genes['position_size'] = random.uniform(0.8, 1.0)

        # Mutate stop loss
        if random.random() < mutation_rate:
            self.genes['use_stop_loss'] = random.choice([True, False])

        if random.random() < mutation_rate:
            self.genes['stop_loss_pct'] = random.uniform(1.0, 5.0)

    def copy(self) -> 'StrategyChromosome':
        """Create a deep copy of the chromosome."""
        import copy
        return StrategyChromosome(copy.deepcopy(self.genes))

    def to_dict(self) -> Dict[str, Any]:
        """Convert chromosome to dictionary for serialization."""
        return {
            'genes': self.genes,
            'fitness': self.fitness,
            'complexity': self.get_complexity()
        }
