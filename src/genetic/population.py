"""
Population Management

Manages a population of strategy chromosomes.
"""

from typing import List, Dict, Any
from .chromosome import StrategyChromosome


class Population:
    """Manages a population of strategy chromosomes."""

    def __init__(self, size: int, config: Dict[str, Any]):
        """
        Initialize population.

        Args:
            size: Population size
            config: Configuration dictionary
        """
        self.size = size
        self.config = config
        self.individuals: List[StrategyChromosome] = []
        self.generation = 0

    def initialize_random(self):
        """Initialize population with random individuals."""
        self.individuals = [
            StrategyChromosome.random(self.config)
            for _ in range(self.size)
        ]
        self.generation = 0

    def add(self, chromosome: StrategyChromosome):
        """Add a chromosome to population."""
        self.individuals.append(chromosome)

    def remove(self, chromosome: StrategyChromosome):
        """Remove a chromosome from population."""
        self.individuals.remove(chromosome)

    def get_best(self, n: int = 1) -> List[StrategyChromosome]:
        """
        Get top N individuals by fitness.

        Args:
            n: Number of individuals to return

        Returns:
            List of best chromosomes
        """
        sorted_pop = sorted(self.individuals, key=lambda x: x.fitness, reverse=True)
        return sorted_pop[:n]

    def get_worst(self, n: int = 1) -> List[StrategyChromosome]:
        """
        Get bottom N individuals by fitness.

        Args:
            n: Number of individuals to return

        Returns:
            List of worst chromosomes
        """
        sorted_pop = sorted(self.individuals, key=lambda x: x.fitness)
        return sorted_pop[:n]

    def get_average_fitness(self) -> float:
        """Calculate average fitness of population."""
        if not self.individuals:
            return 0.0
        return sum(ind.fitness for ind in self.individuals) / len(self.individuals)

    def get_best_fitness(self) -> float:
        """Get best fitness in population."""
        if not self.individuals:
            return 0.0
        return max(ind.fitness for ind in self.individuals)

    def get_diversity(self) -> float:
        """
        Calculate population diversity.

        Diversity = number of unique entry indicator types / population size

        Returns:
            Diversity score (0-1)
        """
        if not self.individuals:
            return 0.0

        unique_entry_types = set(
            ind.genes['entry_indicator_type']
            for ind in self.individuals
        )

        return len(unique_entry_types) / len(StrategyChromosome.INDICATOR_TYPES)

    def get_statistics(self) -> Dict[str, float]:
        """Get population statistics."""
        if not self.individuals:
            return {
                'generation': self.generation,
                'size': 0,
                'best_fitness': 0.0,
                'avg_fitness': 0.0,
                'worst_fitness': 0.0,
                'diversity': 0.0
            }

        fitnesses = [ind.fitness for ind in self.individuals]

        return {
            'generation': self.generation,
            'size': len(self.individuals),
            'best_fitness': max(fitnesses),
            'avg_fitness': sum(fitnesses) / len(fitnesses),
            'worst_fitness': min(fitnesses),
            'diversity': self.get_diversity()
        }

    def __len__(self) -> int:
        """Return population size."""
        return len(self.individuals)

    def __iter__(self):
        """Iterate over individuals."""
        return iter(self.individuals)

    def __repr__(self) -> str:
        """String representation."""
        stats = self.get_statistics()
        return (f"Population(gen={stats['generation']}, "
                f"size={stats['size']}, "
                f"best={stats['best_fitness']:.3f}, "
                f"avg={stats['avg_fitness']:.3f})")
