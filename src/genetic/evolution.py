"""
Genetic Algorithm Engine

Main evolution loop with all anti-overfitting protections.
"""

import pandas as pd
from typing import Dict, Any, List, Optional
from tqdm import tqdm
from .population import Population
from .chromosome import StrategyChromosome
from .operators import GeneticOperators
from .fitness import FitnessFunction


class GeneticAlgorithm:
    """
    Genetic algorithm for evolving trading strategies.

    Implements:
    - Tournament selection
    - Uniform crossover
    - Adaptive mutation
    - Elitism
    - Complexity penalty
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize genetic algorithm.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.ga_config = config.get('genetic', {})

        # GA parameters
        self.population_size = self.ga_config.get('population_size', 100)
        self.generations = self.ga_config.get('generations', 50)
        self.mutation_rate = self.ga_config.get('mutation_rate', 0.15)
        self.crossover_rate = self.ga_config.get('crossover_rate', 0.7)
        self.elitism_rate = self.ga_config.get('elitism', 0.1)
        self.tournament_size = self.ga_config.get('tournament_size', 5)

        # Components
        self.population = Population(self.population_size, config)
        self.fitness_function = FitnessFunction(config)
        self.operators = GeneticOperators()

        # Evolution tracking
        self.history = []
        self.best_ever: Optional[StrategyChromosome] = None

    def evolve(
        self,
        train_data: pd.DataFrame,
        verbose: bool = True
    ) -> StrategyChromosome:
        """
        Run the genetic algorithm evolution.

        Args:
            train_data: Training data for fitness evaluation
            verbose: Print progress

        Returns:
            Best strategy found
        """
        if verbose:
            print(f"\n🧬 Starting Genetic Algorithm Evolution")
            print(f"Population: {self.population_size}, Generations: {self.generations}")
            print(f"Mutation Rate: {self.mutation_rate}, Crossover Rate: {self.crossover_rate}")
            print(f"=" * 60)

        # Initialize random population
        self.population.initialize_random()

        # Evolution loop
        for generation in range(self.generations):
            self.population.generation = generation

            # Evaluate fitness for all individuals
            if verbose:
                print(f"\nGeneration {generation + 1}/{self.generations}")
                print("Evaluating fitness...")

            self._evaluate_population(train_data, verbose)

            # Track statistics
            stats = self.population.get_statistics()
            self.history.append(stats)

            # Update best ever
            best_this_gen = self.population.get_best(1)[0]
            if self.best_ever is None or best_this_gen.fitness > self.best_ever.fitness:
                self.best_ever = best_this_gen.copy()

            if verbose:
                print(f"Best Fitness: {stats['best_fitness']:.4f}")
                print(f"Avg Fitness:  {stats['avg_fitness']:.4f}")
                print(f"Diversity:    {stats['diversity']:.2f}")
                print(f"Best Ever:    {self.best_ever.fitness:.4f}")

            # Create next generation
            if generation < self.generations - 1:
                self._create_next_generation()

        if verbose:
            print(f"\n{'=' * 60}")
            print(f"✅ Evolution Complete!")
            print(f"Best Fitness: {self.best_ever.fitness:.4f}")
            print(f"{'=' * 60}\n")

        return self.best_ever

    def _evaluate_population(self, data: pd.DataFrame, verbose: bool = True):
        """Evaluate fitness with fitness sharing to maintain diversity."""
        if verbose:
            iterator = tqdm(self.population.individuals, desc="Evaluating")
        else:
            iterator = self.population.individuals

        # First pass: calculate base fitness
        for individual in iterator:
            fitness = self.fitness_function.evaluate(individual, data)
            individual.fitness = fitness

        # Second pass: Fitness Sharing - heavily penalize similar strategies
        # This prevents convergence to a single solution
        indicator_counts = {}
        for individual in self.population.individuals:
            entry_type = individual.genes['entry_indicator_type']
            exit_type = individual.genes['exit_indicator_type']
            combo = f"{entry_type}+{exit_type}"
            indicator_counts[combo] = indicator_counts.get(combo, 0) + 1

        # Apply fitness sharing: divide fitness by square root of niche size
        # This maintains diversity while still allowing good strategies to survive
        for individual in self.population.individuals:
            if individual.fitness > 0:  # Only apply to valid strategies
                entry_type = individual.genes['entry_indicator_type']
                exit_type = individual.genes['exit_indicator_type']
                combo = f"{entry_type}+{exit_type}"

                niche_count = indicator_counts[combo]

                # Use square root to avoid overly harsh penalty
                # If 100 strategies use same combo, fitness is divided by 10
                import math
                sharing_factor = math.sqrt(niche_count)
                individual.fitness = individual.fitness / max(1.0, sharing_factor)

    def _create_next_generation(self):
        """Create next generation using genetic operators."""
        new_population = []

        # Elitism: Keep best individuals
        elite_size = int(self.population_size * self.elitism_rate)
        if elite_size > 0:
            elites = self.operators.elitism_selection(
                self.population.individuals,
                elite_size
            )
            new_population.extend(elites)

        # Fill rest with offspring
        while len(new_population) < self.population_size:
            # Selection
            parent1 = self.operators.tournament_selection(
                self.population.individuals,
                self.tournament_size
            )
            parent2 = self.operators.tournament_selection(
                self.population.individuals,
                self.tournament_size
            )

            # Crossover
            child1, child2 = self.operators.crossover(
                parent1,
                parent2,
                self.crossover_rate
            )

            # Mutation
            child1 = self.operators.mutate(child1, self.mutation_rate, self.config)
            child2 = self.operators.mutate(child2, self.mutation_rate, self.config)

            # Add to new population
            new_population.append(child1)
            if len(new_population) < self.population_size:
                new_population.append(child2)

        # Replace old population
        self.population.individuals = new_population[:self.population_size]

    def get_evolution_history(self) -> pd.DataFrame:
        """Get evolution history as DataFrame."""
        return pd.DataFrame(self.history)

    def get_best_strategies(self, n: int = 10) -> List[StrategyChromosome]:
        """
        Get top N strategies from final population.

        Args:
            n: Number of strategies to return

        Returns:
            List of best strategies
        """
        return self.population.get_best(n)

    def evaluate_on_validation(
        self,
        strategies: List[StrategyChromosome],
        validation_data: pd.DataFrame,
        verbose: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Evaluate strategies on validation data (Anti-Overfitting Check).

        Args:
            strategies: List of strategies to evaluate
            validation_data: Validation data
            verbose: Print progress

        Returns:
            List of detailed evaluation results
        """
        if verbose:
            print(f"\n🔍 Validating {len(strategies)} strategies on validation data...")

        results = []

        iterator = tqdm(strategies, desc="Validating") if verbose else strategies

        for i, strategy in enumerate(iterator):
            result = self.fitness_function.evaluate_detailed(strategy, validation_data)
            result['strategy_rank'] = i + 1
            result['strategy'] = strategy
            results.append(result)

        # Sort by validation fitness
        results.sort(key=lambda x: x['fitness'], reverse=True)

        if verbose:
            print(f"\n📊 Validation Results:")
            print(f"{'Rank':<6} {'Train Fit':<12} {'Val Fit':<12} {'Degradation':<12}")
            print("-" * 45)
            for i, result in enumerate(results[:10]):
                train_fit = result['strategy'].fitness
                val_fit = result['fitness']
                degradation = ((train_fit - val_fit) / train_fit * 100) if train_fit > 0 else 0
                print(f"{i+1:<6} {train_fit:<12.4f} {val_fit:<12.4f} {degradation:<12.1f}%")

        return results
