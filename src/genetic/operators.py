"""
Genetic Operators

Selection, crossover, and mutation operations for genetic algorithm.
"""

import random
from typing import List, Tuple
from .chromosome import StrategyChromosome


class GeneticOperators:
    """Genetic algorithm operators."""

    @staticmethod
    def tournament_selection(
        population: List[StrategyChromosome],
        tournament_size: int = 5
    ) -> StrategyChromosome:
        """
        Select individual using tournament selection.

        Args:
            population: List of chromosomes
            tournament_size: Number of individuals in tournament

        Returns:
            Selected chromosome
        """
        # Randomly select tournament_size individuals
        tournament = random.sample(population, min(tournament_size, len(population)))

        # Return the best one
        return max(tournament, key=lambda x: x.fitness)

    @staticmethod
    def crossover(
        parent1: StrategyChromosome,
        parent2: StrategyChromosome,
        crossover_rate: float = 0.7
    ) -> Tuple[StrategyChromosome, StrategyChromosome]:
        """
        Perform crossover between two parents.

        Uses uniform crossover: each gene has 50% chance from each parent.

        Args:
            parent1: First parent
            parent2: Second parent
            crossover_rate: Probability of crossover occurring

        Returns:
            Tuple of two offspring
        """
        # If no crossover, return copies of parents
        if random.random() > crossover_rate:
            return parent1.copy(), parent2.copy()

        # Create offspring with mixed genes
        child1_genes = {}
        child2_genes = {}

        # For each gene, randomly assign from parents
        for key in parent1.genes.keys():
            if random.random() < 0.5:
                child1_genes[key] = parent1.genes[key]
                child2_genes[key] = parent2.genes[key]
            else:
                child1_genes[key] = parent2.genes[key]
                child2_genes[key] = parent1.genes[key]

        child1 = StrategyChromosome(child1_genes)
        child2 = StrategyChromosome(child2_genes)

        return child1, child2

    @staticmethod
    def mutate(
        chromosome: StrategyChromosome,
        mutation_rate: float,
        config: dict
    ) -> StrategyChromosome:
        """
        Mutate a chromosome.

        Args:
            chromosome: Chromosome to mutate
            mutation_rate: Probability of mutation per gene
            config: Configuration for parameter ranges

        Returns:
            Mutated chromosome (modifies in place and returns)
        """
        chromosome.mutate(mutation_rate, config)
        return chromosome

    @staticmethod
    def elitism_selection(
        population: List[StrategyChromosome],
        elite_size: int
    ) -> List[StrategyChromosome]:
        """
        Select top N individuals (elitism).

        Args:
            population: List of chromosomes
            elite_size: Number of elite individuals to keep

        Returns:
            List of elite chromosomes
        """
        # Sort by fitness (descending)
        sorted_pop = sorted(population, key=lambda x: x.fitness, reverse=True)

        # Return top elite_size
        return [chrom.copy() for chrom in sorted_pop[:elite_size]]

    @staticmethod
    def roulette_wheel_selection(
        population: List[StrategyChromosome]
    ) -> StrategyChromosome:
        """
        Select individual using roulette wheel selection (fitness proportionate).

        Args:
            population: List of chromosomes

        Returns:
            Selected chromosome
        """
        # Calculate total fitness (handle negative fitness)
        min_fitness = min(chrom.fitness for chrom in population)
        if min_fitness < 0:
            # Shift all fitness values to be positive
            adjusted_fitness = [chrom.fitness - min_fitness + 1 for chrom in population]
        else:
            adjusted_fitness = [chrom.fitness for chrom in population]

        total_fitness = sum(adjusted_fitness)

        if total_fitness == 0:
            # If all fitness is zero, return random
            return random.choice(population)

        # Spin the wheel
        pick = random.uniform(0, total_fitness)
        current = 0

        for i, fitness in enumerate(adjusted_fitness):
            current += fitness
            if current >= pick:
                return population[i]

        # Fallback (shouldn't reach here)
        return population[-1]
