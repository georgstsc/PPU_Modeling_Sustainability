"""
================================================================================
OPTIMIZATION - Genetic Algorithm for Portfolio Optimization
================================================================================

This module implements the Genetic Algorithm (GA) for finding optimal PPU
portfolios in the Swiss Energy Storage Optimization project.

Key features:
- Energy sovereignty constraint (113 TWh/year target)
- Multi-scenario evaluation (robustness)
- Plateau detection for early stopping
- Immutable data handling

Author: Energy Systems Optimization Project
Date: December 2024
================================================================================
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
import time
import pickle
from pathlib import Path
import copy

from config import Config, DEFAULT_CONFIG
from data_loader import load_all_data, sample_scenario_indices, get_scenario_data, CachedData
from ppu_framework import (
    Portfolio, PPUDefinition, load_all_ppu_data, 
    create_ppu_dictionary, assign_renewable_locations,
    estimate_annual_production, check_energy_sovereignty
)
from dispatch_engine import (
    run_dispatch_simulation, compute_scenario_cost, compute_portfolio_metrics
)


# =============================================================================
# GENETIC ALGORITHM DATA STRUCTURES
# =============================================================================

@dataclass
class Individual:
    """An individual in the GA population (a portfolio)."""
    
    portfolio: Portfolio
    fitness: float = float('inf')  # Lower is better
    
    # Evaluation details
    mean_cost: float = float('inf')
    cvar: float = float('inf')
    annual_production_twh: float = 0.0
    is_sovereign: bool = False
    
    # Scenario results (for debugging)
    scenario_costs: List[float] = field(default_factory=list)
    
    def copy(self) -> 'Individual':
        """Create a deep copy of this individual."""
        return Individual(
            portfolio=self.portfolio.copy(),
            fitness=self.fitness,
            mean_cost=self.mean_cost,
            cvar=self.cvar,
            annual_production_twh=self.annual_production_twh,
            is_sovereign=self.is_sovereign,
            scenario_costs=self.scenario_costs.copy(),
        )


@dataclass
class GAStats:
    """Statistics tracked during GA evolution."""
    
    generation: int = 0
    best_fitness: float = float('inf')
    mean_fitness: float = float('inf')
    worst_fitness: float = float('inf')
    
    best_fitness_history: List[float] = field(default_factory=list)
    mean_fitness_history: List[float] = field(default_factory=list)
    
    generations_without_improvement: int = 0
    total_evaluations: int = 0
    elapsed_time_s: float = 0.0


# =============================================================================
# PORTFOLIO INITIALIZATION
# =============================================================================

def random_portfolio(
    ppu_order: List[str],
    bounds: Dict[str, Tuple[int, int]],
    rng: np.random.Generator,
) -> Portfolio:
    """
    Generate a random portfolio within bounds.
    
    Args:
        ppu_order: Ordered list of PPU names
        bounds: Min/max counts per PPU type
        rng: Random number generator
        
    Returns:
        Random Portfolio
    """
    counts = {}
    for ppu in ppu_order:
        min_v, max_v = bounds.get(ppu, (0, 10))
        counts[ppu] = rng.integers(min_v, max_v + 1)
    
    return Portfolio(ppu_counts=counts)


def generate_initial_population(
    pop_size: int,
    ppu_order: List[str],
    config: Config,
    rng: np.random.Generator,
) -> List[Individual]:
    """
    Generate initial population of random portfolios.
    
    Ensures diversity by varying strategies.
    
    Args:
        pop_size: Population size
        ppu_order: Ordered list of PPU names
        config: Configuration
        rng: Random number generator
        
    Returns:
        List of Individual objects
    """
    bounds = config.ppu.PORTFOLIO_BOUNDS
    population = []
    
    for i in range(pop_size):
        portfolio = random_portfolio(ppu_order, bounds, rng)
        population.append(Individual(portfolio=portfolio))
    
    return population


# =============================================================================
# FITNESS EVALUATION
# =============================================================================

def evaluate_individual(
    individual: Individual,
    data: CachedData,
    ppu_definitions: Dict[str, PPUDefinition],
    config: Config,
    rng: np.random.Generator,
    verbose: bool = False,
) -> Individual:
    """
    Evaluate an individual's fitness.
    
    This runs dispatch simulation across multiple scenarios and computes
    the aggregate cost metric.
    
    Args:
        individual: Individual to evaluate
        data: Cached data
        ppu_definitions: PPU definitions
        config: Configuration
        rng: Random number generator
        verbose: Print details
        
    Returns:
        Updated Individual with fitness
    """
    portfolio = individual.portfolio
    
    # Check energy sovereignty first
    is_sovereign, annual_prod, target = check_energy_sovereignty(
        portfolio, ppu_definitions, config
    )
    individual.annual_production_twh = annual_prod
    individual.is_sovereign = is_sovereign
    
    # Heavy penalty if not sovereign
    if not is_sovereign:
        deficit = target - annual_prod
        penalty = config.fitness.SOVEREIGNTY_PENALTY_MULTIPLIER * deficit
        individual.fitness = penalty
        individual.mean_cost = penalty
        individual.cvar = penalty
        return individual
    
    # Create PPU dictionary
    ppu_dict = create_ppu_dictionary(portfolio, ppu_definitions, config)
    
    # Assign renewable locations
    ppu_dict = assign_renewable_locations(
        ppu_dict,
        data.solar_ranking,
        data.wind_ranking,
    )
    
    # Run multiple scenarios
    n_scenarios = config.scenario.SCENARIOS_PER_EVALUATION
    scenario_results = []
    
    for s in range(n_scenarios):
        # Sample scenario indices (COPY data to avoid mutation)
        indices = sample_scenario_indices(
            n_hours=data.n_hours,
            days_per_scenario=config.scenario.DAYS_PER_SCENARIO,
            n_scenarios=1,
            rng=rng,
        )[0]
        
        # Get COPIES of scenario data
        demand = data.get_demand()[indices]
        spot = data.get_spot_prices()[indices]
        solar = data.get_solar_incidence()[indices]
        wind = data.get_wind_incidence()[indices]
        
        # Run dispatch
        _, results = run_dispatch_simulation(
            scenario_indices=np.arange(len(indices)),  # Local indices
            ppu_dictionary=ppu_dict,
            demand_data=demand,
            spot_data=spot,
            solar_data=solar,
            wind_data=wind,
            portfolio_counts=portfolio.ppu_counts,
            config=config,
            verbose=False,
        )
        
        # Compute cost
        cost = compute_scenario_cost(results, ppu_dict, config)
        scenario_results.append({'net_spot_cost_chf': cost, **results})
        individual.scenario_costs.append(cost)
    
    # Compute aggregate metrics
    metrics = compute_portfolio_metrics(scenario_results, config)
    individual.mean_cost = metrics['mean_cost']
    individual.cvar = metrics['cvar']
    
    # Fitness: use combined cost (harmonic mean approach optional)
    if config.fitness.COST_AGGREGATION == 'harmonic_mean':
        # Harmonic mean is more sensitive to high costs
        costs = [max(c, 1e-6) for c in individual.scenario_costs]  # Avoid division by zero
        individual.fitness = len(costs) / sum(1/c for c in costs)
    else:
        individual.fitness = metrics['combined_cost']
    
    return individual


def evaluate_population(
    population: List[Individual],
    data: CachedData,
    ppu_definitions: Dict[str, PPUDefinition],
    config: Config,
    rng: np.random.Generator,
    verbose: bool = False,
) -> List[Individual]:
    """
    Evaluate fitness for all individuals in population.
    
    Args:
        population: List of individuals
        data: Cached data
        ppu_definitions: PPU definitions
        config: Configuration
        rng: Random number generator
        verbose: Print progress
        
    Returns:
        Population with updated fitness values
    """
    for i, ind in enumerate(population):
        if verbose and i % 10 == 0:
            print(f"  Evaluating individual {i+1}/{len(population)}...")
        
        evaluate_individual(ind, data, ppu_definitions, config, rng)
    
    return population


# =============================================================================
# GENETIC OPERATORS
# =============================================================================

def tournament_selection(
    population: List[Individual],
    tournament_size: int,
    rng: np.random.Generator,
) -> Individual:
    """
    Select an individual via tournament selection.
    
    Args:
        population: Current population
        tournament_size: Number of individuals in tournament
        rng: Random number generator
        
    Returns:
        Selected individual (copy)
    """
    indices = rng.choice(len(population), size=tournament_size, replace=False)
    tournament = [population[i] for i in indices]
    winner = min(tournament, key=lambda ind: ind.fitness)
    return winner.copy()


def crossover(
    parent1: Individual,
    parent2: Individual,
    ppu_order: List[str],
    rng: np.random.Generator,
) -> Tuple[Individual, Individual]:
    """
    Perform crossover between two parents.
    
    Uses uniform crossover at the PPU level.
    
    Args:
        parent1: First parent
        parent2: Second parent
        ppu_order: Ordered list of PPU names
        rng: Random number generator
        
    Returns:
        Two offspring individuals
    """
    child1_counts = {}
    child2_counts = {}
    
    for ppu in ppu_order:
        if rng.random() < 0.5:
            child1_counts[ppu] = parent1.portfolio.get_count(ppu)
            child2_counts[ppu] = parent2.portfolio.get_count(ppu)
        else:
            child1_counts[ppu] = parent2.portfolio.get_count(ppu)
            child2_counts[ppu] = parent1.portfolio.get_count(ppu)
    
    return (
        Individual(portfolio=Portfolio(ppu_counts=child1_counts)),
        Individual(portfolio=Portfolio(ppu_counts=child2_counts)),
    )


def mutate(
    individual: Individual,
    ppu_order: List[str],
    bounds: Dict[str, Tuple[int, int]],
    mutation_rate: float,
    mutation_sigma: float,
    rng: np.random.Generator,
) -> Individual:
    """
    Mutate an individual's portfolio.
    
    Uses Gaussian mutation scaled by bounds.
    
    Args:
        individual: Individual to mutate
        ppu_order: Ordered list of PPU names
        bounds: Min/max counts per PPU
        mutation_rate: Probability of mutating each gene
        mutation_sigma: Standard deviation for Gaussian mutation (relative)
        rng: Random number generator
        
    Returns:
        Mutated individual
    """
    for ppu in ppu_order:
        if rng.random() < mutation_rate:
            min_v, max_v = bounds.get(ppu, (0, 10))
            current = individual.portfolio.get_count(ppu)
            
            # Gaussian mutation
            range_v = max_v - min_v
            delta = rng.normal(0, mutation_sigma * range_v)
            new_value = current + delta
            
            # Clamp to bounds
            new_value = max(min_v, min(max_v, int(round(new_value))))
            individual.portfolio.set_count(ppu, new_value)
    
    # Reset fitness (needs re-evaluation)
    individual.fitness = float('inf')
    
    return individual


# =============================================================================
# MAIN GA LOOP
# =============================================================================

def run_genetic_algorithm(
    config: Config = DEFAULT_CONFIG,
    verbose: bool = True,
    save_progress: bool = True,
    progress_callback: Optional[Callable[[GAStats], None]] = None,
) -> Tuple[Individual, GAStats]:
    """
    Run the genetic algorithm optimization.
    
    Args:
        config: Configuration
        verbose: Print progress
        save_progress: Save intermediate results
        progress_callback: Optional callback for progress updates
        
    Returns:
        Tuple of (best_individual, stats)
    """
    start_time = time.time()
    
    # Get run mode settings
    pop_size, n_generations, plateau_generations = config.run_mode.get_settings()
    
    if verbose:
        print("=" * 60)
        print("GENETIC ALGORITHM OPTIMIZATION")
        print("=" * 60)
        print(f"Mode: {config.run_mode.MODE}")
        print(f"Population: {pop_size}")
        print(f"Max generations: {n_generations}")
        print(f"Plateau detection: {plateau_generations} generations")
        print()
    
    # Initialize RNG
    seed = config.ga.RANDOM_SEED
    rng = np.random.default_rng(seed)
    
    # Load data
    if verbose:
        print("Loading data...")
    data = load_all_data(config)
    
    # Load PPU definitions
    if verbose:
        print("Loading PPU definitions...")
    _, _, ppu_definitions = load_all_ppu_data(config)
    
    # Get PPU order for array operations
    ppu_order = list(config.ppu.PORTFOLIO_BOUNDS.keys())
    
    # Initialize population
    if verbose:
        print(f"Generating initial population of {pop_size} individuals...")
    population = generate_initial_population(pop_size, ppu_order, config, rng)
    
    # Evaluate initial population
    if verbose:
        print("Evaluating initial population...")
    population = evaluate_population(population, data, ppu_definitions, config, rng, verbose)
    
    # Sort by fitness
    population.sort(key=lambda ind: ind.fitness)
    
    # Initialize stats
    stats = GAStats()
    stats.best_fitness = population[0].fitness
    stats.best_fitness_history.append(stats.best_fitness)
    
    best_ever = population[0].copy()
    
    if verbose:
        print(f"\nInitial best fitness: {stats.best_fitness:,.0f}")
        if best_ever.is_sovereign:
            print(f"  Production: {best_ever.annual_production_twh:.1f} TWh/year")
        else:
            print(f"  NOT SOVEREIGN: {best_ever.annual_production_twh:.1f} TWh/year")
        print()
    
    # Evolution loop
    for generation in range(n_generations):
        stats.generation = generation + 1
        gen_start = time.time()
        
        # Create new population
        new_population = []
        
        # Elitism: keep top individuals
        n_elite = max(1, int(pop_size * config.ga.ELITE_FRACTION))
        for i in range(n_elite):
            elite = population[i].copy()
            new_population.append(elite)
        
        # Generate offspring
        while len(new_population) < pop_size:
            # Selection
            parent1 = tournament_selection(population, config.ga.TOURNAMENT_SIZE, rng)
            parent2 = tournament_selection(population, config.ga.TOURNAMENT_SIZE, rng)
            
            # Crossover
            if rng.random() < config.ga.CROSSOVER_RATE:
                child1, child2 = crossover(parent1, parent2, ppu_order, rng)
            else:
                child1, child2 = parent1.copy(), parent2.copy()
            
            # Mutation
            child1 = mutate(child1, ppu_order, config.ppu.PORTFOLIO_BOUNDS,
                           config.ga.MUTATION_RATE, config.ga.MUTATION_SIGMA, rng)
            child2 = mutate(child2, ppu_order, config.ppu.PORTFOLIO_BOUNDS,
                           config.ga.MUTATION_RATE, config.ga.MUTATION_SIGMA, rng)
            
            new_population.extend([child1, child2])
        
        # Trim to population size
        new_population = new_population[:pop_size]
        
        # Evaluate new individuals (those with inf fitness)
        for ind in new_population:
            if ind.fitness == float('inf'):
                evaluate_individual(ind, data, ppu_definitions, config, rng)
                stats.total_evaluations += 1
        
        # Sort and update population
        new_population.sort(key=lambda ind: ind.fitness)
        population = new_population
        
        # Update stats
        current_best = population[0].fitness
        stats.mean_fitness = np.mean([ind.fitness for ind in population])
        stats.worst_fitness = population[-1].fitness
        
        # Check for improvement
        if current_best < stats.best_fitness:
            improvement = stats.best_fitness - current_best
            stats.best_fitness = current_best
            stats.generations_without_improvement = 0
            best_ever = population[0].copy()
            
            if verbose:
                print(f"Gen {stats.generation:3d}: ↓ New best! {stats.best_fitness:,.0f} "
                      f"(improved by {improvement:,.0f})")
        else:
            stats.generations_without_improvement += 1
            
            if verbose and generation % 5 == 0:
                print(f"Gen {stats.generation:3d}: {stats.best_fitness:,.0f} "
                      f"(plateau: {stats.generations_without_improvement})")
        
        stats.best_fitness_history.append(stats.best_fitness)
        stats.mean_fitness_history.append(stats.mean_fitness)
        
        # Progress callback
        if progress_callback:
            progress_callback(stats)
        
        # Check stopping criteria
        if stats.generations_without_improvement >= plateau_generations:
            if verbose:
                print(f"\n🛑 Plateau detected: No improvement for {plateau_generations} generations")
            break
        
        # Time check for overnight mode
        elapsed = time.time() - start_time
        stats.elapsed_time_s = elapsed
        
    # Final summary
    stats.elapsed_time_s = time.time() - start_time
    
    if verbose:
        print()
        print("=" * 60)
        print("OPTIMIZATION COMPLETE")
        print("=" * 60)
        print(f"Total generations: {stats.generation}")
        print(f"Total evaluations: {stats.total_evaluations}")
        print(f"Time elapsed: {stats.elapsed_time_s:.1f}s")
        print()
        print("BEST PORTFOLIO:")
        print(f"  Fitness: {best_ever.fitness:,.0f}")
        print(f"  Annual production: {best_ever.annual_production_twh:.1f} TWh/year")
        print(f"  Sovereign: {'Yes' if best_ever.is_sovereign else 'No'}")
        print(f"  Mean scenario cost: {best_ever.mean_cost:,.0f} CHF")
        print(f"  CVaR: {best_ever.cvar:,.0f} CHF")
        print()
        print("PPU counts:")
        for ppu, count in sorted(best_ever.portfolio.ppu_counts.items()):
            if count > 0:
                print(f"    {ppu}: {count}")
    
    # Save results
    if save_progress:
        results_path = Path('ga_optimization_results.pkl')
        with open(results_path, 'wb') as f:
            pickle.dump({
                'best_individual': best_ever,
                'stats': stats,
                'config': config,
            }, f)
        if verbose:
            print(f"\nResults saved to {results_path}")
    
    return best_ever, stats


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def load_optimization_results(
    filepath: str = 'ga_optimization_results.pkl'
) -> Tuple[Individual, GAStats, Config]:
    """Load saved optimization results."""
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data['best_individual'], data['stats'], data['config']


def portfolio_summary(individual: Individual) -> pd.DataFrame:
    """Create a summary DataFrame of a portfolio."""
    rows = []
    for ppu, count in individual.portfolio.ppu_counts.items():
        if count > 0:
            rows.append({'PPU': ppu, 'Count': count})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    # Quick test run
    config = DEFAULT_CONFIG
    config.run_mode.MODE = "quick"  # Fast test
    
    best, stats = run_genetic_algorithm(config, verbose=True)
    
    print("\n\nFinal portfolio summary:")
    print(portfolio_summary(best))

