"""
================================================================================
PORTFOLIO EXPLORER - Multi-run GA with 3D Metrics Collection
================================================================================

This module runs the Genetic Algorithm multiple times with different seeds
and collects all evaluated portfolios with their (x, y, z) metrics for
3D visualization.

Author: Energy Systems Optimization Project
Date: December 2024
================================================================================
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import json
import time
from tqdm import tqdm

from config import Config, DEFAULT_CONFIG
from optimization import (
    run_genetic_algorithm, Individual, GAStats,
    evaluate_individual, evaluate_population
)
from data_loader import load_all_data, CachedData
from ppu_framework import load_all_ppu_data
from portfolio_metrics import calculate_portfolio_metrics_3d, PortfolioMetrics3D
from risk_calculator import RiskCalculator


# =============================================================================
# GA MODIFICATION TO TRACK ALL EVALUATED PORTFOLIOS
# =============================================================================

class TrackingGA:
    """Wrapper around GA that tracks all evaluated portfolios."""
    
    def __init__(self, config: Config = DEFAULT_CONFIG):
        self.config = config
        self.evaluated_portfolios: List[Individual] = []
        self.risk_calculator = RiskCalculator(config)
    
    def run_with_tracking(
        self,
        n_runs: int = 5,
        seeds: Optional[List[int]] = None,
        verbose: bool = True
    ) -> Tuple[List[Individual], List[GAStats]]:
        """
        Run GA multiple times and collect all evaluated portfolios.
        
        Args:
            n_runs: Number of GA runs
            seeds: Optional list of seeds (if None, generates random seeds)
            verbose: Print progress
            
        Returns:
            Tuple of (all_evaluated_portfolios, all_stats)
        """
        if seeds is None:
            seeds = list(range(42, 42 + n_runs))
        
        all_evaluated = []
        all_stats = []
        
        # Load data once
        if verbose:
            print("Loading data and PPU definitions...")
        data = load_all_data(self.config)
        _, _, ppu_definitions = load_all_ppu_data(self.config)
        
        for run_idx, seed in enumerate(seeds):
            if verbose:
                print(f"\n{'='*60}")
                print(f"GA Run {run_idx + 1}/{n_runs} (seed={seed})")
                print(f"{'='*60}")
            
            # Set seed in config
            self.config.ga.RANDOM_SEED = seed
            
            # Run GA with callback to track evaluations
            best, stats = self._run_ga_with_tracking(
                data, ppu_definitions, verbose=verbose
            )
            
            all_stats.append(stats)
            
            if verbose:
                print(f"Run {run_idx + 1} complete: {stats.total_evaluations} evaluations")
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Total portfolios evaluated: {len(self.evaluated_portfolios)}")
            print(f"{'='*60}")
        
        return self.evaluated_portfolios, all_stats
    
    def _run_ga_with_tracking(
        self,
        data: CachedData,
        ppu_definitions: Dict,
        verbose: bool = True
    ) -> Tuple[Individual, GAStats]:
        """Run GA while tracking all evaluations."""
        from optimization import (
            generate_initial_population, evaluate_population,
            tournament_selection, crossover, mutate,
            GAStats
        )
        import numpy as np
        
        # Get settings
        pop_size, n_generations, plateau_generations = self.config.run_mode.get_settings()
        
        # Initialize RNG
        rng = np.random.default_rng(self.config.ga.RANDOM_SEED)
        
        # Get PPU order
        ppu_order = list(self.config.ppu.PORTFOLIO_BOUNDS.keys())
        
        # Initialize population
        population = generate_initial_population(
            pop_size, ppu_order, self.config, rng
        )
        
        # Evaluate initial population
        population = evaluate_population(
            population, data, ppu_definitions, self.config, rng, verbose
        )
        
        # Track all evaluated portfolios
        for ind in population:
            if ind.fitness != float('inf'):
                self.evaluated_portfolios.append(ind.copy())
        
        # Sort by fitness
        population.sort(key=lambda ind: ind.fitness)
        
        # Initialize stats
        stats = GAStats()
        stats.best_fitness = population[0].fitness
        stats.mean_fitness = float(np.mean([ind.fitness for ind in population]))
        stats.best_fitness_history.append(stats.best_fitness)
        stats.mean_fitness_history.append(stats.mean_fitness)
        
        best_ever = population[0].copy()
        
        # Evolution loop
        for generation in range(n_generations):
            stats.generation = generation + 1
            
            # Create new population
            new_population = []
            
            # Elitism
            n_elite = max(1, int(pop_size * self.config.ga.ELITE_FRACTION))
            for i in range(n_elite):
                elite = population[i].copy()
                new_population.append(elite)
            
            # Generate offspring
            while len(new_population) < pop_size:
                parent1 = tournament_selection(
                    population, self.config.ga.TOURNAMENT_SIZE, rng
                )
                parent2 = tournament_selection(
                    population, self.config.ga.TOURNAMENT_SIZE, rng
                )
                
                if rng.random() < self.config.ga.CROSSOVER_RATE:
                    child1, child2 = crossover(parent1, parent2, ppu_order, rng)
                else:
                    child1, child2 = parent1.copy(), parent2.copy()
                
                child1 = mutate(
                    child1, ppu_order, self.config.ppu.PORTFOLIO_BOUNDS,
                    self.config.ga.MUTATION_RATE, self.config.ga.MUTATION_SIGMA, rng
                )
                child2 = mutate(
                    child2, ppu_order, self.config.ppu.PORTFOLIO_BOUNDS,
                    self.config.ga.MUTATION_RATE, self.config.ga.MUTATION_SIGMA, rng
                )
                
                new_population.extend([child1, child2])
            
            new_population = new_population[:pop_size]
            
            # Evaluate new individuals
            for ind in new_population:
                if ind.fitness == float('inf'):
                    evaluate_individual(
                        ind, data, ppu_definitions, self.config, rng
                    )
                    stats.total_evaluations += 1
                    
                    # Track evaluated portfolio
                    if ind.fitness != float('inf'):
                        self.evaluated_portfolios.append(ind.copy())
            
            # Sort and update
            new_population.sort(key=lambda ind: ind.fitness)
            population = new_population
            
            # Update stats
            current_best = population[0].fitness
            stats.mean_fitness = float(np.mean([ind.fitness for ind in population]))
            
            if current_best < stats.best_fitness:
                stats.best_fitness = current_best
                stats.generations_without_improvement = 0
                best_ever = population[0].copy()
            else:
                stats.generations_without_improvement += 1
            
            stats.best_fitness_history.append(stats.best_fitness)
            stats.mean_fitness_history.append(stats.mean_fitness)
            
            # Check stopping
            if stats.generations_without_improvement >= plateau_generations:
                break
        
        return best_ever, stats


# =============================================================================
# RESULTS COLLECTION AND STORAGE
# =============================================================================

def compute_metrics_for_portfolios(
    portfolios: List[Individual],
    config: Config = DEFAULT_CONFIG,
    verbose: bool = True
) -> List[PortfolioMetrics3D]:
    """
    Compute 3D metrics for a list of portfolios.
    
    Args:
        portfolios: List of evaluated portfolio individuals
        config: Configuration
        verbose: Print progress
        
    Returns:
        List of PortfolioMetrics3D objects
    """
    risk_calculator = RiskCalculator(config)
    metrics_list = []
    
    # Remove duplicates (same portfolio counts)
    unique_portfolios = {}
    for ind in portfolios:
        portfolio_key = tuple(sorted(ind.portfolio.ppu_counts.items()))
        if portfolio_key not in unique_portfolios:
            unique_portfolios[portfolio_key] = ind
    
    unique_list = list(unique_portfolios.values())
    
    if verbose:
        print(f"Computing metrics for {len(unique_list)} unique portfolios...")
    
    for i, ind in enumerate(tqdm(unique_list, disable=not verbose)):
        try:
            metrics = calculate_portfolio_metrics_3d(
                ind, config, risk_calculator
            )
            metrics_list.append(metrics)
        except Exception as e:
            if verbose:
                print(f"Error computing metrics for portfolio {i}: {e}")
            continue
    
    return metrics_list


def save_results_to_csv(
    metrics_list: List[PortfolioMetrics3D],
    config: Config,
    output_path: str = "portfolio_3d_results.csv"
) -> pd.DataFrame:
    """
    Save portfolio results to CSV.
    
    Args:
        metrics_list: List of portfolio metrics
        config: Configuration (for hyperparameters)
        output_path: Output file path
        
    Returns:
        DataFrame with results
    """
    rows = []
    
    for metrics in metrics_list:
        # Serialize portfolio counts as JSON string
        portfolio_json = json.dumps(metrics.portfolio_counts)
        
        # Serialize hyperparameters
        hyperparams = {
            'pop_size': config.run_mode.get_settings()[0],
            'n_generations': config.run_mode.get_settings()[1],
            'plateau_generations': config.run_mode.get_settings()[2],
            'crossover_rate': config.ga.CROSSOVER_RATE,
            'mutation_rate': config.ga.MUTATION_RATE,
            'mutation_sigma': config.ga.MUTATION_SIGMA,
            'tournament_size': config.ga.TOURNAMENT_SIZE,
            'elite_fraction': config.ga.ELITE_FRACTION,
        }
        hyperparams_json = json.dumps(hyperparams)
        
        row = {
            'portfolio_dict': portfolio_json,
            'x_RoT': metrics.x_rot,
            'y_volatility': metrics.y_volatility,
            'z_return': metrics.z_return,
            'hyperparameters': hyperparams_json,
            'total_energy_twh': metrics.total_energy_twh,
            'annual_production_twh': metrics.annual_production_twh,
        }
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    
    return df


# =============================================================================
# MAIN EXPLORATION FUNCTION
# =============================================================================

def explore_portfolio_space(
    config: Config = DEFAULT_CONFIG,
    n_runs: int = 5,
    seeds: Optional[List[int]] = None,
    output_path: str = "portfolio_3d_results.csv",
    verbose: bool = True
) -> pd.DataFrame:
    """
    Main function to explore portfolio space and collect 3D metrics.
    
    Args:
        config: Configuration
        n_runs: Number of GA runs
        seeds: Optional list of seeds
        output_path: Output CSV path
        verbose: Print progress
        
    Returns:
        DataFrame with all results
    """
    start_time = time.time()
    
    if verbose:
        print("="*60)
        print("PORTFOLIO SPACE EXPLORATION")
        print("="*60)
        print(f"Configuration: {config.run_mode.MODE}")
        print(f"Number of runs: {n_runs}")
        print()
    
    # Initialize tracker
    tracker = TrackingGA(config)
    
    # Run GA multiple times
    all_portfolios, all_stats = tracker.run_with_tracking(
        n_runs=n_runs, seeds=seeds, verbose=verbose
    )
    
    if verbose:
        print(f"\nTotal unique portfolios to evaluate: {len(all_portfolios)}")
    
    # Compute 3D metrics for all portfolios
    metrics_list = compute_metrics_for_portfolios(
        all_portfolios, config, verbose=verbose
    )
    
    # Save to CSV
    if verbose:
        print(f"\nSaving results to {output_path}...")
    
    df = save_results_to_csv(metrics_list, config, output_path)
    
    elapsed = time.time() - start_time
    
    if verbose:
        print(f"\n{'='*60}")
        print("EXPLORATION COMPLETE")
        print(f"{'='*60}")
        print(f"Total portfolios evaluated: {len(metrics_list)}")
        print(f"Time elapsed: {elapsed:.1f}s")
        print(f"Results saved to: {output_path}")
        print()
        print("Summary statistics:")
        print(f"  X (RoT) range: [{df['x_RoT'].min():.4f}, {df['x_RoT'].max():.4f}]")
        print(f"  Y (Volatility) range: [{df['y_volatility'].min():.2f}, {df['y_volatility'].max():.2f}]")
        print(f"  Z (Return) range: [{df['z_return'].min():.2f}, {df['z_return'].max():.2f}]")
    
    return df


if __name__ == "__main__":
    # Example usage
    config = DEFAULT_CONFIG
    config.run_mode.MODE = "quick"  # Use quick mode for testing
    
    df = explore_portfolio_space(
        config=config,
        n_runs=3,  # Small number for testing
        output_path="portfolio_3d_results.csv",
        verbose=True
    )
    
    print(f"\nResults DataFrame shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

