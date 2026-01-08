"""
================================================================================
MULTI-OBJECTIVE PORTFOLIO EXPLORER
================================================================================

This module explores the portfolio space using multiple optimization objectives
to better cover the efficiency frontier.

Instead of only optimizing for cost, we run the GA with different objectives:
1. Minimize Cost (original)
2. Minimize RoT (Risk of Technology)
3. Minimize Volatility
4. Weighted combinations

This provides better coverage of the Pareto frontier.

Author: Energy Systems Optimization Project
Date: December 2024
================================================================================
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Callable
from pathlib import Path
import json
import time
from tqdm import tqdm
from dataclasses import dataclass, field

from config import Config, DEFAULT_CONFIG
from optimization import Individual, GAStats, evaluate_portfolio_full_year
from data_loader import load_all_data, CachedData
from ppu_framework import load_all_ppu_data, Portfolio
from portfolio_metrics import calculate_portfolio_metrics_3d, PortfolioMetrics3D
from risk_calculator import RiskCalculator
from pareto_frontier import extract_pareto_frontier_from_df, extract_pareto_frontier_3d_from_df


# =============================================================================
# MULTI-OBJECTIVE FITNESS FUNCTIONS
# =============================================================================

def fitness_cost_only(metrics: PortfolioMetrics3D) -> float:
    """Original fitness: minimize cost (maximize return)."""
    # Return is negative when costs > spot, so we want to maximize return (minimize -return)
    return -metrics.z_return  # Lower is better


def fitness_rot_only(metrics: PortfolioMetrics3D) -> float:
    """Fitness: minimize Risk of Technology."""
    return metrics.x_rot  # Lower is better


def fitness_volatility_only(metrics: PortfolioMetrics3D) -> float:
    """Fitness: minimize Volatility."""
    return metrics.y_volatility  # Lower is better


def fitness_rot_volatility(metrics: PortfolioMetrics3D, w_rot: float = 0.5) -> float:
    """
    Combined fitness: weighted sum of RoT and Volatility.
    
    Args:
        metrics: Portfolio metrics
        w_rot: Weight for RoT (1 - w_rot = weight for Volatility)
        
    Returns:
        Fitness value (lower is better)
    """
    # Normalize to [0, 1] range approximately
    # RoT is already in [0, 1]
    # Volatility needs normalization (typical range: 100-1000)
    norm_vol = metrics.y_volatility / 1000.0  # Rough normalization
    
    return w_rot * metrics.x_rot + (1 - w_rot) * norm_vol


def fitness_all_objectives(
    metrics: PortfolioMetrics3D,
    w_rot: float = 0.33,
    w_vol: float = 0.33,
    w_return: float = 0.34
) -> float:
    """
    Combined fitness: weighted sum of all three objectives.
    
    Args:
        metrics: Portfolio metrics
        w_rot: Weight for RoT (minimize)
        w_vol: Weight for Volatility (minimize)
        w_return: Weight for Return (maximize, so we minimize -return)
        
    Returns:
        Fitness value (lower is better)
    """
    # Normalize all to similar scales
    norm_rot = metrics.x_rot  # [0, 1]
    norm_vol = metrics.y_volatility / 1000.0  # Rough normalization
    norm_return = -metrics.z_return / 100.0  # Flip sign, normalize (return is usually negative)
    
    return w_rot * norm_rot + w_vol * norm_vol + w_return * norm_return


# =============================================================================
# MULTI-OBJECTIVE GA
# =============================================================================

@dataclass
class ObjectiveConfig:
    """Configuration for a single objective."""
    name: str
    fitness_fn: Callable[[PortfolioMetrics3D], float]
    description: str


def get_objective_configs() -> List[ObjectiveConfig]:
    """
    Get predefined objective configurations for exploration.
    
    Returns objectives that optimize across all 3 dimensions:
    - RoT (Risk of Technology - supply chain risk)
    - Volatility (cost stability)
    - Return (economic efficiency)
    """
    return [
        # =====================================================================
        # 3D MULTI-OBJECTIVE FITNESS FUNCTIONS (Primary)
        # =====================================================================
        ObjectiveConfig(
            name="3d_balanced",
            fitness_fn=lambda m: fitness_all_objectives(m, w_rot=0.33, w_vol=0.33, w_return=0.34),
            description="3D Balanced: Equal weights (RoT, Vol, Return)"
        ),
        ObjectiveConfig(
            name="3d_stability",
            fitness_fn=lambda m: fitness_all_objectives(m, w_rot=0.4, w_vol=0.4, w_return=0.2),
            description="3D Stability: 40% RoT + 40% Vol + 20% Return"
        ),
        ObjectiveConfig(
            name="3d_return",
            fitness_fn=lambda m: fitness_all_objectives(m, w_rot=0.2, w_vol=0.2, w_return=0.6),
            description="3D Return: 60% Return + 20% RoT + 20% Vol"
        ),
        ObjectiveConfig(
            name="3d_rot_focus",
            fitness_fn=lambda m: fitness_all_objectives(m, w_rot=0.6, w_vol=0.2, w_return=0.2),
            description="3D RoT Focus: 60% RoT + 20% Vol + 20% Return"
        ),
        ObjectiveConfig(
            name="3d_vol_focus",
            fitness_fn=lambda m: fitness_all_objectives(m, w_rot=0.2, w_vol=0.6, w_return=0.2),
            description="3D Vol Focus: 60% Vol + 20% RoT + 20% Return"
        ),
        
        # =====================================================================
        # SINGLE-OBJECTIVE FITNESS FUNCTIONS (For comparison)
        # =====================================================================
        ObjectiveConfig(
            name="cost_only",
            fitness_fn=fitness_cost_only,
            description="1D: Minimize cost (maximize return only)"
        ),
        ObjectiveConfig(
            name="rot_only",
            fitness_fn=fitness_rot_only,
            description="1D: Minimize Risk of Technology only"
        ),
        ObjectiveConfig(
            name="volatility_only",
            fitness_fn=fitness_volatility_only,
            description="1D: Minimize Volatility only"
        ),
        
        # =====================================================================
        # 2D COMBINED FITNESS FUNCTIONS (Legacy)
        # =====================================================================
        ObjectiveConfig(
            name="rot_vol_50_50",
            fitness_fn=lambda m: fitness_rot_volatility(m, w_rot=0.5),
            description="2D: 50% RoT + 50% Vol (no Return)"
        ),
    ]


class MultiObjectiveGA:
    """
    GA that can optimize for different objectives.
    
    Runs multiple GA sessions with different fitness functions to explore
    the portfolio space more thoroughly.
    """
    
    def __init__(self, config: Config = DEFAULT_CONFIG):
        self.config = config
        self.risk_calculator = RiskCalculator(config)
        self.all_portfolios: List[Tuple[Dict[str, int], PortfolioMetrics3D]] = []
        
        # Load data once
        self.data = load_all_data(config)
        _, _, self.ppu_definitions = load_all_ppu_data(config)
    
    def evaluate_portfolio_with_metrics(
        self,
        portfolio_counts: Dict[str, int]
    ) -> Optional[PortfolioMetrics3D]:
        """
        Evaluate a portfolio and return its 3D metrics.
        
        IMPORTANT: Only evaluates portfolios that meet energy sovereignty (≥113 TWh/year).
        Invalid portfolios are rejected to avoid misleading results.
        
        Args:
            portfolio_counts: Dictionary of PPU counts
            
        Returns:
            PortfolioMetrics3D or None if portfolio is invalid or evaluation fails
        """
        try:
            # Create Individual
            portfolio = Portfolio(ppu_counts=portfolio_counts)
            individual = Individual(portfolio=portfolio)
            
            # CRITICAL: Check energy sovereignty first
            # This prevents evaluating tiny portfolios that minimize RoT/Volatility
            # but don't meet the 113 TWh/year target
            from ppu_framework import check_energy_sovereignty
            
            is_sovereign, annual_prod, target = check_energy_sovereignty(
                portfolio, self.ppu_definitions, self.config
            )
            
            if not is_sovereign:
                # Reject portfolio - doesn't meet energy sovereignty
                # This prevents the GA from finding "optimal" portfolios that are too small
                return None
            
            # Only calculate metrics for valid portfolios
            metrics = calculate_portfolio_metrics_3d(
                individual, self.config, self.risk_calculator, debug=False
            )
            
            return metrics
        except Exception as e:
            # Evaluation failed (e.g., dispatch error)
            return None
    
    def generate_initial_population(
        self,
        pop_size: int = 30,
        seed: int = 42,
        verbose: bool = True
    ) -> List[Tuple[Dict[str, int], PortfolioMetrics3D]]:
        """
        Generate and evaluate a diverse initial population.
        
        This population can be shared across multiple objective runs to ensure
        all optimizers start from the same search space.
        
        Args:
            pop_size: Population size
            seed: Random seed for reproducibility
            verbose: Print progress
            
        Returns:
            List of (portfolio_counts, metrics) tuples
        """
        rng = np.random.default_rng(seed)
        ppu_order = list(self.config.ppu.PORTFOLIO_BOUNDS.keys())
        
        # Identify PPU categories for smart initialization
        incidence_ppus = set(self.config.ppu.INCIDENCE_PPUS)
        
        population = []
        
        # --- TYPE 1: Simple portfolios (dominated by 1-2 cheap incidence PPUs) ---
        n_simple = max(1, pop_size // 4)  # 25%
        cheap_incidence = ['PV', 'WD_ON', 'WD_OFF', 'HYD_R']
        
        for i in range(n_simple):
            counts = {ppu: 0 for ppu in ppu_order}
            n_ppus = rng.integers(1, 3)
            selected_ppus = rng.choice(cheap_incidence, size=min(n_ppus, len(cheap_incidence)), replace=False)
            
            for ppu_name in selected_ppus:
                if ppu_name in ppu_order:
                    _, max_val = self.config.ppu.PORTFOLIO_BOUNDS[ppu_name]
                    counts[ppu_name] = int(rng.integers(max_val // 2, max_val + 1))
            
            for ppu_name in ['HYD_S', 'PHS']:
                if ppu_name in ppu_order:
                    _, max_val = self.config.ppu.PORTFOLIO_BOUNDS[ppu_name]
                    counts[ppu_name] = int(rng.integers(0, max(1, max_val // 4)))
            
            population.append(counts)
        
        # --- TYPE 2: Moderate complexity portfolios (3-5 PPU types) ---
        n_moderate = max(1, pop_size // 4)  # 25%
        
        for i in range(n_moderate):
            counts = {ppu: 0 for ppu in ppu_order}
            n_ppus = rng.integers(3, 6)
            selected_ppus = rng.choice(list(incidence_ppus), size=min(n_ppus, len(incidence_ppus)), replace=False)
            
            for ppu_name in selected_ppus:
                if ppu_name in ppu_order:
                    min_val, max_val = self.config.ppu.PORTFOLIO_BOUNDS[ppu_name]
                    counts[ppu_name] = int(rng.integers(min_val, max_val + 1))
            
            storage_support = ['HYD_S', 'PHS', 'H2P_G']
            for ppu_name in storage_support:
                if ppu_name in ppu_order and rng.random() < 0.5:
                    _, max_val = self.config.ppu.PORTFOLIO_BOUNDS[ppu_name]
                    counts[ppu_name] = int(rng.integers(0, max(1, max_val // 3)))
            
            population.append(counts)
        
        # --- TYPE 3: Full random portfolios ---
        n_random = pop_size - len(population)
        
        for _ in range(n_random):
            counts = {}
            for ppu_name in ppu_order:
                min_val, max_val = self.config.ppu.PORTFOLIO_BOUNDS[ppu_name]
                counts[ppu_name] = int(rng.integers(min_val, max_val + 1))
            population.append(counts)
        
        # Evaluate all portfolios
        evaluated = []
        if verbose:
            print("Generating shared initial population...")
        for counts in tqdm(population, desc="Evaluating initial population", disable=not verbose):
            metrics = self.evaluate_portfolio_with_metrics(counts)
            if metrics is not None:
                evaluated.append((counts.copy(), metrics))
        
        if verbose:
            print(f"Valid portfolios in initial population: {len(evaluated)}/{len(population)}")
        
        return evaluated

    def run_single_objective_ga(
        self,
        objective: ObjectiveConfig,
        n_generations: int = 20,
        pop_size: int = 30,
        seed: int = 42,
        verbose: bool = True,
        initial_population: Optional[List[Tuple[Dict[str, int], PortfolioMetrics3D]]] = None
    ) -> List[Tuple[Dict[str, int], PortfolioMetrics3D]]:
        """
        Run GA with a specific objective function.
        
        Uses a simple GA implementation focused on the given objective.
        
        Args:
            objective: Objective configuration
            n_generations: Number of generations
            pop_size: Population size
            seed: Random seed for evolution (selection, mutation)
            verbose: Print progress
            initial_population: Optional pre-evaluated population to start from.
                               If provided, skips initial population generation.
            
        Returns:
            List of (portfolio_counts, metrics) tuples for all evaluated portfolios
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"Objective: {objective.name}")
            print(f"Description: {objective.description}")
            print(f"{'='*60}")
        
        rng = np.random.default_rng(seed)
        ppu_order = list(self.config.ppu.PORTFOLIO_BOUNDS.keys())
        
        evaluated_portfolios = []
        
        # Use provided initial population or generate new one
        if initial_population is not None:
            # Use shared initial population - this ensures all objectives
            # start from the same search space
            pop_with_metrics = [(counts.copy(), metrics) for counts, metrics in initial_population]
            evaluated_portfolios.extend([(counts.copy(), metrics) for counts, metrics in initial_population])
            
            if verbose:
                print(f"Using shared initial population ({len(pop_with_metrics)} portfolios)")
        else:
            # Generate new population (legacy behavior)
            incidence_ppus = set(self.config.ppu.INCIDENCE_PPUS)
            
            population = []
            
            # --- TYPE 1: Simple portfolios ---
            n_simple = max(1, pop_size // 4)
            cheap_incidence = ['PV', 'WD_ON', 'WD_OFF', 'HYD_R']
            
            for i in range(n_simple):
                counts = {ppu: 0 for ppu in ppu_order}
                n_ppus = rng.integers(1, 3)
                selected_ppus = rng.choice(cheap_incidence, size=min(n_ppus, len(cheap_incidence)), replace=False)
                
                for ppu_name in selected_ppus:
                    if ppu_name in ppu_order:
                        _, max_val = self.config.ppu.PORTFOLIO_BOUNDS[ppu_name]
                        counts[ppu_name] = int(rng.integers(max_val // 2, max_val + 1))
                
                for ppu_name in ['HYD_S', 'PHS']:
                    if ppu_name in ppu_order:
                        _, max_val = self.config.ppu.PORTFOLIO_BOUNDS[ppu_name]
                        counts[ppu_name] = int(rng.integers(0, max(1, max_val // 4)))
                
                population.append(counts)
            
            # --- TYPE 2: Moderate complexity portfolios ---
            n_moderate = max(1, pop_size // 4)
            
            for i in range(n_moderate):
                counts = {ppu: 0 for ppu in ppu_order}
                n_ppus = rng.integers(3, 6)
                selected_ppus = rng.choice(list(incidence_ppus), size=min(n_ppus, len(incidence_ppus)), replace=False)
                
                for ppu_name in selected_ppus:
                    if ppu_name in ppu_order:
                        min_val, max_val = self.config.ppu.PORTFOLIO_BOUNDS[ppu_name]
                        counts[ppu_name] = int(rng.integers(min_val, max_val + 1))
                
                storage_support = ['HYD_S', 'PHS', 'H2P_G']
                for ppu_name in storage_support:
                    if ppu_name in ppu_order and rng.random() < 0.5:
                        _, max_val = self.config.ppu.PORTFOLIO_BOUNDS[ppu_name]
                        counts[ppu_name] = int(rng.integers(0, max(1, max_val // 3)))
                
                population.append(counts)
            
            # --- TYPE 3: Full random portfolios ---
            n_random = pop_size - len(population)
            
            for _ in range(n_random):
                counts = {}
                for ppu_name in ppu_order:
                    min_val, max_val = self.config.ppu.PORTFOLIO_BOUNDS[ppu_name]
                    counts[ppu_name] = int(rng.integers(min_val, max_val + 1))
                population.append(counts)
            
            # Evaluate initial population
            pop_with_metrics = []
            for counts in tqdm(population, desc="Initial population", disable=not verbose):
                metrics = self.evaluate_portfolio_with_metrics(counts)
                if metrics is not None:
                    pop_with_metrics.append((counts, metrics))
                    evaluated_portfolios.append((counts.copy(), metrics))
        
        # Evolution loop
        for gen in range(n_generations):
            if verbose:
                print(f"Generation {gen + 1}/{n_generations}", end="\r")
            
            # Calculate fitness for all individuals
            fitness_values = [
                objective.fitness_fn(m) for _, m in pop_with_metrics
            ]
            
            # Sort by fitness (lower is better)
            sorted_pop = sorted(
                zip(pop_with_metrics, fitness_values),
                key=lambda x: x[1]
            )
            pop_with_metrics = [p for p, f in sorted_pop]
            
            # Selection and reproduction
            n_elite = max(1, int(pop_size * 0.1))
            new_pop_counts = [counts.copy() for counts, _ in pop_with_metrics[:n_elite]]
            
            while len(new_pop_counts) < pop_size:
                # Tournament selection
                tournament_indices = rng.choice(len(pop_with_metrics), size=3, replace=False)
                # Calculate fitness for tournament participants
                tournament_fitness = [
                    objective.fitness_fn(m) 
                    for _, m in [pop_with_metrics[i] for i in tournament_indices]
                ]
                # Select winner with best (lowest) fitness
                winner_local_idx = np.argmin(tournament_fitness)
                winner_idx = tournament_indices[winner_local_idx]
                parent = pop_with_metrics[winner_idx][0]
                
                # Mutation
                child = parent.copy()
                for ppu_name in ppu_order:
                    if rng.random() < 0.2:  # Mutation rate
                        min_val, max_val = self.config.ppu.PORTFOLIO_BOUNDS[ppu_name]
                        # Gaussian mutation
                        current = child[ppu_name]
                        delta = int(rng.normal(0, max(1, (max_val - min_val) * 0.1)))
                        child[ppu_name] = max(min_val, min(max_val, current + delta))
                
                new_pop_counts.append(child)
            
            # Evaluate new population
            pop_with_metrics = []
            for counts in new_pop_counts:
                # Check if we already evaluated this portfolio
                portfolio_key = tuple(sorted(counts.items()))
                existing = None
                for ec, em in evaluated_portfolios:
                    if tuple(sorted(ec.items())) == portfolio_key:
                        existing = (ec, em)
                        break
                
                if existing:
                    pop_with_metrics.append(existing)
                else:
                    metrics = self.evaluate_portfolio_with_metrics(counts)
                    if metrics is not None:
                        pop_with_metrics.append((counts, metrics))
                        evaluated_portfolios.append((counts.copy(), metrics))
                    else:
                        # Re-use elite if evaluation fails
                        pop_with_metrics.append(pop_with_metrics[0])
        
        if verbose:
            best_counts, best_metrics = pop_with_metrics[0]
            best_fitness = objective.fitness_fn(best_metrics)
            print(f"\nBest fitness: {best_fitness:.4f}")
            print(f"  RoT: {best_metrics.x_rot:.4f}")
            print(f"  Volatility: {best_metrics.y_volatility:.2f}")
            print(f"  Return: {best_metrics.z_return:.2f}%")
        
        return evaluated_portfolios
    
    def explore_multi_objective(
        self,
        objectives: Optional[List[ObjectiveConfig]] = None,
        n_generations: int = 15,
        pop_size: int = 25,
        seeds: Optional[List[int]] = None,
        verbose: bool = True,
        share_initial_population: bool = True,
        initial_portfolios: Optional[List['Portfolio']] = None
    ) -> pd.DataFrame:
        """
        Explore portfolio space with multiple objectives.
        
        Args:
            objectives: List of objective configurations (default: all predefined)
            n_generations: Generations per objective
            pop_size: Population size
            seeds: Seeds for each objective run (used for evolution, not initialization)
            verbose: Print progress
            share_initial_population: If True, all objectives start from the same
                                     initial population. This ensures consistent 
                                     comparison across objectives.
            initial_portfolios: Optional list of Portfolio objects to seed the
                               initial population. If provided, these are evaluated
                               and added to the initial population.
            
        Returns:
            DataFrame with all evaluated portfolios and their metrics
        """
        if objectives is None:
            objectives = get_objective_configs()
        
        if seeds is None:
            seeds = list(range(42, 42 + len(objectives)))
        
        start_time = time.time()
        
        if verbose:
            print("="*60)
            print("MULTI-OBJECTIVE PORTFOLIO EXPLORATION")
            print("="*60)
            print(f"Objectives: {len(objectives)}")
            print(f"Generations per objective: {n_generations}")
            print(f"Population size: {pop_size}")
            print(f"Shared initial population: {share_initial_population}")
            if initial_portfolios:
                print(f"Seed portfolios provided: {len(initial_portfolios)}")
            print()
        
        all_evaluated = []
        
        # Generate shared initial population if requested
        shared_population = None
        if share_initial_population:
            # Start with seed portfolios if provided
            if initial_portfolios:
                if verbose:
                    print(f"Evaluating {len(initial_portfolios)} seed portfolios...")
                seed_evaluated = []
                for portfolio in tqdm(initial_portfolios, desc="Evaluating seeds", disable=not verbose):
                    metrics = self.evaluate_portfolio_with_metrics(portfolio.ppu_counts)
                    if metrics is not None:
                        seed_evaluated.append((portfolio.ppu_counts.copy(), metrics))
                
                if verbose:
                    print(f"Valid seed portfolios: {len(seed_evaluated)}/{len(initial_portfolios)}")
                
                # Generate additional portfolios to reach pop_size
                additional_needed = max(0, pop_size - len(seed_evaluated))
                if additional_needed > 0 and verbose:
                    print(f"Generating {additional_needed} additional portfolios...")
                
                additional = self.generate_initial_population(
                    pop_size=additional_needed,
                    seed=42,
                    verbose=verbose
                ) if additional_needed > 0 else []
                
                shared_population = seed_evaluated + additional
            else:
                shared_population = self.generate_initial_population(
                    pop_size=pop_size,
                    seed=42,  # Fixed seed for reproducibility
                    verbose=verbose
                )
            if verbose:
                print()
        
        for obj_idx, objective in enumerate(objectives):
            seed = seeds[obj_idx] if obj_idx < len(seeds) else 42 + obj_idx
            
            evaluated = self.run_single_objective_ga(
                objective=objective,
                n_generations=n_generations,
                pop_size=pop_size,
                seed=seed,
                verbose=verbose,
                initial_population=shared_population
            )
            
            all_evaluated.extend(evaluated)
            
            if verbose:
                print(f"Evaluated {len(evaluated)} portfolios for '{objective.name}'")
        
        # Remove duplicates (same portfolio counts)
        unique_portfolios = {}
        for counts, metrics in all_evaluated:
            portfolio_key = tuple(sorted(counts.items()))
            if portfolio_key not in unique_portfolios:
                unique_portfolios[portfolio_key] = (counts, metrics)
        
        if verbose:
            print(f"\nTotal unique portfolios: {len(unique_portfolios)}")
        
        # Convert to DataFrame
        rows = []
        for counts, metrics in unique_portfolios.values():
            rows.append({
                'portfolio_dict': json.dumps(counts),
                'x_RoT': metrics.x_rot,
                'y_volatility': metrics.y_volatility,
                'z_return': metrics.z_return,
                'total_energy_twh': metrics.total_energy_twh,
                'annual_production_twh': metrics.annual_production_twh,
                # CONSTRAINT FIELDS - Required for compliance filtering in notebook
                'storage_constraint_met': metrics.storage_constraint_met,
                'total_domestic_production_twh': metrics.total_domestic_production_twh,
                'aviation_fuel_constraint_met': metrics.aviation_fuel_constraint_met,
            })
        
        df = pd.DataFrame(rows)
        
        elapsed = time.time() - start_time
        
        if verbose:
            print(f"\n{'='*60}")
            print("EXPLORATION COMPLETE")
            print(f"{'='*60}")
            print(f"Time elapsed: {elapsed:.1f}s")
            print(f"Total portfolios: {len(df)}")
            print()
            print("Metrics ranges:")
            print(f"  RoT: [{df['x_RoT'].min():.4f}, {df['x_RoT'].max():.4f}]")
            print(f"  Volatility: [{df['y_volatility'].min():.2f}, {df['y_volatility'].max():.2f}]")
            print(f"  Return: [{df['z_return'].min():.2f}, {df['z_return'].max():.2f}]")
        
        return df


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def explore_and_find_frontier(
    config: Config = DEFAULT_CONFIG,
    objectives: Optional[List[ObjectiveConfig]] = None,
    n_generations: int = 15,
    pop_size: int = 25,
    output_path: Optional[str] = None,
    verbose: bool = True,
    use_existing_as_seeds: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Complete workflow: explore with multiple objectives and find Pareto frontier.
    
    If use_existing_as_seeds=True and output_path exists, loads existing results
    and uses them as seeds for continued exploration.
    
    Args:
        config: Configuration
        objectives: Objective configurations (default: all predefined)
        n_generations: Generations per objective
        pop_size: Population size
        output_path: Output CSV path
        verbose: Print progress
        use_existing_as_seeds: If True and output_path exists, use existing
                              portfolios as seeds for continuation
        
    Returns:
        Tuple of (all_results_df, pareto_frontier_df)
    """
    import os
    
    # Set default output path with scenario if not provided
    if output_path is None:
        scenario = config.energy_system.DEMAND_SCENARIO
        output_path = f"data/result_plots/multi_objective_results_{scenario}.csv"
    
    # Check if we should continue from existing results
    seed_portfolios = None
    existing_df = None
    
    if use_existing_as_seeds and os.path.exists(output_path):
        if verbose:
            print("="*60)
            print("CONTINUING FROM EXISTING RESULTS")
            print("="*60)
            print(f"Loading existing results from: {output_path}")
        
        existing_df = pd.read_csv(output_path)
        
        if verbose:
            print(f"Found {len(existing_df)} existing portfolios")
        
        # Select seed portfolios: frontier + top performers
        # 1. 2D Frontier portfolios
        existing_frontier = extract_pareto_frontier_from_df(
            existing_df, x_col='x_RoT', y_col='y_volatility',
            minimize_x=True, minimize_y=True
        )
        
        # 2. Top portfolios by return
        top_return = existing_df.nlargest(5, 'z_return')
        
        # 3. Top portfolios by low RoT
        top_rot = existing_df.nsmallest(5, 'x_RoT')
        
        # 4. Top portfolios by low volatility
        top_vol = existing_df.nsmallest(5, 'y_volatility')
        
        # Combine and deduplicate
        seed_df = pd.concat([existing_frontier, top_return, top_rot, top_vol]).drop_duplicates(subset=['portfolio_dict'])
        
        # Limit to reasonable number
        n_seeds = min(20, len(seed_df), pop_size)
        seed_df = seed_df.head(n_seeds)
        
        # Convert to Portfolio objects
        seed_portfolios = []
        for _, row in seed_df.iterrows():
            pf_str = row['portfolio_dict']
            if isinstance(pf_str, str):
                pf_dict = json.loads(pf_str)
            else:
                pf_dict = pf_str
            seed_portfolios.append(Portfolio(pf_dict))
        
        if verbose:
            print(f"Selected {len(seed_portfolios)} seed portfolios:")
            print(f"  - {len(existing_frontier)} from 2D frontier")
            print(f"  - Top 5 by return, RoT, and volatility")
            print()
    
    # Run multi-objective exploration
    explorer = MultiObjectiveGA(config)
    new_df = explorer.explore_multi_objective(
        objectives=objectives,
        n_generations=n_generations,
        pop_size=pop_size,
        verbose=verbose,
        initial_portfolios=seed_portfolios  # Use seeds if available
    )
    
    # Combine with existing if we had seeds
    if existing_df is not None:
        all_df = pd.concat([existing_df, new_df], ignore_index=True)
        # Remove duplicates
        all_df = all_df.drop_duplicates(subset=['portfolio_dict'], keep='last')
        if verbose:
            print(f"Combined results: {len(all_df)} unique portfolios")
            print(f"  - Existing: {len(existing_df)}")
            print(f"  - New: {len(new_df)}")
    else:
        all_df = new_df
    
    # Save all results
    all_df.to_csv(output_path, index=False)
    if verbose:
        print(f"\nAll results saved to: {output_path}")
    
    # Find Pareto frontier
    pareto_df = extract_pareto_frontier_from_df(
        all_df,
        x_col='x_RoT',
        y_col='y_volatility',
        minimize_x=True,
        minimize_y=True
    )
    
    # Save frontier
    frontier_path = output_path.replace('.csv', '_frontier.csv')
    pareto_df.to_csv(frontier_path, index=False)
    
    if verbose:
        print(f"\n{'='*60}")
        print("PARETO FRONTIER")
        print(f"{'='*60}")
        print(f"Frontier portfolios: {len(pareto_df)} out of {len(all_df)}")
        print(f"Frontier saved to: {frontier_path}")
        print()
        print("Frontier ranges:")
        print(f"  RoT: [{pareto_df['x_RoT'].min():.4f}, {pareto_df['x_RoT'].max():.4f}]")
        print(f"  Volatility: [{pareto_df['y_volatility'].min():.2f}, {pareto_df['y_volatility'].max():.2f}]")
    
    return all_df, pareto_df


def continue_exploration_3d_frontier(
    config: Config = DEFAULT_CONFIG,
    existing_results_path: Optional[str] = None,
    objectives: Optional[List[ObjectiveConfig]] = None,
    n_generations: int = 10,
    pop_size: int = 20,
    n_seed_portfolios: int = 20,
    output_path: Optional[str] = None,
    verbose: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Continue exploration from existing results and compute 3D Pareto frontier.
    
    Uses the best portfolios from existing results as seeds for further exploration.
    Then computes the 3D Pareto frontier (RoT, Volatility, Return).
    
    Args:
        config: Configuration
        existing_results_path: Path to existing results CSV
        objectives: Objective configurations (default: all predefined)
        n_generations: Generations per objective
        pop_size: Population size
        n_seed_portfolios: Number of top portfolios to seed from existing results
        output_path: Output CSV path
        verbose: Print progress
        
    Returns:
        Tuple of (all_results_df, pareto_frontier_3d_df)
    """
    import os
    
    # Set default paths with scenario if not provided
    scenario = config.energy_system.DEMAND_SCENARIO
    if existing_results_path is None:
        existing_results_path = f"data/result_plots/multi_objective_results_{scenario}.csv"
    if output_path is None:
        output_path = f"data/result_plots/multi_objective_results_{scenario}_3d.csv"
    
    if verbose:
        print("="*80)
        print("3D PARETO FRONTIER EXPLORATION (Continuation)")
        print("="*80)
        print()
    
    # Load existing results
    if not os.path.exists(existing_results_path):
        raise FileNotFoundError(
            f"ERROR: Existing results not found: {existing_results_path}\n"
            f"Please run the initial multi-objective exploration first."
        )
    
    existing_df = pd.read_csv(existing_results_path)
    if verbose:
        print(f"Loaded {len(existing_df)} existing portfolios from: {existing_results_path}")
    
    # Select seed portfolios from existing results
    # Strategy: use frontier + top portfolios by various criteria
    seed_portfolios = []
    
    # 1. Existing 2D frontier portfolios (minimize RoT, minimize Volatility)
    existing_frontier = extract_pareto_frontier_from_df(
        existing_df, x_col='x_RoT', y_col='y_volatility',
        minimize_x=True, minimize_y=True
    )
    frontier_dicts = existing_frontier['portfolio_dict'].tolist()
    for pf_str in frontier_dicts:
        if isinstance(pf_str, str):
            pf_dict = json.loads(pf_str)
        else:
            pf_dict = pf_str
        seed_portfolios.append(Portfolio(pf_dict))
    
    if verbose:
        print(f"  - {len(frontier_dicts)} frontier portfolios")
    
    # 2. Top portfolios by return (highest return)
    top_return = existing_df.nlargest(5, 'z_return')['portfolio_dict'].tolist()
    for pf_str in top_return:
        if isinstance(pf_str, str):
            pf_dict = json.loads(pf_str)
        else:
            pf_dict = pf_str
        if pf_dict not in [p.ppu_counts for p in seed_portfolios]:
            seed_portfolios.append(Portfolio(pf_dict))
    
    if verbose:
        print(f"  - Added top 5 by return")
    
    # 3. Top portfolios by low RoT
    top_rot = existing_df.nsmallest(5, 'x_RoT')['portfolio_dict'].tolist()
    for pf_str in top_rot:
        if isinstance(pf_str, str):
            pf_dict = json.loads(pf_str)
        else:
            pf_dict = pf_str
        if pf_dict not in [p.ppu_counts for p in seed_portfolios]:
            seed_portfolios.append(Portfolio(pf_dict))
    
    if verbose:
        print(f"  - Added top 5 by low RoT")
    
    # 4. Top portfolios by low volatility
    top_vol = existing_df.nsmallest(5, 'y_volatility')['portfolio_dict'].tolist()
    for pf_str in top_vol:
        if isinstance(pf_str, str):
            pf_dict = json.loads(pf_str)
        else:
            pf_dict = pf_str
        if pf_dict not in [p.ppu_counts for p in seed_portfolios]:
            seed_portfolios.append(Portfolio(pf_dict))
    
    if verbose:
        print(f"  - Added top 5 by low volatility")
        print(f"  Total seed portfolios: {len(seed_portfolios)}")
        print()
    
    # Limit to n_seed_portfolios
    if len(seed_portfolios) > n_seed_portfolios:
        seed_portfolios = seed_portfolios[:n_seed_portfolios]
        if verbose:
            print(f"  Limited to {n_seed_portfolios} seed portfolios")
    
    # Run multi-objective exploration with seeds
    if verbose:
        print()
        print("Running continued exploration...")
        print("="*60)
    
    explorer = MultiObjectiveGA(config)
    
    # Override initial population generation to use seeds
    new_df = explorer.explore_multi_objective(
        objectives=objectives,
        n_generations=n_generations,
        pop_size=pop_size,
        verbose=verbose,
        initial_portfolios=seed_portfolios  # Pass seeds
    )
    
    # Combine existing and new results
    combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    
    # Remove duplicates by portfolio_dict
    combined_df = combined_df.drop_duplicates(subset=['portfolio_dict'], keep='last')
    
    if verbose:
        print()
        print(f"Combined results: {len(combined_df)} unique portfolios")
        print(f"  - Existing: {len(existing_df)}")
        print(f"  - New: {len(new_df)}")
    
    # Save combined results
    combined_df.to_csv(output_path, index=False)
    if verbose:
        print(f"\nAll results saved to: {output_path}")
    
    # Find 3D Pareto frontier
    # Minimize RoT, Minimize Volatility, MAXIMIZE Return
    pareto_3d_df = extract_pareto_frontier_3d_from_df(
        combined_df,
        x_col='x_RoT',
        y_col='y_volatility',
        z_col='z_return',
        minimize_x=True,
        minimize_y=True,
        minimize_z=False  # MAXIMIZE return
    )
    
    # Save 3D frontier
    frontier_path = output_path.replace('.csv', '_frontier_3d.csv')
    pareto_3d_df.to_csv(frontier_path, index=False)
    
    if verbose:
        print()
        print("="*60)
        print("3D PARETO FRONTIER")
        print("="*60)
        print(f"3D Frontier portfolios: {len(pareto_3d_df)} out of {len(combined_df)}")
        print(f"Frontier saved to: {frontier_path}")
        print()
        print("3D Frontier ranges:")
        print(f"  RoT: [{pareto_3d_df['x_RoT'].min():.4f}, {pareto_3d_df['x_RoT'].max():.4f}]")
        print(f"  Volatility: [{pareto_3d_df['y_volatility'].min():.2f}, {pareto_3d_df['y_volatility'].max():.2f}]")
        print(f"  Return: [{pareto_3d_df['z_return'].min():.2f}%, {pareto_3d_df['z_return'].max():.2f}%]")
    
    return combined_df, pareto_3d_df


if __name__ == "__main__":
    # Example usage
    config = DEFAULT_CONFIG
    config.run_mode.MODE = "quick"
    
    all_df, pareto_df = explore_and_find_frontier(
        config=config,
        n_generations=10,  # Quick test
        pop_size=20,
        # output_path=None uses default with scenario from config
        verbose=True
    )
    
    print(f"\n✅ Found {len(pareto_df)} Pareto-optimal portfolios")

