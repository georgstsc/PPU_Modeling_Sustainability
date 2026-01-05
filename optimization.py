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
from tqdm import tqdm

from config import Config, DEFAULT_CONFIG
from data_loader import load_all_data, sample_scenario_indices, get_scenario_data, CachedData
from ppu_framework import (
    Portfolio, PPUDefinition, load_all_ppu_data, 
    create_ppu_dictionary, assign_renewable_locations,
    estimate_annual_production, check_energy_sovereignty,
    check_cumulative_energy_balance, find_minimum_renewable_portfolio
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
    
    # Penalty tracking
    progressive_penalty: float = 0.0
    soc_penalty: float = 0.0  # Penalty for depleting storage below initial SoC
    penalty_breakdown: Dict[str, Any] = field(default_factory=dict)
    
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
# PROGRESSIVE COST CALCULATION
# =============================================================================

def calculate_progressive_cost_penalty(
    portfolio: Portfolio,
    ppu_definitions: Dict[str, PPUDefinition],
    config: Config,
) -> Tuple[float, Dict[str, float]]:
    """
    Calculate progressive cost penalty for exceeding soft caps.
    
    For each PPU type, if units > soft_cap:
        penalty_multiplier = 1 + factor * (units - soft_cap)
        penalty = base_cost * (multiplier - 1) * excess_units
    
    Args:
        portfolio: Portfolio to evaluate
        ppu_definitions: PPU definitions with base costs
        config: Configuration with progressive cost caps
        
    Returns:
        Tuple of (total_penalty_chf_per_year, breakdown_by_ppu)
    """
    progressive_caps = config.ppu.PROGRESSIVE_COST_CAPS
    mw_per_unit = config.ppu.MW_PER_UNIT
    hours_per_year = 8760
    
    total_penalty = 0.0
    breakdown = {}
    
    for ppu_name, count in portfolio.ppu_counts.items():
        if count <= 0 or ppu_name not in progressive_caps:
            continue
        
        cap_info = progressive_caps[ppu_name]
        soft_cap = cap_info.get('soft_cap', 9999)
        factor = cap_info.get('factor', 0.0)
        
        if count <= soft_cap or factor <= 0:
            continue
        
        # Get base cost for this PPU (CHF/MWh)
        # CRITICAL: No fallback prices allowed - must find actual cost
        base_cost = None
        if ppu_name in ppu_definitions:
            ppu_def = ppu_definitions[ppu_name]
            if hasattr(ppu_def, 'cost_per_mwh'):
                # cost_per_mwh is in CHF/kWh, convert to CHF/MWh
                base_cost = ppu_def.cost_per_mwh * 1000.0
            elif isinstance(ppu_def, dict) and 'cost_per_mwh' in ppu_def:
                base_cost = ppu_def['cost_per_mwh'] * 1000.0
        
        # CRITICAL: Raise error if cost not found - no fallback allowed
        if base_cost is None:
            raise ValueError(
                f"CRITICAL: Cannot find cost_per_mwh for PPU '{ppu_name}' for progressive cost calculation. "
                f"Cannot use fallback price as it would falsify costs. "
                f"Please ensure PPU cost is defined in ppu_definitions."
            )
        
        # Calculate excess units
        excess_units = count - soft_cap
        
        # Progressive penalty: each excess unit costs more
        # Unit 1 over cap: base_cost * (1 + factor*1)
        # Unit 2 over cap: base_cost * (1 + factor*2)
        # etc.
        # Sum of penalties = base_cost * factor * (1 + 2 + ... + excess)
        #                  = base_cost * factor * excess * (excess + 1) / 2
        penalty_multiplier = factor * excess_units * (excess_units + 1) / 2
        
        # Convert to annual cost (assume 50% capacity factor average)
        capacity_factor = 0.5
        annual_mwh = excess_units * mw_per_unit * hours_per_year * capacity_factor
        penalty = base_cost * penalty_multiplier * annual_mwh / excess_units
        
        total_penalty += penalty
        breakdown[ppu_name] = {
            'count': count,
            'soft_cap': soft_cap,
            'excess': excess_units,
            'factor': factor,
            'penalty_chf': penalty,
        }
    
    return total_penalty, breakdown


def get_portfolio_cost_multiplier(
    portfolio: Portfolio,
    config: Config,
) -> Dict[str, float]:
    """
    Get cost multipliers for each PPU type based on current counts.
    
    Returns dictionary of {ppu_name: multiplier} where multiplier >= 1.0
    """
    progressive_caps = config.ppu.PROGRESSIVE_COST_CAPS
    multipliers = {}
    
    for ppu_name, count in portfolio.ppu_counts.items():
        if count <= 0:
            multipliers[ppu_name] = 1.0
            continue
        
        if ppu_name not in progressive_caps:
            multipliers[ppu_name] = 1.0
            continue
        
        cap_info = progressive_caps[ppu_name]
        soft_cap = cap_info.get('soft_cap', 9999)
        factor = cap_info.get('factor', 0.0)
        
        if count <= soft_cap or factor <= 0:
            multipliers[ppu_name] = 1.0
        else:
            excess = count - soft_cap
            # Average multiplier for all units above soft cap
            multipliers[ppu_name] = 1.0 + factor * excess
    
    return multipliers


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
    
    Checks performed (in order):
    1. Capacity factor estimate (fast, rough check)
    2. Cumulative energy balance (accurate, uses real incidence data)
    3. Dispatch simulation (detailed cost calculation)
    
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
    
    # --- STEP 1: Quick capacity factor check ---
    is_sovereign, annual_prod, target = check_energy_sovereignty(
        portfolio, ppu_definitions, config
    )
    individual.annual_production_twh = annual_prod
    individual.is_sovereign = is_sovereign
    
    # Heavy penalty if capacity factor estimate fails
    if not is_sovereign:
        deficit = target - annual_prod
        penalty = config.fitness.SOVEREIGNTY_PENALTY_MULTIPLIER * deficit
        individual.fitness = penalty
        individual.mean_cost = penalty
        individual.cvar = penalty
        return individual
    
    # --- STEP 1.5: Aviation compliance check ---
    # Portfolios MUST include aviation-compliant storage PPUs to meet 23 TWh/year requirement
    # Aviation fuel comes from Fuel Tank, which is charged by SYN_FT or SYN_CRACK
    aviation_ppus = ['SYN_FT', 'SYN_CRACK']
    has_aviation = any(portfolio.ppu_counts.get(ppu, 0) > 0 for ppu in aviation_ppus)
    
    if not has_aviation:
        # Heavy penalty: cannot meet 23 TWh/year aviation fuel requirement
        # Use 50% of sovereignty penalty as this is a critical requirement
        penalty = config.fitness.SOVEREIGNTY_PENALTY_MULTIPLIER * 0.5
        individual.fitness = penalty
        individual.mean_cost = penalty
        individual.cvar = penalty
        individual.penalty_breakdown = {
            'aviation_compliance': {
                'missing': True,
                'penalty': penalty,
                'required_ppus': aviation_ppus,
                'message': 'Portfolio must include SYN_FT or SYN_CRACK for aviation fuel'
            }
        }
        if verbose:
            print(f"  ❌ Aviation compliance FAILED: No {aviation_ppus} in portfolio")
            print(f"     Penalty: {penalty:,.0f} CHF/year")
        return individual
    
    # --- STEP 2: Cumulative energy balance check (using real incidence data) ---
    # This ensures total renewable production >= total demand (with storage losses)
    # NOTE: This check is expensive! Disabled by default (config.fitness.CHECK_CUMULATIVE_BALANCE)
    if config.fitness.CHECK_CUMULATIVE_BALANCE:
        # Use pre-computed sums for speed (computed once during data loading)
        precomputed = data.get_precomputed_sums() if hasattr(data, 'get_precomputed_sums') else None
        
        is_balanced, total_prod_mwh, total_demand_mwh, balance_ratio, prod_breakdown = \
            check_cumulative_energy_balance(
                portfolio, ppu_definitions,
                data.get_solar_incidence(copy=False),  # Read-only, no copy needed
                data.get_wind_incidence(copy=False),   # Read-only, no copy needed
                data.get_demand(copy=False),           # Read-only, no copy needed
                data.solar_ranking,
                data.wind_ranking,
                config,
                storage_efficiency=config.fitness.STORAGE_ROUND_TRIP_EFFICIENCY,
                precomputed_sums=precomputed,
            )
        
        # Store balance info
        individual.annual_production_twh = total_prod_mwh / 1e6  # Update with actual calculation
        
        # Penalty if cumulative production can't cover demand
        if not is_balanced:
            # Penalty proportional to how far below requirement
            shortfall_ratio = 1.0 - balance_ratio
            penalty = config.fitness.SOVEREIGNTY_PENALTY_MULTIPLIER * shortfall_ratio * 10
            individual.fitness = penalty
            individual.mean_cost = penalty
            individual.cvar = penalty
            individual.is_sovereign = False
            if verbose:
                print(f"  ⚠️ Cumulative balance ratio: {balance_ratio:.2f} (need ≥ 1.0)")
            return individual
    
    # --- STEP 3: Detailed dispatch simulation ---
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
        
        # Compute cost (DETAILED: uses actual per-PPU production and costs)
        cost = compute_scenario_cost(results, ppu_dict, config, ppu_definitions)
        # NOTE: **results must come FIRST so that our computed 'cost' overrides
        # the raw 'net_spot_cost_chf' from dispatch_engine
        scenario_results.append({**results, 'net_spot_cost_chf': cost})
        individual.scenario_costs.append(cost)
    
    # Compute aggregate metrics
    metrics = compute_portfolio_metrics(scenario_results, config)
    individual.mean_cost = metrics['mean_cost']
    individual.cvar = metrics['cvar']
    
    # --- STEP 4: Check final storage SoC constraint ---
    # CYCLIC CONSTRAINT: Storage must end at same level as it started (±tolerance)
    # This prevents "cheating" by either depleting OR overcharging initial reserves
    # Ensures sustainable year-over-year operation
    initial_soc_default = config.storage.INITIAL_SOC_FRACTION
    tolerance = config.storage.FINAL_SOC_TOLERANCE
    soc_penalty = 0.0
    soc_violations = []
    
    for result in scenario_results:
        initial_socs = result.get('initial_storage_soc', {})
        final_socs = result.get('final_storage_soc', {})
        
        for storage_name, final_soc in final_socs.items():
            # Get initial SoC for this storage (use default if not provided)
            initial_soc = initial_socs.get(storage_name, initial_soc_default)
            
            # CYCLIC CONSTRAINT: Check absolute deviation from initial
            soc_deviation = abs(final_soc - initial_soc)
            max_allowed_deviation = tolerance * initial_soc  # ±5% of initial
            
            if soc_deviation > max_allowed_deviation:
                # Penalty proportional to deviation magnitude
                deviation_ratio = soc_deviation / max(initial_soc, 0.01)
                penalty_amount = config.storage.FINAL_SOC_PENALTY_MULTIPLIER * deviation_ratio
                soc_penalty += penalty_amount
                
                soc_violations.append({
                    'storage': storage_name,
                    'initial_soc': initial_soc,
                    'final_soc': final_soc,
                    'deviation': soc_deviation,
                    'direction': 'depleted' if final_soc < initial_soc else 'overcharged'
                })
    
    # Average penalty across scenarios
    if len(scenario_results) > 0:
        soc_penalty /= len(scenario_results)
    
    # Add SoC penalty to mean cost
    individual.mean_cost += soc_penalty
    individual.soc_penalty = soc_penalty
    
    # --- STEP 5: Add progressive cost penalty for exceeding soft caps ---
    progressive_penalty, penalty_breakdown = calculate_progressive_cost_penalty(
        portfolio, ppu_definitions, config
    )
    
    # Add penalty to mean cost (annual basis)
    individual.mean_cost += progressive_penalty
    
    # Store penalty info for debugging
    individual.progressive_penalty = progressive_penalty
    individual.penalty_breakdown = penalty_breakdown
    
    if verbose and progressive_penalty > 0:
        print(f"  Progressive cost penalty: {progressive_penalty:,.0f} CHF/year")
        for ppu, info in penalty_breakdown.items():
            print(f"    {ppu}: {info['count']} units (cap: {info['soft_cap']}, +{info['excess']} excess)")
    
    if verbose and soc_penalty > 0:
        print(f"  ⚠️  Storage SoC penalty: {soc_penalty:,.0f} CHF/year (storage cyclic constraint violated)")
        for violation in soc_violations[:3]:  # Show first 3 violations
            direction = violation['direction']
            print(f"     {violation['storage']}: {violation['initial_soc']:.2%} → {violation['final_soc']:.2%} ({direction})")
    
    # Total penalty
    total_penalty = progressive_penalty + soc_penalty
    
    # Fitness: use combined cost (harmonic mean approach optional)
    if config.fitness.COST_AGGREGATION == 'harmonic_mean':
        # Harmonic mean is more sensitive to high costs
        costs = [max(c + total_penalty, 1e-6) for c in individual.scenario_costs]
        individual.fitness = len(costs) / sum(1/c for c in costs)
    else:
        individual.fitness = metrics['combined_cost'] + total_penalty
    
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
    pop_iter = tqdm(
        population,
        desc="Evaluating population",
        unit="ind",
        disable=not verbose,
        ncols=80,
        leave=False
    )
    
    for ind in pop_iter:
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
    stats.mean_fitness = float(np.mean([ind.fitness for ind in population]))
    stats.best_fitness_history.append(stats.best_fitness)
    stats.mean_fitness_history.append(stats.mean_fitness)  # Include initial mean
    
    best_ever = population[0].copy()
    
    if verbose:
        print(f"\nInitial best fitness: {stats.best_fitness:,.0f}")
        if best_ever.is_sovereign:
            print(f"  Production: {best_ever.annual_production_twh:.1f} TWh/year")
        else:
            print(f"  NOT SOVEREIGN: {best_ever.annual_production_twh:.1f} TWh/year")
        print()
    
    # Evolution loop with progress bar
    pbar = tqdm(
        range(n_generations),
        desc="Evolving",
        unit="gen",
        disable=not verbose,
        ncols=100,
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
    )
    
    for generation in pbar:
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
        stats.mean_fitness = float(np.mean([ind.fitness for ind in population]))
        stats.worst_fitness = population[-1].fitness
        
        # Check for improvement
        if current_best < stats.best_fitness:
            improvement = stats.best_fitness - current_best
            stats.best_fitness = current_best
            stats.generations_without_improvement = 0
            best_ever = population[0].copy()
            status = f"↓ NEW BEST! -{improvement:,.0f}"
        else:
            stats.generations_without_improvement += 1
            status = f"plateau: {stats.generations_without_improvement}/{plateau_generations}"
        
        # Update progress bar with current status
        pbar.set_postfix({
            'best': f'{stats.best_fitness:,.0f}',
            'status': status,
        }, refresh=True)
        
        stats.best_fitness_history.append(stats.best_fitness)
        stats.mean_fitness_history.append(stats.mean_fitness)
        
        # Progress callback
        if progress_callback:
            progress_callback(stats)
        
        # Check stopping criteria
        if stats.generations_without_improvement >= plateau_generations:
            pbar.set_description("🛑 Plateau detected")
            break
        
        # Time check for overnight mode
        elapsed = time.time() - start_time
        stats.elapsed_time_s = elapsed
    
    # Close progress bar
    pbar.close()
        
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
# FULL YEAR EVALUATION
# =============================================================================

@dataclass
class FullYearResults:
    """Results from a full year evaluation."""
    # Time series (hourly resolution)
    demand: np.ndarray
    total_production: np.ndarray
    spot_prices: np.ndarray
    spot_bought: np.ndarray
    spot_sold: np.ndarray
    deficit: np.ndarray
    surplus: np.ndarray
    
    # Production breakdown (hourly)
    renewable_production: np.ndarray = field(default_factory=lambda: np.array([]))
    dispatchable_production: np.ndarray = field(default_factory=lambda: np.array([]))
    
    # Per-PPU production (dict: ppu_name -> total MWh or hourly array)
    ppu_production: Dict[str, Any] = field(default_factory=dict)
    
    # Storage states (dict: storage_name -> hourly SoC array)
    storage_soc: Dict[str, np.ndarray] = field(default_factory=dict)
    
    # Aggregate metrics
    total_production_twh: float = 0.0
    total_demand_twh: float = 0.0
    total_spot_cost_chf: float = 0.0
    total_spot_bought_mwh: float = 0.0
    total_spot_sold_mwh: float = 0.0
    coverage_ratio: float = 0.0
    peak_deficit_mw: float = 0.0
    hours_in_deficit: int = 0
    
    # =========================================================================
    # AVIATION FUEL METRICS (23 TWh/year biooil requirement)
    # =========================================================================
    aviation_fuel_consumed_mwh: float = 0.0
    aviation_fuel_shortfall_mwh: float = 0.0
    aviation_fuel_import_cost_chf: float = 0.0
    aviation_fuel_constraint_met: bool = False
    aviation_fuel_consumed_series: np.ndarray = field(default_factory=lambda: np.array([]))
    aviation_fuel_shortfall_series: np.ndarray = field(default_factory=lambda: np.array([]))
    
    # =========================================================================
    # FINAL STORAGE CONSTRAINT (must end ±5% of initial SoC)
    # =========================================================================
    storage_constraint_met: bool = False
    storage_constraint_violations: Dict[str, Dict[str, float]] = field(default_factory=dict)
    storage_constraint_penalty: float = 0.0
    
    # Monthly breakdowns
    monthly_production: Optional[np.ndarray] = None
    monthly_demand: Optional[np.ndarray] = None


def evaluate_portfolio_full_year(
    individual: Individual,
    config: Config = DEFAULT_CONFIG,
    verbose: bool = True,
) -> FullYearResults:
    """
    Evaluate a portfolio across the entire year (all 8760 hours).
    
    This is the final validation step after optimization to test
    robustness across non-random data.
    
    Args:
        individual: Optimized portfolio individual
        config: Configuration
        verbose: Print progress
        
    Returns:
        FullYearResults with detailed time series and metrics
    """
    from dispatch_engine import run_dispatch_simulation
    from ppu_framework import load_all_ppu_data, assign_renewable_locations
    
    if verbose:
        print("=" * 60)
        print("FULL YEAR EVALUATION")
        print("=" * 60)
        print()
    
    # Load data
    if verbose:
        print("Loading full year data...")
    data = load_all_data(config)
    
    # Load PPU definitions
    if verbose:
        print("Loading PPU definitions...")
    _, _, ppu_definitions = load_all_ppu_data(config)
    
    # Create PPU dictionary DataFrame from portfolio
    ppu_dict = create_ppu_dictionary(individual.portfolio, ppu_definitions, config)
    
    # Assign renewable locations
    ppu_dict = assign_renewable_locations(
        ppu_dict,
        data.solar_ranking,
        data.wind_ranking,
    )
    
    # Get full year data
    n_hours = data.n_hours
    demand = data.get_demand()
    spot = data.get_spot_prices()
    solar = data.get_solar_incidence()
    wind = data.get_wind_incidence()
    
    if verbose:
        print(f"Simulating {n_hours} hours ({n_hours / 24:.0f} days)...")
    
    # Run dispatch for entire year
    _, results = run_dispatch_simulation(
        scenario_indices=np.arange(n_hours),
        ppu_dictionary=ppu_dict,
        demand_data=demand,
        spot_data=spot,
        solar_data=solar,
        wind_data=wind,
        portfolio_counts=individual.portfolio.ppu_counts,
        config=config,
        verbose=False,
    )
    
    # Extract results
    full_year = FullYearResults(
        demand=demand,
        total_production=results.get('total_production', np.zeros(n_hours)),
        spot_prices=spot,
        spot_bought=results.get('spot_bought', np.zeros(n_hours)),
        spot_sold=results.get('spot_sold', np.zeros(n_hours)),
        deficit=results.get('deficit', np.zeros(n_hours)),
        surplus=results.get('surplus', np.zeros(n_hours)),
        renewable_production=results.get('renewable_production', np.zeros(n_hours)),
        dispatchable_production=results.get('dispatchable_production', np.zeros(n_hours)),
        ppu_production=results.get('ppu_production', {}),
        storage_soc=results.get('storage_soc', {}),
        # Aviation fuel metrics
        aviation_fuel_consumed_mwh=results.get('aviation_fuel_consumed_mwh', 0.0),
        aviation_fuel_shortfall_mwh=results.get('aviation_fuel_shortfall_mwh', 0.0),
        aviation_fuel_import_cost_chf=results.get('aviation_fuel_import_cost_chf', 0.0),
        aviation_fuel_constraint_met=results.get('aviation_fuel_constraint_met', False),
        aviation_fuel_consumed_series=results.get('aviation_fuel_consumed_series', np.zeros(n_hours)),
        aviation_fuel_shortfall_series=results.get('aviation_fuel_shortfall_series', np.zeros(n_hours)),
    )
    
    # Compute aggregate metrics
    full_year.total_demand_twh = np.sum(demand) / 1e6
    full_year.total_production_twh = np.sum(full_year.total_production) / 1e6
    full_year.total_spot_bought_mwh = np.sum(full_year.spot_bought)
    full_year.total_spot_sold_mwh = np.sum(full_year.spot_sold)
    
    # Spot market cost (buy - sell)
    full_year.total_spot_cost_chf = (
        np.sum(full_year.spot_bought * spot) - 
        np.sum(full_year.spot_sold * spot)
    )
    
    # Coverage and deficit stats
    full_year.coverage_ratio = full_year.total_production_twh / max(full_year.total_demand_twh, 1e-9)
    full_year.peak_deficit_mw = float(np.max(full_year.deficit))
    full_year.hours_in_deficit = int(np.sum(full_year.deficit > 0))
    
    # Monthly breakdowns (assuming hourly data, 2024 = 366 days)
    hours_per_month = [744, 696, 744, 720, 744, 720, 744, 744, 720, 744, 720, 744]
    monthly_prod = []
    monthly_dem = []
    idx = 0
    for h in hours_per_month:
        end_idx = min(idx + h, n_hours)
        monthly_prod.append(np.sum(full_year.total_production[idx:end_idx]) / 1e6)
        monthly_dem.append(np.sum(demand[idx:end_idx]) / 1e6)
        idx = end_idx
        if idx >= n_hours:
            break
    
    full_year.monthly_production = np.array(monthly_prod)
    full_year.monthly_demand = np.array(monthly_dem)
    
    # =========================================================================
    # FINAL STORAGE CONSTRAINT CHECK
    # =========================================================================
    # System must end with storage levels within ±5% of initial levels
    # This prevents "cheating" by depleting storage over the simulation year
    tolerance = config.storage.FINAL_SOC_TOLERANCE
    initial_soc = config.storage.INITIAL_SOC_FRACTION
    
    violations = {}
    all_met = True
    total_penalty = 0.0
    
    for storage_name, soc_series in full_year.storage_soc.items():
        if len(soc_series) == 0:
            continue
        
        # Get initial and final SoC (as fractions)
        initial_soc_val = soc_series[0] if len(soc_series) > 0 else initial_soc
        final_soc_val = soc_series[-1] if len(soc_series) > 0 else initial_soc
        
        # Check if final SoC is within tolerance of initial SoC
        soc_deviation = abs(final_soc_val - initial_soc_val)
        max_allowed_deviation = tolerance * initial_soc_val  # ±5% of initial
        
        if soc_deviation > max_allowed_deviation:
            all_met = False
            # Calculate penalty proportional to deviation
            deviation_ratio = soc_deviation / max(initial_soc_val, 0.01)
            penalty = config.storage.FINAL_SOC_PENALTY_MULTIPLIER * deviation_ratio
            total_penalty += penalty
            
            violations[storage_name] = {
                'initial_soc': initial_soc_val,
                'final_soc': final_soc_val,
                'deviation': soc_deviation,
                'max_allowed': max_allowed_deviation,
                'penalty': penalty,
            }
    
    full_year.storage_constraint_met = all_met
    full_year.storage_constraint_violations = violations
    full_year.storage_constraint_penalty = total_penalty
    
    if verbose:
        print()
        print("=" * 60)
        print("FULL YEAR RESULTS")
        print("=" * 60)
        print(f"Total Demand:     {full_year.total_demand_twh:,.1f} TWh")
        print(f"Total Production: {full_year.total_production_twh:,.1f} TWh")
        print(f"Coverage Ratio:   {full_year.coverage_ratio*100:.1f}%")
        print()
        print(f"Spot Market:")
        print(f"  Bought: {full_year.total_spot_bought_mwh/1e6:,.2f} TWh")
        print(f"  Sold:   {full_year.total_spot_sold_mwh/1e6:,.2f} TWh")
        print(f"  Net Cost: {full_year.total_spot_cost_chf/1e6:,.1f} M CHF")
        print()
        print(f"Deficit Analysis:")
        print(f"  Hours in deficit: {full_year.hours_in_deficit} ({full_year.hours_in_deficit/n_hours*100:.1f}%)")
        print(f"  Peak deficit: {full_year.peak_deficit_mw:,.0f} MW")
        print()
        print(f"Aviation Fuel (from Fuel Tank - Synthetic):")
        required_twh = config.energy_system.AVIATION_FUEL_DEMAND_TWH_YEAR
        consumed_twh = full_year.aviation_fuel_consumed_mwh / 1e6
        shortfall_twh = full_year.aviation_fuel_shortfall_mwh / 1e6
        print(f"  Required: {required_twh:,.1f} TWh/year")
        print(f"  Consumed: {consumed_twh:,.2f} TWh ({consumed_twh/required_twh*100:.1f}%)")
        print(f"  Shortfall: {shortfall_twh:,.2f} TWh")
        print(f"  Production Cost: {full_year.aviation_fuel_import_cost_chf/1e6:,.1f} M CHF")
        print(f"  Constraint Met: {'✅ YES' if full_year.aviation_fuel_constraint_met else '❌ NO'}")
        if not full_year.aviation_fuel_constraint_met:
            hours_short = np.sum(full_year.aviation_fuel_shortfall_series > 0)
            print(f"  ⚠️  {hours_short} hours with aviation fuel shortfall")
        
        # Final storage constraint report
        print()
        print(f"Final Storage Constraint (±{tolerance*100:.0f}% of initial SoC):")
        print(f"  Constraint Met: {'✅ YES' if full_year.storage_constraint_met else '❌ NO'}")
        if not full_year.storage_constraint_met:
            print(f"  Total Penalty: {full_year.storage_constraint_penalty:,.0f}")
            for name, v in full_year.storage_constraint_violations.items():
                print(f"  ⚠️  {name}: initial={v['initial_soc']:.2%} → final={v['final_soc']:.2%} "
                      f"(deviation: {v['deviation']:.2%}, max allowed: {v['max_allowed']:.2%})")
        else:
            # Show final SoC for all storages even if constraint is met
            for storage_name, soc_series in full_year.storage_soc.items():
                if len(soc_series) > 0:
                    print(f"  ✓ {storage_name}: {soc_series[0]:.1%} → {soc_series[-1]:.1%}")
    
    return full_year


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

