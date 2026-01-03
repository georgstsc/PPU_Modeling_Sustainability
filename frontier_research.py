"""
Frontier Research Module: Systematic exploration along the Pareto frontier.

This module provides functions to:
1. Identify anchor points (corner portfolios) on the frontier
2. Interpolate between anchor points to find intermediate solutions
3. Evaluate interpolated portfolios using dispatch simulation
4. Identify non-dominated and potentially dominating solutions
5. Export extended frontier results

Author: Energy Portfolio Optimization Project
"""

import pandas as pd
import numpy as np
import json
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass


@dataclass
class FrontierResearchConfig:
    """Configuration for frontier research exploration."""
    alphas: Optional[List[float]] = None
    verbose: bool = True
    output_path: str = 'data/result_plots/extended_frontier_research.csv'
    
    def __post_init__(self):
        if self.alphas is None:
            self.alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


@dataclass
class FrontierResearchResults:
    """Results from frontier research exploration."""
    original_frontier_count: int
    interpolated_count: int
    improved_count: int
    best_return: float
    lowest_rot: float
    lowest_volatility: float
    extended_frontier_df: pd.DataFrame
    improved_portfolios: List[Dict]


def interpolate_portfolios(p1: Dict[str, int], p2: Dict[str, int], alpha: float) -> Dict[str, int]:
    """
    Create weighted combination of two portfolios.
    
    Args:
        p1: First portfolio (PPU name -> unit count)
        p2: Second portfolio (PPU name -> unit count)
        alpha: Weight for p1 (1-alpha for p2)
        
    Returns:
        Interpolated portfolio
    """
    result = {}
    all_keys = set(p1.keys()) | set(p2.keys())
    for key in all_keys:
        v1 = p1.get(key, 0)
        v2 = p2.get(key, 0)
        result[key] = int(round(alpha * v1 + (1 - alpha) * v2))
    return result


def identify_anchor_points(frontier_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Identify corner portfolios (best in each dimension).
    
    Args:
        frontier_df: DataFrame with frontier portfolios
        
    Returns:
        Dict mapping anchor name to portfolio row
    """
    return {
        'Best Return': frontier_df.loc[frontier_df['z_return'].idxmax()],
        'Lowest RoT': frontier_df.loc[frontier_df['x_RoT'].idxmin()],
        'Lowest Volatility': frontier_df.loc[frontier_df['y_volatility'].idxmin()]
    }


def generate_interpolated_portfolios(
    anchors: Dict[str, Any],
    alphas: List[float]
) -> List[Dict]:
    """
    Generate interpolated portfolios along frontier edges.
    
    Args:
        anchors: Dict of anchor portfolios
        alphas: List of interpolation weights
        
    Returns:
        List of interpolated portfolio configs
    """
    edges = [
        ('Best Return', 'Lowest RoT'),
        ('Best Return', 'Lowest Volatility'),
        ('Lowest RoT', 'Lowest Volatility')
    ]
    
    interpolated = []
    for edge_name1, edge_name2 in edges:
        p1 = anchors[edge_name1]['portfolio']
        p2 = anchors[edge_name2]['portfolio']
        
        for alpha in alphas:
            interp_p = interpolate_portfolios(p1, p2, alpha)
            interpolated.append({
                'edge': f"{edge_name1} → {edge_name2}",
                'alpha': alpha,
                'portfolio': interp_p
            })
    
    return interpolated


def check_dominance(
    candidate: Dict,
    frontier_df: pd.DataFrame
) -> Tuple[bool, bool]:
    """
    Check if candidate portfolio is non-dominated or dominates frontier.
    
    Args:
        candidate: Dict with x_RoT, y_volatility, z_return
        frontier_df: Original frontier DataFrame
        
    Returns:
        Tuple of (is_non_dominated, dominates_any)
    """
    is_non_dominated = True
    dominates_any = False
    
    for _, frontier_row in frontier_df.iterrows():
        # Check if frontier dominates candidate
        if (frontier_row['x_RoT'] <= candidate['x_RoT'] and 
            frontier_row['y_volatility'] <= candidate['y_volatility'] and 
            frontier_row['z_return'] >= candidate['z_return'] and
            (frontier_row['x_RoT'] < candidate['x_RoT'] or 
             frontier_row['y_volatility'] < candidate['y_volatility'] or 
             frontier_row['z_return'] > candidate['z_return'])):
            is_non_dominated = False
            break
        
        # Check if candidate dominates frontier
        if (candidate['x_RoT'] <= frontier_row['x_RoT'] and 
            candidate['y_volatility'] <= frontier_row['y_volatility'] and 
            candidate['z_return'] >= frontier_row['z_return'] and
            (candidate['x_RoT'] < frontier_row['x_RoT'] or 
             candidate['y_volatility'] < frontier_row['y_volatility'] or 
             candidate['z_return'] > frontier_row['z_return'])):
            dominates_any = True
    
    return is_non_dominated, dominates_any


def run_frontier_research(
    frontier_path: str,
    run_simulation_func,
    calculate_metrics_func,
    ppu_dictionary: pd.DataFrame,
    demand_data: np.ndarray,
    spot_data: np.ndarray,
    solar_data: np.ndarray,
    wind_data: np.ndarray,
    water_inflow_data: np.ndarray,
    config,
    ppu_definitions: Dict,
    research_config: Optional[FrontierResearchConfig] = None
) -> FrontierResearchResults:
    """
    Main function: Run systematic frontier exploration.
    
    Args:
        frontier_path: Path to frontier CSV file
        run_simulation_func: Function to run full year simulation
        calculate_metrics_func: Function to calculate portfolio metrics
        ppu_dictionary: PPU dictionary DataFrame
        demand_data: Hourly demand array
        spot_data: Hourly spot price array
        solar_data: Solar production data
        wind_data: Wind production data
        water_inflow_data: Water inflow data
        config: System configuration
        ppu_definitions: PPU definitions dict
        research_config: Optional research configuration
        
    Returns:
        FrontierResearchResults with all findings
    """
    if research_config is None:
        research_config = FrontierResearchConfig()
    
    verbose = research_config.verbose
    
    # Load frontier data
    frontier_df = pd.read_csv(frontier_path)
    frontier_df['portfolio'] = frontier_df['portfolio_dict'].apply(
        lambda x: json.loads(x.replace("'", '"'))
    )
    
    if verbose:
        print("=" * 80)
        print("FRONTIER RESEARCH: Pushing Along the Pareto Line")
        print("=" * 80)
        print(f"\nLoaded {len(frontier_df)} frontier portfolios")
        print(f"Return range: {frontier_df['z_return'].min():.2f}% to {frontier_df['z_return'].max():.2f}%")
        print(f"RoT range: {frontier_df['x_RoT'].min():.4f} to {frontier_df['x_RoT'].max():.4f}")
        print(f"Volatility range: {frontier_df['y_volatility'].min():.2f} to {frontier_df['y_volatility'].max():.2f}")
    
    # Step 1: Identify anchor points
    if verbose:
        print("\n" + "-" * 40)
        print("Step 1: Identifying Anchor Points")
        print("-" * 40)
    
    anchors = identify_anchor_points(frontier_df)
    
    if verbose:
        for name, anchor in anchors.items():
            print(f"  {name}: Return={anchor['z_return']:.2f}%, RoT={anchor['x_RoT']:.4f}, Vol={anchor['y_volatility']:.2f}")
    
    # Step 2: Generate interpolated portfolios
    if verbose:
        print("\n" + "-" * 40)
        print("Step 2: Generating Interpolated Portfolios")
        print("-" * 40)
    
    alphas = research_config.alphas or [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    interpolated = generate_interpolated_portfolios(anchors, alphas)
    
    if verbose:
        print(f"  Generated {len(interpolated)} interpolated portfolios")
    
    # Step 3: Evaluate interpolated portfolios
    if verbose:
        print("\n" + "-" * 40)
        print("Step 3: Evaluating Portfolios")
        print("-" * 40)
    
    evaluated = []
    for i, item in enumerate(interpolated):
        try:
            # Run simulation
            full_year_results = run_simulation_func(
                portfolio_counts=item['portfolio'],
                ppu_dictionary=ppu_dictionary,
                demand_data=demand_data,
                spot_data=spot_data,
                solar_data=solar_data,
                wind_data=wind_data,
                water_inflow_data=water_inflow_data,
                config=config,
                ppu_definitions=ppu_definitions,
                verbose=False
            )
            
            # Calculate metrics
            metrics = calculate_metrics_func(
                full_year_results=full_year_results,
                ppu_definitions=ppu_definitions,
                spot_data=spot_data,
                portfolio_counts=item['portfolio'],
                config=config
            )
            
            evaluated.append({
                'edge': item['edge'],
                'alpha': item['alpha'],
                'portfolio': item['portfolio'],
                'x_RoT': metrics.risk_of_technology,
                'y_volatility': metrics.volatility,
                'z_return': metrics.mean_return,
                'total_energy_twh': metrics.total_energy_twh
            })
            
            if verbose and (i + 1) % 9 == 0:
                print(f"  Evaluated {i+1}/{len(interpolated)} portfolios")
                
        except Exception as e:
            if verbose:
                print(f"  Error evaluating portfolio {i}: {e}")
    
    if verbose:
        print(f"  Successfully evaluated: {len(evaluated)} portfolios")
    
    # Step 4: Find improved portfolios
    if verbose:
        print("\n" + "-" * 40)
        print("Step 4: Identifying Improvements")
        print("-" * 40)
    
    improved = []
    for result in evaluated:
        is_non_dominated, dominates_any = check_dominance(result, frontier_df)
        if is_non_dominated or dominates_any:
            improved.append({
                **result,
                'is_non_dominated': is_non_dominated,
                'dominates_frontier': dominates_any
            })
    
    if verbose:
        print(f"  Found {len(improved)} potentially improved portfolios")
        
        if improved:
            improved_df = pd.DataFrame(improved)
            
            print("\n  📊 Top by Return:")
            for _, row in improved_df.nlargest(3, 'z_return').iterrows():
                print(f"     Return: {row['z_return']:.2f}% | RoT: {row['x_RoT']:.4f} | Vol: {row['y_volatility']:.2f}")
            
            print("\n  📊 Top by RoT (lowest):")
            for _, row in improved_df.nsmallest(3, 'x_RoT').iterrows():
                print(f"     RoT: {row['x_RoT']:.4f} | Return: {row['z_return']:.2f}% | Vol: {row['y_volatility']:.2f}")
    
    # Step 5: Build extended frontier
    all_portfolios = []
    
    # Original frontier
    for _, row in frontier_df.iterrows():
        all_portfolios.append({
            'portfolio_dict': row['portfolio_dict'],
            'x_RoT': row['x_RoT'],
            'y_volatility': row['y_volatility'],
            'z_return': row['z_return'],
            'total_energy_twh': row['total_energy_twh'],
            'source': 'original_frontier'
        })
    
    # Interpolated portfolios
    for result in evaluated:
        all_portfolios.append({
            'portfolio_dict': json.dumps(result['portfolio']),
            'x_RoT': result['x_RoT'],
            'y_volatility': result['y_volatility'],
            'z_return': result['z_return'],
            'total_energy_twh': result['total_energy_twh'],
            'source': f"interpolated_{result['edge']}_{result['alpha']}"
        })
    
    extended_df = pd.DataFrame(all_portfolios)
    extended_df.to_csv(research_config.output_path, index=False)
    
    if verbose:
        print("\n" + "-" * 40)
        print("Step 5: Results Saved")
        print("-" * 40)
        print(f"  Output: {research_config.output_path}")
        print(f"  Total portfolios: {len(extended_df)}")
    
    # Build results
    eval_df = pd.DataFrame(evaluated) if evaluated else pd.DataFrame()
    
    results = FrontierResearchResults(
        original_frontier_count=len(frontier_df),
        interpolated_count=len(evaluated),
        improved_count=len(improved),
        best_return=float(eval_df['z_return'].max()) if len(eval_df) > 0 else float('nan'),
        lowest_rot=float(eval_df['x_RoT'].min()) if len(eval_df) > 0 else float('nan'),
        lowest_volatility=float(eval_df['y_volatility'].min()) if len(eval_df) > 0 else float('nan'),
        extended_frontier_df=extended_df,
        improved_portfolios=improved
    )
    
    if verbose:
        print("\n" + "=" * 80)
        print("FRONTIER RESEARCH COMPLETE")
        print("=" * 80)
        print(f"  Original frontier: {results.original_frontier_count} portfolios")
        print(f"  New interpolated: {results.interpolated_count} portfolios")
        print(f"  Potentially improved: {results.improved_count} portfolios")
        print(f"  Best return found: {results.best_return:.2f}%")
        print(f"  Lowest RoT found: {results.lowest_rot:.4f}")
        print(f"  Lowest volatility found: {results.lowest_volatility:.2f}")
    
    return results

