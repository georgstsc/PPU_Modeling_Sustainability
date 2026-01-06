"""
================================================================================
PORTFOLIO METRICS - Return, Volatility, and Risk Calculations
================================================================================

This module calculates the three key metrics for portfolio evaluation:
- Return (Z-axis): Weekly average savings vs spot price
- Volatility (Y-axis): Weekly std of production costs
- Risk of Technology (X-axis): Supply risk (calculated in risk_calculator.py)

Author: Energy Systems Optimization Project
Date: December 2024
================================================================================
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

from config import Config, DEFAULT_CONFIG
from risk_calculator import RiskCalculator
from optimization import FullYearResults, evaluate_portfolio_full_year, Individual
from ppu_framework import load_all_ppu_data


# =============================================================================
# RETURN CALCULATION (Z-axis)
# =============================================================================

def calculate_weekly_return(
    spot_prices: np.ndarray,
    production_costs: np.ndarray,
    production: np.ndarray,
    hours_per_week: int = 168
) -> np.ndarray:
    """
    Calculate weekly return as savings percentage (production-weighted).
    
    Return measures: "If we sold our production at spot, how much would we save/lose
    compared to our production cost?"
    
    Formula per week (production-weighted):
        revenue_at_spot = sum(production * spot_price)  # what we'd get selling at spot
        our_cost = sum(production * production_cost)    # what it cost us to produce
        return = (revenue_at_spot - our_cost) / revenue_at_spot * 100
    
    Positive return = we're producing cheaper than spot (profit margin)
    Negative return = we're producing more expensive than spot (loss margin)
    
    Args:
        spot_prices: Array of spot prices (CHF/MWh) for full year
        production_costs: Array of production costs (CHF/MWh) for full year
        production: Array of total production (MW/MWh) for full year
        hours_per_week: Number of hours per week (default 168)
        
    Returns:
        Array of weekly returns (%)
    """
    n_hours = len(spot_prices)
    n_weeks = n_hours // hours_per_week
    
    weekly_returns = []
    
    for week in range(n_weeks):
        start_idx = week * hours_per_week
        end_idx = min(start_idx + hours_per_week, n_hours)
        
        week_spot = spot_prices[start_idx:end_idx]
        week_costs = production_costs[start_idx:end_idx]
        week_production = production[start_idx:end_idx]
        
        # Only count hours with positive spot price and production
        valid_mask = (week_spot > 0) & (week_production > 0)
        
        if np.sum(valid_mask) > 0:
            # Revenue if we sold all production at spot prices
            revenue_at_spot = np.sum(week_production[valid_mask] * week_spot[valid_mask])
            
            # Our actual production cost
            our_cost = np.sum(week_production[valid_mask] * week_costs[valid_mask])
            
            if revenue_at_spot > 0:
                # Margin = (revenue - cost) / revenue * 100
                margin = (revenue_at_spot - our_cost) / revenue_at_spot * 100
                weekly_returns.append(margin)
            else:
                weekly_returns.append(0.0)
        else:
            weekly_returns.append(0.0)
    
    return np.array(weekly_returns)


def calculate_annual_return(
    weekly_returns: np.ndarray
) -> float:
    """
    Calculate annual return as arithmetic mean of weekly returns.
    
    Args:
        weekly_returns: Array of weekly returns (%)
        
    Returns:
        Annual average return (%)
    """
    if len(weekly_returns) == 0:
        return 0.0
    
    return float(np.mean(weekly_returns))


# =============================================================================
# VOLATILITY CALCULATION (Y-axis)
# =============================================================================

def calculate_weekly_volatility(
    production_costs: np.ndarray,
    hours_per_week: int = 168
) -> np.ndarray:
    """
    Calculate weekly volatility (std) of production costs.
    
    Args:
        production_costs: Array of production costs (CHF/MWh) for full year
        hours_per_week: Number of hours per week (default 168)
        
    Returns:
        Array of weekly standard deviations
    """
    n_hours = len(production_costs)
    n_weeks = n_hours // hours_per_week
    
    weekly_volatilities = []
    
    for week in range(n_weeks):
        start_idx = week * hours_per_week
        end_idx = min(start_idx + hours_per_week, n_hours)
        
        week_costs = production_costs[start_idx:end_idx]
        
        # Calculate std for the week
        if len(week_costs) > 1:
            volatility = np.std(week_costs)
        else:
            volatility = 0.0
        
        weekly_volatilities.append(volatility)
    
    return np.array(weekly_volatilities)


def calculate_annual_volatility(
    weekly_volatilities: np.ndarray
) -> float:
    """
    Calculate annual volatility as arithmetic mean of weekly volatilities.
    
    Args:
        weekly_volatilities: Array of weekly standard deviations
        
    Returns:
        Annual average volatility
    """
    if len(weekly_volatilities) == 0:
        return 0.0
    
    return float(np.mean(weekly_volatilities))


# =============================================================================
# PRODUCTION COST CALCULATION
# =============================================================================

def calculate_production_costs(
    full_year_results: FullYearResults,
    ppu_definitions: Dict,
    config: Config = DEFAULT_CONFIG
) -> np.ndarray:
    """
    Calculate hourly weighted-average production cost per MWh PRODUCED.
    
    This matches the visualization.py approach EXACTLY: the cost per MWh of energy
    that our portfolio produces, weighted by production volume.
    
    Uses ppu_definitions directly (like visualization.py) to ensure consistency.
    
    This is the LCOE (Levelized Cost of Energy) of the portfolio.
    
    For the Return calculation, we compare this cost to spot prices:
    - If production_cost < spot_price → we're saving money (positive return)
    - If production_cost > spot_price → we're paying more (negative return)
    
    Args:
        full_year_results: Full year evaluation results
        ppu_definitions: Dict of PPU definitions (PPUDefinition objects)
        config: Configuration
        
    Returns:
        Array of production costs (CHF/MWh produced) for each hour
    """
    # Conversion factor - cost_per_mwh in PPUDefinition is actually CHF/kWh
    CHF_KWH_TO_CHF_MWH = 1000.0
    
    def get_ppu_cost(ppu_name: str) -> float:
        """Get PPU cost in CHF/MWh. CRITICAL: No fallback prices allowed."""
        ppu_def = ppu_definitions.get(ppu_name)
        if ppu_def is None:
            raise ValueError(
                f"CRITICAL: PPU '{ppu_name}' not found in ppu_definitions. "
                f"Cannot use fallback price as it would falsify costs. "
                f"Please ensure all PPUs are properly defined."
            )
        elif hasattr(ppu_def, 'cost_per_mwh'):
            # Convert from CHF/kWh to CHF/MWh
            return ppu_def.cost_per_mwh * CHF_KWH_TO_CHF_MWH
        elif isinstance(ppu_def, dict):
            if 'cost_per_mwh' not in ppu_def:
                raise ValueError(
                    f"CRITICAL: PPU '{ppu_name}' missing 'cost_per_mwh' in definition. "
                    f"Cannot use fallback price as it would falsify costs."
                )
            return ppu_def['cost_per_mwh'] * CHF_KWH_TO_CHF_MWH
        else:
            raise ValueError(
                f"CRITICAL: Cannot extract cost from PPU '{ppu_name}' definition. "
                f"Cannot use fallback price as it would falsify costs."
            )
    
    n_hours = len(full_year_results.demand)
    weighted_costs = np.zeros(n_hours)  # Sum of (production * cost)
    total_production = np.zeros(n_hours)  # Sum of production
    
    # Get PPU production breakdown
    ppu_production = full_year_results.ppu_production
    
    # Calculate PPU operational costs (weighted by production)
    for ppu_name, production_array in ppu_production.items():
        if not isinstance(production_array, np.ndarray):
            continue
        if len(production_array) != n_hours:
            continue
        
        # Get cost for this PPU (in CHF/MWh)
        cost_per_mwh = get_ppu_cost(ppu_name)
        
        # Production is in MW = MWh for 1 hour
        weighted_costs += production_array * cost_per_mwh
        total_production += production_array
    
    # Calculate weighted average cost per MWh produced
    # This is the portfolio's LCOE for each hour
    with np.errstate(divide='ignore', invalid='ignore'):
        production_cost_per_mwh = np.where(
            total_production > 0,
            weighted_costs / total_production,
            0.0
        )
    
    return production_cost_per_mwh


# =============================================================================
# COMPLETE METRICS CALCULATION
# =============================================================================

@dataclass
class PortfolioMetrics3D:
    """3D metrics for portfolio evaluation."""
    
    x_rot: float  # Risk of Technology [0, 1]
    y_volatility: float  # Price volatility (CHF/MWh)
    z_return: float  # Return as savings % [%]
    
    # Additional metadata
    portfolio_counts: Dict[str, int]
    total_energy_twh: float
    annual_production_twh: float
    
    # === CONSTRAINT COMPLIANCE FIELDS ===
    # These are CRITICAL for filtering portfolios in the notebook
    storage_constraint_met: bool = False  # Cyclic SOC constraint (±25% of initial)
    total_domestic_production_twh: float = 0.0  # Electrical sovereignty (≥113 TWh/year)
    aviation_fuel_constraint_met: bool = False  # Aviation fuel requirement (23 TWh/year)
    
    # Weekly breakdowns (for analysis)
    weekly_returns: Optional[np.ndarray] = None
    weekly_volatilities: Optional[np.ndarray] = None


def calculate_portfolio_metrics_3d(
    individual: Individual,
    config: Config = DEFAULT_CONFIG,
    risk_calculator: Optional[RiskCalculator] = None,
    debug: bool = False
) -> PortfolioMetrics3D:
    """
    Calculate complete 3D metrics for a portfolio.
    
    Args:
        individual: Optimized portfolio individual
        config: Configuration
        risk_calculator: Optional pre-initialized risk calculator
        debug: Print debug information
        
    Returns:
        PortfolioMetrics3D object with (x, y, z) metrics
    """
    # Initialize risk calculator if not provided
    if risk_calculator is None:
        risk_calculator = RiskCalculator(config)
    
    # Run full year evaluation
    full_year_results = evaluate_portfolio_full_year(
        individual, config, verbose=False
    )
    
    # Load PPU definitions for cost calculation
    _, _, ppu_definitions = load_all_ppu_data(config)
    
    # Calculate production costs (CHF/MWh of production - LCOE)
    # Uses ppu_definitions directly like visualization.py for consistency
    production_costs = calculate_production_costs(
        full_year_results, ppu_definitions, config
    )
    
    if debug:
        print(f"\n[DEBUG] Production cost statistics (LCOE):")
        print(f"  Mean: {np.mean(production_costs):.2f} CHF/MWh")
        print(f"  Std:  {np.std(production_costs):.2f} CHF/MWh")
        print(f"  Min:  {np.min(production_costs):.2f} CHF/MWh")
        print(f"  Max:  {np.max(production_costs):.2f} CHF/MWh")
        print(f"\n[DEBUG] Spot price statistics:")
        print(f"  Mean: {np.mean(full_year_results.spot_prices):.2f} CHF/MWh")
        print(f"  Std:  {np.std(full_year_results.spot_prices):.2f} CHF/MWh")
        print(f"\n[DEBUG] Total production: {full_year_results.total_production_twh:.2f} TWh")
        print(f"[DEBUG] Total demand: {full_year_results.total_demand_twh:.2f} TWh")
        print(f"[DEBUG] Spot bought: {full_year_results.total_spot_bought_mwh/1e6:.2f} TWh")
        print(f"[DEBUG] Spot sold: {full_year_results.total_spot_sold_mwh/1e6:.2f} TWh")
    
    # Calculate return (Z-axis)
    # Compare production cost to spot price, weighted by production volume
    weekly_returns = calculate_weekly_return(
        full_year_results.spot_prices,
        production_costs,
        full_year_results.total_production
    )
    annual_return = calculate_annual_return(weekly_returns)
    
    if debug:
        print(f"\n[DEBUG] Weekly returns statistics:")
        print(f"  Mean: {np.mean(weekly_returns):.2f}%")
        print(f"  Min:  {np.min(weekly_returns):.2f}%")
        print(f"  Max:  {np.max(weekly_returns):.2f}%")
    
    # Calculate volatility (Y-axis) - std of production costs
    weekly_volatilities = calculate_weekly_volatility(production_costs)
    annual_volatility = calculate_annual_volatility(weekly_volatilities)
    
    # Calculate Risk of Technology (X-axis)
    # Need energy volumes per PPU
    # For production PPUs: use ppu_production (energy produced)
    # For storage INPUT PPUs: use ppu_consumption (energy handled/absorbed)
    ppu_energy_volumes = {}
    total_energy = 0.0
    
    # Add production from INCIDENCE and FLEX PPUs
    for ppu_name, production_array in full_year_results.ppu_production.items():
        if isinstance(production_array, np.ndarray):
            # Sum over all hours to get total MWh
            ppu_energy = np.sum(production_array)
            ppu_energy_volumes[ppu_name] = ppu_energy
            total_energy += ppu_energy
    
    # Add consumption from STORAGE INPUT PPUs (energy they handled)
    # This is separate from production - storage INPUT PPUs consume electricity to charge storage
    if hasattr(full_year_results, 'ppu_consumption') and full_year_results.ppu_consumption:
        for ppu_name, consumption_array in full_year_results.ppu_consumption.items():
            if isinstance(consumption_array, np.ndarray):
                ppu_energy = np.sum(consumption_array)
                if ppu_energy > 0:
                    # Add to existing or create new entry
                    ppu_energy_volumes[ppu_name] = ppu_energy_volumes.get(ppu_name, 0) + ppu_energy
                    total_energy += ppu_energy
    
    portfolio_rot = risk_calculator.get_portfolio_risk(
        individual.portfolio.ppu_counts,
        ppu_energy_volumes
    )
    
    if debug:
        print(f"\n[DEBUG] Final metrics:")
        print(f"  X (RoT):       {portfolio_rot:.4f}")
        print(f"  Y (Volatility): {annual_volatility:.2f} CHF/MWh")
        print(f"  Z (Return):     {annual_return:.2f}%")
        print(f"\n[DEBUG] Constraints:")
        print(f"  Storage (SOC): {'✅' if full_year_results.storage_constraint_met else '❌'}")
        print(f"  Aviation fuel: {'✅' if full_year_results.aviation_fuel_constraint_met else '❌'}")
        print(f"  Electrical sovereignty: {full_year_results.total_production_twh:.1f} TWh (need ≥113)")
    
    return PortfolioMetrics3D(
        x_rot=portfolio_rot,
        y_volatility=annual_volatility,
        z_return=annual_return,
        portfolio_counts=individual.portfolio.ppu_counts.copy(),
        total_energy_twh=total_energy / 1e6,
        annual_production_twh=full_year_results.total_production_twh,
        # CONSTRAINT FIELDS - CRITICAL for notebook filtering
        storage_constraint_met=full_year_results.storage_constraint_met,
        total_domestic_production_twh=full_year_results.total_production_twh,
        aviation_fuel_constraint_met=full_year_results.aviation_fuel_constraint_met,
        # Weekly analysis
        weekly_returns=weekly_returns,
        weekly_volatilities=weekly_volatilities
    )


if __name__ == "__main__":
    # Test metrics calculation
    print("Portfolio metrics module loaded successfully")
    print("Use calculate_portfolio_metrics_3d() to compute (x, y, z) metrics")

