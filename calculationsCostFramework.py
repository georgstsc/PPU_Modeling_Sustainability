# ============================================================================
# DISPATCH INDEX CALCULATION FUNCTIONS
# ============================================================================

import numpy as np
from typing import Dict, List, Tuple

def calculate_disposition_index(soc: float, 
                                soc_target: float = 0.60, 
                                deadband: float = 0.05,
                                alpha_d: float = 1.0) -> float:
    """
    Calculate disposition index d_stor for a storage PPU.
    
    Measures the storage unit's willingness to discharge based on its State of Charge (SoC).
    
    Parameters:
        soc: Current normalized State of Charge [0, 1]
        soc_target: Target SoC setpoint (default: 0.60)
        deadband: Deadband around target to avoid unnecessary cycling (default: 0.05)
        alpha_d: Scaling parameter for tanh squashing (default: 1.0)
    
    Returns:
        Disposition index in [-1, 1]
        +1: Storage full → strongly willing to discharge
        -1: Storage empty → reluctant to discharge (prefers charging)
         0: At target SoC → neutral
    
    Formula:
        Δs = { s - (s* + δ)  if s > s* + δ  (above target)
             { 0             if |s - s*| ≤ δ  (within deadband)
             { s - (s* - δ)  if s < s* - δ  (below target)
        
        Δs_norm = Δs / max{1 - s* - δ, s* - δ}
        d_stor = tanh(Δs_norm / α_d)
    """
    # Calculate deviation from target with deadband
    if soc > soc_target + deadband:
        # Above target
        delta_s = soc - (soc_target + deadband)
    elif soc < soc_target - deadband:
        # Below target
        delta_s = soc - (soc_target - deadband)
    else:
        # Within deadband
        delta_s = 0.0
    
    # Normalize by maximum possible excursion
    max_excursion = max(1.0 - soc_target - deadband, soc_target - deadband)
    delta_s_normalized = delta_s / max_excursion if max_excursion > 0 else 0.0
    
    # Apply squashing function
    d_stor = np.tanh(delta_s_normalized / alpha_d)
    
    return d_stor


def calculate_utility_indices(phi_t: float, 
                              phi_t_smoothed: float,
                              alpha_u: float = 1.0) -> Tuple[float, float]:
    """
    Calculate utility indices u_dis and u_chg for system-wide dispatch context.
    
    Parameters:
        phi_t: Net system shortfall at timestep t (MW)
               phi_t > 0: shortfall (need more supply)
               phi_t < 0: surplus (excess energy)
        phi_t_smoothed: Exponentially smoothed shortfall (EMA)
        alpha_u: Scaling parameter for tanh squashing (default: 1.0)
    
    Returns:
        (u_dis, u_chg): Tuple of discharge and charge utility indices in [-1, 1]
        
        u_dis: Discharge utility (higher during shortfall)
        u_chg: Charge utility (higher during surplus)
    
    Formula:
        u_dis = tanh(Φ_smoothed / α_u)
        u_chg = 1{Φ < 0} · tanh(-Φ_smoothed / α_u)
    """
    # Discharge utility: higher when system needs energy
    u_dis = np.tanh(phi_t_smoothed / alpha_u)
    
    # Charge utility: only active during surplus
    if phi_t < 0:
        u_chg = np.tanh(-phi_t_smoothed / alpha_u)
    else:
        u_chg = 0.0
    
    return u_dis, u_chg


def calculate_monetary_index(
    price_current: float,
    costate_values: Dict[str, float],
    price_sensitivities: Dict[str, float],
    eta_discharge: float,
    volatility_scale: float = 1.0,
    weight_spread: float = 1.0,
    weight_volatility: float = 1.0,
    alpha_m: float = 1.0,
    horizons: List[str] = ['1d', '3d', '7d', '30d']
) -> Tuple[float, Dict[str, float]]:
    """
    Calculate monetary volatility-aware index m comparing opportunity spread vs. volatility impact.
    
    This index combines:
    1. Opportunity spread: Current price vs. future value (favors discharge when prices are high now)
    2. Volatility impact: How discharging now affects future price volatility (penalizes if it increases volatility)
    
    Parameters:
        price_current: Current electricity price π_t (€/MWh or CHF/MWh)
        costate_values: Dictionary mapping horizon to co-state value λ^(H)
                       e.g., {'1d': 50.0, '3d': 52.0, '7d': 48.0, '30d': 55.0}
        price_sensitivities: Dictionary mapping horizon to ∂E[π_{t+H}]/∂E_i (ζ_i,H)
                            e.g., {'1d': 0.01, '3d': 0.015, '7d': 0.02, '30d': 0.025}
                            Positive means discharging increases future prices
        eta_discharge: Discharge efficiency η_discharge (to convert energy to inventory impact)
        volatility_scale: Normalization factor v_* for volatility (e.g., 30-day MAD)
        weight_spread: Weight 'a' for opportunity spread (default: 1.0)
        weight_volatility: Weight 'b' for volatility penalty (default: 1.0)
        alpha_m: Scaling parameter for tanh squashing (default: 1.0)
        horizons: List of horizon keys to use (default: ['1d', '3d', '7d', '30d'])
    
    Returns:
        Tuple of (m, diagnostics):
        - m: Monetary index in [-1, 1]
             m > 0: Using now reduces volatility or has good spread → favorable
             m < 0: Using now increases volatility or poor spread → unfavorable
        - diagnostics: Dict with intermediate values for analysis
    
    Formula:
        S = (1/|H|) Σ (π_t - λ^(H))  [opportunity spread]
        ΔVar = Σ (1/|H|) · ζ_{i,H}² · (1/η_discharge)²  [volatility impact]
        V = ΔVar / v_*  [normalized volatility]
        X = a·S - b·V  [combined score]
        m = tanh(X / α_m)  [squashed index]
    
    Note: If co-state values not available, use proxy:
        λ^(H) ≈ η_discharge · E[π_{t+H}]
    """
    # (a) Calculate opportunity spread across horizons
    spreads = []
    for horizon in horizons:
        if horizon in costate_values:
            lambda_H = costate_values[horizon]
            spread_H = price_current - lambda_H
            spreads.append(spread_H)
    
    # Average spread
    if len(spreads) > 0:
        S = np.mean(spreads)
    else:
        S = 0.0
    
    # (b) Calculate volatility impact
    variance_terms = []
    for horizon in horizons:
        if horizon in price_sensitivities:
            zeta_H = price_sensitivities[horizon]
            # Inventory impact of discharging 1 MWh
            inventory_impact = 1.0 / eta_discharge if eta_discharge > 0 else 1.0
            # Variance contribution from this horizon
            var_H = (zeta_H ** 2) * (inventory_impact ** 2)
            variance_terms.append(var_H)
    
    # Average variance
    if len(variance_terms) > 0:
        delta_var = np.mean(variance_terms)
    else:
        delta_var = 0.0
    
    # Normalize by volatility scale
    V = delta_var / volatility_scale if volatility_scale > 0 else 0.0
    
    # (c) Combine spread and volatility with weights
    X = weight_spread * S - weight_volatility * V
    
    # Apply squashing function
    m = np.tanh(X / alpha_m)
    
    # Prepare diagnostics
    diagnostics = {
        'spread': S,
        'volatility': V,
        'combined_score': X,
        'num_horizons_spread': len(spreads),
        'num_horizons_volatility': len(variance_terms)
    }
    
    return m, diagnostics


def calculate_discharge_benefit(d_stor: float, 
                                u_dis: float, 
                                m: float) -> Tuple[float, float]:
    """
    Calculate discharge benefit and cost metrics.
    
    Parameters:
        d_stor: Disposition index [-1, 1]
        u_dis: Discharge utility index [-1, 1]
        m: Monetary volatility-aware index [-1, 1]
    
    Returns:
        (B_dis, kappa_dis): Discharge benefit and cost
        B_dis in [-1, 1]: Higher is better for discharging
        kappa_dis in [0, 2]: Lower is better (cost metric)
    
    Formula:
        B_dis = (1/3)(d_stor + u_dis + m)
        κ_dis = 1 - B_dis
    """
    B_dis = (d_stor + u_dis + m) / 3.0
    kappa_dis = 1.0 - B_dis
    
    return B_dis, kappa_dis


def calculate_charge_benefit(d_stor: float, 
                             u_chg: float, 
                             m: float) -> Tuple[float, float]:
    """
    Calculate charge benefit and cost metrics.
    
    Parameters:
        d_stor: Disposition index [-1, 1]
        u_chg: Charge utility index [-1, 1]
        m: Monetary volatility-aware index [-1, 1]
    
    Returns:
        (B_chg, kappa_chg): Charge benefit and cost
        B_chg in [-1, 1]: Higher is better for charging
        kappa_chg in [0, 2]: Lower is better (cost metric)
    
    Formula:
        B_chg = (1/3)((-d_stor) + u_chg + (-m))
        κ_chg = 1 - B_chg
    """
    B_chg = ((-d_stor) + u_chg + (-m)) / 3.0
    kappa_chg = 1.0 - B_chg
    
    return B_chg, kappa_chg


def exponential_moving_average(current_value: float, 
                               previous_ema: float, 
                               beta: float = 0.2) -> float:
    """
    Calculate exponential moving average (EMA) for smoothing time series.
    
    Parameters:
        current_value: Current raw value x_t
        previous_ema: Previous smoothed value x̄_{t-1}
        beta: Smoothing parameter (default: 0.2)
              Higher beta = more smoothing (slower response)
    
    Returns:
        Smoothed value x̄_t
    
    Formula:
        x̄_t = (1 - β)·x_t + β·x̄_{t-1}
    """
    return (1.0 - beta) * current_value + beta * previous_ema
