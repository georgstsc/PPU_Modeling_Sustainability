"""
================================================================================
RISK CALCULATOR - Technology Risk of Supply (RoT) Calculation
================================================================================

This module calculates the Risk of Technology (RoT) for PPUs and portfolios
based on natural resource supply risk.

Author: Energy Systems Optimization Project
Date: December 2024
================================================================================
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import ast

from config import Config, DEFAULT_CONFIG


# =============================================================================
# RESOURCE RISK CALCULATION
# =============================================================================

def load_country_resource_data(data_dir: Path) -> pd.DataFrame:
    """Load country resource production data."""
    filepath = data_dir / 'country_resource_production.csv'
    return pd.read_csv(filepath)


def load_resource_risk_data(data_dir: Path) -> pd.DataFrame:
    """Load natural resources supply risk data."""
    filepath = data_dir / 'natural_resources_supply_risk.csv'
    return pd.read_csv(filepath)


def calculate_resource_risks(
    country_resource_df: pd.DataFrame
) -> Dict[str, float]:
    """
    Calculate resource risk for each resource based on country count.
    
    Formula: resource_risk = sqrt(1 / # producing_countries)
    
    Args:
        country_resource_df: DataFrame with Country and Resource columns
        
    Returns:
        Dictionary mapping resource name to risk value [0, 1]
    """
    # Count unique countries per resource
    resource_counts = country_resource_df.groupby('Resource')['Country'].nunique()
    
    # Calculate risk: sqrt(1 / count)
    resource_risks = {}
    for resource, count in resource_counts.items():
        if count > 0:
            risk = np.sqrt(1.0 / count)
            resource_risks[resource] = risk
        else:
            # If no countries found, assume maximum risk
            resource_risks[resource] = 1.0
    
    return resource_risks


def map_resources_to_components(
    resource_risk_df: pd.DataFrame,
    ppu_constructs_df: Optional[pd.DataFrame] = None
) -> Dict[str, List[str]]:
    """
    Map components to resources based on PPU dependencies.
    
    Uses the PPUs_Dependent column to find which PPUs use each resource,
    then maps to components via the PPU component chains.
    
    Args:
        resource_risk_df: DataFrame with Resource and PPUs_Dependent columns
        ppu_constructs_df: Optional PPU constructs DataFrame (loaded if None)
        
    Returns:
        Dictionary mapping component name to list of resource names
    """
    # Load PPU constructs if not provided
    if ppu_constructs_df is None:
        from config import DEFAULT_CONFIG
        from pathlib import Path
        from ppu_framework import load_ppu_constructs
        ppu_path = Path(DEFAULT_CONFIG.paths.DATA_DIR) / DEFAULT_CONFIG.paths.PPU_CONSTRUCTS
        ppu_constructs_df = load_ppu_constructs(str(ppu_path))
    
    component_to_resources = {}
    
    for _, row in resource_risk_df.iterrows():
        resource = row['Resource']
        ppus_dependent = str(row.get('PPUs_Dependent', ''))
        
        # Parse PPU names (comma-separated)
        ppu_names = [p.strip() for p in ppus_dependent.split(',')]
        
        # For each PPU, get its components and map resource to them
        for ppu_name in ppu_names:
            # Find PPU in constructs
            ppu_row = ppu_constructs_df[ppu_constructs_df['PPU'] == ppu_name]
            if len(ppu_row) > 0:
                components = ppu_row.iloc[0]['Components']
                
                # If components is a string, parse it
                if isinstance(components, str):
                    try:
                        components = ast.literal_eval(components)
                    except:
                        components = [components]
                
                # Map resource to all components in this PPU
                for component in components:
                    if component not in component_to_resources:
                        component_to_resources[component] = []
                    if resource not in component_to_resources[component]:
                        component_to_resources[component].append(resource)
    
    return component_to_resources


def calculate_component_risk(
    component_name: str,
    component_to_resources: Dict[str, List[str]],
    resource_risks: Dict[str, float]
) -> float:
    """
    Calculate risk for a single component using power mean.
    
    Formula: component_risk = (sum(resource_risk^p))^(1/p)
    where p = number of natural resources used by component
    
    Args:
        component_name: Name of the component
        component_to_resources: Mapping of components to resources
        resource_risks: Dictionary of resource risks
        
    Returns:
        Component risk value [0, 1]
    """
    # Get resources for this component
    resources = component_to_resources.get(component_name, [])
    
    if not resources:
        # Component not found in mapping - assume low risk (common component)
        return 0.1
    
    # Get resource risks
    resource_risk_values = []
    for resource in resources:
        # Try exact match first
        if resource in resource_risks:
            resource_risk_values.append(resource_risks[resource])
        else:
            # Try partial match (e.g., "Iron/Steel" vs "Iron Ore")
            matched = False
            for res_name, risk_val in resource_risks.items():
                if resource.lower() in res_name.lower() or res_name.lower() in resource.lower():
                    resource_risk_values.append(risk_val)
                    matched = True
                    break
            if not matched:
                # Resource not found - assume medium risk
                resource_risk_values.append(0.5)
    
    if not resource_risk_values:
        return 0.1
    
    # Calculate power mean: (sum(x^p))^(1/p) where p = number of resources
    p = len(resource_risk_values)
    if p == 0:
        return 0.1
    
    # Power mean formula
    sum_powered = sum(r**p for r in resource_risk_values)
    component_risk = (sum_powered) ** (1.0 / p)
    
    return component_risk


def calculate_ppu_risk(
    ppu_name: str,
    ppu_components: List[str],
    component_to_resources: Dict[str, List[str]],
    resource_risks: Dict[str, float]
) -> float:
    """
    Calculate Risk of Technology (RoT) for a PPU.
    
    Formula: PPU_RoT = arithmetic_mean(all component_risks)
    
    Args:
        ppu_name: PPU name
        ppu_components: List of component names in PPU chain
        component_to_resources: Mapping of components to resources
        resource_risks: Dictionary of resource risks
        
    Returns:
        PPU RoT value [0, 1]
    """
    component_risks = []
    
    for component in ppu_components:
        comp_risk = calculate_component_risk(
            component, component_to_resources, resource_risks
        )
        component_risks.append(comp_risk)
    
    if not component_risks:
        return 0.1  # Default low risk if no components
    
    # Arithmetic mean of component risks
    ppu_rot = np.mean(component_risks)
    
    return ppu_rot


def calculate_portfolio_risk(
    portfolio_counts: Dict[str, int],
    ppu_risks: Dict[str, float],
    ppu_energy_volumes: Dict[str, float]
) -> float:
    """
    Calculate weighted portfolio Risk of Technology.
    
    Formula: Portfolio_RoT = sum(PPU_RoT_i * (energy_volume_i / total_energy))
    
    Args:
        portfolio_counts: PPU counts in portfolio
        ppu_risks: Dictionary of PPU RoT values
        ppu_energy_volumes: Dictionary of energy volumes per PPU (MWh)
        
    Returns:
        Portfolio RoT value [0, 1]
    """
    total_energy = sum(ppu_energy_volumes.values())
    
    if total_energy == 0:
        return 0.0
    
    weighted_risk = 0.0
    for ppu_name, count in portfolio_counts.items():
        if count > 0 and ppu_name in ppu_risks:
            ppu_rot = ppu_risks[ppu_name]
            ppu_energy = ppu_energy_volumes.get(ppu_name, 0.0)
            weight = ppu_energy / total_energy
            weighted_risk += ppu_rot * weight
    
    return weighted_risk


# =============================================================================
# MAIN RISK CALCULATION INTERFACE
# =============================================================================

class RiskCalculator:
    """Main interface for calculating technology risks."""
    
    def __init__(self, config: Config = DEFAULT_CONFIG):
        """Initialize risk calculator with data loading."""
        data_dir = Path(config.paths.DATA_DIR)
        
        # Load data
        self.country_resource_df = load_country_resource_data(data_dir)
        self.resource_risk_df = load_resource_risk_data(data_dir)
        
        # Load PPU constructs for mapping
        from ppu_framework import load_ppu_constructs
        ppu_path = data_dir / config.paths.PPU_CONSTRUCTS
        self.ppu_constructs_df = load_ppu_constructs(str(ppu_path))
        
        # Calculate resource risks
        self.resource_risks = calculate_resource_risks(self.country_resource_df)
        
        # Map components to resources
        self.component_to_resources = map_resources_to_components(
            self.resource_risk_df, self.ppu_constructs_df
        )
        
        # Cache for PPU risks (will be populated as needed)
        self._ppu_risk_cache: Dict[str, float] = {}
    
    def get_ppu_risk(
        self,
        ppu_name: str,
        ppu_components: List[str]
    ) -> float:
        """
        Get or calculate PPU risk.
        
        Args:
            ppu_name: PPU name
            ppu_components: List of component names
            
        Returns:
            PPU RoT value
        """
        # Check cache
        if ppu_name in self._ppu_risk_cache:
            return self._ppu_risk_cache[ppu_name]
        
        # Calculate
        risk = calculate_ppu_risk(
            ppu_name, ppu_components,
            self.component_to_resources, self.resource_risks
        )
        
        # Cache
        self._ppu_risk_cache[ppu_name] = risk
        
        return risk
    
    def get_portfolio_risk(
        self,
        portfolio_counts: Dict[str, int],
        ppu_energy_volumes: Dict[str, float]
    ) -> float:
        """
        Calculate portfolio risk.
        
        Args:
            portfolio_counts: PPU counts
            ppu_energy_volumes: Energy volumes per PPU (MWh)
            
        Returns:
            Portfolio RoT value
        """
        # Get PPU risks (need to load PPU definitions)
        from ppu_framework import load_all_ppu_data
        
        _, _, ppu_definitions = load_all_ppu_data()
        ppu_risks = {}
        
        for ppu_name, count in portfolio_counts.items():
            if count > 0 and ppu_name in ppu_definitions:
                ppu_def = ppu_definitions[ppu_name]
                ppu_risks[ppu_name] = self.get_ppu_risk(
                    ppu_name, ppu_def.components
                )
        
        return calculate_portfolio_risk(
            portfolio_counts, ppu_risks, ppu_energy_volumes
        )


if __name__ == "__main__":
    # Test risk calculation
    calculator = RiskCalculator()
    
    print("Resource Risks (sample):")
    for resource, risk in list(calculator.resource_risks.items())[:10]:
        print(f"  {resource}: {risk:.4f}")
    
    print(f"\nTotal resources: {len(calculator.resource_risks)}")
    print(f"Total components mapped: {len(calculator.component_to_resources)}")

