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


def _map_descriptive_to_short_component(
    descriptive_name: str,
    ppu_name: str,
    ppu_components: List[str]
) -> Optional[str]:
    """
    Map descriptive component name to short component name used in constructs.
    
    Examples:
    - "Wind turbine (generator + structure)" → "Wind (onshore)" or "Wind (offshore)"
    - "Li-ion battery system" → "Battery"
    - "Inverter / power electronics" → "Inverter DC/AC" or "Converter AC/DC"
    - "Grid connection" → "Grid"
    
    Args:
        descriptive_name: Descriptive component name from ppu_supply_risk_resources.csv
        ppu_name: PPU name (e.g., "WD_ON", "PV")
        ppu_components: List of short component names for this PPU
        
    Returns:
        Short component name if match found, None otherwise
    """
    desc_lower = descriptive_name.lower()
    
    # Direct keyword matching
    if 'wind turbine' in desc_lower or 'wind' in desc_lower:
        # Match to wind component
        for comp in ppu_components:
            if 'wind' in comp.lower() and ('onshore' in comp.lower() or 'offshore' in comp.lower()):
                return comp
    
    if 'battery' in desc_lower or 'li-ion' in desc_lower:
        for comp in ppu_components:
            if 'battery' in comp.lower():
                return comp
    
    if 'inverter' in desc_lower or 'power electronics' in desc_lower:
        for comp in ppu_components:
            if 'inverter' in comp.lower() or 'converter' in comp.lower():
                return comp
    
    if 'grid' in desc_lower or 'connection' in desc_lower:
        for comp in ppu_components:
            if comp.lower() == 'grid':
                return comp
    
    if 'hydro turbine' in desc_lower or 'turbine-generator' in desc_lower:
        for comp in ppu_components:
            if 'hydro' in comp.lower() and 'turb' in comp.lower():
                return comp
    
    if 'pv module' in desc_lower or 'pv' in desc_lower:
        for comp in ppu_components:
            if comp.upper() == 'PV':
                return comp
    
    if 'electrolyser' in desc_lower or 'electrolyzer' in desc_lower:
        for comp in ppu_components:
            if 'electrolyzer' in comp.lower() or 'electrolyser' in comp.lower():
                return comp
    
    if 'compressor' in desc_lower:
        for comp in ppu_components:
            if 'compressor' in comp.lower():
                return comp
    
    if 'steam' in desc_lower and 'generator' in desc_lower:
        for comp in ppu_components:
            if 'steam' in comp.lower() and 'generator' in comp.lower():
                return comp
    
    if 'ice' in desc_lower and 'generator' in desc_lower:
        for comp in ppu_components:
            if comp.upper() == 'ICE':
                return comp
    
    if 'solar concentrator' in desc_lower or 'solar thermal' in desc_lower:
        for comp in ppu_components:
            if 'solar concentrator' in comp.lower():
                return comp
    
    if 'molten-salt' in desc_lower or 'salt' in desc_lower:
        for comp in ppu_components:
            if 'salt' in comp.lower():
                return comp
    
    if 'combined-cycle' in desc_lower or 'ccgt' in desc_lower or 'power plant' in desc_lower:
        for comp in ppu_components:
            if 'combined cycle' in comp.lower():
                return comp
    
    # Try partial matching as fallback
    for comp in ppu_components:
        comp_lower = comp.lower()
        # Check if any significant word from descriptive name matches
        desc_words = set(desc_lower.split())
        comp_words = set(comp_lower.split())
        if desc_words.intersection(comp_words):
            return comp
    
    return None


def map_resources_to_components(
    resource_risk_df: pd.DataFrame,
    ppu_constructs_df: Optional[pd.DataFrame] = None,
    ppu_supply_risk_df: Optional[pd.DataFrame] = None
) -> Dict[str, List[str]]:
    """
    Map components to resources based on SPECIFIC component usage.
    
    Uses ppu_supply_risk_resources.csv to map resources to specific components
    within PPUs, rather than mapping to all components in a PPU chain.
    
    Args:
        resource_risk_df: DataFrame with Resource and PPUs_Dependent columns
        ppu_constructs_df: Optional PPU constructs DataFrame (loaded if None)
        ppu_supply_risk_df: Optional PPU supply risk DataFrame (loaded if None)
        
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
    
    # Load PPU supply risk data (has specific component-resource mappings)
    if ppu_supply_risk_df is None:
        from config import DEFAULT_CONFIG
        from pathlib import Path
        data_dir = Path(DEFAULT_CONFIG.paths.DATA_DIR)
        ppu_supply_risk_path = data_dir / 'ppu_supply_risk_resources.csv'
        if ppu_supply_risk_path.exists():
            ppu_supply_risk_df = pd.read_csv(ppu_supply_risk_path)
        else:
            ppu_supply_risk_df = None
    
    component_to_resources = {}
    
    # First, try to use the detailed ppu_supply_risk_resources.csv if available
    if ppu_supply_risk_df is not None:
        # Build comprehensive resource matching patterns
        # Map variations to standardized resource names from resource_risk_df
        def match_resource_in_string(resource_str: str, resource_risk_df: pd.DataFrame) -> List[str]:
            """
            Match resource variations in a string to standardized resource names.
            
            Args:
                resource_str: String containing resource descriptions
                resource_risk_df: DataFrame with standardized resource names
                
            Returns:
                List of matched standardized resource names
            """
            resource_str_lower = resource_str.lower()
            matched_resources = []
            
            for _, res_row in resource_risk_df.iterrows():
                resource_name = res_row['Resource']
                resource_lower = resource_name.lower()
                
                # Direct match
                if resource_lower in resource_str_lower:
                    matched_resources.append(resource_name)
                    continue
                
                # Pattern matching for common variations
                if 'rare earth' in resource_lower:
                    if ('rare earth' in resource_str_lower or 
                        'nd/dy' in resource_str_lower or 
                        'nddy' in resource_str_lower.replace('/', '')):
                        matched_resources.append(resource_name)
                        continue
                
                if 'platinum group' in resource_lower or 'pgm' in resource_lower:
                    if ('platinum' in resource_str_lower or 
                        'pgm' in resource_str_lower or
                        'pt/ir' in resource_str_lower or
                        'ir/pt' in resource_str_lower):
                        matched_resources.append(resource_name)
                        continue
                
                if 'iridium' in resource_lower:
                    # Iridium must be explicitly mentioned (not just "ir" which could match "iron")
                    if 'iridium' in resource_str_lower:
                        matched_resources.append(resource_name)
                        continue
                
                if 'iron' in resource_lower and ('steel' in resource_lower or 'ore' in resource_lower):
                    if ('iron' in resource_str_lower and 
                        ('steel' in resource_str_lower or 'ore' in resource_str_lower)):
                        matched_resources.append(resource_name)
                        continue
                
                # Check for key element names
                element_keywords = {
                    'lithium': 'lithium',
                    'cobalt': 'cobalt',
                    'nickel': 'nickel',
                    'manganese': 'manganese',
                    'graphite': 'graphite',
                    'copper': 'copper',
                    'aluminium': 'aluminium',
                    'aluminum': 'aluminium',
                    'silicon': 'silicon',
                    'silver': 'silver',
                    'titanium': 'titanium',
                    'chromium': 'chromium',
                    'molybdenum': 'molybdenum',
                }
                
                for keyword, element in element_keywords.items():
                    if keyword in resource_lower and keyword in resource_str_lower:
                        matched_resources.append(resource_name)
                        break
            
            return list(set(matched_resources))  # Remove duplicates
        
        # Process ppu_supply_risk_resources.csv
        for _, row in ppu_supply_risk_df.iterrows():
            ppu_name = row['PPU']
            descriptive_component = row['Component']
            resources_str = str(row.get('Primary_Resources_Manufacturing', ''))
            
            # Skip if no resources specified
            if pd.isna(resources_str) or resources_str == 'nan' or resources_str.strip() == '':
                continue
            
            # Get actual component list for this PPU
            ppu_row = ppu_constructs_df[ppu_constructs_df['PPU'] == ppu_name]
            if len(ppu_row) > 0:
                components = ppu_row.iloc[0]['Components']
                if isinstance(components, str):
                    try:
                        components = ast.literal_eval(components)
                    except:
                        components = [components]
                
                # Map descriptive component to short component name
                short_component = _map_descriptive_to_short_component(
                    descriptive_component, ppu_name, components
                )
                
                if short_component:
                    # Match resources in the string to standardized resource names
                    matched_resources = match_resource_in_string(resources_str, resource_risk_df)
                    
                    # Add matched resources to component mapping
                    for resource in matched_resources:
                        if short_component not in component_to_resources:
                            component_to_resources[short_component] = []
                        if resource not in component_to_resources[short_component]:
                            component_to_resources[short_component].append(resource)
    
    # Fallback: For resources not found in ppu_supply_risk_resources.csv,
    # use Components_Using column from natural_resources_supply_risk.csv to
    # intelligently map to specific component types (not all components in PPU chain)
    for _, row in resource_risk_df.iterrows():
        resource = row['Resource']
        components_using = str(row.get('Components_Using', ''))
        ppus_dependent = str(row.get('PPUs_Dependent', ''))
        
        # Check if this resource was already mapped via ppu_supply_risk_resources.csv
        resource_already_mapped = False
        for comp_name in component_to_resources.keys():
            if resource in component_to_resources[comp_name]:
                resource_already_mapped = True
                break
        
        if resource_already_mapped:
            continue  # Skip fallback, already mapped
        
        # Use Components_Using description to map to specific component types
        # Examples: "Permanent magnets (PMSG wind turbines)" → only Wind components
        #           "Li-ion battery cathodes" → only Battery components
        components_using_lower = components_using.lower()
        
        # Parse PPU names
        ppu_names = [p.strip() for p in ppus_dependent.split(',')]
        
        for ppu_name in ppu_names:
            ppu_row = ppu_constructs_df[ppu_constructs_df['PPU'] == ppu_name]
            if len(ppu_row) > 0:
                components = ppu_row.iloc[0]['Components']
                if isinstance(components, str):
                    try:
                        components = ast.literal_eval(components)
                    except:
                        components = [components]
                
                # Map based on Components_Using description
                target_components = []
                
                if 'wind' in components_using_lower or 'pmsg' in components_using_lower:
                    # Only wind components
                    target_components = [c for c in components if 'wind' in c.lower()]
                elif 'battery' in components_using_lower or 'li-ion' in components_using_lower:
                    # Only battery components
                    target_components = [c for c in components if 'battery' in c.lower()]
                elif 'inverter' in components_using_lower or 'power electronics' in components_using_lower:
                    # Only inverter/converter components
                    target_components = [c for c in components if 'inverter' in c.lower() or 'converter' in c.lower()]
                elif 'electrolyser' in components_using_lower or 'electrolyzer' in components_using_lower:
                    # Only electrolyzer components
                    target_components = [c for c in components if 'electrolyzer' in c.lower() or 'electrolyser' in c.lower()]
                elif 'ice' in components_using_lower and 'aftertreatment' in components_using_lower:
                    # Only ICE components
                    target_components = [c for c in components if c.upper() == 'ICE']
                elif 'catalyst' in components_using_lower:
                    # Map to components that use catalysts (synthesis, cracking, etc.)
                    target_components = [c for c in components if any(kw in c.lower() for kw in ['synthesis', 'cracking', 'sabatier', 'ft-', 'reforming'])]
                else:
                    # If no specific component type identified, don't assign (conservative)
                    # This prevents over-assignment
                    continue
                
                # Assign resource only to identified target components
                for component in target_components:
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

