# =============================================================================
# SECTION 4: Interactive Portfolio Analysis - COMPLETE IMPORTS
# =============================================================================
# Add these imports at the TOP of your Section 4 cell:

import time
import json
from pathlib import Path
from IPython.display import clear_output, display
from ipywidgets import Dropdown, Button, Output, VBox, HBox, Layout, Label
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# PPU Framework imports
from ppu_framework import Portfolio

# Optimization imports (MISSING - causes NameError)
from optimization import Individual, evaluate_portfolio_full_year

# Config import (MISSING)
from config import DEFAULT_CONFIG as config

# Visualization imports (MISSING - for plotting functions)
from data_visualization import (
    plot_full_year_production_by_source,
    plot_full_year_storage,
    plot_energy_balance_distribution
)

# Import the compliance check function from Section 3.1
# (or define it inline if not available)
def is_portfolio_compliant(portfolio_dict, row) -> bool:
    """Check if portfolio satisfies constraints (relaxed - no storage check)."""
    if isinstance(portfolio_dict, str):
        try:
            portfolio = json.loads(portfolio_dict)
        except:
            return False
    else:
        portfolio = portfolio_dict
    
    # 1. Aviation: THERM >= 263 + (SYN_FT or SYN_CRACK)
    MIN_THERM_FOR_AVIATION = 263
    therm_count = portfolio.get('THERM', 0)
    syn_ft_count = portfolio.get('SYN_FT', 0)
    syn_crack_count = portfolio.get('SYN_CRACK', 0)
    
    if not (therm_count >= MIN_THERM_FOR_AVIATION and (syn_ft_count > 0 or syn_crack_count > 0)):
        return False
    
    # 2. Sovereignty: >= 113 TWh
    if 'total_domestic_production_twh' in row:
        if row['total_domestic_production_twh'] < 113.0:
            return False
    
    # Storage constraint SKIPPED (CSV has old 5% tolerance)
    return True

print("✅ All imports loaded successfully!")

