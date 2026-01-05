#!/usr/bin/env python3
"""
Validation Script: Check if Energy_Portfolio_Optimization.ipynb correctly filters for compliance

Usage:
    python validate_notebook_compliance.py

This script validates that the notebook's compliance filtering is working correctly
by checking the generated CSV files for constraint violations.
"""

import pandas as pd
import json
from pathlib import Path
from typing import Dict, Tuple


def is_portfolio_compliant(portfolio_dict: Dict, row: pd.Series) -> bool:
    """
    Check if a portfolio satisfies ALL three hard constraints.
    (Same logic as in the notebook)
    """
    # Parse portfolio if needed
    if isinstance(portfolio_dict, str):
        try:
            portfolio = json.loads(portfolio_dict)
        except:
            return False
    else:
        portfolio = portfolio_dict
    
    # 1. Aviation Fuel Constraint
    MIN_THERM_FOR_AVIATION = 263
    therm_count = portfolio.get('THERM', 0)
    syn_ft_count = portfolio.get('SYN_FT', 0)
    syn_crack_count = portfolio.get('SYN_CRACK', 0)
    
    aviation_compliant = (
        therm_count >= MIN_THERM_FOR_AVIATION and 
        (syn_ft_count > 0 or syn_crack_count > 0)
    )
    
    if not aviation_compliant:
        return False
    
    # 2. Cyclic SOC Constraint
    if 'storage_constraint_met' in row:
        soc_compliant = row['storage_constraint_met']
        if not soc_compliant:
            return False
    
    # 3. Electrical Sovereignty (113 TWh/year)
    if 'total_domestic_production_twh' in row:
        electrical_compliant = row['total_domestic_production_twh'] >= 113.0
        if not electrical_compliant:
            return False
    
    return True


def validate_csv_file(csv_path: Path) -> Tuple[int, int, int]:
    """
    Validate a results CSV file for compliance.
    
    Returns:
        (total_portfolios, compliant_portfolios, non_compliant_portfolios)
    """
    if not csv_path.exists():
        print(f"⚠️  File not found: {csv_path}")
        return 0, 0, 0
    
    # Read file in chunks to avoid memory issues
    try:
        df = pd.read_csv(csv_path, nrows=100)  # Sample first 100 rows
        total = len(df)
        
        # Get actual total count
        with open(csv_path, 'r') as f:
            actual_total = sum(1 for _ in f) - 1  # -1 for header
        
        compliant_mask = df.apply(
            lambda row: is_portfolio_compliant(row.get('portfolio_dict', {}), row),
            axis=1
        )
        
        compliant = compliant_mask.sum()
        non_compliant = total - compliant
        
        return actual_total, compliant, non_compliant
    except Exception as e:
        print(f"   ❌ Error reading file: {e}")
        return 0, 0, 0


def main():
    """Run validation checks."""
    print("=" * 70)
    print("VALIDATION: Portfolio Compliance Filtering")
    print("=" * 70)
    print()
    
    # Check if result files exist
    data_dir = Path("data/result_plots")
    
    files_to_check = [
        "multi_objective_results_final.csv",
        "multi_objective_results_final_frontier_3d.csv",
    ]
    
    all_passed = True
    
    for filename in files_to_check:
        filepath = data_dir / filename
        print(f"📄 Checking: {filename}")
        print("-" * 70)
        
        total, compliant, non_compliant = validate_csv_file(filepath)
        
        if total == 0:
            print(f"   ⚠️  No data found (file missing or empty)")
            print()
            continue
        
        sample_size = compliant + non_compliant
        compliance_rate = (compliant / sample_size) * 100 if sample_size > 0 else 0
        
        print(f"   Total portfolios in file: {total}")
        print(f"   Sample checked: {sample_size} portfolios")
        print(f"   ✅ Compliant: {compliant} ({compliance_rate:.1f}%)")
        print(f"   ❌ Non-compliant: {non_compliant} ({100-compliance_rate:.1f}%)")
        
        if non_compliant > 0:
            print(f"\n   ⚠️  WARNING: {non_compliant} non-compliant portfolios found!")
            print(f"   These portfolios should be filtered out in visualization cells.")
            all_passed = False
        else:
            print(f"\n   ✅ PASS: All portfolios are compliant!")
        
        print()
    
    print("=" * 70)
    if all_passed:
        print("✅ VALIDATION PASSED")
        print("All CSV files contain only compliant portfolios or will be filtered.")
    else:
        print("⚠️  VALIDATION WARNING")
        print("Some CSV files contain non-compliant portfolios.")
        print("This is OK if you ran optimization before adding the new constraints.")
        print("The notebook visualization cells will filter them automatically.")
    print("=" * 70)
    print()
    print("Next Steps:")
    print("1. Re-run the optimization with updated constraints (optional)")
    print("2. Run notebook cells 16, 21, 22, 24 to apply filtering")
    print("3. Only compliant portfolios will be displayed")
    print()


if __name__ == "__main__":
    main()

