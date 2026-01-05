#!/usr/bin/env python3
"""
Verification Script: Check if constraint fields are present in CSV files

Usage:
    python verify_constraint_fields.py

This script checks if the multi-objective GA output includes the required
constraint compliance fields.
"""

import pandas as pd
import sys
from pathlib import Path


def verify_csv_has_constraint_fields(csv_path: str) -> bool:
    """
    Check if CSV file has all required constraint fields.
    
    Args:
        csv_path: Path to CSV file
        
    Returns:
        True if all fields present, False otherwise
    """
    required_fields = [
        'storage_constraint_met',
        'total_domestic_production_twh',
        'aviation_fuel_constraint_met'
    ]
    
    if not Path(csv_path).exists():
        print(f"❌ File not found: {csv_path}")
        return False
    
    try:
        df = pd.read_csv(csv_path)
        
        print(f"\n{'='*60}")
        print(f"Checking: {csv_path}")
        print(f"{'='*60}")
        print(f"Total portfolios: {len(df)}")
        print(f"\nColumns in CSV:")
        for col in df.columns:
            print(f"  - {col}")
        
        print(f"\n{'='*60}")
        print("Required Constraint Fields:")
        print(f"{'='*60}")
        
        all_present = True
        for field in required_fields:
            if field in df.columns:
                print(f"  ✅ {field}")
                # Show statistics
                if field == 'storage_constraint_met' or field == 'aviation_fuel_constraint_met':
                    n_true = df[field].sum()
                    n_false = (~df[field]).sum()
                    print(f"     True: {n_true}, False: {n_false}")
                elif field == 'total_domestic_production_twh':
                    print(f"     Range: [{df[field].min():.1f}, {df[field].max():.1f}] TWh/year")
                    n_compliant = (df[field] >= 113.0).sum()
                    print(f"     ≥113 TWh: {n_compliant}/{len(df)}")
            else:
                print(f"  ❌ {field} - MISSING!")
                all_present = False
        
        print(f"\n{'='*60}")
        if all_present:
            print("✅ ALL REQUIRED FIELDS PRESENT")
            
            # Calculate fully compliant portfolios
            compliant_mask = (
                df['storage_constraint_met'] & 
                (df['total_domestic_production_twh'] >= 113.0) &
                df['aviation_fuel_constraint_met']
            )
            n_compliant = compliant_mask.sum()
            
            print(f"\nFully Compliant Portfolios:")
            print(f"  {n_compliant}/{len(df)} ({100*n_compliant/len(df):.1f}%)")
            
            if n_compliant == 0:
                print("\n⚠️  WARNING: No compliant portfolios found!")
                print("   This means ALL portfolios violate at least one constraint.")
                print("   Consider:")
                print("   - Increasing portfolio bounds")
                print("   - Relaxing constraint tolerances")
                print("   - Running GA for more generations")
        else:
            print("❌ MISSING REQUIRED FIELDS")
            print("\n🔧 Fix Required:")
            print("   1. Update portfolio_metrics.py (PortfolioMetrics3D class)")
            print("   2. Update multi_objective_explorer.py (CSV writing)")
            print("   3. Re-run the multi-objective GA")
        
        print(f"{'='*60}\n")
        
        return all_present
        
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        return False


def main():
    """Main verification function."""
    print("\n" + "="*60)
    print("CONSTRAINT FIELDS VERIFICATION")
    print("="*60)
    
    # Check both main results and frontier
    csv_files = [
        'data/result_plots/multi_objective_results.csv',
        'data/result_plots/multi_objective_results_frontier_3d.csv'
    ]
    
    results = {}
    for csv_file in csv_files:
        results[csv_file] = verify_csv_has_constraint_fields(csv_file)
    
    # Final summary
    print("\n" + "="*60)
    print("VERIFICATION SUMMARY")
    print("="*60)
    
    all_passed = all(results.values())
    
    for csv_file, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {csv_file}")
    
    print("="*60)
    
    if all_passed:
        print("\n✅ All CSV files have required constraint fields!")
        print("   The notebook filtering should now work correctly.")
        return 0
    else:
        print("\n❌ Some CSV files are missing constraint fields!")
        print("   The notebook filtering will NOT work correctly.")
        print("\n📝 Action Required:")
        print("   1. Review CONSTRAINT_COMPLIANCE_FIX.md")
        print("   2. Re-run the multi-objective GA")
        print("   3. Run this verification script again")
        return 1


if __name__ == "__main__":
    sys.exit(main())

