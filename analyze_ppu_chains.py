#!/usr/bin/env python3
"""
PPU Chain Analysis: Efficiency and LCOE Calculation.

IMPORTANT CHANGE (v2.0):
- LCOE is now calculated by DIRECTLY SUMMING the 'cost' column from cost_table_tidy.csv
- We NO LONGER calculate from CAPEX/OPEX to avoid double-counting efficiency losses
- Efficiency is ONLY used for calculating chain efficiency (energy output), NOT for cost adjustment
- The 'cost' column already represents the cost per kWh for each component
- Progressive cost escalation (soft caps) remains unchanged
"""

import pandas as pd
import ast

# Load data
ppu_df = pd.read_csv('data/ppu_constructs_components.csv')
cost_df = pd.read_csv('data/cost_table_tidy.csv')

# Create component lookup (component names are in 'item' column)
comp_map = {}
for _, row in cost_df.iterrows():
    comp_map[row['item']] = {
        'eff': float(row['efficiency']) if pd.notna(row['efficiency']) else 1.0,
        'w': float(row['w']) if pd.notna(row['w']) else 0.0,
        'cost': float(row['cost']) if pd.notna(row['cost']) else 0.0,  # Direct cost in CHF/kWh
        'capex': float(row['capex']) if pd.notna(row['capex']) else 0.0,
        'opex': float(row['opex']) if pd.notna(row['opex']) else 0.0,
        'lifetime': float(row['lifetime']) if pd.notna(row['lifetime']) else 20.0,
        'pavg': float(row['pavg_per_pp']) if pd.notna(row['pavg_per_pp']) else 1.0,
    }

# Name aliases for mismatches between PPU components and cost table items
aliases = {
    'Biogas (50% CH4)': 'Biogas  (50% CH4)',  # Double space in cost table
    'Fuel Transport': 'Fuel Taransport',       # Typo in cost table
    'Grid': 'Grid 10km',                        # Generic Grid -> 10km version
    'River': 'Hydro Turb',                      # Use Hydro Turb for River
}

def get_comp(name):
    # Direct lookup
    if name in comp_map:
        return comp_map[name]
    # Alias lookup
    if name in aliases and aliases[name] in comp_map:
        return comp_map[aliases[name]]
    # Fuzzy match - check if name is substring of key or key is substring of name
    for k in comp_map:
        if name.lower() == k.lower():
            return comp_map[k]
    for k in comp_map:
        if name.lower() in k.lower() or k.lower() in name.lower():
            return comp_map[k]
    # Special cases for components without cost data (natural resources, basic infrastructure)
    zero_cost_components = ['River', 'Grid']  # These are "free" natural resources or basic infrastructure
    if name in zero_cost_components:
        return {'eff': 1.0, 'w': 0.0, 'cost': 0.0, 'capex': 0.0, 'opex': 0.0, 'lifetime': 50.0, 'pavg': 1.0}
    print(f"  ⚠️  Component '{name}' not found")
    return {'eff': 1.0, 'w': 0.0, 'cost': 0.0, 'capex': 0.0, 'opex': 0.0, 'lifetime': 20.0, 'pavg': 1.0}

# Parallel PPU configurations: (branch1, branch2, merge, (weight1, weight2))
PARALLEL = {
    'SYN_FT': (
        ['Electrolyzer'],
        ['CO2 Capture 400 ppm'],
        ['H2, CO2 compression', 'FT-Synthesis', 'Refining', 'Fuel Tank'],
        (0.85, 0.15)
    ),
    'SYN_METH': (
        ['Electrolyzer'],
        ['CO2 Capture 400 ppm'],
        ['Sabatier reaction', 'Methane compression', 'CH4 storage 200bar'],
        (0.85, 0.15)
    ),
    'NH3_FULL': (
        ['Electrolyzer'],
        ['N2 separation'],
        ['N2,H2 compression', 'NH3 synthesis', 'Ammonia liquifaction', 'Ammonia storage'],
        (0.75, 0.25)
    ),
    'SYN_CRACK': (
        ['Electrolyzer'],
        ['Palm oil'],
        ['H2, CO2 compression', 'Cracking', 'Refining', 'Fuel Tank'],
        (0.32, 0.68)
    ),
    'CH4_BIO': (
        ['Electrolyzer'],
        ['Biogas purification', 'Biogas (50% CH4)'],
        ['Sabatier reaction', 'Methane compression'],
        (0.70, 0.30)
    ),
}

def chain_eff(comps):
    """Sequential efficiency: Π(η_i) × (1 - Σw_i)."""
    eta = 1.0
    w_tot = 0.0
    for c in comps:
        d = get_comp(c)
        eta *= d['eff']
        w_tot += d['w']
    return eta * (1 - min(w_tot, 0.99))

def branch_eff_raw(comps):
    """Branch efficiency (product only, no w adjustment)."""
    eta = 1.0
    for c in comps:
        eta *= get_comp(c)['eff']
    return eta

def parallel_eff(b1, b2, merge, weights):
    """Parallel chain efficiency: η_m × (w1×η1 + w2×η2) × (1-Σw)."""
    eta1 = branch_eff_raw(b1)
    eta2 = branch_eff_raw(b2)
    eta_m = branch_eff_raw(merge)
    w1, w2 = weights
    
    all_comps = b1 + b2 + merge
    w_tot = sum(get_comp(c)['w'] for c in all_comps)
    
    return eta_m * (w1 * eta1 + w2 * eta2) * (1 - min(w_tot, 0.99))

def calc_lcoe_direct(comps):
    """
    Calculate LCOE by directly summing the 'cost' column from cost_table_tidy.csv.
    
    This avoids double-counting efficiency losses, as the 'cost' column already
    represents the cost per kWh for each component. We simply sum them.
    
    Efficiency is NOT used to adjust costs here - it's only used separately
    to calculate the overall chain efficiency for energy output calculations.
    
    IMPORTANT: The 'cost' column is assumed to already account for component-level
    costs. We do NOT divide by efficiency, as that would double-count losses.
    """
    total_cost = 0.0
    for c in comps:
        d = get_comp(c)
        comp_cost = d.get('cost', 0.0)
        if comp_cost == 0.0 and c not in ['River', 'Grid']:
            print(f"    ⚠️  Warning: Component '{c}' has zero cost")
        total_cost += comp_cost  # Direct sum of component costs (CHF/kWh)
    return total_cost

def calc_lcoe_parallel(b1, b2, merge, weights):
    """
    Calculate LCOE for parallel chains by weighted sum of branch costs.
    
    For parallel chains, we weight the costs by the branch weights:
    - Branch 1 cost × weight1 + Branch 2 cost × weight2 + Merge cost
    
    IMPORTANT: Costs are NOT adjusted by efficiency here. The 'cost' column
    already represents the cost per kWh for each component.
    """
    w1, w2 = weights
    
    # Branch 1 cost (weighted)
    branch1_cost = sum(get_comp(c).get('cost', 0.0) for c in b1) * w1
    
    # Branch 2 cost (weighted)
    branch2_cost = sum(get_comp(c).get('cost', 0.0) for c in b2) * w2
    
    # Merge path cost (full cost, not weighted)
    merge_cost = sum(get_comp(c).get('cost', 0.0) for c in merge)
    
    return branch1_cost + branch2_cost + merge_cost

# Main analysis
print("=" * 120)
print("PPU CHAIN ANALYSIS: EFFICIENCY AND LCOE")
print("=" * 120)

results = []

for _, row in ppu_df.iterrows():
    ppu = row['PPU']
    comps = ast.literal_eval(row['Components'])
    cat = row['Category']
    ext = row['Extract']
    
    if ppu in PARALLEL:
        b1, b2, merge, w = PARALLEL[ppu]
        eff = parallel_eff(b1, b2, merge, w)
        lcoe = calc_lcoe_parallel(b1, b2, merge, w)
        typ = 'Parallel'
        subchains = f"B1({w[0]:.0%}): {b1} | B2({w[1]:.0%}): {b2} | Merge: {merge}"
    else:
        eff = chain_eff(comps)
        lcoe = calc_lcoe_direct(comps)
        typ = 'Sequential'
        subchains = ' → '.join(comps)
    
    results.append({
        'PPU': ppu,
        'Type': typ,
        'Efficiency': eff,
        'LCOE_CHF_kWh': lcoe,
        'Category': cat,
        'Extract': ext,
        'Components': comps,
        'Subchains': subchains
    })

# Print summary table
print(f"\n{'PPU':<15} {'Type':<12} {'Efficiency':>10} {'LCOE(CHF/kWh)':>14} {'Category':<12} {'Extract':<10}")
print("-" * 120)

for r in sorted(results, key=lambda x: -x['Efficiency']):
    print(f"{r['PPU']:<15} {r['Type']:<12} {r['Efficiency']:>10.2%} {r['LCOE_CHF_kWh']:>14.4f} {r['Category']:<12} {r['Extract']:<10}")

# Detailed parallel PPU breakdown
print("\n" + "=" * 120)
print("DETAILED PARALLEL PPU ANALYSIS")
print("=" * 120)

for ppu, (b1, b2, merge, w) in PARALLEL.items():
    print(f"\n📊 {ppu}")
    print("-" * 80)
    
    # Branch efficiencies
    eta1 = branch_eff_raw(b1)
    eta2 = branch_eff_raw(b2)
    eta_m = branch_eff_raw(merge)
    
    # Component-wise breakdown
    print(f"  Branch 1 (weight = {w[0]:.0%}):")
    for c in b1:
        d = get_comp(c)
        print(f"    • {c:<30} η={d['eff']:.4f}  w={d['w']:.4f}")
    print(f"    → Branch 1 η₁ = {eta1:.4f}")
    
    print(f"  Branch 2 (weight = {w[1]:.0%}):")
    for c in b2:
        d = get_comp(c)
        print(f"    • {c:<30} η={d['eff']:.4f}  w={d['w']:.4f}")
    print(f"    → Branch 2 η₂ = {eta2:.4f}")
    
    print(f"  Merged path:")
    for c in merge:
        d = get_comp(c)
        print(f"    • {c:<30} η={d['eff']:.4f}  w={d['w']:.4f}")
    print(f"    → Merge η_m = {eta_m:.4f}")
    
    # Total w
    all_c = b1 + b2 + merge
    w_tot = sum(get_comp(c)['w'] for c in all_c)
    print(f"  Total w losses: Σw = {w_tot:.4f}")
    
    # Final calculation
    eta_final = eta_m * (w[0]*eta1 + w[1]*eta2) * (1 - min(w_tot, 0.99))
    print(f"  Formula: η = η_m × ({w[0]:.2f}×η₁ + {w[1]:.2f}×η₂) × (1-Σw)")
    print(f"         = {eta_m:.4f} × ({w[0]:.2f}×{eta1:.4f} + {w[1]:.2f}×{eta2:.4f}) × {1-w_tot:.4f}")
    print(f"  ✅ Overall Efficiency: {eta_final:.4f} ({eta_final*100:.2f}%)")
    
    # Cost breakdown
    branch1_cost = sum(get_comp(c)['cost'] for c in b1)
    branch2_cost = sum(get_comp(c)['cost'] for c in b2)
    merge_cost = sum(get_comp(c)['cost'] for c in merge)
    print(f"  Cost breakdown:")
    print(f"    Branch 1 cost: {branch1_cost:.6f} CHF/kWh × {w[0]:.2f} = {branch1_cost * w[0]:.6f} CHF/kWh")
    print(f"    Branch 2 cost: {branch2_cost:.6f} CHF/kWh × {w[1]:.2f} = {branch2_cost * w[1]:.6f} CHF/kWh")
    print(f"    Merge cost: {merge_cost:.6f} CHF/kWh")
    
    lcoe = calc_lcoe_parallel(b1, b2, merge, w)
    print(f"  ✅ LCOE (direct sum): {lcoe:.4f} CHF/kWh ({lcoe*1000:.2f} CHF/MWh)")

# Sequential PPU details
print("\n" + "=" * 120)
print("SEQUENTIAL PPU COMPONENT BREAKDOWN")
print("=" * 120)

for r in results:
    if r['Type'] == 'Sequential':
        ppu = r['PPU']
        comps = r['Components']
        print(f"\n📊 {ppu}")
        print("-" * 80)
        
        running_eta = 1.0
        w_tot = 0.0
        total_cost = 0.0
        for c in comps:
            d = get_comp(c)
            running_eta *= d['eff']
            w_tot += d['w']
            total_cost += d['cost']
            print(f"  • {c:<35} η={d['eff']:.4f}  w={d['w']:.4f}  cost={d['cost']:.6f} CHF/kWh  → cumulative η={running_eta:.4f}")
        
        final_eta = running_eta * (1 - min(w_tot, 0.99))
        print(f"  Loss adjustment: (1 - Σw) = (1 - {w_tot:.4f}) = {1-w_tot:.4f}")
        print(f"  ✅ Overall Efficiency: {final_eta:.4f} ({final_eta*100:.2f}%)")
        print(f"  ✅ Total Cost (direct sum): {total_cost:.6f} CHF/kWh")
        print(f"  ✅ LCOE: {r['LCOE_CHF_kWh']:.4f} CHF/kWh ({r['LCOE_CHF_kWh']*1000:.2f} CHF/MWh)")

# Save to CSV
result_df = pd.DataFrame(results)
result_df.to_csv('data/ppu_efficiency_lcoe_analysis.csv', index=False)
print(f"\n✅ Results saved to data/ppu_efficiency_lcoe_analysis.csv")

