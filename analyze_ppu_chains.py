#!/usr/bin/env python3
"""PPU Chain Analysis: Efficiency and LCOE Calculation."""

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
        return {'eff': 1.0, 'w': 0.0, 'capex': 0.0, 'opex': 0.0, 'lifetime': 50.0, 'pavg': 1.0}
    print(f"  ⚠️  Component '{name}' not found")
    return {'eff': 1.0, 'w': 0.0, 'capex': 0.0, 'opex': 0.0, 'lifetime': 20.0, 'pavg': 1.0}

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

def annuity(capex, lifetime, r=0.02):
    """Amortize CAPEX over lifetime."""
    if lifetime <= 0 or capex <= 0:
        return 0.0
    return capex * r * (1+r)**lifetime / ((1+r)**lifetime - 1)

def calc_lcoe(comps, eff):
    """LCOE for 1 GW plant in CHF/kWh."""
    annual = 0.0
    pavg = 1.0
    for c in comps:
        d = get_comp(c)
        cap = d['capex'] * 1e6  # CHF for 1 GW (capex is CHF/kW × 1e6 kW)
        opx = d['opex'] * 1e6   # CHF/yr for 1 GW
        annual += annuity(cap, d['lifetime']) + opx
        if d['pavg'] < pavg:
            pavg = d['pavg']
    
    energy_mwh = 1000 * 8760 * pavg * eff  # 1 GW × 8760h × CF × efficiency
    return (annual / energy_mwh / 1000) if energy_mwh > 0 else 999.0

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
        all_c = b1 + b2 + merge
        lcoe = calc_lcoe(all_c, eff)
        typ = 'Parallel'
        subchains = f"B1({w[0]:.0%}): {b1} | B2({w[1]:.0%}): {b2} | Merge: {merge}"
    else:
        eff = chain_eff(comps)
        lcoe = calc_lcoe(comps, eff)
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
    
    lcoe = calc_lcoe(all_c, eta_final)
    print(f"  ✅ LCOE: {lcoe:.4f} CHF/kWh ({lcoe*1000:.2f} CHF/MWh)")

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
        for c in comps:
            d = get_comp(c)
            running_eta *= d['eff']
            w_tot += d['w']
            print(f"  • {c:<35} η={d['eff']:.4f}  w={d['w']:.4f}  → cumulative η={running_eta:.4f}")
        
        final_eta = running_eta * (1 - min(w_tot, 0.99))
        print(f"  Loss adjustment: (1 - Σw) = (1 - {w_tot:.4f}) = {1-w_tot:.4f}")
        print(f"  ✅ Overall Efficiency: {final_eta:.4f} ({final_eta*100:.2f}%)")
        print(f"  ✅ LCOE: {r['LCOE_CHF_kWh']:.4f} CHF/kWh ({r['LCOE_CHF_kWh']*1000:.2f} CHF/MWh)")

# Save to CSV
result_df = pd.DataFrame(results)
result_df.to_csv('data/ppu_efficiency_lcoe_analysis.csv', index=False)
print(f"\n✅ Results saved to data/ppu_efficiency_lcoe_analysis.csv")

