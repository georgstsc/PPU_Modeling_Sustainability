# PPU Modeling for Sustainability

## Overview

This project develops a comprehensive optimization model to determine the optimal mix of Power Production Units (PPUs) for achieving Switzerland's energy sovereignty with net-zero CO₂ emissions over an 80-year horizon. The model balances economic costs, environmental impacts (CO₂ emissions), and energy sovereignty risks through a portfolio-style optimization approach.

## Project Goal

The primary objective is to create a robust simulation system that identifies the most efficient and sustainable combination of energy production technologies. By the end of the project, we aim to have a complex simulation framework capable of determining the optimal repartition of PPUs, considering various constraints and objectives.

## Key Features

- **Portfolio Optimization**: Maximizes total return over time while meeting demand in every 15-minute interval
- **Multi-Objective Balancing**: Weights monetary costs, CO₂ emissions, and sovereignty penalties
- **Temporal Resolution**: 15-minute timesteps for accurate dispatch and balancing
- **Energy Sources Modeling**: Includes solar, wind, hydro (reservoir and run-of-river), biomass, hydrogen, nuclear, and battery storage
- **Data-Driven**: Uses real 2024 incidence data for solar, wind, and hydrological resources
- **Ranking System**: Identifies optimal locations for PV panels and wind turbines based on resource availability

## Methodology

The optimization uses a composite cost function combining:
- **Price**: Levelized cost of energy (LCOE) for each technology
- **Emissions**: CO₂ equivalent per kWh
- **Sovereignty**: Penalty for import dependence or intermittency

The objective maximizes the harmonic mean of per-slice returns to avoid high-cost scarcity periods.

## Data Sources

- Solar and wind incidence data (hourly, 2024)
- Hydrological data (monthly RoR and reservoir levels)
- Electricity demand profiles
- Cost tables from Prof. Züttel's research
- Exchange rates and energy prices

## Technologies Modeled

### Production PPUs
- HYD_S: Hydro storage
- HYD_R: Hydro run-of-river
- PV: Photovoltaic solar
- WD_ON: Wind onshore
- THERM_G: Thermal biogas
- H2P_G: Hydrogen production (gas)
- And others...

### Storage Systems
- Lake reservoirs
- Battery storage
- Hydrogen storage (various forms)
- Biomass/biofuel storage
- Chemical storage (ammonia, methane)

## Project Structure

```
├── data/                          # Input data files
│   ├── solar_incidence_hourly_2024.csv
│   ├── wind_incidence_hourly_2024.csv
│   ├── ranking_incidence/         # Location rankings
│   └── ...
├── optimization_problem.ipynb     # Main optimization notebook
├── ppu_model_data.ipynb          # Data processing and analysis
├── data_visualization.py         # Visualization functions
└── README.md                     # This file
```

## Supervision

This project is conducted under the supervision of **Prof. Andreas Züttel**, Lead of the Laboratory of Materials for Renewable Energy (LMER) at EPFL.

## Collaborators

- Georgios Stamatopoulos (Primary developer)

## Requirements

- Python 3.8+
- Libraries: pandas, numpy, matplotlib, gurobipy, xarray, networkx
- Jupyter Notebook

## Usage

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt` (if available)
3. Run the notebooks in order:
   - `ppu_model_data.ipynb` for data processing
   - `optimization_problem.ipynb` for optimization

## Future Developments

- Integration of more detailed cost models
- Incorporation of grid infrastructure constraints
- Multi-year optimization with technology evolution
- Uncertainty analysis and robustness testing
- Web-based interface for scenario exploration

## References

- Züttel, A. et al. (2023). Cost analysis of Power Production Units for CO₂-neutral energy systems.
- Swiss Federal Office of Energy (BFE) hydropower statistics
- European energy market data (2024)

## License

This project is part of academic research at EPFL. Please contact the supervisor for usage permissions.