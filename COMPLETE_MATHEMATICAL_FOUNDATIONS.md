# Complete Mathematical Foundations for Energy Portfolio Optimization

**Author:** Georg Schwabedal  
**Supervisor:** Prof. Andreas Züttel, Laboratory of Materials for Renewable Energy (LMER), EPFL  
**Date:** December 2024

---

## Table of Contents

1. [Portfolio Optimization Framework](#1-portfolio-optimization-framework)
2. [Cost Calculation Methodology](#2-cost-calculation-methodology)
3. [Risk Metrics](#3-risk-metrics)
4. [Economic Metrics](#4-economic-metrics)
5. [Storage System Modeling](#5-storage-system-modeling)
6. [Dispatch Simulation](#6-dispatch-simulation)
7. [Optimization Algorithm](#7-optimization-algorithm)
8. [Efficiency Frontier Analysis](#8-efficiency-frontier-analysis)
9. [Constraints and Assumptions](#9-constraints-and-assumptions)
10. [Summary of Key Equations](#10-summary-of-key-equations)

---

## 1. Portfolio Optimization Framework

### 1.1 Problem Formulation

We formulate national energy system design as a **portfolio optimization problem** where:

- **Assets:** Power Production Units (PPUs) representing different energy technologies
- **Portfolio:** Vector of installed capacities (counts) for each PPU type
- **Decision Variables:** $x_k \in \mathbb{N}_0$ for each PPU type $k \in K$, where $K$ is the set of all PPU types (26+ technologies)

A portfolio is defined as:
\[
\mathbf{x} = \{x_1, x_2, \ldots, x_{|K|}\}
\]
where $x_k$ represents the number of units of PPU type $k$ installed.

### 1.2 PPU Decomposition and Structure

Each PPU is decomposed into a **component chain** representing the energy conversion pathway:

1. **Production Component:** Harvests renewable energy (solar, wind, hydro) or generates from fuel
2. **Storage Component (optional):** Converts electricity into storable energy carrier (batteries, hydrogen, chemical storage)
3. **Generation Component (optional):** Converts stored energy back to electricity on demand

**Component Chain Efficiency:**
\[
\eta_{\text{PPU}} = \prod_{i=1}^{n} \eta_i
\]
where $\eta_i$ is the efficiency of component $i$ in the chain, and $n$ is the number of components.

**Total PPU Efficiency:** Accounts for all losses in the conversion chain (production → storage → generation).

### 1.3 Cost Calculation

**Component-Level Costs:**
Each component has:
- **CAPEX:** Capital expenditure (CHF/kW installed)
- **OPEX:** Operational expenditure (CHF/kW/year)
- **Lifetime:** Component lifetime (years)

**PPU-Level Cost (LCOE):**
\[
\text{LCOE}_k = \frac{\sum_{i \in \text{components}} \left( \text{CAPEX}_i + \text{OPEX}_i \cdot \text{lifetime}_i \right)}{\eta_{\text{PPU}} \cdot \text{capacity\_factor}_k \cdot \text{lifetime}_{\text{PPU}}}
\]

**Progressive Cost Escalation:**
To model diminishing returns and resource constraints, costs increase beyond a "soft cap":
\[
\text{Cost}(x_k) = \text{LCOE}_k \cdot \left( 1 + \alpha_k \cdot \max(0, x_k - \text{soft\_cap}_k) \right)
\]
where:
- $\alpha_k$ is the escalation factor for PPU type $k$ (typically 0.0002-0.001 per unit)
- $\text{soft\_cap}_k$ is the threshold beyond which costs escalate (typically 50 units = 0.5 GW)

**Example:** For PV, if soft_cap = 50 and $\alpha = 0.0002$, then:
- Units 1-50: Base cost
- Unit 51: Cost = Base × (1 + 0.0002 × 1) = Base × 1.0002
- Unit 100: Cost = Base × (1 + 0.0002 × 50) = Base × 1.01

### 1.4 Scale Adaptation

**Reference Scale:** 10 MW per unit

**Capacity Calculation:**
\[
\text{Total Capacity}_k = x_k \times 10 \text{ MW}
\]

### 1.5 Grid Assumptions

**Transmission Constraints:** Neglected under the assumption of:
- Well-meshed national grid
- Uniform access to all resources
- Sufficient transmission capacity for modeled scenarios

This simplification allows focus on generation and storage optimization without explicit grid modeling.

---

## 2. Cost Calculation Methodology

### 2.1 Component Chain Cost Calculation

For a PPU with component chain $\{c_1, c_2, \ldots, c_n\}$:

**Step-by-step calculation:**
\[
\text{total\_cost} = \sum_{i=1}^{n} \left( \text{cost}_i + \frac{w_i}{\prod_{j=1}^{i} \eta_j} \right)
\]

where:
- $\text{cost}_i$ is the direct cost of component $i$ (CHF/kWh)
- $w_i$ is the auxiliary energy requirement (kWh/kWh)
- $\eta_j$ is the efficiency of component $j$

**Final efficiency:**
\[
\eta_{\text{PPU}} = \prod_{i=1}^{n} \eta_i
\]

### 2.2 Hourly Production Cost (LCOE)

For each hour $t$, the production cost is a **volume-weighted average**:

\[
C_t = \frac{\sum_{k} P_{k,t} \cdot \text{Cost}_k}{\sum_{k} P_{k,t}}
\]

where:
- $P_{k,t}$ is the power production from PPU $k$ at hour $t$ (MW)
- $\text{Cost}_k$ is the escalated cost per MWh for PPU $k$ (CHF/MWh)

**Cost conversion:** Component costs in CHF/kWh are converted to CHF/MWh by multiplying by 1000.

### 2.3 Total System Cost

**Total cost for hour $t$:**
\[
\text{Total Cost}_t = \sum_{k} P_{k,t} \cdot \text{Cost}_k + \text{Net Spot Cost}_t
\]

where:
\[
\text{Net Spot Cost}_t = \left( \text{Spot Bought}_t - \text{Spot Sold}_t \right) \times S_t
\]

and $S_t$ is the spot price at hour $t$ (CHF/MWh).

**Annual total cost:**
\[
\text{Total Cost}_{\text{annual}} = \sum_{t=1}^{8760} \text{Total Cost}_t
\]

---

## 3. Risk Metrics

### 3.1 Technology Supply Risk (RoT)

#### 3.1.1 Primary Resource Risk (PRR)

For each natural resource $r$ used in PPU components:

\[
\text{PRR}_r = \sqrt{\frac{1}{N_r}}
\]

where $N_r$ is the number of major producing countries for resource $r$.

**Properties:**
- Range: $[0, 1]$ (bounded)
- Interpretation:
  - $\text{PRR} \to 0$: Many producing countries (low risk)
  - $\text{PRR} \to 1$: Few producing countries (high risk)
- Examples:
  - Iron (45 countries): $\text{PRR} = \sqrt{1/45} \approx 0.149$ (low risk)
  - Lithium (8 countries): $\text{PRR} = \sqrt{1/8} \approx 0.354$ (medium risk)
  - Rare Earths (2 countries): $\text{PRR} = \sqrt{1/2} \approx 0.707$ (high risk)

#### 3.1.2 Component Risk

For a component $c$ using multiple resources $\{r_1, r_2, \ldots, r_n\}$, component risk is computed using a **power mean**:

\[
\text{Component Risk}_c = \left( \sum_{i=1}^{n} \text{PRR}_{r_i}^p \right)^{1/p}
\]

where $p = n$ (number of resources used by the component).

**Why Power Mean?**
- Captures worst-case dependency scenarios
- Higher power = more sensitive to high-risk resources
- Additive effect: more resources = higher risk
- Not linear: penalizes single-resource bottlenecks

**Example:**
Component uses Iron (PRR=0.15) and Copper (PRR=0.20):
\[
\text{Component Risk} = \sqrt{0.15^2 + 0.20^2} = \sqrt{0.0625} = 0.250
\]

#### 3.1.3 PPU Risk

For a PPU $k$ with components $\{c_1, c_2, \ldots, c_m\}$, PPU risk is the **arithmetic mean**:

\[
\text{PPU RoT}_k = \frac{1}{m} \sum_{i=1}^{m} \text{Component Risk}_{c_i}
\]

**Rationale:** Average risk across all components in the PPU's conversion chain.

#### 3.1.4 Portfolio Risk

For a portfolio $\mathbf{x}$ with PPU production volumes $\{E_1, E_2, \ldots, E_{|K|}\}$, portfolio risk is **energy-weighted**:

\[
\text{Portfolio RoT}(\mathbf{x}) = \sum_{k=1}^{|K|} \text{PPU RoT}_k \cdot \frac{E_k}{E_{\text{total}}}
\]

where:
- $E_k$ is the annual energy production from PPU type $k$ (MWh)
- $E_{\text{total}} = \sum_{k} E_k$ is total portfolio production

**Rationale:** PPUs that produce more energy have greater influence on portfolio risk.

**Range:** $[0, 1]$ where:
- $0.0-0.2$: Very low risk (diversified resources, many suppliers)
- $0.2-0.4$: Low to medium risk
- $0.4-0.6$: Medium to high risk (some critical dependencies)
- $0.6-1.0$: High risk (concentrated supply chains)

---

## 4. Economic Metrics

### 4.1 Cost Volatility

Cost volatility measures the **temporal variability** of production costs, capturing the economic impact of intermittency and seasonal imbalance.

**Weekly Volatility:**
\[
\sigma_{\text{week},w} = \sqrt{\frac{1}{n_w-1} \sum_{t \in \text{week } w} \left( C_t - \bar{C}_w \right)^2}
\]

where:
- $C_t$ is the production cost per MWh at hour $t$
- $\bar{C}_w$ is the mean cost for week $w$
- $n_w = 168$ hours per week

**Annual Volatility:**
\[
\sigma_{\text{annual}} = \frac{1}{W} \sum_{w=1}^{W} \sigma_{\text{week},w}
\]

where $W$ is the number of weeks in the year (52).

**Production Cost Calculation:**
\[
C_t = \frac{\sum_{k} P_{k,t} \cdot \text{LCOE}_k}{\sum_{k} P_{k,t}}
\]

where $P_{k,t}$ is the power production from PPU $k$ at hour $t$ (MW).

**Interpretation:**
- Low volatility (< 100 CHF/MWh): Stable costs, predictable economics
- Medium volatility (100-300 CHF/MWh): Moderate variation, manageable
- High volatility (> 300 CHF/MWh): High variation, economic uncertainty

### 4.2 Economic Return

Return measures the **economic performance** relative to purchasing all energy from the spot market.

**Weekly Return:**
\[
R_w = \frac{\text{Revenue}_{\text{spot}} - \text{Cost}_{\text{production}}}{\text{Revenue}_{\text{spot}}} \times 100\%
\]

where:
- $\text{Revenue}_{\text{spot}} = \sum_{t \in w} P_t \cdot S_t$ (revenue if sold at spot prices)
- $\text{Cost}_{\text{production}} = \sum_{t \in w} P_t \cdot C_t$ (actual production cost)
- $P_t$ is production at hour $t$ (MWh)
- $S_t$ is spot price at hour $t$ (CHF/MWh)
- $C_t$ is production cost at hour $t$ (CHF/MWh)

**Annual Return:**
\[
R_{\text{annual}} = \frac{1}{W} \sum_{w=1}^{W} R_w
\]

**Interpretation:**
- **Positive return:** Production cheaper than spot (savings)
- **Negative return:** Production more expensive than spot (premium for sovereignty)
- Typical range: -400% to +50% (negative is common for energy sovereignty portfolios)

---

## 5. Storage System Modeling

### 5.1 Storage Capacity Scaling

**Storage capacity scales with the number of PPU units** that input to (charge) the storage system. This reflects the physical reality that more charging infrastructure requires proportionally more storage capacity.

**Capacity Scaling Formula:**
\[
C_m = C_m^{\text{base}} \times \max(1, N_m^{\text{input}})
\]

where:
- $C_m$ is the total capacity of storage type $m$ (MWh)
- $C_m^{\text{base}}$ is the base capacity per unit (MWh)
- $N_m^{\text{input}}$ is the number of PPU units that input to storage $m$

**Power Scaling:**
\[
P_m^{\text{charge}} = N_m^{\text{input}} \times P_{\text{unit}}
\]
\[
P_m^{\text{discharge}} = N_m^{\text{extract}} \times P_{\text{unit}}
\]

where:
- $P_m^{\text{charge}}$ is maximum charge power (MW)
- $P_m^{\text{discharge}}$ is maximum discharge power (MW)
- $N_m^{\text{extract}}$ is the number of PPU units that extract from storage $m$
- $P_{\text{unit}} = 10$ MW per PPU unit

**Special Cases:**
- **Lake Storage:** Capacity does NOT scale (fixed at 8.87 TWh, physical constraint)
- **Physical Power Caps:** Some storages have hard limits (e.g., Lake = 2 GW) regardless of PPU count

**Rationale:**
- More charging PPUs → more energy to store → larger capacity needed
- More extraction PPUs → more discharge power needed
- Models infrastructure scaling realistically

### 5.2 Storage State Evolution

**State of Charge (SOC) Update:**
\[
\text{SOC}_{m}(t+1) = \text{SOC}_{m}(t) \times (1 - \delta_m) + \text{Charge}_m(t) \times \eta_{\text{charge}} - \frac{\text{Discharge}_m(t)}{\eta_{\text{discharge}}}
\]

where:
- $\delta_m$ is self-discharge rate for storage $m$ (per hour)
- $\eta_{\text{charge}}, \eta_{\text{discharge}}$ are round-trip efficiencies
- $\text{Charge}_m(t), \text{Discharge}_m(t)$ are charge/discharge amounts at hour $t$ (MWh)

### 5.3 Disposition Index

The **disposition index** measures storage "fullness" relative to target:

\[
\text{DI}_m(t) = \frac{\text{SOC}_m(t) - \text{SOC}_{\text{target}}}{\text{SOC}_{\text{max}} - \text{SOC}_{\text{target}}}
\]

**Usage:**
- **Discharge Priority:** Higher DI = discharge first (more full)
- **Charge Priority:** Lower DI = charge first (more empty)

---

## 6. Dispatch Simulation

### 6.1 Merit-Order Dispatch Logic

Portfolio operation is simulated using a **merit-order dispatch logic** that prioritizes:
1. Zero marginal cost generation (renewables)
2. Storage charging/discharging
3. Dispatchable generators
4. Spot market imports/exports

### 6.2 Dispatch Algorithm (Per Hour)

For each hour $t$:

1. **Calculate Renewable Production:**
   \[
   P_{\text{renewable}}(t) = P_{\text{PV}}(t) + P_{\text{Wind}}(t) + P_{\text{Hydro}}(t)
   \]

2. **Calculate Total Production:**
   \[
   P_{\text{total}}(t) = P_{\text{renewable}}(t) + P_{\text{dispatchable}}(t) + P_{\text{storage\_discharge}}(t)
   \]

3. **Calculate Deficit/Surplus:**
   \[
   \Delta(t) = D(t) - P_{\text{total}}(t)
   \]

4. **If Deficit ($\Delta(t) > 0$):**
   - Discharge storage (prioritize by disposition index)
   - If still deficit: Buy from spot market
   - Cost: $\text{Cost}_{\text{buy}} = \Delta(t) \times S(t)$

5. **If Surplus ($\Delta(t) < 0$):**
   - Charge storage (prioritize by disposition index)
   - If storage full: Sell to spot market
   - Revenue: $\text{Revenue}_{\text{sell}} = |\Delta(t)| \times S(t)$

6. **Update Storage States:**
   \[
   \text{SOC}_{m}(t+1) = \text{SOC}_{m}(t) \times (1 - \delta_m) + \text{Charge}_m(t) \times \eta_{\text{charge}} - \frac{\text{Discharge}_m(t)}{\eta_{\text{discharge}}}
   \]

### 6.3 Renewable Production Models

**Solar Power:**
\[
P_{\text{PV}}(t, \text{rank}) = \eta_{\text{PV}} \cdot A \cdot I(t, \text{rank})
\]

where:
- $\eta_{\text{PV}}$ is panel efficiency
- $A$ is panel area
- $I(t, \text{rank})$ is solar irradiance at location rank and hour $t$

**Wind Power:**
\[
P_{\text{Wind}}(t, \text{rank}) = \begin{cases}
0 & \text{if } v < v_{\text{cut-in}} \\
P_{\text{rated}} \cdot \left( \frac{v}{v_{\text{rated}}} \right)^3 & \text{if } v_{\text{cut-in}} \leq v < v_{\text{rated}} \\
P_{\text{rated}} & \text{if } v_{\text{rated}} \leq v < v_{\text{cut-out}} \\
0 & \text{if } v \geq v_{\text{cut-out}}
\end{cases}
\]

where:
- $v$ is wind speed
- $v_{\text{cut-in}}, v_{\text{rated}}, v_{\text{cut-out}}$ are turbine speed thresholds
- $P_{\text{rated}}$ is rated power

### 6.4 Spot Market Interactions

**Buy from Spot:**
- When: Deficit and storage insufficient
- Cost: $\text{Cost}_{\text{spot}} = \sum_t \max(0, \Delta(t)) \times S(t)$

**Sell to Spot:**
- When: Surplus and storage full
- Revenue: $\text{Revenue}_{\text{spot}} = \sum_t \max(0, -\Delta(t)) \times S(t)$

**Net Cost:**
\[
\text{Net Cost}_{\text{spot}} = \text{Cost}_{\text{buy}} - \text{Revenue}_{\text{sell}}
\]

---

## 7. Optimization Algorithm

### 7.1 Problem Characteristics

The optimization problem is:
- **High-dimensional:** 26+ decision variables (one per PPU type)
- **Non-convex:** Multiple local optima
- **Discrete:** Integer decision variables (PPU counts)
- **Computationally expensive:** Each evaluation requires full-year simulation

**Solution Approach:** Genetic Algorithm (GA) - population-based metaheuristic

### 7.2 Genetic Algorithm Framework

#### 7.2.1 Encoding

Each portfolio is encoded as an **integer vector**:

\[
\mathbf{x} = [x_1, x_2, \ldots, x_{|K|}]
\]

where $x_k \in [x_k^{\min}, x_k^{\max}]$ is the count of PPU type $k$.

#### 7.2.2 Fitness Function

**Primary Fitness:**
\[
f(\mathbf{x}) = (1 - \lambda) \cdot \bar{C}(\mathbf{x}) + \lambda \cdot \text{CVaR}_{95}(\mathbf{x})
\]

where:
- $\bar{C}(\mathbf{x})$ is mean cost across scenarios
- $\text{CVaR}_{95}(\mathbf{x})$ is Conditional Value-at-Risk (95th percentile)
- $\lambda = 0.3$ is the CVaR weight (tail risk vs mean cost trade-off)

**CVaR Definition:**
\[
\text{CVaR}_{95} = \mathbb{E}[C | C \geq \text{VaR}_{95}]
\]

where $\text{VaR}_{95}$ is the 95th percentile of cost distribution.

**Penalty for Constraint Violation:**
\[
f(\mathbf{x}) = 10^6 \times \max(0, E_{\text{target}} - E(\mathbf{x}))
\]

if energy sovereignty is violated.

#### 7.2.3 Population Initialization

**Random Initialization:**
\[
x_k \sim \text{Uniform}(x_k^{\min}, x_k^{\max}) \quad \forall k
\]

**Bias Toward Renewables:**
- 50% of population: Random
- 50% of population: Biased toward renewable-heavy portfolios

#### 7.2.4 Selection

**Tournament Selection:**
1. Randomly select $s = 3$ individuals
2. Choose individual with best (lowest) fitness
3. Repeat to fill population

**Elitism:**
- Top 10% of population survive unchanged
- Ensures best solutions aren't lost

#### 7.2.5 Crossover

**Uniform Crossover:**
For each PPU type $k$, randomly choose value from parent 1 or parent 2:

\[
x_k^{\text{child}} = \begin{cases}
x_k^{\text{parent1}} & \text{with probability } 0.5 \\
x_k^{\text{parent2}} & \text{with probability } 0.5
\end{cases}
\]

**Crossover Rate:** 0.8 (80% of offspring from crossover)

#### 7.2.6 Mutation

**Gaussian Mutation:**
\[
x_k^{\text{mutated}} = \max\left(x_k^{\min}, \min\left(x_k^{\max}, x_k + \delta\right)\right)
\]

where:
\[
\delta \sim \mathcal{N}(0, \sigma \cdot (x_k^{\max} - x_k^{\min}))
\]

and $\sigma = 0.1$ is the mutation strength.

**Mutation Rate:** 0.2 (20% chance per variable)

#### 7.2.7 Convergence Detection

**Plateau Detection:**
- Track best fitness over generations
- If no improvement for $G_{\text{plateau}}$ generations, stop
- Default: $G_{\text{plateau}} = 3-10$ depending on run mode

**Stopping Criteria:**
\[
\text{Stop if: } f_{\text{best}}(t) = f_{\text{best}}(t-G_{\text{plateau}})
\]

### 7.3 Multi-Objective Extension

#### 7.3.1 Objective Functions

To explore the efficiency frontier, we run GA with different objectives:

1. **Cost Minimization:**
   \[
   f_1(\mathbf{x}) = -\text{Return}(\mathbf{x})
   \]

2. **RoT Minimization:**
   \[
   f_2(\mathbf{x}) = \text{RoT}(\mathbf{x})
   \]

3. **Volatility Minimization:**
   \[
   f_3(\mathbf{x}) = \sigma_{\text{annual}}(\mathbf{x})
   \]

4. **Combined Objectives:**
   \[
   f_4(\mathbf{x}) = w_1 \cdot \text{RoT}(\mathbf{x}) + w_2 \cdot \frac{\sigma_{\text{annual}}(\mathbf{x})}{1000}
   \]

---

## 8. Efficiency Frontier Analysis

### 8.1 Pareto Optimality

After running GA with multiple objectives, we identify **Pareto-optimal portfolios**:

**Definition:** Portfolio $\mathbf{x}^*$ is Pareto-optimal if:
\[
\nexists \mathbf{x}' \text{ such that: } \begin{cases}
\text{RoT}(\mathbf{x}') \leq \text{RoT}(\mathbf{x}^*) \text{ AND } \sigma(\mathbf{x}') < \sigma(\mathbf{x}^*) \\
\text{OR} \\
\text{RoT}(\mathbf{x}') < \text{RoT}(\mathbf{x}^*) \text{ AND } \sigma(\mathbf{x}') \leq \sigma(\mathbf{x}^*)
\end{cases}
\]

**Mathematical Definition:**

For objectives $f_1(\mathbf{x})$ and $f_2(\mathbf{x})$ to minimize:

**Solution $\mathbf{x}^*$ is Pareto-optimal if:**
\[
\nexists \mathbf{x}' \text{ such that: } 
  (f_1(\mathbf{x}') \leq f_1(\mathbf{x}^*) \text{ AND } f_2(\mathbf{x}') < f_2(\mathbf{x}^*)) \text{ OR }
  (f_1(\mathbf{x}') < f_1(\mathbf{x}^*) \text{ AND } f_2(\mathbf{x}') \leq f_2(\mathbf{x}^*))
\]

**Pareto Frontier:**
\[
F = \{\mathbf{x}^* | \mathbf{x}^* \text{ is Pareto-optimal}\}
\]

### 8.2 Dominance Checking

**Algorithm:**
1. For each portfolio $\mathbf{x}_i$:
   - Check if any other portfolio $\mathbf{x}_j$ dominates it
   - If not dominated, add to Pareto frontier
2. Sort frontier by RoT for visualization

**Dominance Definition:**
Portfolio $\mathbf{x}_1$ dominates portfolio $\mathbf{x}_2$ if:
\[
(\text{RoT}(\mathbf{x}_1) \leq \text{RoT}(\mathbf{x}_2) \text{ AND } \sigma(\mathbf{x}_1) < \sigma(\mathbf{x}_2)) \text{ OR }
(\text{RoT}(\mathbf{x}_1) < \text{RoT}(\mathbf{x}_2) \text{ AND } \sigma(\mathbf{x}_1) \leq \sigma(\mathbf{x}_2))
\]

---

## 9. Constraints and Assumptions

### 9.1 Physical Constraints

#### 9.1.1 PPU Deployment Bounds

Each PPU type $k$ has minimum and maximum deployment limits:

\[
x_k^{\min} \leq x_k \leq x_k^{\max} \quad \forall k \in K
\]

**Examples:**
- PV: $x_{\text{PV}}^{\min} = 0$, $x_{\text{PV}}^{\max} = 1000$ (10 GW max)
- Wind Onshore: $x_{\text{WD\_ON}}^{\min} = 0$, $x_{\text{WD\_ON}}^{\max} = 1000$ (10 GW max)
- Hydro Storage: $x_{\text{HYD\_S}}^{\min} = 0$, $x_{\text{HYD\_S}}^{\max} = 300$ (3 GW max, limited by lake capacity)

#### 9.1.2 Resource Availability Limits

**Photovoltaic Area:**
- Maximum deployable area based on land availability
- Diminishing returns as best sites are used first

**Biomass Availability:**
- Limited by sustainable harvest rates
- Escalation factor increases costs beyond soft cap

**Hydropower Expansion:**
- Limited by suitable river sites
- Lake storage capacity: ~2 GW maximum

### 9.2 Energy Sovereignty Constraint

**Primary Constraint:**
\[
\sum_{k} E_k(\mathbf{x}) \geq E_{\text{target}}
\]

where:
- $E_k(\mathbf{x})$ is annual energy production from PPU $k$ (TWh)
- $E_{\text{target}} = 113$ TWh/year (Switzerland's 2050 target)

**Enforcement:**
- Portfolios failing this constraint receive penalty: $\text{Fitness} = 10^6 \times \text{deficit}$
- Ensures all evaluated portfolios meet sovereignty target

### 9.3 Cost Assumptions

**Currency:** All costs in 2024 CHF

**Technology Learning (2100):**
\[
C_{2100} = C_{2024} \times (1 - r)^{n}
\]

where:
- $r$ is learning rate (typically 0.10-0.20)
- $n$ is number of doublings in cumulative capacity

**Inflation:** Not explicitly modeled (all costs in 2024 CHF)

### 9.4 Grid Assumptions

- **Transmission:** Neglected (well-meshed grid assumed)
- **Distribution:** Uniform access to all resources
- **Imports/Exports:** Unlimited at spot prices
- **Grid Stability:** Not explicitly modeled

### 9.5 Storage Assumptions

- **Self-Discharge:** Modeled for chemical storage (hydrogen, batteries)
- **Efficiency Losses:** Round-trip efficiency 50-95% depending on technology
- **Power Limits:** Maximum charge/discharge power constraints (scaled by PPU count)
- **Capacity Limits:** Maximum storage capacity constraints (scaled by input PPU count, except Lake)
- **Scaling:** Storage capacity and power scale linearly with number of PPU units

### 9.6 Aviation Fuel Requirement (Biooil Constraint)

Aviation remains a material residual demand that is not straightforwardly substitutable by direct electricity use. Based on Züttel et al., an aviation-fuel requirement of approximately **23 TWh/year** is specified as a hard constraint.

**Mathematical Formulation:**
\[
\text{Aviation Fuel Demand} = 23 \text{ TWh/year} = \frac{23 \times 10^6 \text{ MWh}}{8760 \text{ h}} \approx 2625.57 \text{ MWh/hour}
\]

**Implementation:**
1. **Hourly Discharge:** Every hour, 2625.57 MWh of biooil must be discharged from storage for aviation
2. **Storage Type:** Biooil storage (discharge efficiency = 1.0 for fuel delivery, not electricity conversion)
3. **Supply Sources:** Biooil can be:
   - Imported at market price (67 CHF/MWh)
   - Produced domestically via pyrolysis pathways (BIO_OIL_FROM_WOOD, BIO_OIL_FROM_PALM)

**Capacity Constraint:**
\[
N_{\text{BIO\_OIL\_ICE}} \times P_{\text{unit}} \geq \frac{23 \times 10^6}{8760} \approx 2625.57 \text{ MW}
\]

With $P_{\text{unit}} = 10$ MW per PPU unit:
\[
N_{\text{BIO\_OIL\_ICE}} \geq \lceil 2625.57 / 10 \rceil = 263 \text{ units}
\]

**Validation:**
- Post-simulation validation checks that every hour met the discharge requirement
- Portfolios failing this constraint are rejected as infeasible
- Aviation fuel import costs are tracked separately and included in total system cost

**Note:** This discharge is **independent of electricity generation**. The biooil goes directly to aviation as fuel, not through BIO_OIL_ICE for power generation. The constraint ensures sufficient storage throughput capacity exists.

---

## 10. Summary of Key Equations

### 10.1 Portfolio Definition
\[
\mathbf{x} = \{x_1, x_2, \ldots, x_{|K|}\}, \quad x_k \in [x_k^{\min}, x_k^{\max}]
\]

### 10.2 Energy Sovereignty
\[
\sum_{k} E_k(\mathbf{x}) \geq 113 \text{ TWh/year}
\]

### 10.3 Component Chain Efficiency
\[
\eta_{\text{PPU}} = \prod_{i=1}^{n} \eta_i
\]

### 10.4 LCOE Calculation
\[
\text{LCOE}_k = \frac{\sum_{i \in \text{components}} \left( \text{CAPEX}_i + \text{OPEX}_i \cdot \text{lifetime}_i \right)}{\eta_{\text{PPU}} \cdot \text{capacity\_factor}_k \cdot \text{lifetime}_{\text{PPU}}}
\]

### 10.5 Progressive Cost Escalation
\[
\text{Cost}(x_k) = \text{LCOE}_k \cdot \left( 1 + \alpha_k \cdot \max(0, x_k - \text{soft\_cap}_k) \right)
\]

### 10.6 Storage Capacity Scaling
\[
C_m = C_m^{\text{base}} \times \max(1, N_m^{\text{input}})
\]
\[
P_m^{\text{charge}} = N_m^{\text{input}} \times P_{\text{unit}}, \quad P_m^{\text{discharge}} = N_m^{\text{extract}} \times P_{\text{unit}}
\]

### 10.7 Storage State Evolution
\[
\text{SOC}_{m}(t+1) = \text{SOC}_{m}(t) \times (1 - \delta_m) + \text{Charge}_m(t) \times \eta_{\text{charge}} - \frac{\text{Discharge}_m(t)}{\eta_{\text{discharge}}}
\]

### 10.8 Disposition Index
\[
\text{DI}_m(t) = \frac{\text{SOC}_m(t) - \text{SOC}_{\text{target}}}{\text{SOC}_{\text{max}} - \text{SOC}_{\text{target}}}
\]

### 10.9 Risk of Technology
\[
\text{PRR}_r = \sqrt{\frac{1}{N_r}}, \quad \text{Component Risk}_c = \left( \sum_{i} \text{PRR}_{r_i}^p \right)^{1/p}
\]
\[
\text{PPU RoT}_k = \frac{1}{m} \sum_{i} \text{Component Risk}_{c_i}, \quad \text{Portfolio RoT} = \sum_{k} \text{PPU RoT}_k \cdot \frac{E_k}{E_{\text{total}}}
\]

### 10.10 Cost Volatility
\[
\sigma_{\text{week},w} = \sqrt{\frac{1}{n_w-1} \sum_{t \in w} (C_t - \bar{C}_w)^2}, \quad \sigma_{\text{annual}} = \frac{1}{W} \sum_{w=1}^{W} \sigma_{\text{week},w}
\]

### 10.11 Aviation Fuel Constraint
\[
\forall t: \text{Biooil\_Discharge}(t) \geq 2625.57 \text{ MWh}, \quad N_{\text{BIO\_OIL\_ICE}} \geq 263
\]

### 10.12 Economic Return
\[
R_w = \frac{\text{Revenue}_{\text{spot}} - \text{Cost}_{\text{production}}}{\text{Revenue}_{\text{spot}}} \times 100\%, \quad R_{\text{annual}} = \frac{1}{W} \sum_{w=1}^{W} R_w
\]

### 10.12 Hourly Production Cost (LCOE)
\[
C_t = \frac{\sum_{k} P_{k,t} \cdot \text{Cost}_k}{\sum_{k} P_{k,t}}
\]

### 10.13 Fitness Function
\[
f(\mathbf{x}) = (1 - \lambda) \cdot \bar{C}(\mathbf{x}) + \lambda \cdot \text{CVaR}_{95}(\mathbf{x}) + \text{Penalty}(\mathbf{x})
\]

### 10.14 Pareto Optimality
\[
\nexists \mathbf{x}' \text{ such that: } 
  (f_1(\mathbf{x}') \leq f_1(\mathbf{x}^*) \text{ AND } f_2(\mathbf{x}') < f_2(\mathbf{x}^*)) \text{ OR }
  (f_1(\mathbf{x}') < f_1(\mathbf{x}^*) \text{ AND } f_2(\mathbf{x}') \leq f_2(\mathbf{x}^*))
\]

---

## References

### Academic References

1. Züttel, A. et al. (2023). Cost analysis of Power Production Units for CO₂-neutral energy systems. *Laboratory of Materials for Renewable Energy, EPFL.*

2. Swiss Federal Office of Energy (BFE). Hydropower statistics and energy demand data (2024).

3. European energy market data (2024). Spot price data.

### Data Sources

- Solar and wind incidence: Weather/climate databases (2024, hourly resolution)
- Spot prices: European energy market (2024, hourly resolution)
- Demand: Swiss Federal Office of Energy (2024, 15-minute resolution, aggregated to hourly)
- Component costs: LMER research data (Züttel et al., 2023)
- Resource production: Country-level production statistics
- Resource dependencies: Natural resource supply chain mappings

---

**End of Mathematical Foundations Document**

