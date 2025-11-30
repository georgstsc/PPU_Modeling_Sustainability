# Code Review: `evaluate_portfolio_ga` Function

## Executive Summary

The `evaluate_portfolio_ga` function is well-structured overall but has **critical issues** with data immutability that could cause incorrect results when running multiple scenarios. There are also opportunities for improvement in error handling, performance, and code clarity.

---

## Critical Issues

### 1. **Data Mutability Violation** ⚠️ **CRITICAL**

**Problem:**
The function's docstring states:
> "This function NEVER mutates them; per-scenario copies are created internally."

However, the function **directly passes** `raw_energy_storage` and `raw_energy_incidence` to `run_complete_pipeline()`, which **does mutate** these objects by:
- Adding `'history'` keys
- Modifying `'current_value'` 
- Adding `'initial_current_value'` keys

**Impact:**
- Scenarios will interfere with each other
- History from previous scenarios accumulates
- `current_value` states leak between scenarios
- Results become non-reproducible and incorrect

**Location:** Lines 4847-4852 in `optimization_problem.ipynb`

**Fix Required:**
```python
# Before calling run_complete_pipeline, create deep copies:
import copy

for s_idx in range(n_scenarios):
    # Create deep copies for this scenario
    scenario_storage = copy.deepcopy(raw_energy_storage)
    scenario_incidence = copy.deepcopy(raw_energy_incidence)
    
    results = run_complete_pipeline(
        scenario,
        ppu_counts,
        scenario_storage,  # Use copies
        scenario_incidence,  # Use copies
        verbose=verbose
    )
```

---

### 2. **Inconsistent Error Handling**

**Problem:**
When a scenario fails (line 4868-4873), the function:
- Appends `1e9` as a cost (very high penalty)
- Appends `1.0` for HHI and spot_dep (maximum values)
- But **continues execution** without logging the actual error details

**Issues:**
- The error message uses `s_idx` but should use `s_idx + 1` for consistency
- No distinction between different error types
- Silent failures could mask systematic issues
- The penalty values (`1e9`, `1.0`) might not be appropriate for all metrics

**Recommendation:**
```python
except Exception as e:
    error_msg = f"Scenario {s_idx + 1} failed: {type(e).__name__}: {str(e)}"
    if verbose:
        print(f"\n  [ERROR] {error_msg}")
    else:
        print(f"✗", end=" ", flush=True)
    
    # Log full error for debugging
    import traceback
    if verbose:
        traceback.print_exc()
    
    # Use NaN or None to indicate failure, then filter later
    scenario_costs.append(np.nan)
    scenario_hhi.append(np.nan)
    scenario_cvar.append(np.nan)
    scenario_spot_dep.append(np.nan)
```

Then filter out NaN values before computing statistics.

---

## Design & Architecture Issues

### 3. **Missing Return Values**

**Problem:**
The function collects `scenario_cvar` but **never uses it** in the return dictionary. Only `cvar_95` (computed from `scenario_costs`) is returned.

**Location:** Lines 4834, 4866, 4881

**Question:**
- Is `scenario_cvar` meant to be per-scenario CVaR? If so, should you return `avg_cvar` or `std_cvar`?
- Or is it redundant since you compute `cvar_95` from the aggregated costs?

**Recommendation:**
Either:
1. Remove `scenario_cvar` if it's not needed
2. Or return additional statistics like `std_cvar` or `min_cvar`, `max_cvar`

---

### 4. **Hardcoded Scenario Parameters**

**Problem:**
Line 4840: `scenario = generate_random_scenario(annual_data=None, num_days=30, seed=seed)`

The `num_days=30` is hardcoded. This should be a parameter.

**Recommendation:**
```python
def evaluate_portfolio_ga(
    ppu_counts: Dict[str, int],
    raw_energy_storage: list[Dict],
    raw_energy_incidence: list[Dict],
    n_scenarios: int = 5,
    num_days: int = 30,  # Add this parameter
    seed_base: Optional[int] = None,
    verbose: bool = False,
    plot_results: bool = True
) -> Dict[str, float]:
```

---

### 5. **Directory Management Issue**

**Problem:**
Lines 4826-4830: The function deletes and recreates the entire `result_plots_dir` at the start if `plot_results=True`.

**Issues:**
- This happens **once** at the start, not per-scenario
- If `plot_results=False` during optimization but `True` for final evaluation, old plots remain
- No check if directory deletion fails

**Recommendation:**
```python
if plot_results:
    result_plots_dir = "data/result_plots/scenario_evolution"
    try:
        if os.path.exists(result_plots_dir):
            shutil.rmtree(result_plots_dir)
    except OSError as e:
        if verbose:
            print(f"  [WARNING] Could not delete {result_plots_dir}: {e}")
    os.makedirs(result_plots_dir, exist_ok=True)
```

---

## Code Quality Issues

### 6. **Type Hints Inconsistency**

**Problem:**
- Line 4807: `raw_energy_storage: list[Dict]` (lowercase `list`)
- Should be: `List[Dict[str, Any]]` for consistency with other type hints

**Recommendation:**
```python
from typing import Dict, Any, Optional, List

def evaluate_portfolio_ga(
    ppu_counts: Dict[str, int],
    raw_energy_storage: List[Dict[str, Any]],
    raw_energy_incidence: List[Dict[str, Any]],
    ...
) -> Dict[str, float]:
```

---

### 7. **Progress Output Logic**

**Problem:**
Lines 4842-4844 and 4861-4863: The progress output logic is inverted.

- When `verbose=False`, it prints progress
- When `verbose=True`, it doesn't print progress (relies on `run_complete_pipeline`)

This is counterintuitive. Typically, `verbose=True` should show more output.

**Recommendation:**
Consider renaming or clarifying:
- `verbose=False`: Minimal output (just checkmarks)
- `verbose=True`: Detailed output (full pipeline logs)

Or add a separate `show_progress` parameter.

---

### 8. **Missing Input Validation**

**Problem:**
No validation of:
- `n_scenarios > 0`
- `ppu_counts` is non-empty
- `raw_energy_storage` and `raw_energy_incidence` are valid lists

**Recommendation:**
```python
if n_scenarios <= 0:
    raise ValueError(f"n_scenarios must be > 0, got {n_scenarios}")
if not ppu_counts:
    raise ValueError("ppu_counts cannot be empty")
if not isinstance(raw_energy_storage, list) or not isinstance(raw_energy_incidence, list):
    raise TypeError("raw_energy_storage and raw_energy_incidence must be lists")
```

---

## Performance Considerations

### 9. **Potential Memory Issues**

**Problem:**
If `plot_results=True`, all scenario results are kept in memory (via `plot_scenario_evolution`). For many scenarios, this could be memory-intensive.

**Recommendation:**
- Consider plotting asynchronously or in batches
- Or add a `max_plots` parameter to limit plotting to first N scenarios

---

### 10. **Timing Information**

**Good:** The function tracks total time when `verbose=True` (line 4876-4878).

**Enhancement:** Consider also tracking:
- Per-scenario timing (min/max/median)
- Time spent in plotting vs. pipeline execution

---

## Documentation Issues

### 11. **Incomplete Docstring**

**Problem:**
The docstring is minimal. Missing:
- Description of return values
- Example usage
- Notes about scenario independence (currently incorrect)

**Recommendation:**
```python
"""
Evaluate a portfolio via full pipeline over multiple random scenarios.

The caller supplies immutable base dictionaries `raw_energy_storage` and
`raw_energy_incidence`. This function creates deep copies internally to ensure
scenario independence.

Parameters:
-----------
ppu_counts : Dict[str, int]
    Dictionary mapping PPU type names to counts
raw_energy_storage : List[Dict[str, Any]]
    List of storage dictionaries. Will be deep-copied per scenario.
raw_energy_incidence : List[Dict[str, Any]]
    List of incidence dictionaries. Will be deep-copied per scenario.
n_scenarios : int, default=5
    Number of random scenarios to evaluate
num_days : int, default=30
    Number of days per scenario
seed_base : Optional[int], default=None
    Base seed for scenario generation. Each scenario uses seed_base + scenario_index
verbose : bool, default=False
    If True, show detailed pipeline output. If False, show minimal progress.
plot_results : bool, default=True
    If True, generate and save plots. Set to False during optimization for performance.

Returns:
--------
Dict[str, float]
    Dictionary with keys:
    - 'mean_cost': Mean cost across all scenarios [CHF]
    - 'cvar_95': Conditional Value-at-Risk at 95% confidence [CHF]
    - 'avg_hhi': Average Herfindahl-Hirschman Index (concentration measure)
    - 'avg_spot_dep': Average spot market dependence (fraction)

Raises:
-------
ValueError
    If n_scenarios <= 0 or ppu_counts is empty

Notes:
------
- Each scenario uses independent deep copies of storage/incidence data
- Failed scenarios are logged and excluded from statistics
- Plotting is disabled during optimization (plot_results=False) for performance

Example:
--------
>>> results = evaluate_portfolio_ga(
...     ppu_counts={'PV': 10, 'HYD_R': 5},
...     raw_energy_storage=storage_list,
...     raw_energy_incidence=incidence_list,
...     n_scenarios=10,
...     seed_base=42
... )
>>> print(f"Mean cost: {results['mean_cost']:,.2f} CHF")
"""
```

---

## Positive Aspects ✅

1. **Good separation of concerns**: Function focuses on evaluation, delegates to `run_complete_pipeline`
2. **Performance optimization**: `plot_results` flag allows disabling expensive plotting during optimization
3. **Progress tracking**: Shows progress even when `verbose=False` (though logic could be clearer)
4. **Error resilience**: Continues execution even if individual scenarios fail
5. **Comprehensive metrics**: Returns multiple relevant metrics (cost, CVaR, HHI, spot dependence)
6. **Timing information**: Tracks execution time when verbose

---

## Recommendations Summary

### Must Fix (Critical):
1. ✅ **Fix data mutability** - Add deep copying of `raw_energy_storage` and `raw_energy_incidence`
2. ✅ **Improve error handling** - Better logging and NaN handling for failed scenarios

### Should Fix (Important):
3. ✅ **Add input validation**
4. ✅ **Make `num_days` a parameter**
5. ✅ **Fix type hints**
6. ✅ **Clarify progress output logic**
7. ✅ **Remove or use `scenario_cvar`**

### Nice to Have:
8. ✅ **Enhance docstring**
9. ✅ **Improve directory management**
10. ✅ **Add per-scenario timing stats**

---

## Testing Recommendations

1. **Test immutability**: Verify that input `raw_energy_storage` and `raw_energy_incidence` are unchanged after function call
2. **Test scenario independence**: Run same portfolio with same `seed_base` twice, verify identical results
3. **Test error handling**: Inject failures in `run_complete_pipeline`, verify graceful handling
4. **Test edge cases**: `n_scenarios=1`, `n_scenarios=0`, empty `ppu_counts`

---

## Related Code Issues

### `run_complete_pipeline` also needs attention:
- It mutates input parameters (adds `history`, modifies `current_value`)
- Should either:
  - Accept copies and mutate those, OR
  - Document that it mutates inputs and callers should copy

Consider refactoring to make immutability explicit:
```python
def run_complete_pipeline(..., copy_inputs: bool = True):
    if copy_inputs:
        raw_energy_storage = copy.deepcopy(raw_energy_storage)
        raw_energy_incidence = copy.deepcopy(raw_energy_incidence)
    # ... rest of function
```

