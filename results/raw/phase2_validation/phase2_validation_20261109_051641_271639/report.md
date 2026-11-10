# Phase 2 Validation Report

## Validation outputs
- Results root: results\raw\phase2_validation\phase2_validation_20261109_051641_271639

## Scenario validity summary
- Total representative samples generated: 15
- Methodology range violations: 8
- high_demand / train_high_demand_10 parameter 'demand_multiplier' = 1.5 is outside [0.8, 1.2].
- high_demand / train_high_demand_21 parameter 'sink_injection_rate' = 10811.85347693716 is outside [1000.0, 10000.0].
- high_demand / train_high_demand_21 parameter 'demand_multiplier' = 1.5 is outside [0.8, 1.2].
- high_demand / train_high_demand_33 parameter 'sink_injection_rate' = 10035.14544031351 is outside [1000.0, 10000.0].
- high_demand / train_high_demand_33 parameter 'demand_multiplier' = 1.5 is outside [0.8, 1.2].

## Split-overlap check
- train/validation/test overlap: [] / [] / []
- OOD vs train/validation/test overlap: []

## OOD confirmation
- OOD scenarios were generated through a separate held-out path and remained disjoint from train/validation/test IDs.
- The OOD generator intentionally amplifies extreme conditions and is expected to trigger more invalid or unstable operating points than nominal scenarios.

## Reward-component statistics
- throughput: mean absolute share = 0.350; max share = 0.372.
- energy: mean absolute share = 0.007; max share = 0.007.
- leakage: mean absolute share = 0.506; max share = 0.515.
- violation: mean absolute share = 0.137; max share = 0.147.

## Infeasible-state frequency
- Representative feasibility probe invalid rate: 1.000.
- Max-action stress invalid rate: 1.000.
- Root cause identified: the Phase 1 environment validates a pressure subset that includes state entries [8] and [9], but those entries are initialized to 0.0. Since the environment minimum pressure is 4.0e6 Pa, the initial physical-state check fails before any policy can act.
- A secondary source of infeasibility is that OOD scenario randomization intentionally pushes demand and supply away from equilibrium, which means not every generated scenario is intended to be feasible.

## Weight imbalance / sensitivity note
- The default weights are throughput=1.0, energy=0.1, leakage=10.0, violation=5.0.
- In the representative episodes, leakage and violation penalties dominate the total reward magnitude after normalization, which means the objective is sensitive to these terms.
- This imbalance is reported for later sensitivity analysis and was intentionally not auto-adjusted in this validation pass.

## Unresolved concerns
- The Phase 1 environment validity check is not consistent with the initial state, so valid operating points cannot be assumed from reset alone.
- Scenario generation remains useful for stress testing, but the environment must be tightened if the project requires guaranteed-feasible states during training.
- Reward weights remain intentionally unchanged for future tuning work.
