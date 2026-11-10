# Open Questions and Unresolved Scientific Ambiguities

This document records all missing, contradictory, or scientifically ambiguous parameters that must be resolved before implementation or final manuscript claims.

Status: unresolved by the current repository state. No values are invented here.

## 1. Primary formulation conflict

### Decision status: resolved at the planning stage

The project contains two materially different formulations:

- Older ICTC formulation: 6 sources, 4 sinks, 8 routes, 11 discrete quantity levels, 2,112 discrete actions
- Later AI-only formulation: 3 sources, 2 sinks, 5 pipeline segments, 2 compressor stations, 29D observation, 7D continuous action

The authoritative working configuration for this repository is the later AI-only formulation.

This resolution is recorded in [docs/CONFIG_DECISIONS.md](docs/CONFIG_DECISIONS.md). The older formulation remains a historical or comparative reference only and must not be used as the active project implementation target.

The remaining open issues below are not about the formulation choice itself, but about the exact parameterization and implementation details that still need to be specified.

## 2. Network and physical-parameter ambiguities

### Question 2: What is the exact physical topology for the final network?

The methodology states the final base config as 3 sources, 2 sinks, 5 segments, and 2 compressors, but it does not specify the exact node locations, line connectivity, source-to-sink routing, or whether injection limits are node-specific or segment-specific.

### Question 3: What are the exact capacity and demand distributions for each node?

The methodology provides ranges for source capacity and sink injection rate, but not the per-node values used in the final study or the assumptions for supply-demand balancing.

### Question 4: What are the exact pipeline segment geotechnical properties?

The methodology names length and diameter ranges, but it does not define the exact set of segment lengths, diameters, material roughness, elevation changes, or allowable flow ranges used by the final simulation.

### Question 5: What exact pressure and compressor operating constraints are enforced?

The methodology defines the equations and general ranges, but the final implementation must still specify: min/max pressure at sources and sinks, minimum compressor pressure ratio, compressor operating envelope, and failed-state handling.

## 3. Scenario-generation ambiguities

### Question 6: What are the exact scenario-generation rules beyond the broad ranges?

The methodology lists broad ranges but not the precise sampling process for each class, including:

- time-dependent demand curves,
- source intermittency schedule,
- exact outage timing rules,
- energy-price waveform generation,
- leakage time profiles,
- correlated failure events.

### Question 7: What constitutes a true out-of-distribution scenario?

The methodology lists examples of OOD conditions, but it does not specify the exact thresholds for training distribution vs OOD test distribution, or how the thresholds are validated.

### Question 8: Are equipment failures and energy-price spikes treated as isolated events or as correlated events?

The methodology states these as scenario classes, but the implementation must define whether failures are independent, partially correlated, or generated jointly with demand/price changes.

## 4. Observation and action-space ambiguities

### Question 9: How are the 29 state dimensions and 7 action dimensions exactly mapped to physical variables?

The methodology lists categories but not the exact ordering of features, units, clipping rules, normalization bounds, or missing-value conventions.

### Question 10: What is the exact action-space projection logic?

The methodology requires action clipping and restoration, but it does not specify whether actions are projected in raw physical units, normalized action units, or after each step with penalty-based repair.

### Question 11: What constitutes a physically feasible action for each compressor and source?

The project requires action feasibility checks, but the exact action bounds and correction logic remain unspecified.

## 5. Reward-function ambiguities

### Question 12: What exact normalization policy is used for each reward component?

The methodology states that reward components must be normalized explicitly and gives a generic min-max form, but it does not provide the actual min/max bounds for throughput, energy cost, leakage, and violation penalties across the training distribution.

### Question 13: How are reward components aggregated at the episode level?

The project requires per-component logging and total reward decomposition, but it does not specify whether decomposition is done by time step, per episode, or per rollout batch.

### Question 14: What is the exact definition of the throughput term Phi(s_t)?

The methodology names it as CO₂ throughput/delivery reward but does not define the precise mathematical expression used for the implementation, especially relative to sink demand satisfaction, delivery rate, and total mass delivered.

### Question 15: What is the exact leakage and violation penalty formulation?

The methodology contains general penalty terms but not the exact formulas, thresholds, or evaluation logic for operational violations, pressure violations, or failure-induced leakage.

## 6. Benchmark and baseline ambiguities

### Question 16: Which additional RL baseline is required, and under what action-space assumptions?

The instructions state:

- if actions remain continuous, prefer SAC;
- if actions are converted to discrete, a DQN-family method may be used.

The final project must document which action formulation is being used before implementation.

### Question 17: What are the exact objective functions for the MILP benchmark?

The methodology states that the MILP benchmark is used for source–sink allocation and network operation, but it does not specify the objective terms, constraints, or whether the MILP objective matches the PPO reward structure exactly.

### Question 18: What is the exact fixed heuristic and greedy policy formulation?

The project calls for a fixed rule-based heuristic and a greedy scheduler, but the actual policy logic and deterministic rules are not defined in the methodology.

## 7. Data, split, and statistical ambiguities

### Question 19: What exactly qualifies as in-distribution vs out-of-distribution for this project?

The methodology names examples but does not define the formal threshold rules for OOD classification. It also does not specify whether OOD conditions are evaluated with separate scenario generators or by overlaying stress parameters onto the base distribution.

### Question 20: Should all reported metrics be aggregated across seeds or selected based on a validation checkpoint?

The methodology requires validation to determine the retained PPO checkpoint, but it does not define whether final reporting uses the best validation seed, the average across seeds, or the median across runs.

### Question 21: What statistical tests and uncertainty metrics are expected for final tables?

The methodology requires robustness, but it does not specify the exact statistical procedure for confidence intervals, significance tests, or bootstrap/stratified sampling across seeds.

## 8. Reproducibility and documentation ambiguities

### Question 22: What exact artifacts are required for each experiment?

The methodology lists logging requirements, but it does not specify the final artifact set for each run: raw metrics, normalized traces, config snapshots, network state logs, seeds per run, and checkpoint retention policy.

### Question 23: What is the final documentation requirement for configuration decisions?

The project states that docs/CONFIG_DECISIONS.md must record decisions if the code/data support a different configuration, but the repository currently does not contain that file or any confirmed decision record.

## 9. Remaining implementation gate items

The following items must be explicitly resolved before the project moves from planning to implementation:

1. final network configuration selection,
2. exact multi-scenario generation logic,
3. exact reward-normalization bounds,
4. final action-space formulation and RL benchmark choice,
5. exact MILP objective and matching constraints,
6. exact OOD definition and thresholds,
7. exact reproducibility and artifact requirements.

Until these are answered or deferred with a documented decision, the project should remain in planning mode and not begin code implementation.
