# Phase 2 Implementation Report

## 1. Files created or updated

- src/scenarios/__init__.py
- src/scenarios/domain_randomization.py
- src/scenarios/generator.py
- src/scenarios/splits.py
- src/environment/reward.py
- configs/scenarios.yaml
- configs/reward.yaml
- configs/experiments.yaml
- configs/base_network.yaml
- tests/test_phase2.py

## 2. Phase 2 scope implemented

### 2.1 Scenario classes

Implemented the required five scenario families:

- Normal
- High demand
- Low supply
- Equipment failure
- Energy-price spike

The scenario generator encodes separate perturbation logic for each class and preserves deterministic behavior by seed.

### 2.2 Methodology-defined domain randomization

Implemented the methodology's randomization ranges for:

- source capacity: 500–5,000
- sink injection rate: 1,000–10,000
- pipeline length: 10–100 km
- pipeline diameter: 0.3–0.8 m
- compressor power: 3,000–8,000 kW
- compressor efficiency: 0.75–0.88
- base energy price: $0.05–$0.12/kWh
- peak price multiplier: 1.5–3.0×
- simulated leakage rate: 0.1–2.0%
- failure probability: 0–5%

### 2.3 Fixed seeds and split enforcement

Implemented fixed split seeds:

- training: 42
- validation: 123
- test: 456

The split logic generates non-overlapping train/validation/test scenario IDs and rejects any overlap.

### 2.4 Held-out OOD generation

Implemented a separate OOD pathway that is explicitly distinct from ordinary test scenarios. OOD scenarios are generated in a separate function and include stronger perturbations such as higher demand multipliers, low supply, equipment outages, and extreme energy-price conditions.

### 2.5 Reward function

Implemented the reward specification:

$$
R = w_1 \Phi - w_2 C_{\mathrm{energy}} - w_3 L - w_4 V
$$

with the required weights:

- $w_1 = 1.0$
- $w_2 = 0.1$
- $w_3 = 10.0$
- $w_4 = 5.0$

The implementation normalizes each component explicitly and stores per-component values, contributions, and total reward in a reward breakdown object.

## 3. Assumptions and parameterization used

- Reward normalization uses min-max scaling over a bounded interval with default bounds of [0, 1] for throughput, energy, leakage, and violation. This is consistent with the methodology's explicit normalization requirement while keeping the design explicit and testable.
- Scenario perturbations are implemented as parameterized multipliers and outage probabilities rather than guessed physical constants.
- The final network configuration remains the later AI-only continuous formulation recorded in [docs/CONFIG_DECISIONS.md](CONFIG_DECISIONS.md).
- Historical ICTC discrete configuration remains a historical reference and is not used in the active implementation.

## 4. Unresolved scientific parameters

The following values remain unresolved and are not guessed:

- exact per-node source capacities and sink demand trajectories,
- exact pipeline elevation changes and segment connectivity,
- exact compressor design-point operating curves beyond the broad range,
- exact per-feature observation normalization bounds for all state dimensions beyond the working implementation defaults,
- exact MILP objective and matching constraints, which remain for a later phase.

These are documented in [docs/OPEN_QUESTIONS.md](OPEN_QUESTIONS.md) and are parameterized in the config files rather than invented.

## 5. Test status

Executed command:

- `f:/Downloads/PPO_ICTC_2026/.venv/Scripts/python.exe -m pytest tests/test_phase2.py -q`

Result:

- 6 passed
- 0 failed

### Test coverage included

- deterministic scenario reproduction by seed,
- train/validation/test non-overlap,
- scenario perturbation ranges,
- OOD generation,
- reward decomposition,
- reward-component normalization,
- total reward equals the sum of component contributions.

## 6. Scope boundary reminder

This is Phase 2 only. PPO, SAC, MILP, greedy scheduling, and experiment orchestration remain intentionally unimplemented in this phase.
