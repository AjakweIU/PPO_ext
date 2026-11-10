# PROJECT_INSTRUCTIONS.md

## Project Title
**AI-Only CO₂ Transport Journal Extension**

## Purpose
This repository implements the journal extension of the CO₂ transport optimization study originally presented at ICTC 2026.

The journal version is **AI/optimization only**.

### Strict Scope Boundary
Do **not** implement, restore, import, or reuse any of the following in this repository:
- Purechain
- blockchain
- smart contracts
- carbon-credit minting
- batch-to-credit assurance
- ledger verification
- blockchain transaction latency
- blockchain transaction success rate
- blockchain provenance experiments
- validator/network-fault blockchain experiments

Those topics belong to a separate manuscript and are outside the scope of this project.

The authoritative scientific scope of this repository is:

1. Physics-informed CO₂ transport simulation
2. Scenario generation and domain randomization
3. MILP-based benchmark/infrastructure optimization
4. PPO-based adaptive operational scheduling
5. Strong baseline comparison
6. Generalization testing
7. Reward-function transparency and ablation
8. Scalability analysis
9. Computational/decision-time comparison
10. Statistical evaluation

---

# 1. Read Before Coding

Before creating or modifying implementation files:

1. Read the complete AI-only methodology document.
2. Inspect the existing repository, if any.
3. Produce an implementation map linking every methodology requirement to:
   - source file,
   - class/function,
   - configuration entry,
   - experiment,
   - expected output.
4. Identify contradictions, missing parameters, or unclear assumptions.
5. Do **not** invent missing scientific parameters.
6. Record unresolved issues in `docs/OPEN_QUESTIONS.md`.
7. Do not run large experiments until unit tests and small smoke tests pass.

The methodology is the scientific specification. The code must reflect it explicitly.

---

# 2. Reviewer-Driven Journal Requirements

The journal extension must directly address the ICTC reviewer concerns.

## 2.1 Stronger Baselines
The original ICTC paper compared PPO against only a fixed rule-based heuristic. The journal version must include stronger benchmarks.

Required comparison methods:

- Fixed rule-based heuristic
- Greedy scheduler
- MILP benchmark
- PPO
- One additional RL baseline appropriate for the final action-space formulation

### Alternative RL choice
If the final action space remains **continuous**, prefer:
- SAC

If the final action space is intentionally converted to **discrete**, a DQN-family method may be used.

Do not mix incompatible action-space assumptions without documenting the conversion.

---

## 2.2 Simulation Environment Transparency
The journal implementation must explicitly define and log:

- source capacities
- sink capacities/injection limits
- pipeline lengths
- pipeline diameters
- pressure constraints
- compressor limits
- compressor efficiencies
- energy-price ranges
- demand profiles
- supply profiles
- failure probabilities
- leakage assumptions
- temporal dynamics
- scenario-generation rules
- normalization bounds

Every experiment must be reproducible from configuration files.

---

## 2.3 Reward Function Transparency
The reward function must not be treated as a black box.

Base form:

\[
R(s_t,a_t)=
w_1\Phi(s_t)
-w_2C_{\mathrm{energy}}(s_t,a_t)
-w_3L(s_t)
-w_4V(s_t)
\]

where:

- \(\Phi(s_t)\): CO₂ throughput/delivery term
- \(C_{\mathrm{energy}}\): energy-cost term
- \(L(s_t)\): simulated leakage penalty
- \(V(s_t)\): operational/pressure violation penalty

Base weights:

- `w1 = 1.0`
- `w2 = 0.1`
- `w3 = 10.0`
- `w4 = 5.0`

The implementation must:

- normalize reward components explicitly,
- log each component separately,
- store total reward and component values per episode,
- support reward-weight sensitivity experiments,
- support ablation experiments.

---

## 2.4 Train/Validation/Test Separation
Use non-overlapping scenario sets.

Default split:

- Training: 70%
- Validation: 15%
- Test: 15%

Default reproducibility seeds:

- Training seed: `42`
- Validation seed: `123`
- Test seed: `456`

The journal must distinguish:

### In-distribution testing
New realizations sampled from the training distribution.

### Out-of-distribution testing
Conditions outside the training range or unseen structural conditions.

Examples:
- higher demand than training range,
- unseen energy-price profiles,
- source outage,
- compressor outage,
- altered sink capacity,
- altered network scale.

Do not report interpolation performance as generalization.

---

# 3. Authoritative Base Environment

Unless later corrected by the actual experiment code and documented in `docs/CONFIG_DECISIONS.md`, use the later AI methodology as the authoritative base configuration:

- 3 industrial CO₂ sources
- 2 geological storage sinks
- 5 pipeline segments
- 2 compressor stations
- 168 hourly time steps per episode
- 29-dimensional observation space
- 7-dimensional continuous action space

## Important Consistency Check
The older ICTC paper also contains a different configuration:

- 6 sources
- 4 sinks
- 8 routes
- 11 quantity levels
- 2,112 discrete actions

Do **not** silently combine these formulations.

Before final journal experiments, determine which configuration is supported by the current code and retained result files.

Document the final choice in:

`docs/CONFIG_DECISIONS.md`

and ensure all manuscript equations, tables, figures, and experiments use one consistent formulation.

---

# 4. Repository Structure

Use the following structure unless a technically superior organization is justified.

```text
co2-ai-journal/
│
├── PROJECT_INSTRUCTIONS.md
├── README.md
├── requirements.txt
├── pyproject.toml
│
├── methodology/
│   └── AI_Only_CO2_Transport_Journal_Updated_Methodology.md
│
├── configs/
│   ├── base_network.yaml
│   ├── ppo.yaml
│   ├── sac.yaml
│   ├── milp.yaml
│   ├── scenarios.yaml
│   ├── reward.yaml
│   ├── scalability.yaml
│   └── experiments.yaml
│
├── src/
│   ├── physics/
│   │   ├── thermodynamics.py
│   │   ├── pipeline.py
│   │   └── compressor.py
│   │
│   ├── environment/
│   │   ├── co2_transport_env.py
│   │   ├── constraints.py
│   │   ├── reward.py
│   │   └── normalization.py
│   │
│   ├── scenarios/
│   │   ├── generator.py
│   │   ├── domain_randomization.py
│   │   └── splits.py
│   │
│   ├── agents/
│   │   ├── ppo_agent.py
│   │   └── sac_agent.py
│   │
│   ├── baselines/
│   │   ├── fixed_policy.py
│   │   ├── greedy.py
│   │   └── milp.py
│   │
│   ├── evaluation/
│   │   ├── metrics.py
│   │   ├── statistics.py
│   │   ├── timing.py
│   │   ├── ablation.py
│   │   ├── generalization.py
│   │   └── scalability.py
│   │
│   └── utils/
│       ├── seeds.py
│       ├── logging.py
│       └── io.py
│
├── experiments/
│   ├── train_ppo.py
│   ├── train_sac.py
│   ├── run_baselines.py
│   ├── run_generalization.py
│   ├── run_ablation.py
│   ├── run_scalability.py
│   ├── run_reward_sensitivity.py
│   └── run_full_evaluation.py
│
├── tests/
│   ├── test_physics.py
│   ├── test_environment.py
│   ├── test_constraints.py
│   ├── test_reward.py
│   ├── test_scenarios.py
│   └── test_baselines.py
│
├── results/
│   ├── raw/
│   ├── processed/
│   ├── tables/
│   ├── figures/
│   └── checkpoints/
│
└── docs/
    ├── IMPLEMENTATION_MAP.md
    ├── OPEN_QUESTIONS.md
    ├── CONFIG_DECISIONS.md
    └── EXPERIMENT_LOG.md
```

---

# 5. Physics-Based CO₂ Transport Model

## 5.1 Thermodynamic Properties
Implement supercritical CO₂ property estimation using the methodology-defined equation of state.

Primary implementation should support the Peng-Robinson cubic EOS.

Density:

\[
\rho=\frac{PM}{ZRT}
\]

Use:
- CO₂ critical temperature: `304.13 K`
- critical pressure: `7.377 MPa`
- acentric factor: `0.225`
- molecular weight: `44.01 g/mol`

If Span-Wagner is used as a reference/high-fidelity comparison, clearly separate it from the computational model used during RL training.

---

## 5.2 Pipeline Flow
Implement Darcy-Weisbach pressure loss:

\[
\Delta P =
f\frac{L}{D}\frac{\rho v^2}{2}
+\rho g\Delta h
\]

Use Swamee-Jain for turbulent friction factor:

\[
f =
\frac{0.25}{
\left[
\log_{10}
\left(
\frac{\epsilon/D}{3.7}
+\frac{5.74}{Re^{0.9}}
\right)
\right]^2
}
\]

Default commercial-steel roughness:
- `epsilon = 0.046 mm`

Constraints must detect:
- insufficient pressure,
- capacity exceedance,
- infeasible flow,
- invalid source/sink allocations.

---

## 5.3 Compressor Model
Use the polytropic compression formulation defined in the methodology.

Compressor efficiency must vary with deviation from the design point.

Default efficiency range:
- `0.75–0.88`

Track:
- compressor power,
- energy consumed,
- pressure ratio,
- operating status,
- constraint violations.

---

# 6. Scenario Generation

Implement five base scenario classes:

1. Normal operation
2. High demand
3. Low supply
4. Equipment failure
5. Energy-price spike

## Default domain-randomization ranges

### Network/physical parameters
- Source capacity: `500–5,000 tonnes/day`
- Sink injection rate: `1,000–10,000 tonnes/day`
- Pipeline length: `10–100 km`
- Pipeline diameter: `0.3–0.8 m`
- Compressor power: `3,000–8,000 kW`
- Compressor efficiency: `0.75–0.88`

### Operational parameters
- Base energy price: `$0.05–0.12/kWh`
- Peak price multiplier: `1.5–3.0x`
- Simulated leakage rate: `0.1–2.0%`
- Failure probability: `0–5% per episode`

### Scenario perturbations
- High demand: increase demand beyond normal levels
- Low supply: reduce source availability
- Equipment failure: compressor or pipeline unavailability
- Energy spike: elevated time-dependent energy prices

Store all sampled parameters in experiment metadata.

---

# 7. MDP Formulation

## 7.1 Observation Space
Base observation vector:

`state_dim = 29`

Components:
- source pressure and temperature,
- sink pressure,
- compressor inlet/outlet pressure,
- compressor power/status,
- pipeline flows,
- supply/demand profiles,
- energy price,
- hour,
- day.

Every state component must have:
- documented unit,
- valid range,
- normalization rule.

---

## 7.2 Action Space
Base action vector:

`action_dim = 7`

Continuous actions:
- 3 source capture-rate adjustments,
- 4 compressor-control variables for two compressors.

Actions must be bounded and projected/clipped into physically feasible ranges.

Constraint handling must be explicit.

Do not silently replace invalid actions with arbitrary values.

Log:
- raw action,
- bounded action,
- whether correction occurred,
- violation reason.

---

# 8. PPO Implementation

Use PPO with actor-critic architecture.

## Network
- Input: 29
- Hidden layer 1: 128 + ReLU
- Hidden layer 2: 64 + ReLU
- Policy output: 7 + bounded activation
- Value output: 1

## Hyperparameters
- Learning rate: `3e-4`
- Gamma: `0.99`
- GAE lambda: `0.95`
- Clip epsilon: `0.2`
- Entropy coefficient: `0.01`
- Value coefficient: `0.5`
- Batch size: `64`
- Rollout length: `2048`
- Total timesteps: `100000`
- Episode length: `168`
- Epochs/update: `10`

Use validation performance for model selection.

Save:
- best checkpoint,
- final checkpoint,
- training history,
- validation history,
- random seeds,
- environment configuration.

---

# 9. Baseline Implementations

## 9.1 Fixed Rule-Based Baseline
Use the original simple baseline only as the weakest benchmark.

Default behavior from the methodology:
- compressor settings: 50% capacity,
- source capture rates: 70% of maximum,
- no dynamic response to energy prices,
- no adaptive demand response.

---

## 9.2 Greedy Scheduler
Implement a transparent greedy scheduler that makes locally optimal decisions based on current conditions.

Document its objective and tie-breaking logic.

Possible priority:
1. satisfy demand,
2. choose feasible lowest-energy route,
3. respect source/sink capacities,
4. avoid constraint violations.

---

## 9.3 MILP Benchmark
Implement a mathematical optimization benchmark using PuLP/CBC.

MILP should operate on the same network and constraints used by the RL environment.

For tractable instances, compare:
- objective quality,
- throughput,
- energy cost,
- violations,
- solution time.

Do not claim MILP is directly equivalent to PPO unless both optimize the same operational objective and constraints.

---

## 9.4 Alternative RL Baseline
Use SAC for continuous actions unless the final action formulation changes.

Use the same:
- environment,
- state variables,
- reward,
- train/validation/test splits,
- evaluation seeds.

Hyperparameter differences must be documented.

---

# 10. Evaluation Metrics

At minimum report:

## Operational
- Total CO₂ throughput
- CO₂ delivered to sinks
- Energy consumption
- Energy cost
- Energy efficiency
- Simulated leakage
- Pressure/constraint violations
- Feasible-action rate

## Learning
- Mean episode reward
- Reward standard deviation
- Convergence rate
- Training stability
- Generalization gap

## Computational
- Mean decision/inference time
- Median decision time
- P95 decision time
- Training time
- MILP solve time
- Model size / parameter count where useful

## Statistical
- Mean
- Standard deviation
- 95% confidence interval
- appropriate paired significance testing
- effect size where practical

Do not use reward alone as evidence of practical superiority.

---

# 11. Required Experiments

## Experiment A — PPO Convergence
Measure:
- episode reward,
- reward components,
- throughput,
- energy,
- violations,
- policy stability.

Clarify whether plotted throughput is:
- training throughput,
- evaluation throughput,
- or validation throughput.

Never label training progression as baseline improvement.

---

## Experiment B — Strong Baseline Comparison
Compare:
- Fixed heuristic
- Greedy
- MILP
- SAC
- PPO

Use identical test scenarios.

Primary output table:

| Method | Reward | Throughput | Energy Cost | Violations | Mean Decision Time |
|---|---:|---:|---:|---:|---:|

---

## Experiment C — Scenario Robustness
Evaluate all methods under:
- Normal
- High demand
- Low supply
- Equipment failure
- Energy spike

Report both reward and physical/operational metrics.

---

## Experiment D — OOD Generalization
Create unseen test conditions.

Examples:
- demand above training maximum,
- energy spike above training range,
- unseen source outage,
- unseen compressor outage,
- altered sink capacity.

Report degradation relative to in-distribution testing.

---

## Experiment E — Reward Sensitivity
Test multiple reward-weight settings.

At minimum:
- default,
- energy-focused,
- safety-focused,
- throughput-focused.

Do not select the best configuration using the test set.

---

## Experiment F — Ablation
Required PPO variants:
- Full PPO
- No energy penalty
- No leakage penalty
- No violation penalty
- No domain randomization

Measure:
- throughput,
- energy cost,
- violations,
- simulated leakage,
- reward.

---

## Experiment G — Decision-Time Comparison
Measure decision latency for:
- heuristic,
- greedy,
- MILP,
- SAC,
- PPO.

Report:
- mean,
- median,
- p95.

The practical journal argument should distinguish:
- offline training cost,
- online inference cost,
- repeated optimization cost.

---

## Experiment H — Scalability
Evaluate progressively larger networks.

Suggested levels:

| Scale | Sources | Sinks | Approx. Routes |
|---|---:|---:|---:|
| Base | 3 | 2 | 5 |
| Medium | 6 | 4 | 8 |
| Large | 12 | 8 | 20 |
| Very Large | 24 | 16 | 40 |

Record:
- reward,
- throughput,
- constraint violations,
- PPO inference time,
- convergence behavior,
- MILP solve time.

If some scales become computationally infeasible, report that explicitly.

---

# 12. Statistical Protocol

Use multiple seeds.

Recommended minimum:
- 5 independent seeds

Preferred:
- 10 independent seeds

For each reported comparison:
- store per-seed raw results,
- calculate mean ± standard deviation,
- calculate 95% CI,
- test paired differences where assumptions allow.

Do not report significance without preserving the underlying sample values.

---

# 13. Output Requirements

Every experiment must save machine-readable results.

Use:
- CSV for tabular results,
- JSON for configuration/metadata,
- model checkpoints for RL,
- PNG/PDF/SVG for publication figures where supported.

Each experiment directory should include:

```text
experiment_name/
├── config.json
├── environment.json
├── seeds.json
├── raw_results.csv
├── summary_metrics.json
├── statistics.json
├── figures/
└── logs/
```

Never hard-code manuscript numbers into plotting scripts.

Figures must be regenerated from saved raw data.

---

# 14. Reproducibility Rules

Every run must record:
- timestamp,
- git commit hash if available,
- Python version,
- package versions,
- seed,
- scenario ID,
- environment configuration,
- model configuration,
- hardware information.

Set deterministic seeds for:
- Python random,
- NumPy,
- PyTorch,
- environment generator.

Where GPU non-determinism remains, document it.

---

# 15. Acceptance Tests Before Full Experiments

Do not launch full training until these pass.

## Physics
- pressure drop increases with flow under fixed conditions,
- compressor power increases appropriately with pressure ratio,
- invalid physical states are rejected.

## Environment
- reset returns valid 29-D observation,
- step accepts valid 7-D action,
- episode terminates after 168 steps,
- constraints are logged correctly,
- reward components sum to total reward.

## Scenarios
- scenario seeds are reproducible,
- train/validation/test scenarios do not overlap,
- OOD scenarios are outside the intended training range.

## Baselines
- heuristic runs without learning,
- greedy decisions are deterministic given a state,
- MILP returns feasible solutions for small test cases.

## RL
- PPO can complete a smoke training run,
- checkpoint save/load preserves behavior,
- evaluation mode disables exploration where appropriate.

---

# 16. Scientific Guardrails

Do not make unsupported claims.

Specifically:

- "simulated leakage" is not physical leakage prevention.
- Synthetic/simulated results are not field deployment results.
- PPO training improvement is not automatically improvement over a baseline.
- Reward improvement is not equivalent to throughput improvement.
- A single successful seed is not evidence of robustness.
- MILP superiority/optimality claims require matched objectives and constraints.
- OOD generalization must use truly held-out conditions.
- Do not describe a stress test as production-scale validation.
- Do not report blockchain results in this repository.

---

# 17. First Task for the VS Code AI Agent

Do **not** begin by implementing the entire project.

First perform only this task:

> Read `PROJECT_INSTRUCTIONS.md` and the AI-only methodology completely. Inspect the repository. Do not modify code yet. Create `docs/IMPLEMENTATION_MAP.md` mapping every methodological requirement to the file/class/function that should implement it. Then create `docs/OPEN_QUESTIONS.md` listing every missing, contradictory, or scientifically ambiguous parameter. Pay particular attention to the conflict between the older 6-source/4-sink/8-route discrete formulation and the later 3-source/2-sink/5-pipeline/2-compressor continuous formulation. Do not invent values to resolve contradictions.

Stop after producing those two documents.

---

# 18. Recommended Implementation Order

After the implementation map is reviewed:

1. Physics module
2. Base transport environment
3. Constraint system
4. Scenario generator
5. Reward implementation
6. Fixed heuristic
7. Greedy baseline
8. MILP benchmark
9. PPO
10. Alternative RL baseline
11. Evaluation pipeline
12. Generalization experiments
13. Reward sensitivity
14. Ablation
15. Scalability
16. Statistical analysis
17. Publication tables and figures

Do not skip directly to final training.

---

# 19. Journal-Level Success Criteria

The project is ready for journal manuscript results only when:

- the environment is fully documented and reproducible,
- reward normalization and weights are transparent,
- PPO is compared against strong baselines,
- throughput is compared directly across methods,
- training/test scenario separation is explicit,
- OOD generalization is evaluated,
- decision-time/computational cost is measured,
- scalability is tested,
- ablation explains the contribution of major design components,
- multiple-seed statistical results are available,
- all final manuscript values can be traced to saved raw result files.

---

# End of Instructions
