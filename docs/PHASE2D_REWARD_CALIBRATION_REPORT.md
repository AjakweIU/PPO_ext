# Phase 2D Reward Calibration and Objective Validation

## Scope and outputs

This pass calibrates reward definitions and normalization only. No PPO, SAC, greedy, MILP, benchmark, or test-set weight selection was performed.

- Raw deterministic calibration: `results/raw/phase2_validation/phase2d_reward_calibration_20261110_045640_933971/phase2d_calibration.json`
- Candidate weights for later training: `configs/reward_sensitivity.yaml`
- Default weights remain throughput 1.0, energy 0.1, simulated leakage 10.0, physical violation 5.0.

Calibration used 100 deterministic 24-step episodes, 20 per scenario class. All 100 reset and transition sequences were physically feasible and included. Reward shares are computed per episode as the absolute component sum divided by that episode's total absolute component sum; table statistics are the mean and min/max across those episode shares.

## Service shortfall versus physical violations

Demand and capture shortfalls are not physical constraint violations. They are logged as service metrics and do not enter `V_t`; lower delivered CO2 already lowers throughput reward. The physical-violation term is:

\[
V_t=\operatorname{clip}_{[0,1]}\left(
\frac{A_t+P_t+R_t+C_t}{4}
\right),
\]

where:

- $A_t$: normalized action correction from projecting an out-of-range action;
- $P_t$: fraction of checked pressure channels outside configured pressure limits;
- $R_t$: fraction of compressor pressure-ratio requests clipped by physical pressure limits;
- $C_t$: maximum normalized source or sink flow capacity excess after dispatch.

Demand shortfall and capture/source shortfall remain separately reported as tonnes and fractions. This avoids applying an additional generic penalty for unavoidable unmet demand. The environment also reports each of the four physical violation components separately. On the feasible calibration trajectories all four averaged zero, while mean demand shortfall was 67.15% and mean capture shortfall was 3.52%. Thus service shortage no longer creates a physical-violation penalty. Deliberately unsafe actions and below-minimum pressures still activate their respective physical terms.

The default objective remains:

\[
R_t=1.0\tilde{\Phi}_t-0.1\tilde{C}_{energy,t}-10.0\tilde{L}_t-5.0V_t.
\]

## Normalization derivations

### Energy cost

Monetary cost uses the implemented physical compressor powers:

\[
C_{energy,t}=\sum_k W_{k,t}[\mathrm{kW}]\;\pi_{e,t}[\mathrm{USD/kWh}]\;\Delta t[\mathrm{h}],
\]

so the raw unit is USD per timestep. The lower bound is $0, attainable at zero flow or unity pressure ratio. The upper bound is derived from the compressor model at the configured maximum envelope: each station at 1.25 times design flow, minimum inlet pressure 4 MPa, maximum pressure ratio 2.2 (subject to 25 MPa outlet limit), maximum configured temperature 350 K, minimum compressor efficiency 0.75, and the configured compressor power ceiling. The summed maximum modeled power is 3,146.27 kW. The maximum configured OOD electricity price is $0.15/kWh times 3.5 = $0.525/kWh, and the timestep is 1 hour, giving a matching cost bound of $1,651.79/timestep. This is cost-to-cost normalization; no kW/kWh bound is used as a USD bound. The same default applies in `DEFAULT_REWARD_BOUNDS` and `configs/reward.yaml`; each environment derives its active bound using its configured compressor rating and timestep.

### Simulated leakage

Leakage remains explicitly a **simulated leakage** loss, not a physical leakage prediction. It is proportional to transported segment flow and the scenario leakage fraction. The maximum simulated transported quantity per timestep is the minimum of configured aggregate source capacity, aggregate sink capacity, and aggregate compressor design flow at the 1.25 throttle limit, multiplied by $\Delta t/24$. The normalization upper bound multiplies that maximum transported quantity by the configured maximum leakage fraction of 2%. For the nominal fixed network this yields 171.875 tonnes throughput and 3.4375 tonnes simulated leakage per one-hour timestep.

### Throughput

The throughput upper bound uses the same feasible hourly flow envelope: $\min(6300, 5200, 4125)/24=171.875$ tonnes per one-hour timestep. Scenario-specific environments use their active source/sink capacities and the configured compressor flow envelope.

## Default-weight calibration

| Component | Raw mean per timestep | Normalization range | Normalized mean | Weighted mean | Mean absolute share | Min episode share | Max episode share |
|---|---:|---:|---:|---:|---:|---:|---:|
| Throughput | 59.754 tonnes | 0-171.875 tonnes | 0.57819 | +0.57819 | 41.46% | 20.03% | 78.51% |
| Energy cost | $9.669 | $0-$1,651.79 | 0.005854 | -0.000585 | 0.044% | 0.0065% | 0.177% |
| Simulated leakage | 0.18697 tonnes | 0-3.4375 tonnes | 0.09444 | -0.94444 | 58.50% | 21.39% | 79.94% |
| Physical violation | 0.0000 fraction | 0-1 | 0.0000 | 0.0000 | 0.00% | 0.00% | 0.00% |

The apparent leakage share is not caused by a unit mismatch: its raw output is tonnes per timestep, the bound is tonnes per timestep, and the normalized value is dimensionless. It arises from the specified leakage weight of 10 applied to a normalized simulated loss (mean penalty magnitude 0.944), compared with a mean normalized throughput reward of 0.578. Leakage is a scientifically meaningful objective term, but this calibration shows that it often exceeds the throughput reward under the default weight; no weight adjustment is made here.

The energy bound is also unit-consistent, but the measured energy signal is negligible: $9.67 against a physical envelope of $1,651.79, then multiplied by $w_2=0.1$. Its mean absolute contribution is 0.044% (maximum episode share 0.177%). The energy term can technically vary with compressor action and price, but this default calibration does **not** support claiming materially energy-aware optimization. The weight was not altered to conceal this result. The supplied Energy-aware candidate is retained for later training/sensitivity evaluation; it is not selected or declared sufficient in this pass.

## Service shortfall results

Mean demand shortfall in the feasible deterministic episodes was 67.15% overall. By scenario: Normal 67.67%, High Demand 76.01%, Low Supply 70.30%, Equipment Failure 62.80%, Energy-Price Spike 58.98%. This is service performance, not physical infeasibility. The throughput term already rewards delivered quantity, and these shortages did not raise $V_t$. Capacity/source shortfall is still reported explicitly so future studies can assess service-level requirements separately.

## Candidate sensitivity configurations

`configs/reward_sensitivity.yaml` contains the exact requested candidates:

- Default: $(1.0, 0.1, 10.0, 5.0)$
- Energy-aware: $(1.0, 0.2, 10.0, 5.0)$
- Safety-oriented: $(1.0, 0.1, 10.0, 10.0)$
- Throughput-oriented: $(1.5, 0.1, 10.0, 5.0)$

They are unranked candidates for later training, not selected configurations. No test set was used for weight selection.

## Verification

- Calibration episodes: 100; physically feasible episodes included: 100; excluded: 0.
- Reward returned by `step()` equals the sum of its weighted component contributions (covered by Phase 2C tests).
- Service shortfall produces no physical violation penalty when pressure, ratio, capacity, and action-correction terms are zero; deliberate physical violations remain penalized (covered by tests).
- Full Phase 1/2/2C test run after the final changes: `27 passed`.
