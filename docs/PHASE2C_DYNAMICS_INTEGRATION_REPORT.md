# Phase 2C Environment-Reward-Dynamics Integration Report

## Scope and outputs

Phase 2C integrates the existing reward function, compressor and pipeline physics, scenario parameters, topology flows, failures, and constraint penalties. No PPO, SAC, greedy, MILP, or benchmark experiments were implemented.

- Raw deterministic audit: `results/raw/phase2_validation/phase2c_dynamics_20261110_043021_785163/phase2c_validation.json`
- Topology assumption: [NETWORK_TOPOLOGY.md](NETWORK_TOPOLOGY.md)
- Observation contract: [OBSERVATION_SPACE_MAPPING.md](OBSERVATION_SPACE_MAPPING.md)

The Phase 1, Phase 2, and Phase 2C suites passed: `27 passed`.

## Integrated reward

`CO2TransportEnv.step()` now calls `compute_reward()` with unchanged weights: throughput 1.0, energy 0.1, leakage 10.0, violation 5.0. Reward is exactly the sum of weighted contributions. The `info` record exposes raw and normalized throughput/energy/leakage, violation, per-component weighted terms, total/final reward, energy use/cost, segment flows, shortfalls, and failure state.

Across 2,400 deterministic one-hour transitions (20 episodes per scenario class), aggregate component means were:

| Component | Raw mean per step | Normalization bounds | Normalized mean | Weighted mean | Absolute contribution share |
|---|---:|---:|---:|---:|---:|
| Throughput | 1,365.57 tonnes/day | 0-216.67 tonnes/timestep | 0.26261 | +0.26261 | 10.73% |
| Energy | $26.17/timestep | $0-8,400/timestep | 0.003115 | -0.000312 | 0.013% |
| Leakage | 0.2054 tonnes/timestep | 0-4.333 tonnes/timestep | 0.04740 | -0.47400 | 19.36% |
| Violation | 0.3423 fraction | 0-1 | 0.34232 | -1.71159 | 69.90% |

Absolute contribution shares sum magnitudes over the audit, not signed reward. Energy is currently negligible under the selected physical envelope and unchanged weight; do not treat it as a meaningful policy signal without later normalization/sensitivity review. Violation is large because its defined value includes demand shortfall, capture shortfall, action correction, pressure violations, and pressure-ratio clipping. It is not a count of physical invalid states alone.

Scenario means over 20 episodes (480 transitions per class):

| Scenario | Mean delivery (tonnes/day) | Mean demand shortfall | Physical invalid steps | Mean violation value |
|---|---:|---:|---:|---:|
| Normal | 1,553.38 | 53.1% | 0 | 0.3281 |
| High demand | 1,716.34 | 75.5% | 0 | 0.3558 |
| Low supply | 1,046.71 | 68.2% | 0 | 0.3467 |
| Equipment failure | 1,139.44 | 70.6% | 0 | 0.3456 |
| Energy-price spike | 1,372.00 | 60.9% | 0 | 0.3354 |

## Compressor and energy coupling

The existing Phase 1 polytropic compressor power function is used with CO2 properties, flow, inlet temperature, efficiency versus design flow, and pressure ratio. The paired deterministic probe changed pressure ratios from `[1.12, 1.12]` to `[1.487, 1.250]`; compressor powers rose from approximately `[33.40, 48.41]` kW to `[201.46, 119.85]` kW. Delivered flow rose from 2,469.77 to 3,557.65 tonnes/day because pressure-ratio requests and throttles affect permitted route capacity.

A price-only paired probe held all physical parameters and actions fixed: energy use remained 87.898 kWh while energy cost rose from $14.06 to $21.97 per timestep. Thus price changes monetary cost, not the physical energy calculation.

## Topology, flows, and source/sink dynamics

The five configured segments follow the explicit simplified radial topology in `docs/NETWORK_TOPOLOGY.md`: three source feeders to two compressor stations and two station discharge branches to sinks. Compressor 1 serves Sources 1-2/Sink 1; Compressor 2 serves Source 3/Sink 2. Segment channels are separately populated. Their audit mean rates were approximately `[314.40, 470.53, 585.74, 783.49, 584.71]` tonnes/day, confirming they are not duplicated common-flow values.

Source capture controls withdrawal. Sink delivery is limited by source availability, compressor design-flow throttle, segment/compressor availability, sink capacity, and sink demand. Demand and capture shortfalls are reported in tonnes and fractions. Configured tonnes/day capacities are rates, not finite daily inventories; each step reapplies the rate limits and reports post-dispatch available rate headroom. High-demand and low-supply parameters now change demand/available capacity used by transitions.

## Failure and leakage behavior

Source, compressor, and pipeline failures are sampled at reset from scenario parameters and remain fixed for the episode. Failed sources have zero available rate; failed compressors or relevant segments have zero route capacity. In the forced-failure probe, healthy delivery was 4,116.28 tonnes/day, while simultaneous compressor failure and all-segment failure each reduced delivery to zero. Failure IDs are returned in reset and step info.

Leakage is a transparent simulated loss, not a physical leakage prediction. Each segment loses a fraction of its transported flow equal to the scenario leakage rate apportioned by segment length. Leakage therefore increases with transported flow and scenario leakage rate, is deducted before downstream delivery, and is used in the reward.

## Feasibility and shortfall

All sampled scenario resets were feasible; the 2,400 audit steps had zero physical invalid states after the pressure-limit roundoff correction. Scenario mean demand-shortfall fractions were:

- Normal: 53.1%
- High demand: 75.5%
- Low supply: 68.2%
- Equipment failure: 70.6%
- Energy-price spike: 60.9%

These are unmet targets, not physical capacity violations. They activate the operational shortfall component of the violation penalty. The generator samples network-total source capacity over 500-5,000 tonnes/day and network-total sink demand over 1,000-10,000 tonnes/day (in-distribution, with scenario perturbations), independently. Since demand may exceed supply by construction, high shortfall is expected and should be treated as a scenario-calibration decision, not a failure of physical constraint enforcement. The fixed nominal config still has 6,300 tonnes/day source capacity and 5,200 tonnes/day sink capacity.

A deliberately above-range source action is projected and logged and yields a nonzero correction/violation penalty. A deliberately below-minimum source pressure is detected and penalized. Pressure bounds remain 4-25 MPa; only a tiny floating-point comparison tolerance is used at the configured boundary.

## Assumptions and limitations

- Network topology and compressor action pairing are explicit integration assumptions because the methodology supplies dimensions but no connectivity or per-action semantics.
- Pipeline pressure losses use the Phase 1 Darcy-Weisbach implementation, configured lengths/diameters, and Peng-Robinson CO2 density. Elevation changes remain zero because values are unspecified.
- The current model is a deterministic simplified radial simulator, not a transient linepack, compressor-map, or calibrated asset model. Results must not be presented as plant predictions.
- Reward weights are unchanged. Energy's contribution is negligible under current envelope/weight; violation share is large because demand/capture shortfall are included. Any retuning belongs in a later sensitivity study, not this integration pass.
