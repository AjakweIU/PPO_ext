# Phase 2B Scientific Alignment Report

## Scope and evidence

This pass aligns the observation contract and audits reward scaling and dynamic coupling. No PPO, SAC, greedy, MILP, or full experiments were implemented. Fresh probe data are in `results/raw/phase2_validation/phase2b_alignment_20261110_035946_787782/phase2b_alignment.json`.

The observation mapping is detailed in [OBSERVATION_SPACE_MAPPING.md](OBSERVATION_SPACE_MAPPING.md). The methodology and configuration now agree on the 29-feature ordering and the use of entries 26-28 for price/hour/day. The environment and both config range matrices report 29 entries. The Phase 1 and Phase 2 splits have no ID intersections; OOD IDs also remain disjoint.

## Observation mapping

The methodology's groups add up to 29: source pressure/temperature (6), sink pressure (2), compressor state (8), pipeline flow (5), supply/demand (5), and price/time (3). Compressor channels are now documented four per station: inlet pressure, outlet pressure, power, and status. The methodology/project configuration does not specify compressor station topology, individual design points, or the meaning of the four compressor action dimensions, so those dynamics remain unresolved.

The prior use of entries 25-28 for action echoes and episode progress was inconsistent with the authoritative state. They now represent sink 2 demand, energy price, hour, and day. Five segment-flow slots currently carry the simplified network's common delivered flow; exact segment connectivity is not provided, so this is not a hydraulic segment solution.

## Reward scaling on feasible episodes

The audit ran 100 in-distribution representative episodes of 168 steps each; all had valid reset and zero invalid steps. These statistics decompose the standalone `compute_reward` over observed environment throughput and a leakage proxy. They do **not** represent the environment's active reward: `CO2TransportEnv.step` currently returns `0.1 * delivered_flow` when valid and does not call `compute_reward`.

| Component | Raw mean | Normalization bounds | Normalized mean | Weighted mean | Mean absolute contribution |
|---|---:|---:|---:|---:|---:|
| Throughput | 3,242.5 tonnes/day | 0-5,200 tonnes/day | 0.6236 | +0.6236 | 22.08% |
| Energy | $0 per episode reported by environment | $0-1,411,200 per episode | 0.0000 | 0.0000 | 0.00% |
| Leakage | 192.32 tonnes/episode, derived proxy only | 0-728 tonnes/episode | 0.2642 | -2.6417 | 77.92% |
| Violation | 0.000 step fraction | 0-1 | 0.0000 | 0.0000 | 0.00% |

Weights were unchanged: throughput 1.0, energy 0.1, leakage 10.0, violation 5.0. Absolute percentages are the mean of per-episode shares, so they need not equal shares computed from the component means.

### Why leakage dominates

The underlying scenario leakage rates are sampled independently in the 0.1%-2% range; the environment does not apply or expose a leakage process. For this audit only, leakage mass was derived as episode delivery multiplied by the sampled leakage fraction. The resulting raw proxy averages 192.32 tonnes per episode. Against the capacity-derived 728-tonne upper bound, the normalized mean is 0.2642; multiplying by `w3=10` gives a penalty magnitude of 2.6417, over four times the mean normalized throughput reward of 0.6236. A diagnostic-only calculation using `w3=1` reduces leakage's mean absolute share from 77.92% to 28.31%; no weight was changed or optimized.

Thus the observed dominance is primarily the combination of `w3=10` and the selected leakage normalization ceiling, not evidence that the physical leakage fraction itself is 73%. Physical leakage is not yet a dynamically simulated environment quantity. There is also a separate bounds problem: `configs/reward.yaml` uses `[0,1]` for every component. Feeding the raw physical means into those bounds saturates positive throughput and leakage to 1 and gives leakage 90.91% of absolute weighted magnitude. Those bounds are dimensionally incompatible with raw tonnes/day and episode USD; training must not use that mixed-unit path without defining training-distribution scales.

### Energy and violation terms

- Energy does not currently influence the environment: the compressor-power observation entries remain zero, step info has no energy consumption, scenario energy prices are not passed into the environment, and the active step reward has no energy term. The standalone weighted energy term also uses `w2=0.1`; its measured share is 0% because the environment-reported power is zero. Its ability to influence a policy cannot be established until compressor power and price-dependent cost are coupled.
- Violation is zero in all feasible episodes because pressure state does not evolve and action projection clips source controls to [0,1]. Sink flow is constructed within sink capacity. The pressure detector itself does catch a forced 3 MPa state under the 4 MPa minimum. Therefore, zero observed violations reflect a static, projected operating path, not proof that violation risk is absent. Capacity constraints should be tested on unprojected/transition-generated states as dynamics are added.

## Dynamic coupling results

| Requested dependency | Audit result | Evidence / limitation |
|---|---|---|
| Source availability and capture action affect delivery | Partial: action affects flow; scenario availability does not | Source action probe changes delivery from 1,260 to 5,200 tonnes/day. Scenario capacities/outages are not injected into the environment. |
| Transport decision affects delivered CO2 | Yes, in the simplified flow calculation | Delivered flow responds to source capture fractions and is capped by aggregate sink capacity. |
| Capacity limits feasible throughput | Yes, for configured source and sink caps | Default delivery caps at 5,200 tonnes/day; halving source capacities reduces delivery to 3,150 tonnes/day. Flows are allocated proportionally to sink capacity; no route topology is modeled. |
| Increased compressor effort changes energy consumption | No | Compressor actions at 0.75 versus 1.25 give identical delivery/reward; compressor power remains 0 kW. Action bounds are enforced, but these actions are otherwise unused. |
| Equipment failure changes available control/transport capacity | No | Failure probabilities and outage fields exist in scenarios but are not consumed by `CO2TransportEnv`. |
| Energy price changes cost, not physical energy use | Not implemented | The environment has neither physical compressor energy use nor an energy-cost term. Scenario price changes are not applied to state or reward. |
| Leakage responds to transport conditions | No in the environment | Scenario leakage rate is sampled but unused by environment state/transition/reward. The audit proxy is proportional to delivered mass, but is post-hoc only. |

## Demand shortfall and nominal difficulty

The previous reported shortfall frequencies (about 60% normal/energy spike, 80% high-demand/low-supply, 70% equipment failure, and 90% OOD) were not physical infeasibility measurements. They assumed the generator's scalar `source_capacity` is replicated across three sources and scalar `sink_injection_rate` across two sinks. Under that same implicit per-node interpretation, a fresh 500-sample normal screen finds 64.2% of samples with requested demand above aggregate sampled supply.

The scenario generator does not document whether its scalar sink rate is per sink or network total, and it samples supply and demand independently. Consequently, the shortfall frequency is interpretation-dependent and the nominal distribution is not demonstrably calibrated. The authoritative fixed network itself has 6,300 tonnes/day aggregate source capacity versus 5,200 tonnes/day aggregate sink capacity, so its configured capacity screen is feasible with 1,100 tonnes/day margin. This points to an unresolved scenario-to-node mapping/calibration issue rather than an inherently undersized fixed network. The environment currently does not consume scenario demand or calculate its shortfall, so the prior percentages cannot be interpreted as measured environment behavior.

## Conclusions and unresolved work

1. Observation ordering is explicit and consistent across the methodology, base-network config, Phase 1 defaults, normalization matrix, and environment slots. Compressor action semantics and network topology remain unspecified.
2. The current environment couples source capture to simplified delivered throughput and enforces source/sink capacity bounds. It does not yet couple compressor controls, outages, price, energy, or leakage into transitions/reward.
3. Leakage dominance in the earlier ~73% diagnostic comes from normalized leakage multiplied by `w3=10`; the leakage fraction itself is modest, and the environment has no implemented leakage physics. The audit-specific bounds give 77.92% here; default `[0,1]` bounds would saturate physical-valued inputs and produce 90.91% leakage share.
4. Energy currently has no effect; the zero violation term reflects action projection and static pressure, although the pressure detector catches a deliberately injected below-minimum pressure.
5. Scenario shortfalls are high because random supply and demand are unbalanced under an undocumented scalar-to-node mapping. Keep this as an open scientific calibration issue; do not tune against test data.
6. The default reward weights remain unchanged. No test-set weight selection was performed.
