# Simplified Network Topology for Phase 2C

The methodology defines three sources, two sinks, five pipeline segments, and two compressor stations, but does not provide node-to-segment connectivity. For Phase 2C integration, the environment uses the following explicit **simplified radial topology**. It is an engineering integration assumption for a deterministic environment, not a validated representation of a specific CO2 transport asset.

| Segment | From | To | Configured length / diameter index |
|---:|---|---|---|
| 0 | Source 1 | Compressor 1 inlet | 0 |
| 1 | Source 2 | Compressor 1 inlet | 1 |
| 2 | Source 3 | Compressor 2 inlet | 2 |
| 3 | Compressor 1 outlet | Sink 1 | 3 |
| 4 | Compressor 2 outlet | Sink 2 | 4 |

Compressor 1 receives Sources 1 and 2 and delivers to Sink 1. Compressor 2 receives Source 3 and delivers to Sink 2. Source network-total capacity and sink network-total demand from scenarios are distributed across their nodes proportional to the configured per-node capacities. Sink injection remains capped by configured sink capacity. Any excess target is recorded as capacity shortfall, not injected.

## Compressor action interpretation

The four compressor controls are paired by station. The first control requests a pressure ratio; action 0.75 maps to the configured minimum ratio and 1.25 maps to the configured maximum ratio. The second control multiplies the configured design-flow capacity and is clipped to the action limits. Requested ratios are further limited by the configured maximum pressure. This interpretation is explicit because the scientific formulation specifies four controls but does not define their individual meanings.

## Flows, losses, and failures

Each feeder flow follows the corresponding source's captured flow. Compressor discharge flow is the sum of its connected feeder flows after feeder leakage. Discharge segment flow follows that compressor's branch. Segment flows therefore need not be equal. Per-segment pressure loss uses the existing Darcy-Weisbach implementation, configured segment length and diameter, and Peng-Robinson CO2 density. Elevation changes remain zero because no configured elevations exist.

The configured scenario leakage fraction is applied to transported flow as a simple proportional loss weighted by each segment's share of total configured pipeline length. This is a transparent simulated loss model only; it is not a physical leakage prediction.

At reset, source and compressor outages use their scenario probabilities. Pipeline segments independently fail with probability `failure_probability / number_of_segments`, giving approximately the configured per-episode failure probability of at least one segment outage for small probabilities. A failed component has zero availability/capacity for that episode and its effect is recorded in `info`.

## Limitations

The topology does not model loops, cross-connects, transient linepack, geospatial elevation, valve dynamics, compressor maps, or empirically calibrated leak locations. Those require engineering data and should replace this assumption before claiming network-specific results. The configuration source is `configs/base_network.yaml`.
