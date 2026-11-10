# Observation Space Mapping

The observation has 29 values, matching the methodology group sizes: 6 source pressure/temperature values, 2 sink pressures, 8 compressor values, 5 pipeline flows, 5 supply/demand values, and 3 time/price values.

| Index | Variable | Physical meaning | Unit | Source component | Valid physical range | Normalization range |
|---:|---|---|---|---|---:|---:|
| 0 | `source_1_pressure` | Source 1 pressure | Pa | Source 1 | 4.0e6-2.5e7 | 4.0e6-2.5e7 |
| 1 | `source_2_pressure` | Source 2 pressure | Pa | Source 2 | 4.0e6-2.5e7 | 4.0e6-2.5e7 |
| 2 | `source_3_pressure` | Source 3 pressure | Pa | Source 3 | 4.0e6-2.5e7 | 4.0e6-2.5e7 |
| 3 | `source_1_temperature` | Source 1 temperature | K | Source 1 | 250-350 | 250-350 |
| 4 | `source_2_temperature` | Source 2 temperature | K | Source 2 | 250-350 | 250-350 |
| 5 | `source_3_temperature` | Source 3 temperature | K | Source 3 | 250-350 | 250-350 |
| 6 | `sink_1_pressure` | Sink 1 pressure | Pa | Sink 1 | 4.0e6-2.5e7 | 4.0e6-2.5e7 |
| 7 | `sink_2_pressure` | Sink 2 pressure | Pa | Sink 2 | 4.0e6-2.5e7 | 4.0e6-2.5e7 |
| 8 | `compressor_1_inlet_pressure` | Compressor 1 inlet pressure | Pa | Compressor station 1 | 4.0e6-2.5e7 | 4.0e6-2.5e7 |
| 9 | `compressor_1_outlet_pressure` | Compressor 1 outlet pressure | Pa | Compressor station 1 | 4.0e6-2.5e7 | 4.0e6-2.5e7 |
| 10 | `compressor_1_power` | Compressor 1 electrical power | kW | Compressor station 1 | 0-8,000 | 0-8,000 |
| 11 | `compressor_1_status` | Compressor 1 operating-status fraction (0 off, 1 on) | fraction | Compressor station 1 | 0-1 | 0-1 |
| 12 | `compressor_2_inlet_pressure` | Compressor 2 inlet pressure | Pa | Compressor station 2 | 4.0e6-2.5e7 | 4.0e6-2.5e7 |
| 13 | `compressor_2_outlet_pressure` | Compressor 2 outlet pressure | Pa | Compressor station 2 | 4.0e6-2.5e7 | 4.0e6-2.5e7 |
| 14 | `compressor_2_power` | Compressor 2 electrical power | kW | Compressor station 2 | 0-8,000 | 0-8,000 |
| 15 | `compressor_2_status` | Compressor 2 operating-status fraction (0 off, 1 on) | fraction | Compressor station 2 | 0-1 | 0-1 |
| 16 | `pipeline_segment_1_flow` | Segment 1 transported CO2 flow | tonnes/day | Pipeline segment 1 | 0-5,200 | 0-5,200 |
| 17 | `pipeline_segment_2_flow` | Segment 2 transported CO2 flow | tonnes/day | Pipeline segment 2 | 0-5,200 | 0-5,200 |
| 18 | `pipeline_segment_3_flow` | Segment 3 transported CO2 flow | tonnes/day | Pipeline segment 3 | 0-5,200 | 0-5,200 |
| 19 | `pipeline_segment_4_flow` | Segment 4 transported CO2 flow | tonnes/day | Pipeline segment 4 | 0-5,200 | 0-5,200 |
| 20 | `pipeline_segment_5_flow` | Segment 5 transported CO2 flow | tonnes/day | Pipeline segment 5 | 0-5,200 | 0-5,200 |
| 21 | `source_1_available_capacity` | Source 1 available rate after outage availability is applied | tonnes/day | Source 1 | 0-5,000 | 0-5,000 |
| 22 | `source_2_available_capacity` | Source 2 available rate after outage availability is applied | tonnes/day | Source 2 | 0-5,000 | 0-5,000 |
| 23 | `source_3_available_capacity` | Source 3 available rate after outage availability is applied | tonnes/day | Source 3 | 0-5,000 | 0-5,000 |
| 24 | `sink_1_demand` | Sink 1 requested injection rate (target, not guaranteed flow) | tonnes/day | Sink 1 | 0-15,000 | 0-15,000 |
| 25 | `sink_2_demand` | Sink 2 requested injection rate (target, not guaranteed flow) | tonnes/day | Sink 2 | 0-15,000 | 0-15,000 |
| 26 | `energy_price` | Electricity price, including scenario peak multiplier | USD/kWh | Energy market | 0.05-0.525 | 0.05-0.525 |
| 27 | `hour_of_day` | Hour of day | hour | Simulation clock | 0-23 | 0-23 |
| 28 | `day_of_week` | Day of week, Monday=0 | day | Simulation clock | 0-6 | 0-6 |

## Compressor Channels 8-15

The methodology and project instructions define the compressor group as compressor inlet/outlet pressure, power, and status. This mapping allocates four values per station in that order, and the environment updates those values from each transition using the compressor model and documented topology. The station-to-segment topology and compressor-control action semantics are explicit integration assumptions documented in [NETWORK_TOPOLOGY.md](NETWORK_TOPOLOGY.md), not facts supplied by the methodology.

## Range and normalization notes

- Pressure limits come from `configs/base_network.yaml` (`min_pressure_pa`, `max_pressure_pa`). Temperatures use the existing observation-range configuration.
- Compressor power uses the configured compressor power ceiling; status is a unit fraction.
- The 5,200 tonnes/day nominal segment-flow ceiling is the sum of the configured sink capacities, 2,200 + 3,000 tonnes/day. Scenario-specific flow ceilings must be recalculated from the active capacities. Segment flow channels map to the explicitly documented topology in [NETWORK_TOPOLOGY.md](NETWORK_TOPOLOGY.md), and therefore can differ by segment.
- Source availability normalization uses the 5,000 tonnes/day upper edge of the methodology randomization range. The fixed nominal capacities are 1,500, 2,200, and 2,600 tonnes/day.
- Sink values are requested demand targets. The high-demand scenario can exceed configured physical sink injection capacities; that is unmet demand pressure, not permission to inject above capacity.
- Electricity price upper bound is the OOD envelope: 0.15 USD/kWh base price times 3.5 peak multiplier. In-distribution values remain within the methodology base and peak ranges.
- Values outside normalization bounds are clipped for the policy observation by the current normalizer. Such clipping must be logged and must not be interpreted as physical constraint handling.

## Alignment and unresolved details

This ordering reconciles the methodology group sizes and resolves the old use of indices 25-28 for source-action echoes and episode progress. Those are not part of the specified 29-feature state. The environment now uses 26-28 for energy price, hour, and day.

The methodology does not specify pipeline connectivity, individual compressor design points, exact sink demand profiles, or compressor action semantics. The selected topology/action interpretation is explicitly labeled as an integration assumption in [NETWORK_TOPOLOGY.md](NETWORK_TOPOLOGY.md). Scenario source capacities, sink-demand targets, outages, price, compressor efficiency, and leakage now enter environment transitions; the radial topology remains simplified and is not a calibrated network solution.
