# Phase 2 Corrective Validation Report

Generated: 2026-11-09T05:32:29.492332+00:00
New results: `results\raw\phase2_validation\phase2_corrective_validation_20261109_053226_550328`

## State mapping and reset correction
- Methodology Section 7.1 defines indices 0-5 as source pressure/temperature channels, 6-7 as sink pressures, 8-15 as compressor-state channels, 16-20 as pipeline flow rates, 21-25 as supply/demand profiles, and 26-28 as energy price/hour/day.
- The methodology does not specify the individual meanings of compressor channels 8 and 9. They are not pressure channels and must not be checked against pressure limits.
- Reset now validates source pressure entries 0-2 and sink pressure entries 6-7. The existing nominal values are within configured 4-25 MPa limits; the minimum-pressure constraint remains unchanged.
- Reset feasibility: verified through the test suite and directly recorded in each episode result.

## Scenario-conditioned ranges
- Generated 100 in-distribution samples (20 per class) and 80 OOD samples (20 per non-normal class).
- Conditioned range violations: 0.
- Nominal base ranges are separated from scenario overrides. High-demand multiplier 1.5 and derived sink-rate range 1,500-15,000 are intentional in-distribution conditions; OOD high demand uses multiplier 2.0-2.8.
- OOD parameter excursions are checked against explicit OOD-conditioned ranges and are labeled OOD, not counted as methodology violations.
- Derived sink injection rates are demand targets. Exceeding available supply is recorded as demand shortfall, not as a physical flow-capacity violation; environment flows remain clipped by configured sink capacities.

## Split and OOD separation
- Split counts: {'train': 70, 'validation': 15, 'test': 15, 'ood': 20}.
- train/validation overlap: []; train/test: []; validation/test: []; OOD vs all in-distribution splits: [].
- All OOD IDs are marked OOD with split='ood': True.

## Physical feasibility
- Physical infeasibility here means a reset or environment step violating source capacity, sink capacity, or configured pressure constraints. Demand shortfall is reported separately.
- normal: 20 episodes; reset invalid 0; episodes with invalid steps 0; invalid-step rate 0.0000; demand-shortfall rate 0.600.
- high_demand: 20 episodes; reset invalid 0; episodes with invalid steps 0; invalid-step rate 0.0000; demand-shortfall rate 0.800.
- low_supply: 20 episodes; reset invalid 0; episodes with invalid steps 0; invalid-step rate 0.0000; demand-shortfall rate 0.800.
- equipment_failure: 20 episodes; reset invalid 0; episodes with invalid steps 0; invalid-step rate 0.0000; demand-shortfall rate 0.700.
- energy_price_spike: 20 episodes; reset invalid 0; episodes with invalid steps 0; invalid-step rate 0.0000; demand-shortfall rate 0.600.
- OOD: 80 episodes; reset invalid 0; episodes with invalid steps 0; invalid-step rate 0.0000; demand-shortfall rate 0.900.
- Any physical invalidity reason is retained per episode in `physical_episode_details` and `ood_episode_details` in the JSON output.
- The current Phase 1 transition does not apply sampled outages, compressor efficiency, pressure dynamics, pipeline hydraulics, or scenario demand as dynamic constraints; zero physical invalidity is therefore a check of the implemented environment contract, not proof of full-network physical feasibility.

## Reward analysis on feasible episodes
- Reward components were recomputed for 100 scenario episodes only after reset and episode validity checks.
- Raw and normalized metrics use explicit audit bounds derived from scenario maxima; weights remain unchanged (throughput 1.0, energy 0.1, leakage 10.0, violation 5.0). The energy metric is a transparent compressor-power/price proxy because the environment does not currently return physical compressor energy use.
- throughput: mean absolute-reward share 20.84% (range 9.07-35.40%); negligible in all episodes (<1%): False; dominant in all (>50%): False.
- energy: mean absolute-reward share 6.26% (range 0.62-25.93%); negligible in all episodes (<1%): False; dominant in all (>50%): False.
- leakage: mean absolute-reward share 72.89% (range 44.62-89.13%); negligible in all episodes (<1%): False; dominant in all (>50%): False.
- violation: mean absolute-reward share 0.00% (range 0.00-0.00%); negligible in all episodes (<1%): True; dominant in all (>50%): False.
- Per-episode records contain raw values, normalization bounds, normalized values, weighted contributions, total reward, and percentage share of total absolute component magnitude.

## Comparison with previous validation
- Previous report: `docs/PHASE2_VALIDATION_REPORT.md`; previous result bundle: `results/raw/phase2_validation/phase2_validation_20261109_051641_271639`.
- Previously reported representative physical invalid rate: not recorded and max-action stress invalid rate: 1.0.
- The previous 100% invalid finding was caused by checking compressor-state entries 8 and 9 as pressures. After correcting that mapping, this pass reports reset and episode feasibility using the actual pressure channels; this is a code-path correction, not a relaxation of the pressure floor.
- The previous range audit applied nominal ranges indiscriminately and counted intentional high-demand values as violations. This pass uses class-specific ranges and keeps OOD excursions separate.
- The previous reward statistics used synthetic component inputs. This pass derives throughput, leakage, violation frequency, and a documented energy proxy from feasible environment episodes and records complete component detail.

## Unresolved concerns
- Individual physical definitions for compressor observation channels 8-15 remain unspecified by the methodology.
- The Phase 1 environment remains a simplified feasibility checker: sampled scenario pressure, outage, compressor, leakage, and demand dynamics are not fully coupled into state transitions.
- Reward bounds in `configs/reward.yaml` remain generic [0,1]; this audit uses explicit physical audit scales, but the project should define training-distribution normalization bounds before training or sensitivity experiments.
- Demand-shortfall rates are screening indicators under the current scalar-per-source/sink expansion, not a network-flow optimization result.
