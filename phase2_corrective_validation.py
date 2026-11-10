from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.environment.co2_transport_env import CO2TransportEnv
from src.environment.reward import compute_reward
from src.scenarios.domain_randomization import DEFAULT_DOMAIN_RANGES
from src.scenarios.generator import ScenarioGenerator
from src.scenarios.splits import generate_ood_scenarios, split_scenarios

SCENARIO_TYPES = [
    "normal",
    "high_demand",
    "low_supply",
    "equipment_failure",
    "energy_price_spike",
]
SEEDS = range(500)
EPISODE_STEPS = 168


def conditioned_ranges(scenario_type: str, is_ood: bool = False) -> dict[str, tuple[float, float]]:
    ranges = dict(DEFAULT_DOMAIN_RANGES)
    if is_ood:
        if scenario_type == "high_demand":
            ranges.update(demand_multiplier=(2.0, 2.8), sink_injection_rate=(2000.0, 15000.0))
        elif scenario_type == "low_supply":
            ranges.update(source_capacity=(175.0, 2750.0), supply_multiplier=(0.35, 0.55), source_outage_probability=(0.05, 0.05))
        elif scenario_type == "equipment_failure":
            ranges.update(source_capacity=(350.0, 3500.0), failure_probability=(0.05, 0.07), source_outage_probability=(0.06, 0.06), compressor_outage_probability=(0.08, 0.08), compressor_efficiency=(0.75, 0.80))
        elif scenario_type == "energy_price_spike":
            ranges.update(base_energy_price=(0.12, 0.15), peak_price_multiplier=(3.0, 3.5))
        return ranges

    if scenario_type == "normal":
        ranges.update(demand_multiplier=(1.0, 1.0), supply_multiplier=(1.0, 1.0), base_energy_price=(0.08, 0.08), peak_price_multiplier=(2.0, 2.0), leakage_rate=(0.005, 0.005), failure_probability=(0.01, 0.01))
    elif scenario_type == "high_demand":
        ranges.update(demand_multiplier=(1.5, 1.5), sink_injection_rate=(1500.0, 15000.0), supply_multiplier=(1.0, 1.0), failure_probability=(0.01, 0.01))
    elif scenario_type == "low_supply":
        ranges.update(source_capacity=(350.0, 3500.0), supply_multiplier=(0.7, 0.7), source_outage_probability=(0.02, 0.02), compressor_outage_probability=(0.0, 0.0), leakage_rate=(0.01, 0.01))
    elif scenario_type == "equipment_failure":
        ranges.update(source_capacity=(450.0, 4500.0), compressor_efficiency=(0.75, 0.80), failure_probability=(0.03, 0.03), source_outage_probability=(0.03, 0.03), compressor_outage_probability=(0.04, 0.04))
    elif scenario_type == "energy_price_spike":
        ranges.update(base_energy_price=(0.10, 0.10), peak_price_multiplier=(2.5, 2.5), leakage_rate=(0.006, 0.006))
    return ranges


def env_for(spec: object) -> CO2TransportEnv:
    params = spec.parameters
    return CO2TransportEnv(
        source_capacity=np.full(3, float(params["source_capacity"])),
        sink_capacity=np.full(2, float(params["sink_injection_rate"])),
    )


def spec_range_findings(spec: object, ranges: dict[str, tuple[float, float]]) -> list[dict[str, object]]:
    findings = []
    for key, (lower, upper) in ranges.items():
        if key not in spec.parameters:
            continue
        value = float(spec.parameters[key])
        if not lower - 1e-10 <= value <= upper + 1e-10:
            findings.append({"parameter": key, "value": value, "allowed": [lower, upper]})
    return findings


def episode_metrics(spec: object, seed: int) -> dict[str, object]:
    params = spec.parameters
    env = env_for(spec)
    _, reset_info = env.reset(seed=seed)
    valid_steps = 0
    invalid_reasons: dict[str, int] = {}
    throughput_rates = []
    compressor_activity = []
    for step in range(EPISODE_STEPS):
        source_fraction = [0.55 + 0.1 * (step % 3), 0.6, 0.5]
        compressor_controls = [1.0, 1.0, 1.0, 1.0]
        action = np.asarray(source_fraction + compressor_controls, dtype=float)
        _, _, _, _, info = env.step(action)
        if info["valid"]:
            valid_steps += 1
        for group, reasons in info["constraint_check"].items():
            for reason in reasons:
                invalid_reasons[reason] = invalid_reasons.get(reason, 0) + 1
        throughput_rates.append(float(np.sum(env.state[23:25])))
        compressor_activity.append(float(np.mean(compressor_controls)))

    mean_delivery_rate = float(np.mean(throughput_rates))
    total_delivery_tonnes = mean_delivery_rate * EPISODE_STEPS / 24.0
    leakage_rate = float(params["leakage_rate"])
    raw_leakage_tonnes = total_delivery_tonnes * leakage_rate
    unit_energy_price = float(params["base_energy_price"] * params["peak_price_multiplier"])
    compressor_power_kw = float(params["compressor_power_kw"])
    raw_energy_cost = float(sum(compressor_activity) * 2.0 * compressor_power_kw * unit_energy_price)
    raw_violation_fraction = float((EPISODE_STEPS - valid_steps) / EPISODE_STEPS)

    # Bounds are explicit audit scales derived from configured physical and OOD maxima.
    raw_values = {
        "throughput": mean_delivery_rate,
        "energy": raw_energy_cost,
        "leakage": raw_leakage_tonnes,
        "violation": raw_violation_fraction,
    }
    normalization_bounds = {
        "throughput": (0.0, 2.0 * 15000.0),
        "energy": (0.0, EPISODE_STEPS * 2.0 * 8000.0 * 1.25 * 0.15 * 3.5),
        "leakage": (0.0, 2.0 * 15000.0 * EPISODE_STEPS / 24.0 * 0.02),
        "violation": (0.0, 1.0),
    }
    reward_computed = bool(reset_info["valid"] and valid_steps == EPISODE_STEPS)
    reward = compute_reward(
        throughput=raw_values["throughput"],
        energy_cost=raw_values["energy"],
        leakage=raw_values["leakage"],
        violation=raw_values["violation"],
        bounds=normalization_bounds,
    ) if reward_computed else None
    step_contributions = reward.component_contributions if reward is not None else {}
    episode_contributions = {key: value * EPISODE_STEPS for key, value in step_contributions.items()}
    absolute_sum = sum(abs(v) for v in episode_contributions.values())
    shares = {
        key: (abs(value) / absolute_sum * 100.0 if absolute_sum else 0.0)
        for key, value in episode_contributions.items()
    }
    demand_rate = 2.0 * float(params["sink_injection_rate"])
    max_supply_rate = 3.0 * float(params["source_capacity"])
    return {
        "scenario_id": spec.scenario_id,
        "scenario_type": spec.scenario_type,
        "is_ood": spec.is_ood,
        "seed": spec.seed,
        "reset_valid": bool(reset_info["valid"]),
        "valid_steps": valid_steps,
        "invalid_steps": EPISODE_STEPS - valid_steps,
        "reward_computed": reward_computed,
        "invalid_reasons": invalid_reasons,
        "mean_delivery_rate_tonnes_per_day": mean_delivery_rate,
        "demand_rate_tonnes_per_day": demand_rate,
        "max_supply_rate_tonnes_per_day": max_supply_rate,
        "demand_shortfall_at_full_capacity": bool(max_supply_rate < demand_rate),
        "raw_values": raw_values,
        "normalization_bounds": normalization_bounds,
        "normalized_values": reward.normalized_values if reward is not None else {},
        "weighted_contributions_per_step_equivalent": step_contributions,
        "weighted_episode_contributions": episode_contributions,
        "total_reward_per_step_equivalent": reward.total_reward if reward is not None else None,
        "total_episode_reward": reward.total_reward * EPISODE_STEPS if reward is not None else None,
        "absolute_contribution_percent": shares,
    }


def main() -> None:
    root = Path("results/raw/phase2_validation")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    out_dir = root / f"phase2_corrective_validation_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=False)

    samples = []
    range_findings = []
    for scenario_type in SCENARIO_TYPES:
        for seed in range(10, 30):
            spec = ScenarioGenerator(seed=seed).generate_scenario(scenario_type, split="train", seed=seed)
            findings = spec_range_findings(spec, conditioned_ranges(scenario_type))
            range_findings.extend({"scenario_id": spec.scenario_id, "scenario_type": scenario_type, **finding} for finding in findings)
            samples.append({"scenario": spec.as_dict(), "range_findings": findings})

    ood_specs = []
    for scenario_type in SCENARIO_TYPES[1:]:
        for seed in range(1000, 1020):
            spec = ScenarioGenerator(seed=seed).generate_ood_scenario(scenario_type, split="ood", seed=seed)
            findings = spec_range_findings(spec, conditioned_ranges(scenario_type, is_ood=True))
            range_findings.extend({"scenario_id": spec.scenario_id, "scenario_type": scenario_type, "is_ood": True, **finding} for finding in findings)
            ood_specs.append({"scenario": spec.as_dict(), "ood_range_findings": findings})

    train, val, test = split_scenarios(ScenarioGenerator(seed=42), n_train=70, n_val=15, n_test=15)
    split_ids = {name: {s.scenario_id for s in specs} for name, specs in [("train", train), ("validation", val), ("test", test)]}
    ood = generate_ood_scenarios(ScenarioGenerator(seed=999), n_scenarios=20)
    split_overlap = {
        "train_validation": sorted(split_ids["train"] & split_ids["validation"]),
        "train_test": sorted(split_ids["train"] & split_ids["test"]),
        "validation_test": sorted(split_ids["validation"] & split_ids["test"]),
        "ood_train_validation_test": sorted({s.scenario_id for s in ood} & set.union(*split_ids.values())),
        "counts": {"train": len(train), "validation": len(val), "test": len(test), "ood": len(ood)},
        "ood_marked": all(s.is_ood and s.split == "ood" for s in ood),
    }

    representative_specs = []
    for scenario_type in SCENARIO_TYPES:
        for seed in range(2000, 2020):
            representative_specs.append(ScenarioGenerator(seed=seed).generate_scenario(scenario_type, split="validation", seed=seed))
    ood_representative_specs = [
        ScenarioGenerator(seed=3000 + i).generate_ood_scenario(SCENARIO_TYPES[1 + i % 4], split="ood", seed=3000 + i)
        for i in range(80)
    ]
    physical_results = [episode_metrics(spec, spec.seed) for spec in representative_specs]
    ood_physical_results = [episode_metrics(spec, spec.seed) for spec in ood_representative_specs]
    reward_stats = [episode_metrics(spec, spec.seed) for spec in representative_specs]
    feasible_reward_stats = [record for record in reward_stats if record["reward_computed"]]

    summaries = {}
    for scenario_type in SCENARIO_TYPES:
        records = [r for r in physical_results if r["scenario_type"] == scenario_type]
        summaries[scenario_type] = {
            "sample_count": len(records),
            "reset_invalid_count": sum(not r["reset_valid"] for r in records),
            "episodes_with_physical_invalid_steps": sum(r["invalid_steps"] > 0 for r in records),
            "physical_invalid_step_count": sum(r["invalid_steps"] for r in records),
            "physical_invalid_step_rate": sum(r["invalid_steps"] for r in records) / (len(records) * EPISODE_STEPS),
            "demand_shortfall_count": sum(r["demand_shortfall_at_full_capacity"] for r in records),
            "demand_shortfall_rate": sum(r["demand_shortfall_at_full_capacity"] for r in records) / len(records),
        }
    ood_summary = {
        "sample_count": len(ood_physical_results),
        "reset_invalid_count": sum(not r["reset_valid"] for r in ood_physical_results),
        "episodes_with_physical_invalid_steps": sum(r["invalid_steps"] > 0 for r in ood_physical_results),
        "physical_invalid_step_count": sum(r["invalid_steps"] for r in ood_physical_results),
        "physical_invalid_step_rate": sum(r["invalid_steps"] for r in ood_physical_results) / (len(ood_physical_results) * EPISODE_STEPS),
        "demand_shortfall_count": sum(r["demand_shortfall_at_full_capacity"] for r in ood_physical_results),
        "demand_shortfall_rate": sum(r["demand_shortfall_at_full_capacity"] for r in ood_physical_results) / len(ood_physical_results),
    }

    component_names = ["throughput", "energy", "leakage", "violation"]
    component_stats = {}
    for name in component_names:
        shares = [r["absolute_contribution_percent"][name] for r in feasible_reward_stats]
        weighted = [abs(r["weighted_episode_contributions"][name]) for r in feasible_reward_stats]
        raw = [float(r["raw_values"][name]) for r in feasible_reward_stats]
        normalized = [float(r["normalized_values"][name]) for r in feasible_reward_stats]
        component_stats[name] = {
            "mean_raw_value": float(np.mean(raw)),
            "mean_normalized_value": float(np.mean(normalized)),
            "mean_absolute_contribution_percent": float(np.mean(shares)),
            "min_absolute_contribution_percent": float(np.min(shares)),
            "max_absolute_contribution_percent": float(np.max(shares)),
            "dominant_episode_fraction_above_50_percent": float(np.mean(np.asarray(shares) > 50.0)),
            "negligible_episode_fraction_below_1_percent": float(np.mean(np.asarray(shares) < 1.0)),
            "mean_absolute_weighted_episode_contribution": float(np.mean(weighted)),
            "negligible_in_all_episodes_below_1_percent": bool(max(shares) < 1.0),
            "dominant_in_all_episodes_above_50_percent": bool(min(shares) > 50.0),
        }

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scenario_sample_count": len(samples),
        "ood_sample_count": len(ood_specs),
        "configured_nominal_ranges": {key: list(value) for key, value in DEFAULT_DOMAIN_RANGES.items()},
        "conditioned_range_violations": range_findings,
        "split_overlap": split_overlap,
        "physical_feasibility_by_class": summaries,
        "ood_physical_feasibility": ood_summary,
        "representative_episode_count": len(reward_stats),
        "feasible_reward_episode_count": len(feasible_reward_stats),
        "reward_component_statistics": component_stats,
        "representative_episodes": reward_stats,
        "physical_episode_details": physical_results,
        "ood_episode_details": ood_physical_results,
        "scenario_samples": samples,
        "ood_samples": ood_specs,
    }
    (out_dir / "corrective_validation.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    (out_dir / "scenario_samples.json").write_text(json.dumps(samples, indent=2), encoding="utf-8")
    (out_dir / "ood_samples.json").write_text(json.dumps(ood_specs, indent=2), encoding="utf-8")
    (out_dir / "episode_rewards.json").write_text(json.dumps(reward_stats, indent=2), encoding="utf-8")

    previous_dir = Path("results/raw/phase2_validation/phase2_validation_20261109_051641_271639")
    previous_summary_path = previous_dir / "summary.json"
    previous_summary = json.loads(previous_summary_path.read_text(encoding="utf-8")) if previous_summary_path.exists() else {}

    lines = [
        "# Phase 2 Corrective Validation Report",
        "",
        f"Generated: {output['generated_at']}",
        f"New results: `{out_dir}`",
        "",
        "## State mapping and reset correction",
        "- Methodology Section 7.1 defines indices 0-5 as source pressure/temperature channels, 6-7 as sink pressures, 8-15 as compressor-state channels, 16-20 as pipeline flow rates, 21-25 as supply/demand profiles, and 26-28 as energy price/hour/day.",
        "- The methodology does not specify the individual meanings of compressor channels 8 and 9. They are not pressure channels and must not be checked against pressure limits.",
        "- Reset now validates source pressure entries 0-2 and sink pressure entries 6-7. The existing nominal values are within configured 4-25 MPa limits; the minimum-pressure constraint remains unchanged.",
        "- Reset feasibility: verified through the test suite and directly recorded in each episode result.",
        "",
        "## Scenario-conditioned ranges",
        f"- Generated {len(samples)} in-distribution samples (20 per class) and {len(ood_specs)} OOD samples (20 per non-normal class).",
        f"- Conditioned range violations: {len(range_findings)}.",
        "- Nominal base ranges are separated from scenario overrides. High-demand multiplier 1.5 and derived sink-rate range 1,500-15,000 are intentional in-distribution conditions; OOD high demand uses multiplier 2.0-2.8.",
        "- OOD parameter excursions are checked against explicit OOD-conditioned ranges and are labeled OOD, not counted as methodology violations.",
        "- Derived sink injection rates are demand targets. Exceeding available supply is recorded as demand shortfall, not as a physical flow-capacity violation; environment flows remain clipped by configured sink capacities.",
        "",
        "## Split and OOD separation",
        f"- Split counts: {split_overlap['counts']}.",
        f"- train/validation overlap: {split_overlap['train_validation']}; train/test: {split_overlap['train_test']}; validation/test: {split_overlap['validation_test']}; OOD vs all in-distribution splits: {split_overlap['ood_train_validation_test']}.",
        f"- All OOD IDs are marked OOD with split='ood': {split_overlap['ood_marked']}.",
        "",
        "## Physical feasibility",
        "- Physical infeasibility here means a reset or environment step violating source capacity, sink capacity, or configured pressure constraints. Demand shortfall is reported separately.",
    ]
    for name, stats in summaries.items():
        lines.append(f"- {name}: {stats['sample_count']} episodes; reset invalid {stats['reset_invalid_count']}; episodes with invalid steps {stats['episodes_with_physical_invalid_steps']}; invalid-step rate {stats['physical_invalid_step_rate']:.4f}; demand-shortfall rate {stats['demand_shortfall_rate']:.3f}.")
    lines.append(f"- OOD: {ood_summary['sample_count']} episodes; reset invalid {ood_summary['reset_invalid_count']}; episodes with invalid steps {ood_summary['episodes_with_physical_invalid_steps']}; invalid-step rate {ood_summary['physical_invalid_step_rate']:.4f}; demand-shortfall rate {ood_summary['demand_shortfall_rate']:.3f}.")
    lines.extend([
        "- Any physical invalidity reason is retained per episode in `physical_episode_details` and `ood_episode_details` in the JSON output.",
        "- The current Phase 1 transition does not apply sampled outages, compressor efficiency, pressure dynamics, pipeline hydraulics, or scenario demand as dynamic constraints; zero physical invalidity is therefore a check of the implemented environment contract, not proof of full-network physical feasibility.",
        "",
        "## Reward analysis on feasible episodes",
        f"- Reward components were recomputed for {len(feasible_reward_stats)} of {len(reward_stats)} representative episodes; only episodes with valid reset and all {EPISODE_STEPS} valid steps were included.",
        "- Raw and normalized metrics use explicit audit bounds derived from scenario maxima; weights remain unchanged (throughput 1.0, energy 0.1, leakage 10.0, violation 5.0). The energy metric is a transparent compressor-power/price proxy because the environment does not currently return physical compressor energy use.",
    ])
    for name, stats in component_stats.items():
        lines.append(f"- {name}: mean raw {stats['mean_raw_value']:.6g}; mean normalized {stats['mean_normalized_value']:.6g}; mean absolute weighted episode contribution {stats['mean_absolute_weighted_episode_contribution']:.6g}; mean absolute-reward share {stats['mean_absolute_contribution_percent']:.2f}% (range {stats['min_absolute_contribution_percent']:.2f}-{stats['max_absolute_contribution_percent']:.2f}%); >50% share in {stats['dominant_episode_fraction_above_50_percent']:.1%} of episodes; <1% share in {stats['negligible_episode_fraction_below_1_percent']:.1%} of episodes.")
    lines.extend([
        "- Per-episode records contain raw values, normalization bounds, normalized values, weighted contributions per step and summed over 168 steps, total episode reward, and percentage share of total absolute component magnitude.",
        "",
        "## Comparison with previous validation",
        "- Previous report: `docs/PHASE2_VALIDATION_REPORT.md`; previous result bundle: `results/raw/phase2_validation/phase2_validation_20261109_051641_271639`.",
        f"- Previously reported representative physical invalid rate: {previous_summary.get('infeasible_feasible_rate', 'not recorded')} and max-action stress invalid rate: {previous_summary.get('stress_test', {}).get('invalid_rate', 'not recorded')}.",
        "- The previous 100% invalid finding was caused by checking compressor-state entries 8 and 9 as pressures. After correcting that mapping, this pass reports reset and episode feasibility using the actual pressure channels; this is a code-path correction, not a relaxation of the pressure floor.",
        "- The previous range audit applied nominal ranges indiscriminately and counted intentional high-demand values as violations. This pass uses class-specific ranges and keeps OOD excursions separate.",
        "- The previous reward statistics used synthetic component inputs. This pass derives throughput, leakage, violation frequency, and a documented energy proxy from feasible environment episodes and records complete component detail.",
        "",
        "## Unresolved concerns",
        "- Individual physical definitions for compressor observation channels 8-15 remain unspecified by the methodology.",
        "- The environment's assignments to observation entries 25-28 do not match the methodology's energy-price/hour/day group; the full observation-slot mapping needs a separate documented correction before training.",
        "- The Phase 1 environment remains a simplified feasibility checker: sampled scenario pressure, outage, compressor, leakage, and demand dynamics are not fully coupled into state transitions.",
        "- Reward bounds in `configs/reward.yaml` remain generic [0,1]; this audit uses explicit physical audit scales, but the project should define training-distribution normalization bounds before training or sensitivity experiments.",
        "- Demand-shortfall rates are screening indicators under the current scalar-per-source/sink expansion, not a network-flow optimization result.",
    ])
    report = "\n".join(lines) + "\n"
    Path("docs/PHASE2_CORRECTIVE_VALIDATION_REPORT.md").write_text(report, encoding="utf-8")
    (out_dir / "report.md").write_text(report, encoding="utf-8")
    print(json.dumps({
        "output_dir": str(out_dir),
        "sample_count": len(samples),
        "ood_sample_count": len(ood_specs),
        "conditioned_range_violations": len(range_findings),
        "split_overlap": split_overlap,
        "physical_feasibility_by_class": summaries,
        "ood_physical_feasibility": ood_summary,
        "reward_component_statistics": component_stats,
    }, indent=2))


if __name__ == "__main__":
    main()
