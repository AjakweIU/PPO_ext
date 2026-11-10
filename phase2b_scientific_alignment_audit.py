from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from src.environment.co2_transport_env import CO2TransportEnv
from src.environment.normalization import default_observation_ranges
from src.environment.reward import DEFAULT_REWARD_WEIGHTS, compute_reward
from src.scenarios.generator import ScenarioGenerator
from src.scenarios.splits import generate_ood_scenarios, split_scenarios

SCENARIO_TYPES = ["normal", "high_demand", "low_supply", "equipment_failure", "energy_price_spike"]
STEPS = 168


def run_episode(scenario_type: str, seed: int) -> dict[str, object]:
    spec = ScenarioGenerator(seed=seed).generate_scenario(scenario_type, split="validation", seed=seed)
    env = CO2TransportEnv()
    _, reset_info = env.reset(seed=seed)
    delivered = []
    power_kw = []
    violations = []
    for t in range(STEPS):
        source_fraction = [0.35 + (0.25 * ((t % 4) / 3.0)), 0.5, 0.55]
        compressor_actions = [1.0, 1.0, 1.0, 1.0]
        _, reward, _, _, info = env.step(source_fraction + compressor_actions)
        delivered.append(float(info["delivered_flow"]))
        power_kw.append(float(env.state[10] + env.state[14]))
        violations.append(not bool(info["valid"]))
    mean_daily_delivery = float(np.mean(delivered))
    leakage_fraction = float(spec.parameters["leakage_rate"])
    episode_delivery_tonnes = mean_daily_delivery * (STEPS / 24.0)
    derived_leakage_tonnes = episode_delivery_tonnes * leakage_fraction
    energy_price = float(spec.parameters["base_energy_price"] * spec.parameters["peak_price_multiplier"])
    environment_energy_cost = float(sum(power_kw) * energy_price)
    raw_metrics = {
        "throughput_tonnes_per_day": mean_daily_delivery,
        "energy_cost_usd_per_episode_reported_by_environment": environment_energy_cost,
        "leakage_tonnes_per_episode_derived_proxy_not_environment_output": derived_leakage_tonnes,
        "violation_step_fraction": float(np.mean(violations)),
    }
    bounds = {
        "throughput": (0.0, float(np.sum(env.sink_capacity))),
        "energy": (0.0, 2.0 * 8000.0 * 0.525 * STEPS),
        "leakage": (0.0, float(np.sum(env.sink_capacity)) * (STEPS / 24.0) * 0.02),
        "violation": (0.0, 1.0),
    }
    breakdown = compute_reward(
        throughput=raw_metrics["throughput_tonnes_per_day"],
        energy_cost=raw_metrics["energy_cost_usd_per_episode_reported_by_environment"],
        leakage=raw_metrics["leakage_tonnes_per_episode_derived_proxy_not_environment_output"],
        violation=raw_metrics["violation_step_fraction"],
        bounds=bounds,
    )
    abs_total = sum(abs(v) for v in breakdown.component_contributions.values())
    shares = {key: (abs(value) / abs_total * 100.0 if abs_total else 0.0) for key, value in breakdown.component_contributions.items()}
    return {
        "scenario_id": spec.scenario_id,
        "scenario_type": scenario_type,
        "reset_valid": bool(reset_info["valid"]),
        "invalid_steps": int(sum(violations)),
        "raw_metrics": raw_metrics,
        "normalization_bounds": {key: list(value) for key, value in bounds.items()},
        "normalized": breakdown.normalized_values,
        "weighted": breakdown.component_contributions,
        "total_weighted_reward": breakdown.total_reward,
        "absolute_component_share_percent": shares,
        "scenario_leakage_rate": leakage_fraction,
    }


def main() -> None:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    output_dir = Path("results/raw/phase2_validation") / f"phase2b_alignment_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)

    range_matrix = default_observation_ranges()
    with Path("configs/base_network.yaml").open("r", encoding="utf-8") as handle:
        base_config = yaml.safe_load(handle)
    with Path("configs/phase1_defaults.yaml").open("r", encoding="utf-8") as handle:
        phase1_config = yaml.safe_load(handle)
    feature_count = len(base_config["observation"]["features"])
    config_range_count = len(phase1_config["observation"]["ranges"])

    train, val, test = split_scenarios(ScenarioGenerator(seed=42), n_train=70, n_val=15, n_test=15)
    ood = generate_ood_scenarios(ScenarioGenerator(seed=999), n_scenarios=20)
    all_ids = {name: {s.scenario_id for s in group} for name, group in [("train", train), ("validation", val), ("test", test), ("ood", ood)]}
    overlap = {
        "train_validation": sorted(all_ids["train"] & all_ids["validation"]),
        "train_test": sorted(all_ids["train"] & all_ids["test"]),
        "validation_test": sorted(all_ids["validation"] & all_ids["test"]),
        "ood_in_distribution": sorted(all_ids["ood"] & (all_ids["train"] | all_ids["validation"] | all_ids["test"])),
    }

    low_env = CO2TransportEnv()
    low_env.reset(seed=1)
    _, _, _, _, low_info = low_env.step([0.2, 0.2, 0.2, 0.75, 0.75, 0.75, 0.75])
    high_env = CO2TransportEnv()
    high_env.reset(seed=1)
    _, _, _, _, high_info = high_env.step([1.0, 1.0, 1.0, 1.25, 1.25, 1.25, 1.25])
    compressor_low = CO2TransportEnv()
    compressor_low.reset(seed=2)
    _, reward_low, _, _, c_low_info = compressor_low.step([0.5, 0.5, 0.5, 0.75, 0.75, 0.75, 0.75])
    compressor_high = CO2TransportEnv()
    compressor_high.reset(seed=2)
    _, reward_high, _, _, c_high_info = compressor_high.step([0.5, 0.5, 0.5, 1.25, 1.25, 1.25, 1.25])
    reduced_capacity_env = CO2TransportEnv(source_capacity=np.asarray([750.0, 1100.0, 1300.0]))
    reduced_capacity_env.reset(seed=3)
    _, _, _, _, reduced_capacity_info = reduced_capacity_env.step([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    pressure_probe_env = CO2TransportEnv()
    pressure_probe_env.reset(seed=4)
    pressure_probe_env.state[6] = 3.0e6
    _, _, _, _, pressure_probe_info = pressure_probe_env.step([0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0])

    scenario_probes = []
    for scenario_type in SCENARIO_TYPES:
        for seed in range(500, 520):
            spec = ScenarioGenerator(seed=seed).generate_scenario(scenario_type, split="validation", seed=seed)
            env = CO2TransportEnv()
            _, reset_info = env.reset(seed=seed)
            _, _, _, _, info = env.step([0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0])
            scenario_probes.append({
                "scenario_type": scenario_type,
                "scenario_id": spec.scenario_id,
                "reset_valid": bool(reset_info["valid"]),
                "step_valid": bool(info["valid"]),
                "delivered_flow": float(info["delivered_flow"]),
                "scenario_energy_price": float(spec.parameters["base_energy_price"] * spec.parameters["peak_price_multiplier"]),
                "scenario_failure_probability": float(spec.parameters["failure_probability"]),
                "source_outage_probability": float(spec.parameters.get("source_outage_probability", 0.0)),
                "compressor_outage_probability": float(spec.parameters.get("compressor_outage_probability", 0.0)),
                "scenario_leakage_rate": float(spec.parameters["leakage_rate"]),
                "env_energy_state": float(env.state[10] + env.state[14]),
                "env_price_state": float(env.state[26]),
            })

    episodes = [run_episode(scenario_type, seed) for scenario_type in SCENARIO_TYPES for seed in range(600, 620)]
    component_keys = ["throughput", "energy", "leakage", "violation"]
    component_stats = {}
    for key in component_keys:
        component_stats[key] = {
            "mean_raw": float(np.mean([r["raw_metrics"][{"throughput": "throughput_tonnes_per_day", "energy": "energy_cost_usd_per_episode_reported_by_environment", "leakage": "leakage_tonnes_per_episode_derived_proxy_not_environment_output", "violation": "violation_step_fraction"}[key]] for r in episodes])),
            "normalization_bounds": episodes[0]["normalization_bounds"][key],
            "mean_normalized": float(np.mean([r["normalized"][key] for r in episodes])),
            "mean_weighted": float(np.mean([r["weighted"][key] for r in episodes])),
            "mean_absolute_contribution_percent": float(np.mean([r["absolute_component_share_percent"][key] for r in episodes])),
        }
    leakage_share_if_w3_one = []
    for episode in episodes:
        normalized = episode["normalized"]
        terms = {
            "throughput": normalized["throughput"],
            "energy": 0.1 * normalized["energy"],
            "leakage": -normalized["leakage"],
            "violation": -5.0 * normalized["violation"],
        }
        denom = sum(abs(v) for v in terms.values())
        leakage_share_if_w3_one.append(abs(terms["leakage"]) / denom * 100.0 if denom else 0.0)
    default_bound_breakdown = compute_reward(
        throughput=component_stats["throughput"]["mean_raw"],
        energy_cost=component_stats["energy"]["mean_raw"],
        leakage=component_stats["leakage"]["mean_raw"],
        violation=component_stats["violation"]["mean_raw"],
    )
    default_abs_total = sum(abs(v) for v in default_bound_breakdown.component_contributions.values())

    params = [ScenarioGenerator(seed=s).generate_scenario("normal", seed=s).parameters for s in range(1000, 1500)]
    shortfalls = {
        "interpretation": "Each sampled scalar source_capacity is treated as per-source and sink_injection_rate as per-sink, matching the prior validation audit's expansion assumption; scenario generator does not explicitly define this mapping.",
        "normal_rate": float(np.mean([3 * p["source_capacity"] < 2 * p["sink_injection_rate"] for p in params])),
        "configured_nominal_aggregate_source_capacity_tonnes_per_day": float(np.sum(base_config["physical"]["source_capacity_tonnes_per_day"])),
        "configured_nominal_aggregate_sink_capacity_tonnes_per_day": float(np.sum(base_config["physical"]["sink_capacity_tonnes_per_day"])),
        "nominal_configured_capacity_screen_feasible": float(np.sum(base_config["physical"]["source_capacity_tonnes_per_day"])) >= float(np.sum(base_config["physical"]["sink_capacity_tonnes_per_day"])),
    }

    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(output_dir),
        "observation_dimensions": {"normalization_ranges": int(range_matrix.shape[0]), "base_config_features": feature_count, "phase1_config_ranges": config_range_count},
        "split_overlap": overlap,
        "action_coupling_probes": {
            "source_low_delivery": float(low_info["delivered_flow"]),
            "source_high_delivery": float(high_info["delivered_flow"]),
            "source_controls_change_delivery": bool(high_info["delivered_flow"] > low_info["delivered_flow"]),
            "compressor_low_delivery": float(c_low_info["delivered_flow"]),
            "compressor_high_delivery": float(c_high_info["delivered_flow"]),
            "compressor_control_delivery_change": bool(c_low_info["delivered_flow"] != c_high_info["delivered_flow"]),
            "compressor_low_reward": float(reward_low),
            "compressor_high_reward": float(reward_high),
            "compressor_power_state_low": float(compressor_low.state[10] + compressor_low.state[14]),
            "compressor_power_state_high": float(compressor_high.state[10] + compressor_high.state[14]),
            "configured_capacity_limited_delivery": float(high_info["delivered_flow"]),
            "half_source_capacity_delivery": float(reduced_capacity_info["delivered_flow"]),
            "action_projection_limits_source_and_compressor": True,
            "forced_below_minimum_pressure_is_detected": not bool(pressure_probe_info["valid"]),
            "forced_pressure_violation_reasons": pressure_probe_info["constraint_check"]["pressure"],
        },
        "scenario_coupling_probes": scenario_probes,
        "reward_component_statistics": component_stats,
        "leakage_weight_diagnostic": {
            "observed_mean_episode_share_with_w3_10_percent": component_stats["leakage"]["mean_absolute_contribution_percent"],
            "mean_episode_share_counterfactual_w3_1_percent": float(np.mean(leakage_share_if_w3_one)),
            "counterfactual_is_diagnostic_not_weight_update": True,
            "default_config_bounds_mean_normalized": default_bound_breakdown.normalized_values,
            "default_config_bounds_weighted": default_bound_breakdown.component_contributions,
            "default_config_bounds_leakage_absolute_share_percent": abs(default_bound_breakdown.component_contributions["leakage"]) / default_abs_total * 100.0 if default_abs_total else 0.0,
        },
        "reward_weights": DEFAULT_REWARD_WEIGHTS,
        "feasible_episode_count": sum(r["reset_valid"] and r["invalid_steps"] == 0 for r in episodes),
        "episode_count": len(episodes),
        "episodes": episodes,
        "demand_shortfall_screen": shortfalls,
        "active_environment_reward": "CO2TransportEnv.step returns 0.1 * delivered_flow when valid; it does not call compute_reward.",
        "compute_reward_default_bounds": {k: [0.0, 1.0] for k in component_keys},
        "violation_detector_probe": {
            "action_projection_bounds": {"source": [0.0, 1.0], "compressor": [0.75, 1.25]},
            "pressure_limits_still_checked": True,
            "pressure_state_dynamics_present": False,
            "capacity_check_is_effectively_bypassed_by_projection_and_clipping": True,
        },
    }
    (output_dir / "phase2b_alignment.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key not in ["scenario_coupling_probes", "episodes"]}, indent=2))


if __name__ == "__main__":
    main()
