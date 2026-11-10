from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.environment.co2_transport_env import CO2TransportEnv
from src.environment.reward import DEFAULT_REWARD_WEIGHTS
from src.scenarios.generator import ScenarioGenerator

SCENARIO_TYPES = ["normal", "high_demand", "low_supply", "equipment_failure", "energy_price_spike"]
AUDIT_ACTION = np.asarray([0.6, 0.6, 0.6, 0.80, 1.25, 0.80, 1.25], dtype=float)
AUDIT_STEPS = 24


def run_episode(scenario_type: str, seed: int) -> dict[str, object]:
    spec = ScenarioGenerator(seed=seed).generate_scenario(scenario_type, split="validation", seed=seed)
    env = CO2TransportEnv(scenario_parameters=spec.parameters)
    _, reset_info = env.reset(seed=seed)
    step_rows = []
    for _ in range(AUDIT_STEPS):
        _, reward, _, _, info = env.step(AUDIT_ACTION)
        step_rows.append({
            "reward": float(reward),
            "physical_valid": bool(info["physical_valid"]),
            "raw_throughput": float(info["raw_throughput"]),
            "normalized_throughput": float(info["normalized_throughput"]),
            "raw_energy_usd": float(info["raw_energy"]),
            "energy_kwh": float(info["energy_kwh"]),
            "normalized_energy": float(info["normalized_energy"]),
            "raw_simulated_leakage_tonnes": float(info["raw_leakage"]),
            "normalized_simulated_leakage": float(info["normalized_leakage"]),
            "violation": float(info["violation"]),
            "normalized_violation": float(info["normalized_violation"]),
            "violation_components": {key: float(value) for key, value in info["violation_components"].items()},
            "demand_shortfall_fraction": float(info["demand_shortfall_fraction"]),
            "demand_shortfall_tonnes": float(info["demand_shortfall_tonnes"]),
            "capture_shortfall_fraction": float(info["capture_shortfall_fraction"]),
            "capture_shortfall_tonnes": float(info["capture_shortfall_tonnes"]),
            "weighted_contributions": {key: float(value) for key, value in info["weighted_contributions"].items()},
            "reward_bounds": {key: list(value) for key, value in env.reward_bounds.items()},
            "total_reward": float(info["final_reward"]),
        })
    return {
        "scenario_type": scenario_type,
        "scenario_id": spec.scenario_id,
        "seed": seed,
        "reset_valid": bool(reset_info["valid"]),
        "failed_sources": reset_info["failed_sources"],
        "failed_compressors": reset_info["failed_compressors"],
        "failed_segments": reset_info["failed_segments"],
        "all_steps_physically_valid": all(row["physical_valid"] for row in step_rows),
        "steps": step_rows,
    }


def summarize_component(episodes: list[dict[str, object]], component: str) -> dict[str, object]:
    component_values = []
    normalized_values = []
    weighted_values = []
    episode_shares = []
    lower_bounds = []
    upper_bounds = []
    for episode in episodes:
        steps = episode["steps"]
        component_values.extend(
            row[{"throughput": "raw_throughput", "energy": "raw_energy_usd", "leakage": "raw_simulated_leakage_tonnes", "violation": "violation"}[component]]
            for row in steps
        )
        normalized_values.extend(row[{"throughput": "normalized_throughput", "energy": "normalized_energy", "leakage": "normalized_simulated_leakage", "violation": "normalized_violation"}[component]] for row in steps)
        weighted_values.extend(row["weighted_contributions"][component] for row in steps)
        for row in steps:
            bounds = row["reward_bounds"][component]
            lower_bounds.append(bounds[0])
            upper_bounds.append(bounds[1])
        episode_abs = {
            key: sum(abs(row["weighted_contributions"][key]) for row in steps)
            for key in ["throughput", "energy", "leakage", "violation"]
        }
        denominator = sum(episode_abs.values())
        episode_shares.append(episode_abs[component] / denominator * 100.0 if denominator else 0.0)
    return {
        "raw_mean_per_timestep": float(np.mean(component_values)),
        "normalization_range_envelope": [float(min(lower_bounds)), float(max(upper_bounds))],
        "normalized_mean": float(np.mean(normalized_values)),
        "weighted_mean": float(np.mean(weighted_values)),
        "mean_absolute_contribution_share_percent": float(np.mean(episode_shares)),
        "minimum_episode_contribution_share_percent": float(np.min(episode_shares)),
        "maximum_episode_contribution_share_percent": float(np.max(episode_shares)),
    }


def main() -> None:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    output_dir = Path("results/raw/phase2_validation") / f"phase2d_reward_calibration_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    episodes = [
        run_episode(scenario_type, seed)
        for type_index, scenario_type in enumerate(SCENARIO_TYPES)
        for seed in range(2200 + type_index * 100, 2220 + type_index * 100)
    ]
    feasible = [episode for episode in episodes if episode["reset_valid"] and episode["all_steps_physically_valid"]]
    excluded = [episode for episode in episodes if episode not in feasible]
    component_stats = {
        component: summarize_component(feasible, component)
        for component in ["throughput", "energy", "leakage", "violation"]
    }
    violation_category_means = {
        key: float(np.mean([row["violation_components"][key] for episode in feasible for row in episode["steps"]]))
        for key in ["action_correction", "pressure_violation", "pressure_ratio_violation", "capacity_violation"]
    }
    service_stats = {
        "mean_demand_shortfall_fraction": float(np.mean([row["demand_shortfall_fraction"] for episode in feasible for row in episode["steps"]])),
        "mean_demand_shortfall_tonnes_per_timestep": float(np.mean([row["demand_shortfall_tonnes"] for episode in feasible for row in episode["steps"]])),
        "mean_capture_shortfall_fraction": float(np.mean([row["capture_shortfall_fraction"] for episode in feasible for row in episode["steps"]])),
        "mean_capture_shortfall_tonnes_per_timestep": float(np.mean([row["capture_shortfall_tonnes"] for episode in feasible for row in episode["steps"]])),
    }
    per_scenario = {}
    for scenario_type in SCENARIO_TYPES:
        scenario_episodes = [episode for episode in feasible if episode["scenario_type"] == scenario_type]
        per_scenario[scenario_type] = {
            "feasible_episode_count": len(scenario_episodes),
            "mean_demand_shortfall_fraction": float(np.mean([row["demand_shortfall_fraction"] for ep in scenario_episodes for row in ep["steps"]])) if scenario_episodes else None,
            "mean_physical_violation": float(np.mean([row["violation"] for ep in scenario_episodes for row in ep["steps"]])) if scenario_episodes else None,
        }
    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(output_dir),
        "scenario_episode_count": len(episodes),
        "feasible_episode_count": len(feasible),
        "excluded_episode_count": len(excluded),
        "audit_steps_per_episode": AUDIT_STEPS,
        "audit_action": AUDIT_ACTION.tolist(),
        "weights": DEFAULT_REWARD_WEIGHTS,
        "normalization_bounds_by_scenario": {
            episode["scenario_id"]: episode["steps"][0]["reward_bounds"] for episode in feasible
        },
        "component_statistics": component_stats,
        "physical_violation_category_means": violation_category_means,
        "service_shortfall_statistics": service_stats,
        "scenario_summary": per_scenario,
        "episodes": episodes,
        "excluded_episodes": excluded,
    }
    (output_dir / "phase2d_calibration.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key not in ["episodes", "excluded_episodes", "normalization_bounds_by_scenario"]}, indent=2))


if __name__ == "__main__":
    main()
