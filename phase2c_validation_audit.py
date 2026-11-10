from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.environment.co2_transport_env import CO2TransportEnv
from src.scenarios.generator import ScenarioGenerator

SCENARIO_TYPES = ["normal", "high_demand", "low_supply", "equipment_failure", "energy_price_spike"]
BASE_ACTION = np.asarray([0.6, 0.6, 0.6, 1.0, 1.0, 1.0, 1.0], dtype=float)


def run_steps(env: CO2TransportEnv, action: np.ndarray, seed: int, n_steps: int = 24) -> list[dict[str, object]]:
    _, reset_info = env.reset(seed=seed)
    rows = []
    for _ in range(n_steps):
        _, reward, _, _, info = env.step(action)
        invalid_reasons = [reason for reasons in info["constraint_check"].values() for reason in reasons]
        rows.append({
            "reward": float(reward),
            "valid": bool(info["physical_valid"]),
            "constraint_reasons": invalid_reasons,
            "delivered_flow": float(info["delivered_flow"]),
            "energy_kwh": float(info["energy_kwh"]),
            "energy_cost": float(info["energy_cost"]),
            "leakage_tonnes": float(info["leakage_tonnes"]),
            "violation": float(info["violation"]),
            "demand_shortfall_fraction": float(info["demand_shortfall_fraction"]),
            "capacity_shortfall_tonnes": float(np.sum(info["capacity_shortfall_tonnes"])),
            "compressor_power_kw": [float(x) for x in info["compressor_power_kw"]],
            "compressor_pressure_ratios": [float(x) for x in info["compressor_pressure_ratios"]],
            "segment_flows": [float(x) for x in info["segment_flows"]],
            "source_flow": [float(x) for x in info["source_flow"]],
            "sink_flow": [float(x) for x in info["sink_flow"]],
            "weighted_contributions": {key: float(value) for key, value in info["weighted_contributions"].items()},
            "correction": info["correction"],
            "failed_sources": info["failed_sources"],
            "failed_compressors": info["failed_compressors"],
            "failed_segments": info["failed_segments"],
        })
    return [{"reset_info": reset_info, "steps": rows}]


def summarise(records: list[dict[str, object]]) -> dict[str, object]:
    steps = [step for episode in records for step in episode["steps"]]
    component_names = ["throughput", "energy", "leakage", "violation"]
    contribution_totals = {name: 0.0 for name in component_names}
    absolute_totals = {name: 0.0 for name in component_names}
    for step in steps:
        for name, value in step["weighted_contributions"].items():
            contribution_totals[name] += value
            absolute_totals[name] += abs(value)
    denominator = sum(absolute_totals.values())
    reason_counts: dict[str, int] = {}
    for step in steps:
        for reason in step["constraint_reasons"]:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
    return {
        "episode_count": len(records),
        "step_count": len(steps),
        "reset_valid_count": sum(bool(episode["reset_info"]["valid"]) for episode in records),
        "physical_invalid_step_count": sum(not step["valid"] for step in steps),
        "physical_invalidity_reasons": reason_counts,
        "mean_delivered_flow_tonnes_per_day": float(np.mean([step["delivered_flow"] for step in steps])) if steps else 0.0,
        "mean_energy_kwh": float(np.mean([step["energy_kwh"] for step in steps])) if steps else 0.0,
        "mean_energy_cost": float(np.mean([step["energy_cost"] for step in steps])) if steps else 0.0,
        "mean_leakage_tonnes": float(np.mean([step["leakage_tonnes"] for step in steps])) if steps else 0.0,
        "mean_violation": float(np.mean([step["violation"] for step in steps])) if steps else 0.0,
        "mean_demand_shortfall_fraction": float(np.mean([step["demand_shortfall_fraction"] for step in steps])) if steps else 0.0,
        "weighted_contribution_sums": contribution_totals,
        "absolute_component_shares_percent": {name: (value / denominator * 100.0 if denominator else 0.0) for name, value in absolute_totals.items()},
        "mean_reward": float(np.mean([step["reward"] for step in steps])) if steps else 0.0,
    }


def main() -> None:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    out_dir = Path("results/raw/phase2_validation") / f"phase2c_dynamics_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=False)
    per_scenario = {}
    raw_episodes = {}
    for scenario_index, scenario_type in enumerate(SCENARIO_TYPES):
        episodes = []
        for seed in range(1200 + scenario_index * 100, 1220 + scenario_index * 100):
            spec = ScenarioGenerator(seed=seed).generate_scenario(scenario_type, split="validation", seed=seed)
            env = CO2TransportEnv(scenario_parameters=spec.parameters)
            episodes.extend(run_steps(env, BASE_ACTION, seed, n_steps=24))
        raw_episodes[scenario_type] = episodes
        per_scenario[scenario_type] = summarise(episodes)

    paired_seed = 777
    nominal_spec = ScenarioGenerator(seed=paired_seed).generate_scenario("normal", split="audit", seed=paired_seed)
    price_parameters = dict(nominal_spec.parameters)
    price_parameters["base_energy_price"] = 0.10
    price_parameters["peak_price_multiplier"] = 2.5
    nominal_env = CO2TransportEnv(scenario_parameters=nominal_spec.parameters)
    price_env = CO2TransportEnv(scenario_parameters=price_parameters)
    nominal_steps = run_steps(nominal_env, BASE_ACTION, paired_seed, n_steps=1)[0]["steps"]
    price_steps = run_steps(price_env, BASE_ACTION, paired_seed, n_steps=1)[0]["steps"]

    compressor_low = CO2TransportEnv()
    compressor_high = CO2TransportEnv()
    low_steps = run_steps(compressor_low, np.asarray([0.65, 0.65, 0.65, 0.80, 0.75, 0.80, 0.75]), 998, n_steps=1)[0]["steps"]
    high_steps = run_steps(compressor_high, np.asarray([0.65, 0.65, 0.65, 1.20, 1.25, 1.20, 1.25]), 998, n_steps=1)[0]["steps"]

    healthy_env = CO2TransportEnv()
    compressor_failed_env = CO2TransportEnv(scenario_parameters={"compressor_outage_probability": 1.0})
    pipeline_failed_env = CO2TransportEnv(scenario_parameters={"pipeline_outage_probability": 1.0})
    healthy = run_steps(healthy_env, np.asarray([0.9, 0.9, 0.9, 1.1, 1.25, 1.1, 1.25]), 999, 1)[0]["steps"][0]
    compressor_failed = run_steps(compressor_failed_env, np.asarray([0.9, 0.9, 0.9, 1.1, 1.25, 1.1, 1.25]), 999, 1)[0]["steps"][0]
    pipeline_failed = run_steps(pipeline_failed_env, np.asarray([0.9, 0.9, 0.9, 1.1, 1.25, 1.1, 1.25]), 999, 1)[0]["steps"][0]

    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(out_dir),
        "reward_weights": {"throughput": 1.0, "energy": 0.1, "leakage": 10.0, "violation": 5.0},
        "scenario_summary": per_scenario,
        "reward_and_dynamics_responses": {
            "energy_price": {
                "nominal_energy_kwh": nominal_steps[0]["energy_kwh"],
                "spike_energy_kwh": price_steps[0]["energy_kwh"],
                "nominal_cost": nominal_steps[0]["energy_cost"],
                "spike_cost": price_steps[0]["energy_cost"],
            },
            "compressor_effort": {
                "low_pressure_ratios": low_steps[0]["compressor_pressure_ratios"],
                "high_pressure_ratios": high_steps[0]["compressor_pressure_ratios"],
                "low_power_kw": low_steps[0]["compressor_power_kw"],
                "high_power_kw": high_steps[0]["compressor_power_kw"],
                "low_delivery": low_steps[0]["delivered_flow"],
                "high_delivery": high_steps[0]["delivered_flow"],
            },
            "equipment_failure": {
                "healthy_delivery": healthy["delivered_flow"],
                "compressor_failed_delivery": compressor_failed["delivered_flow"],
                "pipeline_failed_delivery": pipeline_failed["delivered_flow"],
                "compressor_failed_status": compressor_failed["failed_compressors"],
                "pipeline_failed_segments": pipeline_failed["failed_segments"],
            },
            "transport_capacity": {
                "healthy_delivery": healthy["delivered_flow"],
                "healthy_segment_flows": healthy["segment_flows"],
            },
        },
        "raw_episodes": raw_episodes,
    }
    (out_dir / "phase2c_validation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "raw_episodes"}, indent=2))


if __name__ == "__main__":
    main()
