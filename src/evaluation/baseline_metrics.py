from __future__ import annotations

import time
from typing import Any, Iterable

import numpy as np

from src.environment.co2_transport_env import CO2TransportEnv
from src.scenarios.generator import ScenarioSpec


def evaluate_policy(
    policy: Any,
    scenarios: Iterable[ScenarioSpec],
    *,
    episode_steps: int | None = None,
    env_factory: Any = CO2TransportEnv,
) -> dict[str, Any]:
    """Evaluate any policy implementing ``act(observation, env)`` on shared episodes."""
    episode_results = []
    for spec in scenarios:
        env = env_factory(scenario_parameters=spec.parameters)
        observation, reset_info = env.reset(seed=spec.seed)
        target_steps = int(episode_steps or env.episode_steps)
        decision_times_ms = []
        step_records = []
        episode_reward = 0.0
        throughput_tonnes = 0.0
        energy_kwh = 0.0
        energy_cost_usd = 0.0
        simulated_leakage_tonnes = 0.0
        demand_shortfall_tonnes = 0.0
        physical_violations = 0
        action_corrections = 0
        solver_statuses = []

        for _ in range(target_steps):
            decision_start = time.perf_counter_ns()
            decision = policy.act(observation, env)
            decision_time_ms = (time.perf_counter_ns() - decision_start) / 1e6
            decision_times_ms.append(decision_time_ms)
            observation, reward, terminated, truncated, info = env.step(decision.action)
            episode_reward += float(reward)
            throughput_tonnes += float(np.sum(info["sink_received_tonnes"]))
            energy_kwh += float(info["energy_kwh"])
            energy_cost_usd += float(info["energy_cost"])
            simulated_leakage_tonnes += float(info["leakage_tonnes"])
            demand_shortfall_tonnes += float(info["demand_shortfall_tonnes"])
            violation_components = info["violation_components"]
            physical_violation = any(
                violation_components[name] > 0.0
                for name in ["pressure_violation", "pressure_ratio_violation", "capacity_violation"]
            )
            physical_violations += int(physical_violation)
            action_corrections += int(bool(info["correction"]["was_clipped"]))
            solver_status = decision.diagnostics.get("solver_status")
            if solver_status is not None:
                solver_statuses.append(solver_status)
            step_records.append({
                "step": env.current_step,
                "action": np.asarray(decision.action, dtype=float).tolist(),
                "decision_time_ms": decision_time_ms,
                "policy_diagnostics": decision.diagnostics,
                "reward": float(reward),
                "throughput_tonnes": float(np.sum(info["sink_received_tonnes"])),
                "energy_kwh": float(info["energy_kwh"]),
                "energy_cost_usd": float(info["energy_cost"]),
                "simulated_leakage_tonnes": float(info["leakage_tonnes"]),
                "demand_shortfall_tonnes": float(info["demand_shortfall_tonnes"]),
                "violation_components": {key: float(value) for key, value in violation_components.items()},
                "physical_valid": bool(info["physical_valid"]),
                "action_correction": info["correction"],
            })
            if terminated or truncated:
                break

        times = np.asarray(decision_times_ms, dtype=float)
        episode_results.append({
            "scenario_id": spec.scenario_id,
            "scenario_type": spec.scenario_type,
            "split": spec.split,
            "is_ood": spec.is_ood,
            "seed": spec.seed,
            "method": policy.name,
            "reset_valid": bool(reset_info["valid"]),
            "reward": episode_reward,
            "throughput_tonnes": throughput_tonnes,
            "energy_consumption_kwh": energy_kwh,
            "energy_cost_usd": energy_cost_usd,
            "simulated_leakage_tonnes": simulated_leakage_tonnes,
            "demand_shortfall_tonnes": demand_shortfall_tonnes,
            "physical_violations": physical_violations,
            "action_corrections": action_corrections,
            "mean_decision_time_ms": float(np.mean(times)) if times.size else 0.0,
            "median_decision_time_ms": float(np.median(times)) if times.size else 0.0,
            "p95_decision_time_ms": float(np.percentile(times, 95)) if times.size else 0.0,
            "milp_solver_statuses": solver_statuses,
            "valid_episode": bool(reset_info["valid"] and all(row["physical_valid"] for row in step_records)),
            "steps": step_records,
        })

    return {
        "method": policy.name,
        "episode_count": len(episode_results),
        "episodes": episode_results,
        "aggregate": {
            key: float(np.mean([episode[key] for episode in episode_results])) if episode_results else 0.0
            for key in [
                "reward",
                "throughput_tonnes",
                "energy_consumption_kwh",
                "energy_cost_usd",
                "simulated_leakage_tonnes",
                "demand_shortfall_tonnes",
                "physical_violations",
                "action_corrections",
                "mean_decision_time_ms",
                "median_decision_time_ms",
                "p95_decision_time_ms",
            ]
        },
        "pooled_decision_timing_ms": {
            "mean": float(np.mean([row["decision_time_ms"] for episode in episode_results for row in episode["steps"]])) if episode_results else 0.0,
            "median": float(np.median([row["decision_time_ms"] for episode in episode_results for row in episode["steps"]])) if episode_results else 0.0,
            "p95": float(np.percentile([row["decision_time_ms"] for episode in episode_results for row in episode["steps"]], 95)) if episode_results else 0.0,
        },
    }
