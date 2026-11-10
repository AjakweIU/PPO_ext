from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import yaml

from src.baselines import FixedRulePolicy, GreedyScheduler, MILPScheduler
from src.evaluation.baseline_metrics import evaluate_policy
from src.scenarios.generator import ScenarioGenerator


def main() -> None:
    with Path("configs/baselines.yaml").open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    output_dir = Path("results/raw/phase3_baselines") / f"baseline_validation_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)

    scenario_types = ["normal", "high_demand", "low_supply", "equipment_failure", "energy_price_spike"]
    scenarios = [
        ScenarioGenerator(seed=42).generate_scenario(scenario_type, split="validation", seed=3100 + index)
        for index, scenario_type in enumerate(scenario_types)
    ]
    policies = [
        FixedRulePolicy(),
        GreedyScheduler(config["greedy"]["energy_cost_budget_usd_per_delivered_tonne"]),
        MILPScheduler(
            time_limit_seconds=config["milp"]["time_limit_seconds"],
            mip_gap=config["milp"]["mip_gap"],
        ),
    ]
    results = [evaluate_policy(policy, scenarios, episode_steps=24) for policy in policies]

    fixed_policy = FixedRulePolicy()
    fixed_actions = []
    for spec in scenarios:
        from src.environment.co2_transport_env import CO2TransportEnv
        env = CO2TransportEnv(scenario_parameters=spec.parameters)
        observation, _ = env.reset(seed=spec.seed)
        fixed_actions.append(fixed_policy.act(observation, env).action.tolist())

    greedy_policy = GreedyScheduler(config["greedy"]["energy_cost_budget_usd_per_delivered_tonne"])
    greedy_actions = []
    for spec in scenarios:
        from src.environment.co2_transport_env import CO2TransportEnv
        env = CO2TransportEnv(scenario_parameters=spec.parameters)
        observation, _ = env.reset(seed=spec.seed)
        decision = greedy_policy.act(observation, env)
        greedy_actions.append({"scenario_type": spec.scenario_type, "action": decision.action.tolist(), "diagnostics": decision.diagnostics})

    solver_statuses = [
        status
        for result in results if result["method"] == "milp"
        for episode in result["episodes"]
        for status in episode["milp_solver_statuses"]
    ]
    validation = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(output_dir),
        "evaluation_scenario_ids": [scenario.scenario_id for scenario in scenarios],
        "episode_steps": 24,
        "scenario_count": len(scenarios),
        "policies": {result["method"]: result for result in results},
        "fixed_policy_actions_by_scenario": fixed_actions,
        "fixed_policy_is_nonadaptive": len({tuple(action) for action in fixed_actions}) == 1,
        "greedy_actions_by_scenario": greedy_actions,
        "greedy_action_patterns": len({tuple(round(value, 6) for value in row["action"]) for row in greedy_actions}),
        "greedy_varies_across_scenario_conditions": len({tuple(round(value, 6) for value in row["action"]) for row in greedy_actions}) > 1,
        "milp_solver_statuses": solver_statuses,
        "milp_all_solutions_optimal": bool(solver_statuses) and all(status == "Optimal" for status in solver_statuses),
        "all_episodes_physically_valid": {
            result["method"]: all(episode["valid_episode"] for episode in result["episodes"])
            for result in results
        },
    }
    (output_dir / "phase3_baseline_validation.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
    print(json.dumps({
        "output_dir": str(output_dir),
        "aggregates": {result["method"]: result["aggregate"] for result in results},
        "fixed_policy_is_nonadaptive": validation["fixed_policy_is_nonadaptive"],
        "greedy_varies_across_scenario_conditions": validation["greedy_varies_across_scenario_conditions"],
        "milp_status_counts": {status: solver_statuses.count(status) for status in sorted(set(solver_statuses))},
        "all_episodes_physically_valid": validation["all_episodes_physically_valid"],
    }, indent=2))


if __name__ == "__main__":
    main()
