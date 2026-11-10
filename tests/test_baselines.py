import numpy as np
import yaml

from src.baselines import FixedRulePolicy, GreedyScheduler, MILPScheduler
from src.environment.co2_transport_env import CO2TransportEnv
from src.evaluation.baseline_metrics import evaluate_policy
from src.scenarios.generator import ScenarioGenerator


def test_fixed_rule_uses_methodology_action_without_adaptation():
    policy = FixedRulePolicy()
    generator = ScenarioGenerator(seed=42)
    actions = []
    for index, scenario_type in enumerate(["normal", "high_demand", "energy_price_spike"]):
        spec = generator.generate_scenario(scenario_type, seed=100 + index)
        env = CO2TransportEnv(scenario_parameters=spec.parameters)
        observation, _ = env.reset(seed=spec.seed)
        actions.append(policy.act(observation, env).action)
    assert all(np.array_equal(action, np.asarray([0.7, 0.7, 0.7, 1.0, 1.0, 1.0, 1.0])) for action in actions)


def test_greedy_changes_with_demand_and_energy_price():
    generator = ScenarioGenerator(seed=4)
    normal = generator.generate_scenario("normal", seed=4)
    high_demand = generator.generate_scenario("high_demand", seed=4)
    low_price_parameters = dict(normal.parameters)
    high_price_parameters = dict(normal.parameters)
    low_price_parameters["base_energy_price"] = 0.05
    low_price_parameters["peak_price_multiplier"] = 1.0
    high_price_parameters["base_energy_price"] = 0.15
    high_price_parameters["peak_price_multiplier"] = 3.5

    def get_action(parameters, seed=4):
        env = CO2TransportEnv(scenario_parameters=parameters)
        observation, _ = env.reset(seed=seed)
        return GreedyScheduler().act(observation, env).action

    assert not np.array_equal(get_action(normal.parameters), get_action(high_demand.parameters))
    assert not np.array_equal(get_action(low_price_parameters), get_action(high_price_parameters))


def test_milp_returns_solver_status_and_schedule():
    spec = ScenarioGenerator(seed=8).generate_scenario("normal", split="test", seed=8)
    env = CO2TransportEnv(scenario_parameters=spec.parameters)
    observation, _ = env.reset(seed=spec.seed)
    decision = MILPScheduler().act(observation, env)
    assert decision.diagnostics["solver_status"] == "Optimal"
    assert decision.diagnostics["objective_value"] is not None
    assert decision.diagnostics["solve_time_seconds"] >= 0.0
    assert decision.action.shape == (7,)
    assert np.all(np.isfinite(decision.action))


def test_common_evaluation_records_metrics_and_timing():
    spec = ScenarioGenerator(seed=15).generate_scenario("normal", split="validation", seed=15)
    result = evaluate_policy(FixedRulePolicy(), [spec], episode_steps=6)
    episode = result["episodes"][0]
    for key in [
        "reward", "throughput_tonnes", "energy_consumption_kwh", "energy_cost_usd",
        "simulated_leakage_tonnes", "demand_shortfall_tonnes", "physical_violations",
        "action_corrections", "mean_decision_time_ms", "median_decision_time_ms", "p95_decision_time_ms",
    ]:
        assert key in episode
    assert episode["mean_decision_time_ms"] >= 0.0
    assert episode["median_decision_time_ms"] >= 0.0
    assert episode["p95_decision_time_ms"] >= 0.0


def test_baseline_configuration_declares_fixed_and_greedy_rules():
    with open("configs/baselines.yaml", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    assert config["fixed_rule"]["source_capture_fraction"] == 0.7
    assert config["fixed_rule"]["compressor_control_midpoint"] == 1.0
    assert config["greedy"]["energy_cost_budget_usd_per_delivered_tonne"] > 0.0
