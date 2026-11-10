import numpy as np
import pytest

from src.environment.co2_transport_env import CO2TransportEnv


def run_once(env, action, seed=13):
    env.reset(seed=seed)
    return env.step(action)


def test_compressor_pressure_ratio_increases_physical_power():
    env_low = CO2TransportEnv()
    _, _, _, _, low = run_once(env_low, [0.5, 0.5, 0.5, 0.75, 1.25, 0.75, 1.25])
    env_high = CO2TransportEnv()
    _, _, _, _, high = run_once(env_high, [0.5, 0.5, 0.5, 1.25, 1.25, 1.25, 1.25])
    assert np.all(high["compressor_pressure_ratios"] >= low["compressor_pressure_ratios"])
    assert np.sum(high["compressor_power_kw"]) > np.sum(low["compressor_power_kw"])


def test_energy_price_changes_cost_not_physical_energy():
    low_price = CO2TransportEnv(scenario_parameters={"base_energy_price": 0.05, "peak_price_multiplier": 1.0})
    high_price = CO2TransportEnv(scenario_parameters={"base_energy_price": 0.10, "peak_price_multiplier": 2.0})
    action = [0.5, 0.5, 0.5, 1.1, 1.25, 1.1, 1.25]
    _, _, _, _, low = run_once(low_price, action)
    _, _, _, _, high = run_once(high_price, action)
    assert high["energy_kwh"] == pytest.approx(low["energy_kwh"])
    assert high["energy_cost"] > low["energy_cost"]


def test_source_and_compressor_actions_increase_feasible_delivery():
    low_env = CO2TransportEnv()
    high_env = CO2TransportEnv()
    _, _, _, _, low = run_once(low_env, [0.2, 0.2, 0.2, 1.0, 0.75, 1.0, 0.75])
    _, _, _, _, high = run_once(high_env, [1.0, 1.0, 1.0, 1.0, 1.25, 1.0, 1.25])
    assert high["delivered_flow"] > low["delivered_flow"]
    assert np.sum(high["sink_received_tonnes"]) > np.sum(low["sink_received_tonnes"])


def test_source_availability_and_scenario_demand_are_dynamic_state():
    env = CO2TransportEnv(scenario_parameters={"sink_injection_rate": 9000.0, "demand_multiplier": 1.8})
    env.reset(seed=29)
    initial_available = float(np.sum(env.state[21:24]))
    _, _, _, _, first = env.step([0.8, 0.8, 0.8, 0.75, 1.25, 0.75, 1.25])
    assert np.sum(env.state[24:26]) > np.sum(env.sink_capacity)
    assert float(np.sum(env.state[21:24])) == pytest.approx(initial_available)
    assert np.sum(first["source_headroom_rate"]) < initial_available
    assert np.sum(first["sink_received_tonnes"]) > 0.0
    assert np.sum(first["capacity_shortfall_tonnes"]) > 0.0
    assert first["demand_shortfall_fraction"] > 0.0
    assert first["violation"] == 0.0
    assert first["weighted_violation"] == 0.0
    assert first["violation_components"] == {
        "action_correction": 0.0,
        "pressure_violation": 0.0,
        "pressure_ratio_violation": 0.0,
        "capacity_violation": 0.0,
    }


def test_unsafe_action_is_projected_and_penalized():
    env = CO2TransportEnv()
    _, _, _, _, info = run_once(env, [1.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0])
    assert info["correction"]["was_clipped"]
    assert info["violation"] > 0.0
    assert info["weighted_violation"] < 0.0


def test_compressor_failure_reduces_available_transport():
    healthy = CO2TransportEnv()
    failed = CO2TransportEnv(scenario_parameters={"compressor_outage_probability": 1.0})
    action = [1.0, 1.0, 1.0, 1.25, 1.25, 1.25, 1.25]
    _, _, _, _, healthy_info = run_once(healthy, action, seed=17)
    _, _, _, _, failed_info = run_once(failed, action, seed=17)
    assert failed_info["failed_compressors"] == [0, 1]
    assert failed_info["delivered_flow"] < healthy_info["delivered_flow"]


def test_source_failure_is_visible_from_reset():
    env = CO2TransportEnv(scenario_parameters={"source_outage_probability": 1.0})
    _, info = env.reset(seed=31)
    assert info["failed_sources"] == [0, 1, 2]
    assert np.all(env.state[21:24] == 0.0)


def test_pipeline_channels_are_segment_specific_and_failure_is_observable():
    env = CO2TransportEnv()
    _, _, _, _, healthy = run_once(env, [0.2, 0.8, 0.6, 1.0, 1.25, 1.0, 1.25])
    assert len(set(np.round(healthy["segment_flows"], 6))) > 1
    failed_env = CO2TransportEnv(scenario_parameters={"pipeline_outage_probability": 1.0})
    _, _, _, _, failed = run_once(failed_env, [0.2, 0.8, 0.6, 1.0, 1.25, 1.0, 1.25])
    assert failed["failed_segments"] == [0, 1, 2, 3, 4]
    assert failed["delivered_flow"] == 0.0


def test_leakage_scales_with_transport_and_scenario_rate():
    low_leak = CO2TransportEnv(scenario_parameters={"leakage_rate": 0.001})
    high_leak = CO2TransportEnv(scenario_parameters={"leakage_rate": 0.02})
    action = [0.8, 0.8, 0.8, 1.0, 1.25, 1.0, 1.25]
    _, _, _, _, low = run_once(low_leak, action)
    _, _, _, _, high = run_once(high_leak, action)
    assert low["leakage_tonnes"] > 0.0
    assert high["leakage_tonnes"] > low["leakage_tonnes"]
    _, _, _, _, low_flow = run_once(CO2TransportEnv(scenario_parameters={"leakage_rate": 0.01}), [0.2, 0.2, 0.2, 1.0, 1.25, 1.0, 1.25])
    _, _, _, _, high_flow = run_once(CO2TransportEnv(scenario_parameters={"leakage_rate": 0.01}), [0.9, 0.9, 0.9, 1.0, 1.25, 1.0, 1.25])
    assert high_flow["leakage_tonnes"] > low_flow["leakage_tonnes"]


def test_step_reward_is_exact_weighted_decomposition():
    env = CO2TransportEnv()
    _, reward, _, _, info = run_once(env, [0.7, 0.7, 0.7, 1.1, 1.2, 1.1, 1.2])
    assert reward == pytest.approx(sum(info["weighted_contributions"].values()))
    assert info["final_reward"] == pytest.approx(reward)
    assert info["raw_throughput"] >= 0.0
    assert info["normalized_throughput"] >= 0.0
    assert info["raw_energy"] >= 0.0
    assert info["normalized_energy"] >= 0.0
    assert info["raw_leakage"] >= 0.0
    assert info["normalized_leakage"] >= 0.0


def test_below_minimum_pressure_produces_nonzero_violation_penalty():
    env = CO2TransportEnv()
    env.reset(seed=19)
    env.state[0] = 3.0e6
    _, reward, _, _, info = env.step([0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0])
    assert not info["physical_valid"]
    assert info["violation"] > 0.0
    assert info["pressure_violation_fraction"] > 0.0
    assert info["weighted_violation"] < 0.0
    assert reward == pytest.approx(sum(info["weighted_contributions"].values()))


def test_non_finite_action_is_rejected():
    env = CO2TransportEnv()
    env.reset(seed=23)
    with pytest.raises(ValueError, match="finite"):
        env.step([np.nan, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0])
