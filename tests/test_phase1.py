import numpy as np

from src.environment.co2_transport_env import CO2TransportEnv
from src.environment.constraints import check_capacity_limits, check_physical_constraints, check_pressure_limits, project_action
from src.environment.normalization import default_observation_ranges, normalize_observation
from src.physics.compressor import compressor_efficiency, compressor_power, pressure_ratio
from src.physics.pipeline import darcy_weisbach_pressure_drop, swamee_jain_friction_factor
from src.physics.thermodynamics import compute_supercritical_properties, density_from_peng_robinson, solve_z_factor


def test_peng_robinson_density_and_z():
    z = solve_z_factor(300.0, 8.0e6)
    density = density_from_peng_robinson(300.0, 8.0e6, z_factor=z)
    assert np.isfinite(z)
    assert z > 0.0
    assert density > 0.0


def test_supercritical_properties_dict():
    props = compute_supercritical_properties(300.0, 8.0e6)
    assert all(key in props for key in ["temperature_k", "pressure_pa", "z_factor", "density_kg_m3", "viscosity_pa_s"])
    assert props["density_kg_m3"] > 0.0


def test_pipeline_pressure_drop_and_friction():
    friction = swamee_jain_friction_factor(1.0e5, 0.5, roughness_m=4.6e-5)
    drop = darcy_weisbach_pressure_drop(
        density_kg_m3=800.0,
        velocity_m_s=1.5,
        length_m=1000.0,
        diameter_m=0.5,
        roughness_m=4.6e-5,
        elevation_change_m=0.0,
    )
    assert np.isfinite(friction)
    assert friction > 0.0
    assert drop > 0.0


def test_compressor_equations():
    ratio = pressure_ratio(5.0e6, 9.0e6)
    eta = compressor_efficiency(design_efficiency=0.82, flow_rate=1.2, design_flow=1.0)
    power = compressor_power(
        mass_flow_kg_s=1.0,
        inlet_pressure_pa=5.0e6,
        outlet_pressure_pa=9.0e6,
        inlet_temperature_k=300.0,
        efficiency=eta,
    )
    assert ratio > 1.0
    assert 0.75 <= eta <= 0.88
    assert power > 0.0


def test_constraints_and_projection():
    valid, reasons = check_capacity_limits(np.array([100.0, 200.0]), np.array([150.0, 250.0]))
    assert valid
    assert reasons == []

    valid2, reasons2 = check_pressure_limits(np.array([3e6, 1.5e7]), 4e6, 2.5e7)
    assert not valid2
    assert len(reasons2) >= 1

    projected, info = project_action(np.array([1.5, -0.1, 0.5, 0.5, 2.0, 1.0, 1.5]))
    assert projected.shape == (7,)
    assert np.all(projected >= np.array([0.0, 0.0, 0.0, 0.75, 0.75, 0.75, 0.75]))
    assert np.all(projected <= np.array([1.0, 1.0, 1.0, 1.25, 1.25, 1.25, 1.25]))
    assert info["was_clipped"]


def test_normalization_ranges_and_shapes():
    ranges = default_observation_ranges()
    obs = np.linspace(0.0, 1.0, num=29)
    normalized = normalize_observation(obs, ranges=ranges)
    assert normalized.shape == (29,)
    assert np.all((normalized >= 0.0) & (normalized <= 1.0))


def test_env_reset_and_step_shapes():
    env = CO2TransportEnv()
    obs, info = env.reset(seed=42)
    assert obs.shape == (29,)
    assert np.all((obs >= 0.0) & (obs <= 1.0))
    assert info["valid"]
    assert info["constraint_check"]["pressure"] == []
    assert np.all(env.state[[0, 1, 2, 6, 7]] >= env.min_pressure_pa)
    assert np.all(env.state[[0, 1, 2, 6, 7]] <= env.max_pressure_pa)

    action = np.array([0.3, 0.4, 0.5, 0.8, 0.9, 1.0, 1.1], dtype=float)
    next_obs, reward, terminated, truncated, info = env.step(action)
    assert next_obs.shape == (29,)
    assert isinstance(reward, float)
    assert terminated in (True, False)
    assert truncated in (True, False)
    assert info["raw_action"].shape == (7,)


def test_env_physical_constraints_accept_valid_action():
    env = CO2TransportEnv()
    env.reset(seed=7)
    action = np.array([0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0], dtype=float)
    obs, reward, terminated, truncated, info = env.step(action)
    assert info["valid"] in (True, False)
    assert obs.shape == (29,)


def test_observation_group_mapping_and_action_coupling():
    env = CO2TransportEnv()
    obs, reset_info = env.reset(seed=11)
    assert reset_info["valid"]
    assert env.state.shape == (29,)
    assert np.allclose(env.state[8:16], [1.8e7, 1.8e7, 0.0, 0.0, 2.0e7, 2.0e7, 0.0, 0.0])
    assert np.allclose(env.state[21:24], env.source_capacity)
    assert np.allclose(env.state[24:26], env.sink_capacity)
    assert np.allclose(env.state[26:29], [0.08, 0.0, 0.0])

    _, _, _, _, low_info = env.step([0.2, 0.2, 0.2, 0.75, 0.75, 0.75, 0.75])
    low_delivery = low_info["delivered_flow"]
    env.reset(seed=11)
    _, _, _, _, high_info = env.step([1.0, 1.0, 1.0, 1.25, 1.25, 1.25, 1.25])
    assert high_info["delivered_flow"] > low_delivery
    assert high_info["delivered_flow"] <= np.sum(env.sink_capacity)
    assert env.state[26] == 0.08
    assert env.state[27] == 1.0
    assert env.state[28] == 0.0

    env.reset(seed=11)
    _, _, _, _, low_compressor_info = env.step([0.5, 0.5, 0.5, 0.75, 0.75, 0.75, 0.75])
    low_compressor_power = float(np.sum(low_compressor_info["compressor_power_kw"]))
    env.reset(seed=11)
    _, _, _, _, high_compressor_info = env.step([0.5, 0.5, 0.5, 1.25, 1.25, 1.25, 1.25])
    assert high_compressor_info["delivered_flow"] > low_compressor_info["delivered_flow"]
    assert float(np.sum(high_compressor_info["compressor_power_kw"])) > low_compressor_power
    assert env.state[10] > 0.0 or env.state[14] > 0.0
