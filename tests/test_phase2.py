import numpy as np

from src.environment.reward import DEFAULT_REWARD_BOUNDS, RewardNormalizer, compute_reward
from src.scenarios.domain_randomization import DEFAULT_DOMAIN_RANGES, randomize_network_parameters
from src.scenarios.generator import ScenarioGenerator
from src.scenarios.splits import split_scenarios


def test_seeded_scenario_reproduction():
    gen_a = ScenarioGenerator(seed=42)
    gen_b = ScenarioGenerator(seed=42)
    scenario_a = gen_a.generate_scenario("normal")
    scenario_b = gen_b.generate_scenario("normal")
    assert scenario_a == scenario_b


def test_train_validation_test_non_overlap():
    generator = ScenarioGenerator(seed=42)
    train, val, test = split_scenarios(generator, n_train=12, n_val=3, n_test=3)
    train_ids = {s.scenario_id for s in train}
    val_ids = {s.scenario_id for s in val}
    test_ids = {s.scenario_id for s in test}
    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)


def test_scenario_perturbation_ranges():
    rngs = DEFAULT_DOMAIN_RANGES
    params = randomize_network_parameters(rngs, seed=7)
    assert params["source_capacity"] >= 500.0
    assert params["source_capacity"] <= 5000.0
    assert params["sink_injection_rate"] >= 1000.0
    assert params["sink_injection_rate"] <= 10000.0
    assert params["pipeline_length_km"] >= 10.0
    assert params["pipeline_length_km"] <= 100.0
    assert params["pipeline_diameter_m"] >= 0.3
    assert params["pipeline_diameter_m"] <= 0.8
    assert params["compressor_power_kw"] >= 3000.0
    assert params["compressor_power_kw"] <= 8000.0
    assert 0.75 <= params["compressor_efficiency"] <= 0.88
    assert 0.05 <= params["base_energy_price"] <= 0.12
    assert 1.5 <= params["peak_price_multiplier"] <= 3.0
    assert 0.001 <= params["leakage_rate"] <= 0.02
    assert 0.0 <= params["failure_probability"] <= 0.05


def test_ood_scenario_generation():
    generator = ScenarioGenerator(seed=99)
    ood = generator.generate_ood_scenario("high_demand")
    assert ood.is_ood is True
    assert ood.parameters["demand_multiplier"] > 1.0
    assert ood.parameters["source_outage_probability"] >= 0.0


def test_reward_decomposition_and_total():
    reward = compute_reward(
        throughput=0.8,
        energy_cost=0.2,
        leakage=0.1,
        violation=0.05,
        weights={"throughput": 1.0, "energy": 0.1, "leakage": 10.0, "violation": 5.0},
        bounds={
            "throughput": (0.0, 1.0),
            "energy": (0.0, 1.0),
            "leakage": (0.0, 1.0),
            "violation": (0.0, 1.0),
        },
    )
    assert np.isclose(reward.total_reward, reward.component_contributions["throughput"] + reward.component_contributions["energy"] + reward.component_contributions["leakage"] + reward.component_contributions["violation"])
    assert np.isfinite(reward.total_reward)


def test_reward_component_normalization():
    normalizer = RewardNormalizer()
    scaled = normalizer.normalize_component("throughput", 0.6, bounds=(0.0, 1.0))
    assert 0.0 <= scaled <= 1.0
    assert normalizer.normalize_component("energy", 0.0, bounds=(0.0, 1.0)) == 0.0
    assert normalizer.normalize_component("leakage", 1.0, bounds=(0.0, 1.0)) == 1.0
    assert np.isclose(DEFAULT_REWARD_BOUNDS["energy"][1], 1651.7936572613246)
