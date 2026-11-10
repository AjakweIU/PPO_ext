from __future__ import annotations

from typing import Iterable

from src.scenarios.generator import ScenarioGenerator, ScenarioSpec

DEFAULT_SPLIT_SEEDS = {
    "train": 42,
    "validation": 123,
    "test": 456,
}


def split_scenarios(
    generator: ScenarioGenerator | None = None,
    n_train: int = 70,
    n_val: int = 15,
    n_test: int = 15,
    scenario_types: Iterable[str] | None = None,
) -> tuple[list[ScenarioSpec], list[ScenarioSpec], list[ScenarioSpec]]:
    generator = generator or ScenarioGenerator(seed=42)
    scenario_types = tuple(scenario_types or ["normal", "high_demand", "low_supply", "equipment_failure", "energy_price_spike"])

    train: list[ScenarioSpec] = []
    val: list[ScenarioSpec] = []
    test: list[ScenarioSpec] = []

    for idx in range(n_train):
        scenario_name = scenario_types[idx % len(scenario_types)]
        train.append(generator.generate_scenario(scenario_name, split="train", is_ood=False, seed=DEFAULT_SPLIT_SEEDS["train"] + idx * 7))

    for idx in range(n_val):
        scenario_name = scenario_types[idx % len(scenario_types)]
        val.append(generator.generate_scenario(scenario_name, split="validation", is_ood=False, seed=DEFAULT_SPLIT_SEEDS["validation"] + idx * 11))

    for idx in range(n_test):
        scenario_name = scenario_types[idx % len(scenario_types)]
        test.append(generator.generate_scenario(scenario_name, split="test", is_ood=False, seed=DEFAULT_SPLIT_SEEDS["test"] + idx * 13))

    train_ids = {s.scenario_id for s in train}
    val_ids = {s.scenario_id for s in val}
    test_ids = {s.scenario_id for s in test}
    if not (train_ids.isdisjoint(val_ids) and train_ids.isdisjoint(test_ids) and val_ids.isdisjoint(test_ids)):
        raise ValueError("Train/validation/test scenario IDs overlap; split policy failed.")
    return train, val, test


def generate_ood_scenarios(
    generator: ScenarioGenerator | None = None,
    n_scenarios: int = 10,
    scenario_types: Iterable[str] | None = None,
) -> list[ScenarioSpec]:
    generator = generator or ScenarioGenerator(seed=999)
    scenario_types = tuple(scenario_types or ["high_demand", "low_supply", "equipment_failure", "energy_price_spike"])
    out: list[ScenarioSpec] = []
    for idx in range(n_scenarios):
        scenario_name = scenario_types[idx % len(scenario_types)]
        out.append(generator.generate_ood_scenario(scenario_name, split="ood", seed=9000 + idx * 101))
    return out
