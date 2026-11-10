from __future__ import annotations

import copy
import random
from dataclasses import asdict, dataclass
from typing import Any

from src.scenarios.domain_randomization import DEFAULT_DOMAIN_RANGES, randomize_network_parameters


@dataclass(frozen=True)
class ScenarioSpec:
    scenario_id: str
    scenario_type: str
    split: str
    is_ood: bool
    seed: int
    parameters: dict[str, float]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class ScenarioGenerator:
    def __init__(self, seed: int = 42):
        self.seed = int(seed)
        self._rng = random.Random(self.seed)

    def generate_scenario(self, scenario_type: str, split: str = "train", is_ood: bool = False, seed: int | None = None) -> ScenarioSpec:
        scenario_seed = self.seed if seed is None else int(seed)
        rng = random.Random(scenario_seed)
        params = randomize_network_parameters(seed=scenario_seed)

        scenario_type = scenario_type.lower().replace(" ", "_")
        switch = {
            "normal": self._normal_scenario,
            "high_demand": self._high_demand_scenario,
            "low_supply": self._low_supply_scenario,
            "equipment_failure": self._equipment_failure_scenario,
            "energy_price_spike": self._energy_price_spike_scenario,
        }
        if scenario_type not in switch:
            raise ValueError(f"Unsupported scenario type: {scenario_type}")

        params = switch[scenario_type](params, rng, is_ood=is_ood)
        scenario_id = f"{split}_{scenario_type}_{scenario_seed}"
        return ScenarioSpec(
            scenario_id=scenario_id,
            scenario_type=scenario_type,
            split=split,
            is_ood=bool(is_ood),
            seed=scenario_seed,
            parameters=params,
        )

    def generate_ood_scenario(self, scenario_type: str, split: str = "test", seed: int | None = None) -> ScenarioSpec:
        scenario_seed = self.seed + 10000 if seed is None else int(seed)
        return self.generate_scenario(scenario_type, split=split, is_ood=True, seed=scenario_seed)

    def _normal_scenario(self, params: dict[str, float], rng: random.Random, is_ood: bool = False) -> dict[str, float]:
        params["demand_multiplier"] = 1.0
        params["supply_multiplier"] = 1.0
        params["base_energy_price"] = 0.08
        params["peak_price_multiplier"] = 2.0
        params["failure_probability"] = 0.01
        params["source_outage_probability"] = 0.0
        params["compressor_outage_probability"] = 0.0
        params["leakage_rate"] = 0.005
        return params

    def _high_demand_scenario(self, params: dict[str, float], rng: random.Random, is_ood: bool = False) -> dict[str, float]:
        params["demand_multiplier"] = 1.5 if not is_ood else 2.0 + rng.uniform(0.0, 0.8)
        params["sink_injection_rate"] = params["sink_injection_rate"] * params["demand_multiplier"]
        params["supply_multiplier"] = 1.0
        params["failure_probability"] = 0.01
        params["source_outage_probability"] = 0.0
        params["compressor_outage_probability"] = 0.0
        if is_ood:
            params["sink_injection_rate"] = min(params["sink_injection_rate"], 15000.0)
        return params

    def _low_supply_scenario(self, params: dict[str, float], rng: random.Random, is_ood: bool = False) -> dict[str, float]:
        params["supply_multiplier"] = 0.7 if not is_ood else 0.35 + rng.uniform(0.0, 0.2)
        params["source_capacity"] = params["source_capacity"] * params["supply_multiplier"]
        params["source_outage_probability"] = 0.05 if is_ood else 0.02
        params["compressor_outage_probability"] = 0.0
        params["leakage_rate"] = 0.01
        return params

    def _equipment_failure_scenario(self, params: dict[str, float], rng: random.Random, is_ood: bool = False) -> dict[str, float]:
        params["failure_probability"] = 0.03 if not is_ood else 0.05 + rng.uniform(0.0, 0.02)
        params["compressor_efficiency"] = max(0.75, params["compressor_efficiency"] - 0.08)
        params["source_capacity"] = params["source_capacity"] * (0.9 if not is_ood else 0.7)
        params["source_outage_probability"] = 0.06 if is_ood else 0.03
        params["compressor_outage_probability"] = 0.08 if is_ood else 0.04
        return params

    def _energy_price_spike_scenario(self, params: dict[str, float], rng: random.Random, is_ood: bool = False) -> dict[str, float]:
        params["base_energy_price"] = 0.10 if not is_ood else 0.12 + rng.uniform(0.0, 0.03)
        params["peak_price_multiplier"] = 2.5 if not is_ood else 3.0 + rng.uniform(0.0, 0.5)
        params["source_outage_probability"] = 0.0
        params["compressor_outage_probability"] = 0.0
        params["leakage_rate"] = 0.006
        return params


def generate_scenario(scenario_type: str, seed: int = 42, split: str = "train", is_ood: bool = False) -> ScenarioSpec:
    generator = ScenarioGenerator(seed=seed)
    return generator.generate_scenario(scenario_type=scenario_type, split=split, is_ood=is_ood, seed=seed)
