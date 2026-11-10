from __future__ import annotations

import random
from typing import Any

DEFAULT_DOMAIN_RANGES: dict[str, tuple[float, float]] = {
    "source_capacity": (500.0, 5000.0),
    "sink_injection_rate": (1000.0, 10000.0),
    "pipeline_length_km": (10.0, 100.0),
    "pipeline_diameter_m": (0.3, 0.8),
    "compressor_power_kw": (3000.0, 8000.0),
    "compressor_efficiency": (0.75, 0.88),
    "base_energy_price": (0.05, 0.12),
    "peak_price_multiplier": (1.5, 3.0),
    "leakage_rate": (0.001, 0.02),
    "failure_probability": (0.0, 0.05),
    "demand_multiplier": (0.8, 1.2),
    "supply_multiplier": (0.8, 1.2),
}


def randomize_network_parameters(ranges: dict[str, tuple[float, float]] | None = None, seed: int | None = None) -> dict[str, float]:
    rng = random.Random(seed)
    selected_ranges = DEFAULT_DOMAIN_RANGES if ranges is None else ranges
    params: dict[str, float] = {}
    for key, (lower, upper) in selected_ranges.items():
        params[key] = rng.uniform(lower, upper)
    return params


def randomize_scenario_vector(base_values: dict[str, float], perturbation: dict[str, float], seed: int | None = None) -> dict[str, float]:
    rng = random.Random(seed)
    output = dict(base_values)
    for key, magnitude in perturbation.items():
        if key not in output:
            output[key] = 0.0
        output[key] = float(output[key] * (1.0 + rng.uniform(-magnitude, magnitude)))
    return output
