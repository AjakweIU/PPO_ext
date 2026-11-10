from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

DEFAULT_REWARD_WEIGHTS = {
    "throughput": 1.0,
    "energy": 0.1,
    "leakage": 10.0,
    "violation": 5.0,
}

DEFAULT_REWARD_BOUNDS = {
    "throughput": (0.0, 4125.0 / 24.0),
    "energy": (0.0, 1651.7936572613246),
    "leakage": (0.0, 4125.0 / 24.0 * 0.02),
    "violation": (0.0, 1.0),
}


class RewardNormalizer:
    def normalize_component(self, component_name: str, value: float, bounds: tuple[float, float] | None = None) -> float:
        if bounds is None:
            bounds = DEFAULT_REWARD_BOUNDS.get(component_name, (0.0, 1.0))
        lower, upper = bounds
        if upper <= lower:
            return 0.0
        normalized = (value - lower) / (upper - lower)
        return float(max(0.0, min(1.0, normalized)))


@dataclass(frozen=True)
class RewardBreakdown:
    component_values: dict[str, float]
    component_contributions: dict[str, float]
    normalized_values: dict[str, float]
    total_reward: float

    def as_log(self) -> dict[str, float]:
        log_payload = dict(self.component_values)
        log_payload["total_reward"] = self.total_reward
        return log_payload


def compute_reward(
    throughput: float,
    energy_cost: float,
    leakage: float,
    violation: float,
    weights: dict[str, float] | None = None,
    bounds: dict[str, tuple[float, float]] | None = None,
) -> RewardBreakdown:
    weights = DEFAULT_REWARD_WEIGHTS if weights is None else weights
    bounds = DEFAULT_REWARD_BOUNDS if bounds is None else bounds
    normalizer = RewardNormalizer()

    normalized = {
        "throughput": normalizer.normalize_component("throughput", throughput, bounds.get("throughput", (0.0, 1.0))),
        "energy": normalizer.normalize_component("energy", energy_cost, bounds.get("energy", (0.0, 1.0))),
        "leakage": normalizer.normalize_component("leakage", leakage, bounds.get("leakage", (0.0, 1.0))),
        "violation": normalizer.normalize_component("violation", violation, bounds.get("violation", (0.0, 1.0))),
    }

    component_contributions = {
        "throughput": weights.get("throughput", 1.0) * normalized["throughput"],
        "energy": -weights.get("energy", 0.1) * normalized["energy"],
        "leakage": -weights.get("leakage", 10.0) * normalized["leakage"],
        "violation": -weights.get("violation", 5.0) * normalized["violation"],
    }

    total_reward = sum(component_contributions.values())
    component_values = {
        "throughput": normalized["throughput"],
        "energy": normalized["energy"],
        "leakage": normalized["leakage"],
        "violation": normalized["violation"],
    }
    return RewardBreakdown(
        component_values=component_values,
        component_contributions=component_contributions,
        normalized_values=normalized,
        total_reward=float(total_reward),
    )


def normalize_reward_components(raw_values: dict[str, float], bounds: dict[str, tuple[float, float]] | None = None) -> dict[str, float]:
    bounds = DEFAULT_REWARD_BOUNDS if bounds is None else bounds
    normalizer = RewardNormalizer()
    return {name: normalizer.normalize_component(name, value, bounds.get(name, (0.0, 1.0))) for name, value in raw_values.items()}
