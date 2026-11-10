from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np


@dataclass(frozen=True)
class PolicyDecision:
    action: np.ndarray
    diagnostics: dict[str, Any] = field(default_factory=dict)


class SchedulingPolicy(Protocol):
    name: str

    def act(self, observation: np.ndarray, env: Any) -> PolicyDecision:
        """Return a bounded continuous action using only current environment state."""
