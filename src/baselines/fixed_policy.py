from __future__ import annotations

from typing import Any

import numpy as np

from .base import PolicyDecision


class FixedRulePolicy:
    """Non-adaptive 70% source capture, with compressor controls at their midpoint."""

    name = "fixed_rule"

    def act(self, observation: np.ndarray, env: Any) -> PolicyDecision:
        action = np.asarray([0.70, 0.70, 0.70, 1.0, 1.0, 1.0, 1.0], dtype=float)
        return PolicyDecision(action=action, diagnostics={"rule": "fixed_70_percent_capture_midpoint_compressors"})
