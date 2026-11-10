from __future__ import annotations

import numpy as np


def validate_state(state: np.ndarray, expected_dim: int = 29) -> np.ndarray:
    array = np.asarray(state, dtype=float)
    if array.shape != (expected_dim,):
        raise ValueError(f"Expected a state vector of length {expected_dim}, received shape {array.shape}.")
    if not np.all(np.isfinite(array)):
        raise ValueError("State contains non-finite values.")
    return array


def check_capacity_limits(source_flow: np.ndarray, source_capacity: np.ndarray) -> tuple[bool, list[str]]:
    source_flow = np.asarray(source_flow, dtype=float)
    source_capacity = np.asarray(source_capacity, dtype=float)
    if source_flow.shape != source_capacity.shape:
        raise ValueError("Source flow and source capacity shapes must match.")

    reasons: list[str] = []
    for idx, (flow, cap) in enumerate(zip(source_flow, source_capacity)):
        if flow > cap + 1e-9:
            reasons.append(f"source_{idx}_capacity_exceeded")
    return len(reasons) == 0, reasons


def check_sink_limits(sink_flow: np.ndarray, sink_capacity: np.ndarray) -> tuple[bool, list[str]]:
    sink_flow = np.asarray(sink_flow, dtype=float)
    sink_capacity = np.asarray(sink_capacity, dtype=float)
    if sink_flow.shape != sink_capacity.shape:
        raise ValueError("Sink flow and sink capacity shapes must match.")

    reasons: list[str] = []
    for idx, (flow, cap) in enumerate(zip(sink_flow, sink_capacity)):
        if flow > cap + 1e-9:
            reasons.append(f"sink_{idx}_capacity_exceeded")
    return len(reasons) == 0, reasons


def check_pressure_limits(pressure_values: np.ndarray, min_pressure_pa: float, max_pressure_pa: float) -> tuple[bool, list[str]]:
    pressure_values = np.asarray(pressure_values, dtype=float)
    reasons: list[str] = []
    tolerance = max(abs(min_pressure_pa), abs(max_pressure_pa), 1.0) * 1e-12
    for idx, pressure in enumerate(pressure_values):
        if pressure < min_pressure_pa - tolerance:
            reasons.append(f"pressure_{idx}_below_min")
        if pressure > max_pressure_pa + tolerance:
            reasons.append(f"pressure_{idx}_above_max")
    return len(reasons) == 0, reasons


def project_action(action: np.ndarray, source_bounds: tuple[float, float] = (0.0, 1.0), compressor_bounds: tuple[float, float] = (0.75, 1.25)) -> tuple[np.ndarray, dict]:
    action_arr = np.asarray(action, dtype=float)
    if action_arr.shape != (7,):
        raise ValueError("Action vector must have length 7.")
    if not np.all(np.isfinite(action_arr)):
        raise ValueError("Action vector must contain only finite values.")

    projected = action_arr.copy()
    lower_source, upper_source = source_bounds
    lower_comp, upper_comp = compressor_bounds

    reasons: list[str] = []
    for idx in range(3):
        if projected[idx] < lower_source:
            projected[idx] = lower_source
            reasons.append(f"source_{idx}_clipped_lower")
        elif projected[idx] > upper_source:
            projected[idx] = upper_source
            reasons.append(f"source_{idx}_clipped_upper")

    for idx in range(3, 7):
        if projected[idx] < lower_comp:
            projected[idx] = lower_comp
            reasons.append(f"compressor_{idx - 3}_clipped_lower")
        elif projected[idx] > upper_comp:
            projected[idx] = upper_comp
            reasons.append(f"compressor_{idx - 3}_clipped_upper")

    return projected, {"reasons": reasons, "was_clipped": len(reasons) > 0}


def check_physical_constraints(
    source_flow: np.ndarray,
    sink_flow: np.ndarray,
    pressure_values: np.ndarray,
    source_capacity: np.ndarray,
    sink_capacity: np.ndarray,
    min_pressure_pa: float,
    max_pressure_pa: float,
) -> tuple[bool, dict]:
    source_ok, source_reasons = check_capacity_limits(source_flow, source_capacity)
    sink_ok, sink_reasons = check_sink_limits(sink_flow, sink_capacity)
    pressure_ok, pressure_reasons = check_pressure_limits(pressure_values, min_pressure_pa, max_pressure_pa)

    is_valid = source_ok and sink_ok and pressure_ok
    reasons = {
        "source": source_reasons,
        "sink": sink_reasons,
        "pressure": pressure_reasons,
    }
    return is_valid, reasons
