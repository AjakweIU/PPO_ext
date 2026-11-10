from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml


DEFAULT_RANGES = np.array(
    [
        [4.0e6, 2.5e7],
        [4.0e6, 2.5e7],
        [4.0e6, 2.5e7],
        [250.0, 350.0],
        [250.0, 350.0],
        [250.0, 350.0],
        [4.0e6, 2.5e7],
        [4.0e6, 2.5e7],
        [4.0e6, 2.5e7],
        [4.0e6, 2.5e7],
        [0.0, 8000.0],
        [0.0, 1.0],
        [4.0e6, 2.5e7],
        [4.0e6, 2.5e7],
        [0.0, 8000.0],
        [0.0, 1.0],
        [0.0, 5200.0],
        [0.0, 5200.0],
        [0.0, 5200.0],
        [0.0, 5200.0],
        [0.0, 5200.0],
        [0.0, 5000.0],
        [0.0, 5000.0],
        [0.0, 5000.0],
        [0.0, 15000.0],
        [0.0, 15000.0],
        [0.05, 0.525],
        [0.0, 23.0],
        [0.0, 6.0],
    ],
    dtype=float,
)


def default_observation_ranges() -> np.ndarray:
    return DEFAULT_RANGES.copy()


def _load_yaml_ranges(config_path: str | Path | None = None) -> np.ndarray:
    if config_path is None:
        return DEFAULT_RANGES.copy()

    path = Path(config_path)
    if not path.exists():
        return DEFAULT_RANGES.copy()

    with path.open("r", encoding="utf-8") as handle:
        content = yaml.safe_load(handle)

    range_list = content.get("observation", {}).get("ranges", None)
    if range_list is None:
        return DEFAULT_RANGES.copy()

    return np.asarray(range_list, dtype=float)


def normalize_observation(raw_observation: np.ndarray, ranges: np.ndarray | None = None, config_path: str | Path | None = None) -> np.ndarray:
    observation = np.asarray(raw_observation, dtype=float)
    if ranges is None:
        ranges = _load_yaml_ranges(config_path)

    if ranges.ndim == 1:
        if observation.shape != ranges.shape:
            raise ValueError(f"Observation shape {observation.shape} does not match ranges shape {ranges.shape}.")
        return np.clip(observation, 0.0, 1.0)

    if ranges.ndim != 2 or ranges.shape[1] != 2:
        raise ValueError(f"Range matrix must have shape (n_features, 2); received {ranges.shape}.")

    if observation.shape != (ranges.shape[0],):
        raise ValueError(f"Observation shape {observation.shape} does not match ranges shape {ranges.shape}.")

    lower = ranges[:, 0]
    upper = ranges[:, 1]
    width = upper - lower
    width[width == 0.0] = 1.0
    normalized = (observation - lower) / width
    return np.clip(normalized, 0.0, 1.0)


def denormalize_observation(normalized_observation: np.ndarray, ranges: np.ndarray | None = None, config_path: str | Path | None = None) -> np.ndarray:
    observation = np.asarray(normalized_observation, dtype=float)
    if ranges is None:
        ranges = _load_yaml_ranges(config_path)

    if ranges.ndim == 1:
        return np.clip(observation, 0.0, 1.0)

    if ranges.ndim != 2 or ranges.shape[1] != 2:
        raise ValueError(f"Range matrix must have shape (n_features, 2); received {ranges.shape}.")

    if observation.shape != (ranges.shape[0],):
        raise ValueError(f"Normalized observation shape {observation.shape} does not match ranges shape {ranges.shape}.")

    lower = ranges[:, 0]
    upper = ranges[:, 1]
    width = upper - lower
    width[width == 0.0] = 1.0
    return observation * width + lower
