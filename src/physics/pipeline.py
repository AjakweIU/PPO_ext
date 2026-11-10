from __future__ import annotations

import math

import numpy as np

from src.physics.thermodynamics import estimate_chung_viscosity


def swamee_jain_friction_factor(reynolds_number: float, diameter_m: float, roughness_m: float = 4.6e-5) -> float:
    """Compute Darcy friction factor using the Swamee-Jain approximation."""
    if reynolds_number <= 0.0:
        raise ValueError("Reynolds number must be positive.")
    if diameter_m <= 0.0:
        raise ValueError("Pipeline diameter must be positive.")

    if reynolds_number < 2300.0:
        return 64.0 / reynolds_number

    relative_roughness = roughness_m / diameter_m
    log_term = math.log10((relative_roughness / 3.7) + (5.74 / (reynolds_number ** 0.9)))
    return 0.25 / (log_term ** 2)


def darcy_weisbach_pressure_drop(
    density_kg_m3: float,
    velocity_m_s: float,
    length_m: float,
    diameter_m: float,
    roughness_m: float = 4.6e-5,
    gravity_m_s2: float = 9.81,
    elevation_change_m: float = 0.0,
    viscosity_pa_s: float | None = None,
) -> float:
    """Compute pressure drop using Darcy-Weisbach, including elevation head if provided."""
    if density_kg_m3 <= 0.0:
        raise ValueError("Density must be positive.")
    if diameter_m <= 0.0:
        raise ValueError("Diameter must be positive.")
    if length_m < 0.0:
        raise ValueError("Length must be non-negative.")

    velocity_mag = abs(float(velocity_m_s))
    if viscosity_pa_s is None:
        viscosity_pa_s = estimate_chung_viscosity(300.0, density_kg_m3)

    reynolds_number = density_kg_m3 * velocity_mag * diameter_m / viscosity_pa_s
    friction_factor = swamee_jain_friction_factor(reynolds_number, diameter_m, roughness_m)

    pressure_drop = friction_factor * (length_m / diameter_m) * (density_kg_m3 * velocity_mag**2 / 2.0)
    pressure_drop += density_kg_m3 * gravity_m_s2 * elevation_change_m
    return pressure_drop


def compute_flow_velocity(flow_rate_kg_s: float, density_kg_m3: float, diameter_m: float) -> float:
    if density_kg_m3 <= 0.0 or diameter_m <= 0.0:
        raise ValueError("Density and diameter must be positive.")
    area = math.pi * (diameter_m / 2.0) ** 2
    return abs(flow_rate_kg_s) / (density_kg_m3 * area)
