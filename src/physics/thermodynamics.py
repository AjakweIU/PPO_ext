from __future__ import annotations

import math
from typing import Iterable

import numpy as np

R_UNIVERSAL = 8.314462618  # J mol^-1 K^-1

CO2_CRITICAL_TEMPERATURE_K = 304.13
CO2_CRITICAL_PRESSURE_PA = 7.377e6
CO2_ACENTRIC_FACTOR = 0.225
CO2_MOLAR_MASS_KG_PER_MOL = 0.04401


def _peng_robinson_a_and_b(temperature_k: float, pressure_pa: float) -> tuple[float, float]:
    if temperature_k <= 0.0 or pressure_pa <= 0.0:
        raise ValueError("Temperature and pressure must be positive.")

    kappa = 0.37464 + 1.54226 * CO2_ACENTRIC_FACTOR - 0.26992 * CO2_ACENTRIC_FACTOR ** 2
    alpha = (1.0 + kappa * (1.0 - math.sqrt(temperature_k / CO2_CRITICAL_TEMPERATURE_K))) ** 2
    a = 0.45724 * (R_UNIVERSAL**2 * CO2_CRITICAL_TEMPERATURE_K**2 / CO2_CRITICAL_PRESSURE_PA) * alpha
    b = 0.07780 * R_UNIVERSAL * CO2_CRITICAL_TEMPERATURE_K / CO2_CRITICAL_PRESSURE_PA
    return a, b


def solve_z_factor(temperature_k: float, pressure_pa: float) -> float:
    """Return the physically relevant compressibility factor for supercritical CO2."""
    if temperature_k <= 0.0 or pressure_pa <= 0.0:
        raise ValueError("Temperature and pressure must be positive.")

    a, b = _peng_robinson_a_and_b(temperature_k, pressure_pa)
    A = a * pressure_pa / (R_UNIVERSAL**2 * temperature_k**2)
    B = b * pressure_pa / (R_UNIVERSAL * temperature_k)

    coeffs = [
        1.0,
        -(1.0 - B),
        A - 3.0 * B**2 - 2.0 * B,
        -(A * B - B**2 - B**3),
    ]
    roots = np.roots(coeffs)

    real_candidates = []
    for root in roots:
        candidate = complex(root)
        if abs(candidate.imag) < 1e-8:
            value = float(candidate.real)
            if value > 0.0:
                real_candidates.append(value)

    if not real_candidates:
        raise ValueError("No valid positive compressibility factor was found.")

    return max(real_candidates)


def density_from_peng_robinson(temperature_k: float, pressure_pa: float, z_factor: float | None = None) -> float:
    """Compute CO2 density in kg/m^3 using the Peng-Robinson EOS."""
    if temperature_k <= 0.0 or pressure_pa <= 0.0:
        raise ValueError("Temperature and pressure must be positive.")

    z_value = solve_z_factor(temperature_k, pressure_pa) if z_factor is None else z_factor
    return pressure_pa * CO2_MOLAR_MASS_KG_PER_MOL / (z_value * R_UNIVERSAL * temperature_k)


def estimate_chung_viscosity(temperature_k: float, density_kg_m3: float) -> float:
    """Approximate high-pressure CO2 dynamic viscosity in Pa·s using a simplified Chung-style temperature scaling."""
    if temperature_k <= 0.0:
        raise ValueError("Temperature must be positive.")

    base_viscosity = 1.5e-5 * (temperature_k / 300.0) ** 0.6
    density_factor = 1.0 + 0.05 * max(density_kg_m3 / 1000.0, 0.0)
    return base_viscosity * density_factor


def compute_supercritical_properties(temperature_k: float, pressure_pa: float) -> dict[str, float]:
    z_factor = solve_z_factor(temperature_k, pressure_pa)
    density = density_from_peng_robinson(temperature_k, pressure_pa, z_factor=z_factor)
    viscosity = estimate_chung_viscosity(temperature_k, density)
    return {
        "temperature_k": float(temperature_k),
        "pressure_pa": float(pressure_pa),
        "z_factor": float(z_factor),
        "density_kg_m3": float(density),
        "viscosity_pa_s": float(viscosity),
    }


def create_pressure_grid(start_pressure_pa: float, end_pressure_pa: float, steps: int) -> np.ndarray:
    if steps < 2:
        raise ValueError("At least two pressure points are required.")
    return np.linspace(start_pressure_pa, end_pressure_pa, steps)
