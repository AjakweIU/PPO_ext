from __future__ import annotations

import math

from src.physics.thermodynamics import R_UNIVERSAL


DEFAULT_COMPRESSOR_EFFICIENCY_MIN = 0.75
DEFAULT_COMPRESSOR_EFFICIENCY_MAX = 0.88


def pressure_ratio(inlet_pressure_pa: float, outlet_pressure_pa: float) -> float:
    if inlet_pressure_pa <= 0.0:
        raise ValueError("Inlet pressure must be positive.")
    return max(outlet_pressure_pa / inlet_pressure_pa, 1.0)


def compressor_efficiency(
    design_efficiency: float = 0.82,
    flow_rate: float = 1.0,
    design_flow: float = 1.0,
    alpha: float = 0.15,
    min_efficiency: float = DEFAULT_COMPRESSOR_EFFICIENCY_MIN,
    max_efficiency: float = DEFAULT_COMPRESSOR_EFFICIENCY_MAX,
) -> float:
    if design_flow <= 0.0:
        raise ValueError("Design flow must be positive.")
    deviation = (flow_rate - design_flow) / design_flow
    eta = design_efficiency * (1.0 - alpha * deviation ** 2)
    return max(min_efficiency, min(max_efficiency, eta))


def compressor_power(
    mass_flow_kg_s: float,
    inlet_pressure_pa: float,
    outlet_pressure_pa: float,
    inlet_temperature_k: float,
    efficiency: float = 0.82,
    z_factor: float = 1.0,
    polytropic_index: float = 1.3,
    molar_mass_kg_per_mol: float = 0.04401,
) -> float:
    """Return compressor power in watts using the methodology's polytropic compression equation."""
    if mass_flow_kg_s < 0.0:
        raise ValueError("Mass flow must be non-negative.")
    if inlet_pressure_pa <= 0.0 or outlet_pressure_pa <= 0.0:
        raise ValueError("Pressure values must be positive.")
    if inlet_temperature_k <= 0.0:
        raise ValueError("Inlet temperature must be positive.")
    if efficiency <= 0.0:
        raise ValueError("Efficiency must be positive.")

    ratio = outlet_pressure_pa / inlet_pressure_pa
    exponent = (polytropic_index - 1.0) / polytropic_index
    term = ratio ** exponent - 1.0
    numerator = mass_flow_kg_s * z_factor * R_UNIVERSAL * inlet_temperature_k
    denominator = efficiency * molar_mass_kg_per_mol * exponent
    return (numerator / denominator) * term


def compressor_power_kw(
    mass_flow_kg_s: float,
    inlet_pressure_pa: float,
    outlet_pressure_pa: float,
    inlet_temperature_k: float,
    efficiency: float = 0.82,
    z_factor: float = 1.0,
    polytropic_index: float = 1.3,
) -> float:
    return compressor_power(
        mass_flow_kg_s,
        inlet_pressure_pa,
        outlet_pressure_pa,
        inlet_temperature_k,
        efficiency=efficiency,
        z_factor=z_factor,
        polytropic_index=polytropic_index,
    ) / 1000.0
