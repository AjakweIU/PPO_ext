from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

from src.environment.constraints import check_physical_constraints, project_action
from src.environment.normalization import default_observation_ranges, normalize_observation
from src.environment.reward import DEFAULT_REWARD_WEIGHTS, compute_reward
from src.physics.compressor import compressor_efficiency, compressor_power_kw
from src.physics.pipeline import compute_flow_velocity, darcy_weisbach_pressure_drop
from src.physics.thermodynamics import compute_supercritical_properties


class CO2TransportEnv(gym.Env):
    """Physics-informed CO2 transport environment for the journal-study baseline implementation."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        n_sources: int = 3,
        n_sinks: int = 2,
        n_segments: int = 5,
        n_compressors: int = 2,
        episode_steps: int = 168,
        source_capacity: np.ndarray | list[float] | None = None,
        sink_capacity: np.ndarray | list[float] | None = None,
        min_pressure_pa: float = 4.0e6,
        max_pressure_pa: float = 2.5e7,
        scenario_parameters: dict[str, float] | None = None,
        timestep_hours: float = 1.0,
    ):
        self.n_sources = int(n_sources)
        self.n_sinks = int(n_sinks)
        self.n_segments = int(n_segments)
        self.n_compressors = int(n_compressors)
        self.episode_steps = int(episode_steps)
        if (self.n_sources, self.n_sinks, self.n_segments, self.n_compressors) != (3, 2, 5, 2):
            raise ValueError("The configured radial topology requires 3 sources, 2 sinks, 5 segments, and 2 compressors.")

        self.scenario_parameters = dict(scenario_parameters or {})
        base_source_capacity = np.asarray(source_capacity if source_capacity is not None else [1500.0, 2200.0, 2600.0], dtype=float)
        base_sink_capacity = np.asarray(sink_capacity if sink_capacity is not None else [2200.0, 3000.0], dtype=float)
        if source_capacity is None and "source_capacity" in self.scenario_parameters:
            shares = base_source_capacity / np.sum(base_source_capacity)
            base_source_capacity = shares * float(self.scenario_parameters["source_capacity"])
        if sink_capacity is None and "sink_injection_rate" in self.scenario_parameters:
            shares = base_sink_capacity / np.sum(base_sink_capacity)
            base_sink_capacity = shares * float(self.scenario_parameters["sink_injection_rate"])
        self.source_capacity = base_source_capacity
        self.sink_capacity = np.asarray(sink_capacity if sink_capacity is not None else [2200.0, 3000.0], dtype=float)
        self.sink_demand = base_sink_capacity if "sink_injection_rate" in self.scenario_parameters else self.sink_capacity.copy()
        self.min_pressure_pa = float(min_pressure_pa)
        self.max_pressure_pa = float(max_pressure_pa)
        self.timestep_hours = float(timestep_hours)
        if self.timestep_hours <= 0.0:
            raise ValueError("timestep_hours must be positive.")

        self.segment_lengths_m = np.asarray([18000.0, 24000.0, 32000.0, 20000.0, 28000.0], dtype=float)
        self.segment_diameters_m = np.asarray([0.45, 0.50, 0.55, 0.60, 0.52], dtype=float)
        if "pipeline_length_km" in self.scenario_parameters:
            self.segment_lengths_m[:] = float(self.scenario_parameters["pipeline_length_km"]) * 1000.0
        if "pipeline_diameter_m" in self.scenario_parameters:
            self.segment_diameters_m[:] = float(self.scenario_parameters["pipeline_diameter_m"])
        self.design_flows_tonnes_per_day = np.asarray([1500.0, 1800.0], dtype=float)
        scenario_efficiency = float(self.scenario_parameters.get("compressor_efficiency", 0.82))
        self.design_efficiencies = np.full(self.n_compressors, np.clip(scenario_efficiency, 0.75, 0.88), dtype=float)
        compressor_power_limit = float(self.scenario_parameters.get("compressor_power_kw", 8000.0))
        self.compressor_power_limits_kw = np.full(self.n_compressors, compressor_power_limit, dtype=float)
        self.max_pressure_ratio = 2.2
        self.min_pressure_ratio = 1.0
        self.leakage_rate = float(self.scenario_parameters.get("leakage_rate", 0.005))
        self.base_energy_price = float(self.scenario_parameters.get("base_energy_price", 0.08))
        self.peak_price_multiplier = float(self.scenario_parameters.get("peak_price_multiplier", 1.0))
        self.reward_weights = DEFAULT_REWARD_WEIGHTS.copy()
        maximum_rate = min(
            float(np.sum(self.source_capacity)),
            float(np.sum(self.sink_capacity)),
            float(np.sum(self.design_flows_tonnes_per_day * 1.25)),
        )
        maximum_throughput = maximum_rate * self.timestep_hours / 24.0
        maximum_energy_cost = self._maximum_energy_cost_usd()
        self.reward_bounds = {
            "throughput": (0.0, maximum_throughput),
            "energy": (0.0, maximum_energy_cost),
            "leakage": (0.0, maximum_throughput * 0.02),
            "violation": (0.0, 1.0),
        }

        if self.source_capacity.shape[0] != self.n_sources:
            raise ValueError(f"Expected {self.n_sources} source capacities, received {self.source_capacity.shape[0]}.")
        if self.sink_capacity.shape[0] != self.n_sinks:
            raise ValueError(f"Expected {self.n_sinks} sink capacities, received {self.sink_capacity.shape[0]}.")

        self.observation_space = gym.spaces.Box(
            low=np.zeros(29, dtype=float),
            high=np.ones(29, dtype=float),
            dtype=np.float64,
        )
        self.action_space = gym.spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.75, 0.75, 0.75, 0.75], dtype=float),
            high=np.array([1.0, 1.0, 1.0, 1.25, 1.25, 1.25, 1.25], dtype=float),
            dtype=np.float64,
        )

        self.observation_ranges = default_observation_ranges()
        self.current_step = 0
        self.state = None
        self._failed_sources = np.zeros(self.n_sources, dtype=bool)
        self._failed_compressors = np.zeros(self.n_compressors, dtype=bool)
        self._failed_segments = np.zeros(self.n_segments, dtype=bool)

    def _initial_state(self) -> np.ndarray:
        base = np.zeros(29, dtype=float)
        base[0:3] = np.asarray([1.8e7, 1.6e7, 2.0e7], dtype=float)[:self.n_sources]
        base[3:6] = np.asarray([290.0, 295.0, 300.0], dtype=float)[:self.n_sources]
        base[6:8] = np.asarray([1.2e7, 1.1e7], dtype=float)[:self.n_sinks]
        base[8:16] = np.asarray([base[0], base[0], 0.0, 0.0, base[2], base[2], 0.0, 0.0], dtype=float)
        base[16:21] = 0.0
        base[21:24] = self.source_capacity
        base[24:26] = self.sink_demand
        base[26:29] = np.asarray([self.base_energy_price * self.peak_price_multiplier, 0.0, 0.0], dtype=float)
        return base

    def _sample_failures(self) -> None:
        source_probability = float(self.scenario_parameters.get("source_outage_probability", 0.0))
        compressor_probability = float(self.scenario_parameters.get("compressor_outage_probability", 0.0))
        pipeline_probability = float(self.scenario_parameters.get(
            "pipeline_outage_probability",
            self.scenario_parameters.get("failure_probability", 0.0) / max(self.n_segments, 1),
        ))
        self._failed_sources = self.np_random.random(self.n_sources) < source_probability
        self._failed_compressors = self.np_random.random(self.n_compressors) < compressor_probability
        self._failed_segments = self.np_random.random(self.n_segments) < pipeline_probability

    def operational_context(self) -> dict[str, Any]:
        return {
            "source_available_rate": np.where(self._failed_sources, 0.0, self.source_capacity).copy(),
            "sink_demand": self.sink_demand.copy(),
            "sink_capacity": self.sink_capacity.copy(),
            "failed_sources": self._failed_sources.copy(),
            "failed_compressors": self._failed_compressors.copy(),
            "failed_segments": self._failed_segments.copy(),
            "source_pressures_pa": self.state[0:3].copy() if self.state is not None else None,
            "source_temperatures_k": self.state[3:6].copy() if self.state is not None else None,
            "segment_lengths_m": self.segment_lengths_m.copy(),
            "segment_diameters_m": self.segment_diameters_m.copy(),
            "timestep_hours": self.timestep_hours,
            "energy_price_usd_per_kwh": self.base_energy_price * self.peak_price_multiplier,
        }

    def _segment_pressure_drop(self, flow_tonnes_per_day: float, segment_index: int, inlet_pressure_pa: float, temperature_k: float) -> float:
        if flow_tonnes_per_day <= 0.0:
            return 0.0
        properties = compute_supercritical_properties(temperature_k, inlet_pressure_pa)
        mass_flow_kg_s = flow_tonnes_per_day * 1000.0 / 86400.0
        velocity = compute_flow_velocity(mass_flow_kg_s, properties["density_kg_m3"], self.segment_diameters_m[segment_index])
        return darcy_weisbach_pressure_drop(
            density_kg_m3=properties["density_kg_m3"],
            velocity_m_s=velocity,
            length_m=self.segment_lengths_m[segment_index],
            diameter_m=self.segment_diameters_m[segment_index],
            viscosity_pa_s=properties["viscosity_pa_s"],
        )

    def _compressor_power(self, compressor_index: int, flow_tonnes_per_day: float, inlet_pressure_pa: float, outlet_pressure_pa: float, temperature_k: float) -> float:
        if flow_tonnes_per_day <= 0.0 or outlet_pressure_pa <= inlet_pressure_pa:
            return 0.0
        properties = compute_supercritical_properties(temperature_k, inlet_pressure_pa)
        design_flow = self.design_flows_tonnes_per_day[compressor_index]
        efficiency = compressor_efficiency(
            design_efficiency=self.design_efficiencies[compressor_index],
            flow_rate=flow_tonnes_per_day,
            design_flow=design_flow,
        )
        return compressor_power_kw(
            mass_flow_kg_s=flow_tonnes_per_day * 1000.0 / 86400.0,
            inlet_pressure_pa=inlet_pressure_pa,
            outlet_pressure_pa=outlet_pressure_pa,
            inlet_temperature_k=temperature_k,
            efficiency=efficiency,
            z_factor=properties["z_factor"],
        )

    def _maximum_energy_cost_usd(self) -> float:
        inlet_pressure = self.min_pressure_pa
        inlet_temperature = 350.0
        outlet_pressure = min(self.max_pressure_pa, inlet_pressure * self.max_pressure_ratio)
        properties = compute_supercritical_properties(inlet_temperature, inlet_pressure)
        price_ceiling = 0.15 * 3.5
        maximum_power_kw = 0.0
        for index, design_flow in enumerate(self.design_flows_tonnes_per_day):
            model_power_kw = compressor_power_kw(
                mass_flow_kg_s=design_flow * 1.25 * 1000.0 / 86400.0,
                inlet_pressure_pa=inlet_pressure,
                outlet_pressure_pa=outlet_pressure,
                inlet_temperature_k=inlet_temperature,
                efficiency=0.75,
                z_factor=properties["z_factor"],
            )
            maximum_power_kw += min(model_power_kw, self.compressor_power_limits_kw[index])
        return maximum_power_kw * price_ceiling * self.timestep_hours

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        self.current_step = 0
        self._sample_failures()
        self.state = self._initial_state()
        self.state[21:24] = np.where(self._failed_sources, 0.0, self.source_capacity)
        observation = normalize_observation(self.state, self.observation_ranges)
        pressure_values = np.concatenate((self.state[0:3], self.state[6:8], self.state[[8, 9, 12, 13]]))
        valid, constraints = check_physical_constraints(
            source_flow=np.zeros(self.n_sources, dtype=float),
            sink_flow=np.zeros(self.n_sinks, dtype=float),
            pressure_values=pressure_values,
            source_capacity=self.source_capacity,
            sink_capacity=self.sink_capacity,
            min_pressure_pa=self.min_pressure_pa,
            max_pressure_pa=self.max_pressure_pa,
        )
        return observation, {
            "valid": valid,
            "constraint_check": constraints,
            "failed_sources": np.flatnonzero(self._failed_sources).tolist(),
            "failed_compressors": np.flatnonzero(self._failed_compressors).tolist(),
            "failed_segments": np.flatnonzero(self._failed_segments).tolist(),
        }

    def step(self, action: np.ndarray | list[float]):
        if self.state is None:
            raise RuntimeError("Environment has not been reset yet.")

        raw_action = np.asarray(action, dtype=float)
        if raw_action.shape != (7,):
            raise ValueError(f"Action length must be 7, got shape {raw_action.shape}.")

        projected_action, correction = project_action(raw_action)
        self.current_step += 1

        available_source_rate = np.where(self._failed_sources, 0.0, self.source_capacity)
        requested_source_rate = projected_action[:3] * available_source_rate

        ratio_request = np.asarray([projected_action[3], projected_action[5]], dtype=float)
        throttle = np.asarray([projected_action[4], projected_action[6]], dtype=float)
        ratio_fraction = (ratio_request - 0.75) / (1.25 - 0.75)
        requested_ratios = self.min_pressure_ratio + ratio_fraction * (self.max_pressure_ratio - self.min_pressure_ratio)
        requested_ratios = np.clip(requested_ratios, self.min_pressure_ratio, self.max_pressure_ratio)
        compressor_capacity = self.design_flows_tonnes_per_day * throttle
        compressor_capacity[self._failed_compressors] = 0.0

        route_sources = ([0, 1], [2])
        outlet_segments = (3, 4)
        source_flow = np.zeros(self.n_sources, dtype=float)
        segment_flows = np.zeros(self.n_segments, dtype=float)
        leakage_by_segment = np.zeros(self.n_segments, dtype=float)
        sink_flow = np.zeros(self.n_sinks, dtype=float)
        compressor_inlet_pressures = np.asarray([self.state[0], self.state[2]], dtype=float)
        compressor_outlet_pressures = compressor_inlet_pressures.copy()
        compressor_powers_kw = np.zeros(self.n_compressors, dtype=float)
        compressor_status = np.zeros(self.n_compressors, dtype=float)
        ratio_clipped = np.zeros(self.n_compressors, dtype=bool)
        network_length = float(np.sum(self.segment_lengths_m))

        for compressor_index, sources in enumerate(route_sources):
            outlet_segment = outlet_segments[compressor_index]
            sink_index = compressor_index
            feeder_segments = tuple(sources)
            if self._failed_compressors[compressor_index] or self._failed_segments[outlet_segment]:
                continue

            source_inlet_rates = requested_source_rate[list(sources)].copy()
            for local_index, source_index in enumerate(sources):
                if self._failed_segments[source_index]:
                    source_inlet_rates[local_index] = 0.0
            retention = 1.0
            for segment_index in (*feeder_segments, outlet_segment):
                segment_loss_fraction = min(0.99, self.leakage_rate * self.segment_lengths_m[segment_index] / network_length)
                retention *= 1.0 - segment_loss_fraction
            max_route_input = min(
                compressor_capacity[compressor_index],
                self.sink_capacity[sink_index] / max(retention, 1e-12),
                self.sink_demand[sink_index] / max(retention, 1e-12),
            )
            requested_route_flow = float(np.sum(source_inlet_rates))
            if requested_route_flow > max_route_input and requested_route_flow > 0.0:
                source_inlet_rates *= max_route_input / requested_route_flow
            source_flow[list(sources)] = source_inlet_rates

            feeder_outlet_rates = []
            feeder_outlet_pressures = []
            for local_index, source_index in enumerate(sources):
                segment_index = source_index
                segment_flow = float(source_inlet_rates[local_index])
                segment_flows[segment_index] = segment_flow
                loss_fraction = min(0.99, self.leakage_rate * self.segment_lengths_m[segment_index] / network_length)
                leakage_by_segment[segment_index] = segment_flow * loss_fraction
                inlet_pressure = float(self.state[source_index])
                drop = self._segment_pressure_drop(segment_flow, segment_index, inlet_pressure, float(self.state[3 + source_index]))
                feeder_outlet_rates.append(segment_flow - leakage_by_segment[segment_index])
                feeder_outlet_pressures.append(max(1.0, inlet_pressure - drop))

            compressor_flow = float(np.sum(feeder_outlet_rates))
            if compressor_flow > 0.0:
                compressor_inlet = float(np.average(feeder_outlet_pressures, weights=np.maximum(feeder_outlet_rates, 1e-12)))
            else:
                compressor_inlet = float(self.state[sources[0]])
            compressor_inlet_pressures[compressor_index] = compressor_inlet
            maximum_ratio = min(self.max_pressure_ratio, self.max_pressure_pa / max(compressor_inlet, 1.0))
            actual_ratio = min(float(requested_ratios[compressor_index]), maximum_ratio)
            ratio_clipped[compressor_index] = actual_ratio + 1e-12 < requested_ratios[compressor_index]
            compressor_outlet = compressor_inlet * actual_ratio
            compressor_outlet_pressures[compressor_index] = compressor_outlet
            compressor_status[compressor_index] = float(compressor_flow > 0.0)
            compressor_powers_kw[compressor_index] = self._compressor_power(
                compressor_index,
                compressor_flow,
                compressor_inlet,
                compressor_outlet,
                float(np.mean(self.state[3 + np.asarray(sources)])),
            )

            segment_flows[outlet_segment] = compressor_flow
            outlet_loss_fraction = min(0.99, self.leakage_rate * self.segment_lengths_m[outlet_segment] / network_length)
            leakage_by_segment[outlet_segment] = compressor_flow * outlet_loss_fraction
            outlet_pressure_drop = self._segment_pressure_drop(
                compressor_flow,
                outlet_segment,
                max(compressor_outlet, self.min_pressure_pa),
                float(np.mean(self.state[3 + np.asarray(sources)])),
            )
            sink_pressure = compressor_outlet - outlet_pressure_drop
            delivered_rate = min(
                max(0.0, compressor_flow - leakage_by_segment[outlet_segment]),
                float(self.sink_capacity[sink_index]),
                float(self.sink_demand[sink_index]),
            )
            sink_flow[sink_index] = delivered_rate
            self.state[6 + sink_index] = sink_pressure

        delivered_flow = float(np.sum(sink_flow))
        delivered_tonnes = delivered_flow * self.timestep_hours / 24.0
        leakage_tonnes = float(np.sum(leakage_by_segment) * self.timestep_hours / 24.0)
        energy_kwh = float(np.sum(compressor_powers_kw) * self.timestep_hours)
        energy_price = self.base_energy_price * self.peak_price_multiplier
        energy_cost = energy_kwh * energy_price
        demand_total = float(np.sum(self.sink_demand))
        demand_shortfall = max(0.0, demand_total - delivered_flow) / max(demand_total, 1e-12)
        requested_total = float(np.sum(requested_source_rate))
        capture_shortfall = max(0.0, requested_total - float(np.sum(source_flow))) / max(requested_total, 1e-12)
        action_ranges = np.asarray([1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], dtype=float)
        action_correction_fraction = float(np.mean(np.abs(raw_action - projected_action) / action_ranges))
        pressure_values = np.asarray([
            self.state[0], self.state[1], self.state[2], self.state[6], self.state[7],
            compressor_inlet_pressures[0], compressor_outlet_pressures[0],
            compressor_inlet_pressures[1], compressor_outlet_pressures[1],
        ], dtype=float)
        physically_valid, constraints = check_physical_constraints(
            source_flow=source_flow,
            sink_flow=sink_flow,
            pressure_values=pressure_values,
            source_capacity=available_source_rate,
            sink_capacity=self.sink_capacity,
            min_pressure_pa=self.min_pressure_pa,
            max_pressure_pa=self.max_pressure_pa,
        )
        pressure_violation_fraction = float(sum(p < self.min_pressure_pa or p > self.max_pressure_pa for p in pressure_values) / len(pressure_values))
        ratio_violation_fraction = float(np.mean(ratio_clipped))
        source_capacity_excess = np.maximum(0.0, source_flow - available_source_rate) / np.maximum(available_source_rate, 1.0)
        sink_capacity_excess = np.maximum(0.0, sink_flow - self.sink_capacity) / np.maximum(self.sink_capacity, 1.0)
        capacity_violation_fraction = float(np.clip(max(np.max(source_capacity_excess), np.max(sink_capacity_excess)), 0.0, 1.0))
        violation_components = {
            "action_correction": action_correction_fraction,
            "pressure_violation": pressure_violation_fraction,
            "pressure_ratio_violation": ratio_violation_fraction,
            "capacity_violation": capacity_violation_fraction,
        }
        violation = float(np.mean(list(violation_components.values())))
        reward_breakdown = compute_reward(
            throughput=delivered_tonnes,
            energy_cost=energy_cost,
            leakage=leakage_tonnes,
            violation=violation,
            weights=self.reward_weights,
            bounds=self.reward_bounds,
        )
        reward = reward_breakdown.total_reward

        source_headroom_rate = np.maximum(0.0, available_source_rate - source_flow)
        available_next = available_source_rate.copy()
        self.state[8:16] = np.asarray([
            compressor_inlet_pressures[0], compressor_outlet_pressures[0], compressor_powers_kw[0], compressor_status[0],
            compressor_inlet_pressures[1], compressor_outlet_pressures[1], compressor_powers_kw[1], compressor_status[1],
        ], dtype=float)
        self.state[16:21] = segment_flows
        self.state[21:24] = available_next
        self.state[24:26] = self.sink_demand
        self.state[26] = energy_price
        self.state[27] = float(self.current_step % 24)
        self.state[28] = float((self.current_step // 24) % 7)

        observation = normalize_observation(self.state, self.observation_ranges)
        terminated = self.current_step >= self.episode_steps
        truncated = False
        return observation, reward, terminated, truncated, {
            "raw_action": raw_action,
            "projected_action": projected_action,
            "constraint_check": constraints,
            "correction": correction,
            "valid": physically_valid,
            "failed_sources": np.flatnonzero(self._failed_sources).tolist(),
            "failed_compressors": np.flatnonzero(self._failed_compressors).tolist(),
            "failed_segments": np.flatnonzero(self._failed_segments).tolist(),
            "source_flow": source_flow,
            "available_source_rate": available_source_rate,
            "source_headroom_rate": source_headroom_rate,
            "sink_flow": sink_flow,
            "delivered_flow": delivered_flow,
            "sink_received_tonnes": sink_flow * self.timestep_hours / 24.0,
            "capacity_shortfall_tonnes": np.maximum(0.0, self.sink_demand - sink_flow) * self.timestep_hours / 24.0,
            "segment_flows": segment_flows,
            "segment_leakage_tonnes": leakage_by_segment * self.timestep_hours / 24.0,
            "leakage_tonnes": leakage_tonnes,
            "compressor_pressure_ratios": np.divide(compressor_outlet_pressures, compressor_inlet_pressures, out=np.ones_like(compressor_outlet_pressures), where=compressor_inlet_pressures > 0.0),
            "compressor_power_kw": compressor_powers_kw,
            "energy_kwh": energy_kwh,
            "energy_price_usd_per_kwh": energy_price,
            "energy_cost": energy_cost,
            "violation": violation,
            "violation_components": violation_components,
            "raw_throughput": delivered_tonnes,
            "normalized_throughput": reward_breakdown.normalized_values["throughput"],
            "raw_energy": energy_cost,
            "normalized_energy": reward_breakdown.normalized_values["energy"],
            "raw_leakage": leakage_tonnes,
            "normalized_leakage": reward_breakdown.normalized_values["leakage"],
            "normalized_violation": reward_breakdown.normalized_values["violation"],
            "weighted_contributions": reward_breakdown.component_contributions,
            "weighted_throughput": reward_breakdown.component_contributions["throughput"],
            "weighted_energy": reward_breakdown.component_contributions["energy"],
            "weighted_leakage": reward_breakdown.component_contributions["leakage"],
            "weighted_violation": reward_breakdown.component_contributions["violation"],
            "component_values": reward_breakdown.component_values,
            "total_reward": reward,
            "final_reward": reward,
            "reward_breakdown": reward_breakdown,
            "physical_valid": physically_valid,
            "demand_shortfall_fraction": demand_shortfall,
            "demand_shortfall_tonnes": max(0.0, demand_total - delivered_flow) * self.timestep_hours / 24.0,
            "capture_shortfall_fraction": capture_shortfall,
            "capture_shortfall_tonnes": max(0.0, requested_total - float(np.sum(source_flow))) * self.timestep_hours / 24.0,
            "action_correction_fraction": action_correction_fraction,
            "pressure_violation_fraction": pressure_violation_fraction,
            "pressure_ratio_violation_fraction": ratio_violation_fraction,
            "capacity_violation_fraction": capacity_violation_fraction,
        }

    def render(self):
        return None
