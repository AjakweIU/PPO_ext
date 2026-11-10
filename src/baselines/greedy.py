from __future__ import annotations

from typing import Any

import numpy as np

from .base import PolicyDecision


ROUTES = ((0, (0, 1), 0, (0, 1), 3), (1, (2,), 1, (2,), 4))


class GreedyScheduler:
    """One-step deterministic scheduler with an explicit energy-cost cutoff."""

    name = "greedy"

    def __init__(self, energy_cost_budget_usd_per_delivered_tonne: float = 0.50):
        if energy_cost_budget_usd_per_delivered_tonne <= 0.0:
            raise ValueError("The greedy energy-cost budget must be positive.")
        self.energy_cost_budget = float(energy_cost_budget_usd_per_delivered_tonne)

    @staticmethod
    def _ratio_action(env: Any, requested_ratio: float) -> float:
        ratio_fraction = (requested_ratio - env.min_pressure_ratio) / (env.max_pressure_ratio - env.min_pressure_ratio)
        return float(np.clip(0.75 + 0.5 * ratio_fraction, 0.75, 1.25))

    def act(self, observation: np.ndarray, env: Any) -> PolicyDecision:
        context = env.operational_context()
        available = np.maximum(0.0, np.asarray(observation[21:24], dtype=float))
        sink_demand = np.maximum(0.0, np.asarray(observation[24:26], dtype=float))
        source_pressures = np.asarray(context["source_pressures_pa"], dtype=float)
        source_temperatures = np.asarray(context["source_temperatures_k"], dtype=float)
        failed_sources = np.asarray(context["failed_sources"], dtype=bool)
        failed_compressors = np.asarray(context["failed_compressors"], dtype=bool)
        failed_segments = np.asarray(context["failed_segments"], dtype=bool)
        action = np.asarray([0.0, 0.0, 0.0, 0.75, 0.75, 0.75, 0.75], dtype=float)
        route_candidates = []

        for compressor_index, sources, sink_index, feeder_segments, outlet_segment in ROUTES:
            if failed_compressors[compressor_index] or failed_segments[outlet_segment]:
                continue
            source_caps = available[list(sources)].copy()
            for source_index in sources:
                if failed_sources[source_index] or failed_segments[source_index]:
                    source_caps[sources.index(source_index)] = 0.0
            if float(np.sum(source_caps)) <= 0.0:
                continue

            network_length = float(np.sum(context["segment_lengths_m"]))
            max_input = min(
                float(np.sum(source_caps)),
                float(env.design_flows_tonnes_per_day[compressor_index] * 1.25),
                float(env.sink_capacity[sink_index]),
                float(sink_demand[sink_index]),
            )
            if max_input <= 0.0:
                continue
            source_rates = source_caps * (max_input / float(np.sum(source_caps)))
            feeder_rates = []
            feeder_pressures = []
            route_retention = 1.0
            for local_index, source_index in enumerate(sources):
                segment_index = feeder_segments[local_index]
                loss_fraction = min(0.99, env.leakage_rate * context["segment_lengths_m"][segment_index] / network_length)
                route_retention *= 1.0 - loss_fraction
                drop = env._segment_pressure_drop(
                    float(source_rates[local_index]),
                    segment_index,
                    float(source_pressures[source_index]),
                    float(source_temperatures[source_index]),
                )
                feeder_rates.append(float(source_rates[local_index]) * (1.0 - loss_fraction))
                feeder_pressures.append(max(1.0, float(source_pressures[source_index]) - drop))
            compressor_flow = float(np.sum(feeder_rates))
            if compressor_flow <= 0.0:
                continue
            compressor_inlet = float(np.average(feeder_pressures, weights=np.maximum(feeder_rates, 1e-12)))
            outlet_loss = min(0.99, env.leakage_rate * context["segment_lengths_m"][outlet_segment] / network_length)
            route_retention *= 1.0 - outlet_loss
            outlet_drop = env._segment_pressure_drop(
                compressor_flow,
                outlet_segment,
                min(env.max_pressure_pa, compressor_inlet * env.max_pressure_ratio),
                float(np.mean(source_temperatures[list(sources)])),
            )
            minimum_ratio = max(env.min_pressure_ratio, (env.min_pressure_pa + outlet_drop) / max(compressor_inlet, 1.0))
            maximum_ratio = min(env.max_pressure_ratio, env.max_pressure_pa / max(compressor_inlet, 1.0))
            if minimum_ratio > maximum_ratio:
                continue
            pressure_ratio = minimum_ratio
            outlet_pressure = compressor_inlet * pressure_ratio
            power_kw = env._compressor_power(
                compressor_index,
                compressor_flow,
                compressor_inlet,
                outlet_pressure,
                float(np.mean(source_temperatures[list(sources)])),
            )
            midpoint_ratio = min(1.6, maximum_ratio)
            reference_power_kw = env._compressor_power(
                compressor_index,
                compressor_flow,
                compressor_inlet,
                compressor_inlet * midpoint_ratio,
                float(np.mean(source_temperatures[list(sources)])),
            )
            delivered_rate = min(
                compressor_flow * (1.0 - outlet_loss),
                float(env.sink_capacity[sink_index]),
                float(sink_demand[sink_index]),
            )
            delivered_tonnes = delivered_rate * env.timestep_hours / 24.0
            energy_cost = reference_power_kw * context["energy_price_usd_per_kwh"] * env.timestep_hours
            unit_energy_cost = energy_cost / max(delivered_tonnes, 1e-12)
            route_candidates.append({
                "compressor_index": compressor_index,
                "sources": sources,
                "source_rates": source_rates,
                "target_input_rate": max_input,
                "compressor_flow": compressor_flow,
                "delivered_tonnes": delivered_tonnes,
                "pressure_ratio": pressure_ratio,
                "pressure_action": self._ratio_action(env, pressure_ratio),
                "throttle_action": float(np.clip(max_input / env.design_flows_tonnes_per_day[compressor_index], 0.75, 1.25)),
                "estimated_energy_cost_usd_per_tonne": unit_energy_cost,
                "estimated_energy_cost_usd": energy_cost,
                "minimum_feasible_ratio_power_kw": power_kw,
                "midpoint_reference_power_kw": reference_power_kw,
                "estimated_delivered_tonnes": delivered_tonnes,
            })

        route_candidates.sort(key=lambda route: (route["estimated_energy_cost_usd_per_tonne"], route["compressor_index"]))
        selected = []
        for route in route_candidates:
            if route["estimated_energy_cost_usd_per_tonne"] > self.energy_cost_budget:
                continue
            compressor_index = route["compressor_index"]
            for local_index, source_index in enumerate(route["sources"]):
                source_capacity = float(available[source_index])
                if source_capacity > 0.0:
                    action[source_index] = min(1.0, float(route["source_rates"][local_index]) / source_capacity)
            action[3 + 2 * compressor_index] = route["pressure_action"]
            action[4 + 2 * compressor_index] = route["throttle_action"]
            selected.append({
                "compressor_index": compressor_index,
                "estimated_energy_cost_usd_per_tonne": route["estimated_energy_cost_usd_per_tonne"],
                "estimated_delivered_tonnes": route["estimated_delivered_tonnes"],
                "selected_pressure_ratio": route["pressure_ratio"],
            })

        return PolicyDecision(
            action=action,
            diagnostics={
                "selected_routes": selected,
                "route_priority_order": [route["compressor_index"] for route in route_candidates],
                "tie_break": "ascending_compressor_index_after_ascending_incremental_energy_cost",
                "energy_cost_budget_usd_per_delivered_tonne": self.energy_cost_budget,
                "energy_estimate_reference_control": "action midpoint 1.0, mapped to pressure ratio 1.6 or the maximum physically allowed ratio",
                "used_future_information": False,
            },
        )
