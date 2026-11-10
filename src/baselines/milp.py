from __future__ import annotations

import time
from typing import Any

import numpy as np
import pulp

from src.environment.reward import DEFAULT_REWARD_WEIGHTS

from .base import PolicyDecision
from .greedy import ROUTES, GreedyScheduler


class MILPScheduler:
    """One-step PuLP/CBC operational MILP with linearized physical compressor energy."""

    name = "milp"

    def __init__(self, time_limit_seconds: float = 10.0, mip_gap: float = 0.0):
        self.time_limit_seconds = float(time_limit_seconds)
        self.mip_gap = float(mip_gap)
        self._fallback = GreedyScheduler()

    @staticmethod
    def _ratio_action(env: Any, ratio: float) -> float:
        ratio_fraction = (ratio - env.min_pressure_ratio) / (env.max_pressure_ratio - env.min_pressure_ratio)
        return float(np.clip(0.75 + 0.5 * ratio_fraction, 0.75, 1.25))

    def act(self, observation: np.ndarray, env: Any) -> PolicyDecision:
        started = time.perf_counter()
        context = env.operational_context()
        available = np.maximum(0.0, np.asarray(observation[21:24], dtype=float))
        sink_demand = np.maximum(0.0, np.asarray(observation[24:26], dtype=float))
        source_pressures = np.asarray(context["source_pressures_pa"], dtype=float)
        source_temperatures = np.asarray(context["source_temperatures_k"], dtype=float)
        failed_sources = np.asarray(context["failed_sources"], dtype=bool)
        failed_compressors = np.asarray(context["failed_compressors"], dtype=bool)
        failed_segments = np.asarray(context["failed_segments"], dtype=bool)
        problem = pulp.LpProblem("co2_transport_operational_dispatch", pulp.LpMaximize)
        source_flow = {
            index: pulp.LpVariable(f"source_rate_{index}", lowBound=0.0, upBound=float(available[index]))
            for index in range(3)
        }
        route_on = {route[0]: pulp.LpVariable(f"route_open_{route[0]}", cat=pulp.LpBinary) for route in ROUTES}
        route_data = {}
        network_length = float(np.sum(context["segment_lengths_m"]))

        for compressor_index, sources, sink_index, feeder_segments, outlet_segment in ROUTES:
            y = route_on[compressor_index]
            if failed_compressors[compressor_index] or failed_segments[outlet_segment]:
                problem += y == 0
            feeder_losses = {}
            route_retention = 1.0
            for source_index, segment_index in zip(sources, feeder_segments):
                loss_fraction = min(0.99, env.leakage_rate * context["segment_lengths_m"][segment_index] / network_length)
                feeder_losses[source_index] = loss_fraction
                route_retention *= 1.0 - loss_fraction
                if failed_sources[source_index] or failed_segments[segment_index]:
                    problem += source_flow[source_index] == 0.0
                else:
                    problem += source_flow[source_index] <= float(available[source_index]) * y
            outlet_loss = min(0.99, env.leakage_rate * context["segment_lengths_m"][outlet_segment] / network_length)
            route_retention *= 1.0 - outlet_loss
            compressor_flow = pulp.lpSum(source_flow[index] * (1.0 - feeder_losses[index]) for index in sources)
            compressor_capacity = float(env.design_flows_tonnes_per_day[compressor_index] * 1.25)
            problem += compressor_flow <= compressor_capacity * y
            delivered_rate = compressor_flow * (1.0 - outlet_loss)
            problem += delivered_rate <= float(env.sink_capacity[sink_index])
            problem += delivered_rate <= float(sink_demand[sink_index])
            route_data[compressor_index] = {
                "sources": sources,
                "sink_index": sink_index,
                "feeder_losses": feeder_losses,
                "outlet_loss": outlet_loss,
                "route_retention": route_retention,
                "compressor_flow": compressor_flow,
                "delivered_rate": delivered_rate,
                "compressor_capacity": compressor_capacity,
            }

        maximum_feasible_ratios = {}
        route_energy_costs = {}
        route_leakage_tonnes = {}
        for compressor_index, data in route_data.items():
            sources = data["sources"]
            max_source_rate = sum(float(available[index]) for index in sources if not failed_sources[index] and not failed_segments[index])
            max_route_flow = min(max_source_rate, data["compressor_capacity"])
            source_weights = np.asarray([available[index] if not failed_sources[index] and not failed_segments[index] else 0.0 for index in sources], dtype=float)
            if float(np.sum(source_weights)) > 0.0:
                planned_source_rates = source_weights * max_route_flow / float(np.sum(source_weights))
            else:
                planned_source_rates = np.zeros(len(sources), dtype=float)
            feeder_pressures = []
            feeder_rates = []
            for local_index, source_index in enumerate(sources):
                pressure_drop = env._segment_pressure_drop(
                    float(planned_source_rates[local_index]),
                    source_index,
                    float(source_pressures[source_index]),
                    float(source_temperatures[source_index]),
                )
                feeder_pressures.append(max(1.0, float(source_pressures[source_index]) - pressure_drop))
                feeder_rates.append(float(planned_source_rates[local_index]) * (1.0 - data["feeder_losses"][source_index]))
            if float(np.sum(feeder_rates)) > 0.0:
                inlet_pressure = float(np.average(feeder_pressures, weights=np.maximum(feeder_rates, 1e-12)))
                compressor_flow_rate = float(np.sum(feeder_rates))
            else:
                inlet_pressure = float(source_pressures[sources[0]])
                compressor_flow_rate = 0.0
            outlet_segment = ROUTES[compressor_index][4]
            outlet_drop = env._segment_pressure_drop(
                compressor_flow_rate,
                outlet_segment,
                min(env.max_pressure_pa, inlet_pressure * env.max_pressure_ratio),
                float(np.mean(source_temperatures[list(sources)])),
            )
            required_ratio = max(env.min_pressure_ratio, (env.min_pressure_pa + outlet_drop) / max(inlet_pressure, 1.0))
            allowed_ratio = min(env.max_pressure_ratio, env.max_pressure_pa / max(inlet_pressure, 1.0))
            if required_ratio > allowed_ratio:
                problem += route_on[compressor_index] == 0
                required_ratio = allowed_ratio
            maximum_feasible_ratios[compressor_index] = float(required_ratio)

            design_flow = float(env.design_flows_tonnes_per_day[compressor_index])
            reference_power_kw = env._compressor_power(
                compressor_index,
                design_flow,
                inlet_pressure,
                inlet_pressure * required_ratio,
                float(np.mean(source_temperatures[list(sources)])),
            )
            power_kw_per_tonne_per_day = reference_power_kw / max(design_flow, 1e-12)
            energy_usd_per_rate = power_kw_per_tonne_per_day * context["energy_price_usd_per_kwh"] * env.timestep_hours
            route_energy_costs[compressor_index] = data["compressor_flow"] * energy_usd_per_rate
            route_leakage_tonnes[compressor_index] = (
                pulp.lpSum(source_flow[index] * data["feeder_losses"][index] for index in sources)
                + data["compressor_flow"] * data["outlet_loss"]
            ) * env.timestep_hours / 24.0

        timestep_fraction = env.timestep_hours / 24.0
        delivered_tonnes = pulp.lpSum(data["delivered_rate"] for data in route_data.values()) * timestep_fraction
        energy_cost = pulp.lpSum(route_energy_costs.values())
        simulated_leakage = pulp.lpSum(route_leakage_tonnes.values())
        throughput_bound = max(float(env.reward_bounds["throughput"][1]), 1e-12)
        energy_bound = max(float(env.reward_bounds["energy"][1]), 1e-12)
        leakage_bound = max(float(env.reward_bounds["leakage"][1]), 1e-12)
        problem += (
            DEFAULT_REWARD_WEIGHTS["throughput"] * delivered_tonnes / throughput_bound
            - DEFAULT_REWARD_WEIGHTS["energy"] * energy_cost / energy_bound
            - DEFAULT_REWARD_WEIGHTS["leakage"] * simulated_leakage / leakage_bound
        )

        solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=self.time_limit_seconds, gapRel=self.mip_gap)
        problem.solve(solver)
        solve_seconds = time.perf_counter() - started
        status = pulp.LpStatus.get(problem.status, str(problem.status))
        if status not in {"Optimal", "Integer Feasible"}:
            fallback = self._fallback.act(observation, env)
            return PolicyDecision(
                action=fallback.action,
                diagnostics={
                    "solver_status": status,
                    "objective_value": None,
                    "solve_time_seconds": solve_seconds,
                    "fallback": "greedy",
                    "objective_definition": "normalized throughput - normalized energy cost - normalized simulated leakage; hard physical feasibility constraints",
                },
            )

        action = np.asarray([0.0, 0.0, 0.0, 0.75, 0.75, 0.75, 0.75], dtype=float)
        for source_index in range(3):
            if available[source_index] > 0.0:
                action[source_index] = float(np.clip((pulp.value(source_flow[source_index]) or 0.0) / available[source_index], 0.0, 1.0))
        estimated_delivered = 0.0
        estimated_energy = 0.0
        estimated_leakage = 0.0
        for compressor_index, data in route_data.items():
            sources = data["sources"]
            flow = sum(float(pulp.value(source_flow[index]) or 0.0) * (1.0 - data["feeder_losses"][index]) for index in sources)
            if flow <= 1e-9:
                continue
            ratio = maximum_feasible_ratios[compressor_index]
            pressure_fraction = (ratio - env.min_pressure_ratio) / (env.max_pressure_ratio - env.min_pressure_ratio)
            action[3 + 2 * compressor_index] = float(np.clip(0.75 + 0.5 * pressure_fraction, 0.75, 1.25))
            action[4 + 2 * compressor_index] = float(np.clip(flow / env.design_flows_tonnes_per_day[compressor_index], 0.75, 1.25))
            estimated_delivered += float(pulp.value(data["delivered_rate"]) or 0.0) * timestep_fraction
            estimated_energy += float(pulp.value(route_energy_costs[compressor_index]) or 0.0)
            estimated_leakage += float(pulp.value(route_leakage_tonnes[compressor_index]) or 0.0)
        return PolicyDecision(
            action=action,
            diagnostics={
                "solver_status": status,
                "objective_value": float(pulp.value(problem.objective)),
                "solve_time_seconds": solve_seconds,
                "milp_estimated_delivered_tonnes": estimated_delivered,
                "milp_estimated_energy_cost_usd": estimated_energy,
                "milp_estimated_simulated_leakage_tonnes": estimated_leakage,
                "fixed_feasible_pressure_ratios": maximum_feasible_ratios,
                "fallback": None,
                "objective_definition": "normalized throughput - normalized energy cost - normalized simulated leakage; hard physical feasibility constraints",
                "objective_matches_environment_reward_exactly": False,
            },
        )
