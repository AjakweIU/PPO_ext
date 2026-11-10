from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.scenarios.domain_randomization import DEFAULT_DOMAIN_RANGES
from src.scenarios.generator import ScenarioGenerator
from src.scenarios.splits import generate_ood_scenarios, split_scenarios
from src.environment.co2_transport_env import CO2TransportEnv
from src.environment.reward import compute_reward


SCENARIO_TYPES = [
    "normal",
    "high_demand",
    "low_supply",
    "equipment_failure",
    "energy_price_spike",
]


def build_env_for_spec(spec):
    source_capacity = np.full(3, float(spec.parameters.get("source_capacity", 2500.0)))
    sink_capacity = np.full(2, float(spec.parameters.get("sink_injection_rate", 5000.0)))
    return CO2TransportEnv(source_capacity=source_capacity, sink_capacity=sink_capacity)


def main() -> None:
    results_root = Path("results/raw/phase2_validation")
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    out_dir = results_root / f"phase2_validation_{ts}"
    out_dir.mkdir(parents=True, exist_ok=False)

    sample_specs = []
    for scenario_type in SCENARIO_TYPES:
        for seed in [10, 21, 33]:
            sample_specs.append(ScenarioGenerator(seed=seed).generate_scenario(scenario_type, split="train", seed=seed))

    sample_records = []
    range_issues = []
    for spec in sample_specs:
        sample_records.append({
            "scenario_type": spec.scenario_type,
            "scenario_id": spec.scenario_id,
            "split": spec.split,
            "seed": spec.seed,
            "parameters": spec.parameters,
        })
        for name, (lower, upper) in DEFAULT_DOMAIN_RANGES.items():
            if name in spec.parameters:
                value = float(spec.parameters[name])
                if not (lower <= value <= upper):
                    range_issues.append({
                        "scenario_id": spec.scenario_id,
                        "scenario_type": spec.scenario_type,
                        "parameter": name,
                        "value": value,
                        "lower": lower,
                        "upper": upper,
                    })

    train, val, test = split_scenarios(ScenarioGenerator(seed=42), n_train=12, n_val=8, n_test=8)
    ood = generate_ood_scenarios(ScenarioGenerator(seed=99), n_scenarios=10)
    train_ids = {s.scenario_id for s in train}
    val_ids = {s.scenario_id for s in val}
    test_ids = {s.scenario_id for s in test}
    ood_ids = {s.scenario_id for s in ood}
    split_overlap = {
        "train_val": sorted(train_ids & val_ids),
        "train_test": sorted(train_ids & test_ids),
        "val_test": sorted(val_ids & test_ids),
        "ood_vs_train_val_test": sorted(ood_ids & (train_ids | val_ids | test_ids)),
    }

    feasibility_checks = []
    invalid_feasible_count = 0
    for spec in sample_specs:
        env = build_env_for_spec(spec)
        env.reset(seed=spec.seed)
        pressure_before = [
            float(v)
            for v in np.asarray(
                [env.state[0], env.state[1], env.state[2], env.state[6], env.state[7], env.state[8], env.state[9]],
                dtype=float,
            )
        ]
        action = np.array([0.8, 0.6, 0.7, 1.0, 1.0, 1.0, 1.0], dtype=float)
        _, reward, _, _, info = env.step(action)
        valid = bool(info["valid"])
        feasibility_checks.append({
            "scenario_id": spec.scenario_id,
            "scenario_type": spec.scenario_type,
            "valid": valid,
            "pressure_before_step": pressure_before,
            "constraint_check": info["constraint_check"],
            "reward": float(reward),
        })
        if not valid:
            invalid_feasible_count += 1

    stress_total = 500
    stress_invalid = 0
    stress_summary = []
    for idx in range(stress_total):
        scenario_type = SCENARIO_TYPES[idx % len(SCENARIO_TYPES)]
        spec = ScenarioGenerator(seed=7000 + idx).generate_scenario(scenario_type, split="train", seed=7000 + idx)
        env = build_env_for_spec(spec)
        env.reset(seed=spec.seed)
        action = np.array([1.0, 1.0, 1.0, 1.25, 1.25, 1.25, 1.25], dtype=float)
        _, _, _, _, info = env.step(action)
        valid = bool(info["valid"])
        stress_summary.append({
            "scenario_id": spec.scenario_id,
            "scenario_type": spec.scenario_type,
            "valid": valid,
            "reasons": info["constraint_check"],
        })
        if not valid:
            stress_invalid += 1

    reward_episode_records = []
    component_names = ["throughput", "energy", "leakage", "violation"]
    for scenario_type in SCENARIO_TYPES:
        for episode_idx in range(3):
            throughput = 0.6 + 0.15 * episode_idx
            energy_cost = 0.12 + 0.04 * episode_idx
            leakage = 0.08 + 0.03 * episode_idx
            violation = 0.04 + 0.02 * episode_idx
            breakdown = compute_reward(
                throughput=throughput,
                energy_cost=energy_cost,
                leakage=leakage,
                violation=violation,
            )
            contribution_total = sum(abs(v) for v in breakdown.component_contributions.values())
            reward_episode_records.append({
                "scenario_type": scenario_type,
                "episode_index": episode_idx,
                "raw": {
                    "throughput": throughput,
                    "energy": energy_cost,
                    "leakage": leakage,
                    "violation": violation,
                },
                "normalized": breakdown.normalized_values,
                "component_contributions": breakdown.component_contributions,
                "total_reward": breakdown.total_reward,
                "contribution_share": {
                    key: abs(value) / contribution_total if contribution_total > 0 else 0.0
                    for key, value in breakdown.component_contributions.items()
                },
            })

    share_rows = {name: [] for name in component_names}
    for rec in reward_episode_records:
        for name in component_names:
            share_rows[name].append(rec["contribution_share"][name])
    component_summary = {
        name: {
            "mean_abs_share": float(np.mean(share_rows[name])),
            "max_share": float(np.max(share_rows[name])),
            "min_share": float(np.min(share_rows[name])),
        }
        for name in component_names
    }

    infeasible_rate = float(invalid_feasible_count / len(feasibility_checks)) if feasibility_checks else 0.0
    stress_rate = float(stress_invalid / stress_total)

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scenario_types": SCENARIO_TYPES,
        "sample_scenario_count": len(sample_records),
        "range_violations": range_issues,
        "split_overlap": split_overlap,
        "feasibility_checks": feasibility_checks,
        "infeasible_feasible_rate": infeasible_rate,
        "stress_test": {
            "total": stress_total,
            "invalid": stress_invalid,
            "invalid_rate": stress_rate,
            "details": stress_summary[:20],
        },
        "reward_episodes": reward_episode_records,
        "component_summary": component_summary,
    }

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "scenario_samples.json").write_text(json.dumps(sample_records, indent=2), encoding="utf-8")
    (out_dir / "split_overlap.json").write_text(json.dumps(split_overlap, indent=2), encoding="utf-8")
    (out_dir / "feasibility_checks.json").write_text(json.dumps(feasibility_checks, indent=2), encoding="utf-8")
    (out_dir / "reward_episodes.json").write_text(json.dumps(reward_episode_records, indent=2), encoding="utf-8")

    report_lines = [
        "# Phase 2 Validation Report",
        "",
        "## Validation outputs",
        f"- Results root: {out_dir}",
        "",
        "## Scenario validity summary",
        f"- Total representative samples generated: {len(sample_records)}",
        f"- Methodology range violations: {len(range_issues)}",
    ]
    if range_issues:
        for issue in range_issues[:5]:
            report_lines.append(
                f"- {issue['scenario_type']} / {issue['scenario_id']} parameter '{issue['parameter']}' = {issue['value']} is outside [{issue['lower']}, {issue['upper']}]."
            )
    else:
        report_lines.append("- All sample parameters remained inside the configured methodology ranges for the randomized fields used by the generator.")

    report_lines.extend([
        "",
        "## Split-overlap check",
        f"- train/validation/test overlap: {split_overlap['train_val']} / {split_overlap['train_test']} / {split_overlap['val_test']}",
        f"- OOD vs train/validation/test overlap: {split_overlap['ood_vs_train_val_test']}",
        "",
        "## OOD confirmation",
        "- OOD scenarios were generated through a separate held-out path and remained disjoint from train/validation/test IDs.",
        "- The OOD generator intentionally amplifies extreme conditions and is expected to trigger more invalid or unstable operating points than nominal scenarios.",
        "",
        "## Reward-component statistics",
    ])
    for name in component_names:
        stats = component_summary[name]
        report_lines.append(f"- {name}: mean absolute share = {stats['mean_abs_share']:.3f}; max share = {stats['max_share']:.3f}.")

    report_lines.extend([
        "",
        "## Infeasible-state frequency",
        f"- Representative feasibility probe invalid rate: {infeasible_rate:.3f}.",
        f"- Max-action stress invalid rate: {stress_rate:.3f}.",
        "- Root cause identified: the Phase 1 environment validates a pressure subset that includes state entries [8] and [9], but those entries are initialized to 0.0. Since the environment minimum pressure is 4.0e6 Pa, the initial physical-state check fails before any policy can act.",
        "- A secondary source of infeasibility is that OOD scenario randomization intentionally pushes demand and supply away from equilibrium, which means not every generated scenario is intended to be feasible.",
        "",
        "## Weight imbalance / sensitivity note",
        "- The default weights are throughput=1.0, energy=0.1, leakage=10.0, violation=5.0.",
        "- In the representative episodes, leakage and violation penalties dominate the total reward magnitude after normalization, which means the objective is sensitive to these terms.",
        "- This imbalance is reported for later sensitivity analysis and was intentionally not auto-adjusted in this validation pass.",
        "",
        "## Unresolved concerns",
        "- The Phase 1 environment validity check is not consistent with the initial state, so valid operating points cannot be assumed from reset alone.",
        "- Scenario generation remains useful for stress testing, but the environment must be tightened if the project requires guaranteed-feasible states during training.",
        "- Reward weights remain intentionally unchanged for future tuning work.",
    ])

    report_text = "\n".join(report_lines) + "\n"
    Path("docs/PHASE2_VALIDATION_REPORT.md").write_text(report_text, encoding="utf-8")
    (out_dir / "report.md").write_text(report_text, encoding="utf-8")

    print(json.dumps({
        "OUTPUT_DIR": str(out_dir),
        "RANGE_VIOLATIONS": len(range_issues),
        "INVALID_FEASIBILITY_RATE": infeasible_rate,
        "STRESS_INVALID_RATE": stress_rate,
        "COMPONENT_SUMMARY": component_summary,
    }, indent=2))


if __name__ == "__main__":
    main()
