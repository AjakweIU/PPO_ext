# Implementation Map

Status: pre-implementation gate satisfied. The repository has not yet been modified beyond this planning document.

This map links the methodology and project instructions to the concrete implementation targets that should exist in the final repository. It is intentionally written as a design artifact for a future implementation; it does not assume that the code has already been created.

## Working policy

- The later AI-only methodology is the scientific baseline unless actual experiment code and saved results identify a different supported configuration.
- The older 6-source / 4-sink / 8-route discrete formulation is not to be silently merged with the later 3-source / 2-sink / 5-pipeline / 2-compressor continuous formulation.
- Any final choice must be captured in docs/CONFIG_DECISIONS.md and used consistently across all equations, tables, figures, and experiment logs.
- Missing scientific values are not guessed. They remain open questions until the project team confirms them.

## Implementation map

| Methodology requirement | Planned implementation target | Class / function / config entry | Experiment or validation | Expected output |
|---|---|---|---|---|
| Scope boundary excludes blockchain and related components | repository/source tree, README, experiment docs | all modules; no blockchain classes or imports | repo review / CI smoke check | no blockchain artifacts in this project |
| Physics-informed CO₂ transport simulation | src/physics/ | PengRobinsonEOS, pipeline pressure loss, compressor power model | unit tests, smoke environment run | stable thermodynamic and flow estimates |
| Peng–Robinson EOS model | src/physics/thermodynamics.py | PengRobinsonEOS, compressibility_factor, density_from_eos | physics unit tests | physical density and compressibility values |
| Darcy–Weisbach pressure loss and Swamee–Jain friction | src/physics/pipeline.py | DarcyWeisbachPressureLoss, swamee_jain_friction_factor | pressure-loss validation | feasible pressure drop estimates |
| Compressor power and efficiency model | src/physics/compressor.py | CompressorModel, compressor_power, efficiency_from_deviation | compressor unit tests | power, efficiency, and pressure-ratio outputs |
| Base network configuration | configs/base_network.yaml | network topology, capacities, node counts, segment geometry | env initialization and smoke run | valid base network definition |
| Authoritative base network selection | docs/CONFIG_DECISIONS.md, configs/base_network.yaml | final configuration decision record | code/data consistency check | final single formulation used across all outputs |
| 3 sources / 2 sinks / 5 segments / 2 compressors / 168 steps | configs/base_network.yaml, src/environment/co2_transport_env.py | CO2TransportEnv.__init__, network metadata | environment reset / step | 168-step episode and 29D/7D state/action contract |
| 29-D observation space definition | src/environment/co2_transport_env.py, src/environment/normalization.py | observation_builder, normalize_obs | environment reset, state validation | valid observation vector and normalization range |
| 7-D continuous action space | src/environment/co2_transport_env.py, src/environment/constraints.py | action_projection, action_bounds, feasibility checks | step / smoke test | bounded, feasible actions |
| Reward function transparency | src/environment/reward.py, configs/reward.yaml | compute_reward, normalize_component, log component breakdown | reward audit test | total reward decomposed into throughput, energy, leakage, and violation terms |
| Base reward weights w1=1.0, w2=0.1, w3=10.0, w4=5.0 | configs/reward.yaml | reward_weights | reward sensitivity run | documented and reproducible reward weights |
| Reward component logging and storage | src/utils/logging.py, src/environment/reward.py | log_episode_reward, store_reward_trace | training/evaluation logs | per-episode component values and total reward |
| Scenario generation classes | src/scenarios/generator.py | generate_scenario, scenario_class dispatch | scenario smoke tests | valid normal, high-demand, low-supply, failure, energy-spike scenarios |
| Domain randomization ranges | configs/scenarios.yaml | source_capacity_range, sink_rate_range, etc. | scenario generation on training seeds | stochastic but bounded scenario realizations |
| Split into train/validation/test | src/scenarios/splits.py | split_scenarios, seed assignment | hold-out evaluation | non-overlapping 70/15/15 sets |
| Fixed seeds 42, 123, 456 | configs/experiments.yaml and utils/seeds.py | reproducibility settings | train/val/test runs | exact reseeding and reproducibility |
| In-distribution vs OOD evaluation | src/evaluation/generalization.py, experiments/run_generalization.py | evaluate_in_distribution, evaluate_ood | generalization benchmark | separate ID and OOD reports |
| PPO architecture and objective | src/agents/ppo_agent.py | PPOActorCritic, policy_loss, value_loss, clip objective | smoke PPO training | 29→128→64→7 actor and 1 value head |
| PPO hyperparameters | configs/ppo.yaml | learning_rate, gamma, gae_lambda, clip_range, entropy_coef, value_coef, batch_size, rollout_steps, epochs, total_timesteps, episode_length | PPO training run | reproducible PPO training settings |
| Strong baseline comparison | src/baselines/fixed_policy.py, src/baselines/greedy.py, src/baselines/milp.py, src/agents/sac_agent.py | baseline policy classes and RL baseline | baseline benchmark experiment | PPO compared against fixed heuristic, greedy, MILP, and one additional RL method |
| Alternative RL baseline choice | configs/sac.yaml or configs/experiments.yaml | action_space_assumption, algorithm selection | baseline selection check | either SAC for continuous actions or DQN-family for discrete actions, with explicit documentation |
| Constraint checking and repair | src/environment/constraints.py | validate_state, project_action, detect_violation | env step and training runs | invalid states and actions are rejected or corrected with reason log |
| Energy price, demand, supply, failure, leakage assumptions | configs/base_network.yaml, configs/scenarios.yaml | operational profiles and randomization settings | scenario generation and evaluation | reproducible environment dynamics |
| Normalization and missing-data handling | src/environment/normalization.py | normalize_state, clip_value, missing_value_policy | observation validation | consistent feature ranges and stable state values |
| Reward sensitivity and ablation experiments | src/evaluation/ablation.py, experiments/run_ablation.py, experiments/run_reward_sensitivity.py | ablation_grid, reward_weight_sweep | sensitivity analysis | contribution of reward terms isolated |
| Decision-time comparison | src/evaluation/timing.py, experiments/run_full_evaluation.py | measure_inference_latency, compare_methods | benchmark timing | inference and decision latency tables |
| Scalability analysis | src/evaluation/scalability.py, configs/scalability.yaml, experiments/run_scalability.py | scale_network, compute_runtime | network-size sweep | runtime and performance as network scale changes |
| Statistical evaluation | src/evaluation/statistics.py, experiments/run_full_evaluation.py | aggregate_runs, confidence_intervals, significance tests | multi-seed evaluation | robust mean/variance and uncertainty-aware reporting |
| Result tables and figures generation | results/processed, results/tables, results/figures | post-processing pipeline | publication-ready outputs | machine-generated tables and figures from raw outputs |
| Experiment reproducibility logging | docs/EXPERIMENT_LOG.md, src/utils/io.py, src/utils/logging.py | save_config_snapshot, save_metrics, save_artifacts | every experiment | config, seed, environment, metrics, and artifact traceability |
| Output preservation and no-overwrite protocol | results/, docs/EXPERIMENT_LOG.md, src/utils/io.py | timestamped_run_dir, save_raw_artifacts, save_checkpoint_manifest | every experiment | unique run directories, no overwritten outputs, full raw and processed traceability |
| Open questions tracking | docs/OPEN_QUESTIONS.md | unresolved scientific assumptions and contradictions | before implementation and before publication | explicit list of unknowns requiring review |

## File-by-file target structure

Planned repository structure and primary responsibilities:

- configs/base_network.yaml — authoritative network geometry and device configuration
- configs/scenarios.yaml — scenario classes and randomized ranges
- configs/reward.yaml — reward weights and normalization policy
- configs/ppo.yaml — PPO specification
- configs/sac.yaml — alternative agent if continuous action space is retained
- configs/experiments.yaml — train/validation/test and run orchestration
- src/physics/thermodynamics.py — EOS, viscosity, density, pressure/temperature utilities
- src/physics/pipeline.py — pipeline flow and pressure-loss equations
- src/physics/compressor.py — compressor power, operating limits, efficiency model
- src/environment/co2_transport_env.py — environment, state/action handling, step/reset, trajectory management
- src/environment/constraints.py — feasibility, clipping, violation accounting
- src/environment/reward.py — reward decomposition and normalization
- src/environment/normalization.py — observation scaling and data ranges
- src/scenarios/generator.py — scenario generators and randomization hooks
- src/scenarios/domain_randomization.py — per-parameter randomization logic
- src/scenarios/splits.py — train/val/test partitioning
- src/agents/ppo_agent.py — actor–critic policy and PPO training loop
- src/agents/sac_agent.py — optional continuous-control RL baseline
- src/baselines/fixed_policy.py — deterministic heuristic baseline
- src/baselines/greedy.py — greedy benchmark policy
- src/baselines/milp.py — MILP benchmark wrapper
- src/evaluation/metrics.py — throughput, cost, violation metrics
- src/evaluation/statistics.py — bootstrapping, confidence intervals, significance checks
- src/evaluation/timing.py — runtime and decision-time instrumentation
- src/evaluation/generalization.py — ID/OOD reporting
- src/evaluation/ablation.py — reward-weight and component ablation
- src/evaluation/scalability.py — network scaling experiments
- src/utils/seeds.py — reproducibility and random seeding
- src/utils/logging.py — structured experiment logs and artifact emission
- src/utils/io.py — config, results, and checkpoint persistence
- experiments/train_ppo.py, train_sac.py, run_baselines.py, run_generalization.py, run_ablation.py, run_scalability.py, run_reward_sensitivity.py, run_full_evaluation.py — orchestration scripts
- tests/test_physics.py, test_environment.py, test_constraints.py, test_reward.py, test_scenarios.py, test_baselines.py — validation tests
- docs/CONFIG_DECISIONS.md — final decision record for network formulation and scientific configuration
- docs/IMPLEMENTATION_MAP.md — this document
- docs/OPEN_QUESTIONS.md — unresolved scientific choices and contradictions

## High-risk conflict requiring explicit resolution

The methodology contains an older discrete formulation and a newer continuous formulation:

- Older ICTC formulation: 6 sources, 4 sinks, 8 routes, 11 quantity levels, 2,112 discrete actions
- Later AI-only formulation: 3 sources, 2 sinks, 5 pipeline segments, 2 compressor stations, 29D observation, 7D continuous action

This conflict must be resolved before publication and before code implementation. The current repository does not yet contain evidence that one formulation is the actual code-supported option. As a result, the implementation target should be treated as a configuration decision rather than a silent assumption.

## Implementation gate for this repository

This repo must not proceed to code implementation until:

1. docs/IMPLEMENTATION_MAP.md is reviewed.
2. docs/OPEN_QUESTIONS.md is reviewed.
3. docs/CONFIG_DECISIONS.md is created or updated to make the final formulation decision.
4. Remaining scientific ambiguities are either resolved with evidence or explicitly deferred.

Only after this gate is passed should implementation begin.
