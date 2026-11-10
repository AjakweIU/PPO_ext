# Configuration Decisions

Status: final review decision for the planning stage.
Date: 2026-11-09

This document resolves the primary configuration conflict between the older ICTC discrete formulation and the later AI-only continuous formulation. It is intended to be the authoritative decision record for the repository and for the journal extension.

## 1. Final authoritative formulation

The project will use the later AI-only methodology as the authoritative working configuration unless later evidence from actual experiment code or saved result artifacts clearly overrides it.

Selected base configuration:

- 3 industrial CO₂ sources
- 2 geological storage sinks
- 5 pipeline segments
- 2 compressor stations
- 168 hourly simulation steps per episode
- 29-dimensional observation space
- 7-dimensional continuous action space

This configuration is the final working baseline for the journal extension.

## 2. Resolution of the formulation conflict

The older ICTC formulation is treated as a historical or comparative reference only. It is not the active implementation target for this repository.

Historical ICTC formulation (not used as the active final formulation):

- 6 sources
- 4 sinks
- 8 routes
- 11 quantity levels
- 2,112 discrete actions

Reason for exclusion:

- It conflicts with the later AI-only methodology.
- It is incompatible with the later 29D observation and 7D continuous action specification.
- The project instructions explicitly require the later methodology to be used as the authoritative base configuration unless actual experiment code or retained outputs justify a different choice.

Therefore, the repository must not silently mix the discrete formulation with the continuous formulation. All later code, experiments, tables, and figures must use one consistent formulation.

## 3. Action-space decision

The final action space will remain continuous.

This implies:

- the primary RL benchmark should prefer SAC as the additional RL baseline,
- a discrete-action formulation is not the default path,
- any future discrete conversion must be explicitly documented and justified before implementation.

The continuous formulation is consistent with:

- 7D action vector in the later methodology,
- continuous operational control of source capture rates and compressor variables,
- the project requirement to prefer SAC when the final action space remains continuous.

## 4. Consequences for implementation

The following are required from this point onward:

1. The repository must implement the later continuous formulation as the default and documented authority.
2. The older discrete formulation must not be used as active code or as a hidden compatibility layer.
3. All manuscript equations, tables, figures, and experimental results must use the selected formulation consistently.
4. If the actual code or retained result files later show that the older formulation is the only supported one, that decision must be documented in a new revision of this file and the resulting manuscript must be updated accordingly.
5. No experiment output may overwrite previous results. Every run must be stored in a uniquely named or timestamped directory and retained with its raw outputs, configuration snapshot, seeds, logs, model checkpoints, and generated figures.

## 5. Output preservation policy

The experiment protocol must preserve all machine-readable artifacts needed for reproducibility.

Required behavior:

- every run must write to a unique directory name or a timestamped subdirectory,
- previous outputs must never be overwritten,
- raw outputs, processed outputs, checkpoints, logs, seeds, and figure artifacts must all be retained,
- final manuscript tables and figures must be regenerated from the stored raw outputs rather than from hand-entered values,
- any reproducibility audit must be able to trace a result back to its config file, seed set, software versions, and run metadata.

## 6. Remaining open scientific issues

This decision resolves the major formulation conflict, but it does not resolve all scientific ambiguities. The remaining unresolved items are still recorded in [docs/OPEN_QUESTIONS.md](docs/OPEN_QUESTIONS.md) and must be reviewed before implementation begins.

The planning-stage open items include:

- exact per-node capacity and demand values,
- exact topology and pipeline geometry,
- exact pressure/temperature limits,
- exact scenario-generation distributions and failure logic,
- exact state-feature ordering and normalization bounds,
- exact reward normalization and component formulas,
- exact MILP objective and benchmark matching,
- exact OOD thresholds and statistical reporting rules.

These items are not resolved by the formulation decision, and they must not be guessed or silently filled in.

## 6. Implementation gate status

The repository may proceed to implementation only after:

- this decision is reviewed,
- the remaining scientific questions are either resolved with evidence or explicitly deferred,
- [docs/CONFIG_DECISIONS.md](docs/CONFIG_DECISIONS.md) is treated as the controlling record for the final formulation choice,
- the next implementation phase is explicitly approved.

At this stage, the active planning decision is clear: the later AI-only continuous formulation is the authoritative project configuration.
