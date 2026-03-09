# Magnetic Core Backend/Frontend Contract (MVP)

This document defines ownership boundaries for `component.magnetic_core` so GUI
and backend stay deterministic and physically consistent.

## Backend Responsibilities (Core)

- Parse and validate canonical `component.magnetic_core` fields.
- Reject unsupported component/model combinations with typed diagnostics.
- Compute magnetic channels from simulation state (no GUI reconstruction).
- Export canonical channels in `result.virtual_channels` with metadata in
  `result.virtual_channel_metadata`.
- Keep deterministic channel naming and ordering for repeated runs.

## Frontend Responsibilities (GUI)

- Provide structured editors for `magnetic_core` fields:
  - `enabled`
  - `model`
  - saturation parameters
  - hysteresis parameters (`hysteresis_band`, `hysteresis_strength`, `hysteresis_loss_coeff`,
    `hysteresis_state_init`)
  - loss parameters (`core_loss_k`, `core_loss_alpha`, `core_loss_freq_coeff`)
  - policy/initialization (`loss_policy`, `i_equiv_init`)
- Persist values exactly to YAML/Python request payload.
- Render backend-provided channels/metadata as-is (example: `Lsat.core_loss`).
- Surface backend diagnostics with code + field path to users.

## Frontend Non-Responsibilities (Must Not Do)

- Must not synthesize or fake magnetic loss waveforms.
- Must not infer magnetic equations from component names or heuristics.
- Must not silently change invalid magnetic parameters.
- Must not apply hidden fallback from nonlinear model to linear model.

## Canonical Channel Conventions (Current MVP)

- Loss channel: `<component>.core_loss`
- Hysteresis memory channel (when `model: hysteresis`): `<component>.h_state`
- Metadata:
  - `domain = "loss"`
  - `unit = "W"`
  - `source_component = <component_name>`

When `loss_policy: loss_summary` and `simulation.enable_losses: true`:

- Backend also appends a deterministic summary row in `loss_summary.device_losses`:
  - `device_name = "<component>.core"`

When `simulation.enable_losses: true` and `simulation.thermal.enabled: true`:

- Backend exports thermal channel for magnetic loss row:
  - `T(<component>.core)`
- Backend appends/updates deterministic thermal summary row:
  - `thermal_summary.device_temperatures[].device_name = "<component>.core"`
- Backend keeps `component_electrothermal` consistent with that thermal channel.

## Current MVP Limits

- Thermal RC for magnetic virtual rows currently uses simulation-level defaults.
- Advanced hysteresis families beyond the bounded state model are pending.
