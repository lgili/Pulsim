# Supported Runtime Surface (Python-Only)

This document defines the supported user-facing runtime contract for Pulsim.

## Supported

- Python package workflows through `pulsim` bindings.
- YAML (`pulsim-v1`) netlist loading and programmatic circuit construction from Python.
- Python benchmark and validation tooling.

## Internal-Only (Not User-Facing Contract)

- C++ implementation details in `core/` and pybind bindings.
- Internal benchmark/test harness helpers.

## Deprecated / Legacy Surfaces to Remove

- User-facing CLI flow as primary execution path.
- User-facing grpc/remote workflow documentation in the default product path.
- JSON netlist loading workflows.
- Any legacy core runtime path not backed by `pulsim/v1`.

## Migration Policy

- New user guidance must reference Python runtime only.
- Legacy guidance remains temporary and must be removed during Phase 7 of
  `refactor-python-only-v1-hardening`.
