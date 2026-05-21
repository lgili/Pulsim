## Why

Between V5 and V15 we shipped 11 feature commits — source helpers (V5-V10), device models (V11-V15) — without writing the corresponding OpenSpec proposals first. The features are validated by C++ + Python tests, but the spec backlog is empty. This blocks:

1. Anyone reading `openspec/specs/` to understand v2's documented capabilities.
2. The `openspec validate --strict` workflow that the project uses as its release gate.
3. Future contributors who need to know what behaviour is contractually guaranteed.

## What Changes

This is a **documentation-catchup** change. We DO NOT touch implementation. We add spec files retroactively to `openspec/specs/` for each shipped V5-V15 feature, plus a tasks.md tracking the catch-up effort.

For each capability, we create/update `openspec/specs/<capability>/spec.md` with:
- One or more `### Requirement:` blocks describing the contracted behaviour
- At least one `#### Scenario:` per requirement reflecting how the existing tests exercise it

Capabilities to backfill:
- `pulsim-v2-source-helpers` (V5-V10: pwm_switch_fn, dead_time_pwm_pair_fn, spwm_pair_fn, three_phase_spwm_fn, phase_shift_full_bridge_fn, combined_switch_fn)
- `pulsim-v2-sine-source` (V11)
- `pulsim-v2-pulse-source` (V12)
- `pulsim-v2-mosfet-level1` (V13 / V13.1)
- `pulsim-v2-igbt-level1` (V14)
- `pulsim-v2-vcvs-opamp` (V15)

## Impact

- **Affected specs:** 6 new capability spec files under `openspec/specs/`.
- **Affected code:** zero — pure documentation.
- **Risk:** zero — non-invasive.
