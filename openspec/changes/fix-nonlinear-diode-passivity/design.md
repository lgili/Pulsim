## Context

`IdealDiode` is the oldest Newton device: an AD-differentiated smooth
blend used by the DC continuation strategies (κ- and `V_F0`-homotopy,
`pwl/initial_guess.hpp`, `pwl/nonlinear_refresh_diode.hpp`), by the
MOSFET Level-1 body diode, and, until this change, by the SPICE
importer. Its law multiplies a logistic by a line through `V_F0`; the
product is negative on the low side of the line, so the device sources
power for `0 < v < V_F0`.

## Goals / Non-Goals

- Goals: no public entry point builds the device without an explicit
  opt-in; the refusal names the replacements; the importer stops
  producing it; the existing hostile-circuit fixtures keep their
  premise.
- Non-Goals: changing the law (see the rejected alternative), removing
  `IdealDiode` from the kernel, changing the Shockley or PWL diodes.

## Decisions

- **Refuse in Python, not in the binding.** `_builder_ergonomics.install`
  wraps the native `add_nonlinear_diode` (as it already wraps
  `add_shockley_diode`): without `allow_non_passive=True` it raises
  `RetiredDeviceError(ValueError)` before anything is stamped. The C++
  builder method and the binding are untouched, so the C++ tests and
  the stubs are unchanged.
- **YAML refuses in the loader.** `type: nonlinear_diode` without
  `allow_non_passive: true` throws the loader's `std::runtime_error`
  (Python `RuntimeError`), with the same text as the Python refusal.
- **Importer.** LTspice's PWL diode is an ideal switch with a series
  drop: `add_diode(area/Ron, area/Roff, V_th=0, V_f=Vfwd)`.
- **Opt-in name.** `allow_non_passive`, after `allow_synthetic`: it
  names the hazard the caller accepts.

## Alternatives considered

- **Passive softplus law** `[sp(κ(v − V_F0)) − sp(−κ·V_F0)]/(κ·R_d) + G_off·v`.
  Implemented and measured (`implementation-attempt.patch`,
  `DECISIONS.md`): all proposal numbers reproduced, but the knee is
  rounded in current, not in voltage, so a 2 V / 1 kΩ load line moved
  from ≈0.70 V to 0.30 V, and 27 tests that need a circuit Newton
  cannot solve lost their premise. Rejected by the owner.

## Risks / Trade-offs

- Python/YAML callers break loudly → the message names the
  replacements; the opt-in keeps old scripts running.
- The legacy law stays reachable with the opt-in → the name says what
  the caller accepts; `docs/known-limitations.md` records the defect.
