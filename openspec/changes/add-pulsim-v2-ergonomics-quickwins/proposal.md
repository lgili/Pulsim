## Why

Three small but high-leverage UX gaps in v2:

1. **Body diode boilerplate** — every realistic MOSFET use case requires the user to explicitly add an anti-parallel diode (V13 docs even spell this out). This is friction for new users and source of off-by-one bugs.
2. **Python ergonomics** — current `pulsim.v2` API forces users to manually wire `PwlStateSpaceCache`, `SimulationOptions`, `switch_fn`, `nl_refresh`. A one-liner `simulate(builder, t_end, dt)` would lower the activation energy.
3. **`start_from_dc_op` doesn't support V11/V12 time-varying sources** — when a user has a sine or pulse source and asks for DC warm-start, `compute_dc_op` throws. The fix is small: treat the source's `value_at(t_start)` as the DC value during the OP solve.

## What Changes

- **MODIFY** `CircuitBuilder::add_mosfet_level1` — accept an optional `with_body_diode = false` keyword arg. When true, automatically adds an anti-parallel `SwitchedDiode` (source→drain) with sensible defaults (`V_th = 0.5`, `g_on = 1e3`, `g_off = 1e-9`).
- **MODIFY** `compute_dc_op` to handle `StoredKind::PWMVoltageSource`, `StoredKind::SineVoltageSource`, `StoredKind::PulseVoltageSource` by evaluating each source's `value_at(opts.t_start)` and treating that as the DC value during the operating-point solve.
- **ADD** Python module `pulsim.v2.simulate(builder, t_end, dt, switch_fn=None, **kwargs)` — a high-level wrapper that:
  - Creates `PwlStateSpaceCache` automatically
  - Builds `SimulationOptions` from kwargs
  - Defaults `switch_fn` to all-OFF if not provided
  - Auto-enables `enable_nonlinear_refresh` if circuit has nonlinear branches
  - Returns the `SimulationResult` directly
- **ADD** Python convenience: `result.plot(node="vout")` returns a matplotlib figure (lazy import).
- **ADD** Tests for all three.

## Impact

- **Affected specs:** modifies `pulsim-v2-builder-api` (body diode helper), `pulsim-v2-python-bindings` (`simulate` wrapper), `pulsim-v2-dc-operating-point` (V11/V12 support).
- **Affected code:**
  - `core/include/pulsim/v2/builder/circuit_builder.hpp` (modify `add_mosfet_level1`)
  - `core/include/pulsim/v2/pwl/dc_assemble.hpp` (extend source dispatch)
  - `python/pulsim/v2.py` (new `simulate` function)
  - `python/bindings_v2_kernel.cpp` (potentially extend run_transient binding)
  - Tests: `core/tests/v2/builder/test_body_diode_helper.cpp`, `python/tests/v2/test_simulate_wrapper.py`
- **Risk:** Low — all three changes are additive (body diode and `simulate` are opt-in; DC OP fix only triggers when those source types are present).
