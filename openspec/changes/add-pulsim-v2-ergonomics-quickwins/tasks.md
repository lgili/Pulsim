## 1. Body diode auto-helper

- [x] 1.1 Modify `CircuitBuilder::add_mosfet_level1` signature with optional `with_body_diode = false`
- [x] 1.2 When true, auto-add `add_diode(source, drain, g_on=1e3, g_off=1e-9, V_th=0.5)` after the MOSFET
- [ ] 1.3 Same change for `add_igbt_level1` — **deferred** (real IGBTs have no body diode; user can call `add_diode(...)` explicitly if needed)
- [ ] 1.4 YAML support: `with_body_diode: true` — **deferred** (Python + C++ surfaces are enough for now)
- [x] 1.5 Python binding default arg
- [x] 1.6 Test: MOSFET w/ body diode produces 2 branches (mosfet + diode)

## 2. DC operating point for V11/V12

- [x] 2.1 Extend `dc_assemble.hpp` to evaluate `PWMVoltageSource::value_at(t_eval)`, `SineVoltageSource::value_at(t_eval)`, `PulseVoltageSource::value_at(t_eval)` and use those as the source values. Also wired CurrentSource + VCVS dispatch. `run_transient` threads `opts.t_start` to `compute_dc_op`.
- [x] 2.2 Test: DC OP for sine source at T/4 matches +amplitude
- [x] 2.3 Test: DC OP for pulse source (before, during, after t_start)

## 3. Python `simulate` wrapper

- [x] 3.1 Write `pulsim.v2.simulate(builder, t_end, dt, **kwargs)` in `python/pulsim/v2.py`
- [x] 3.2 Auto-detect nonlinear branches via `DevicePool.has_nonlinear_devices()` → set `enable_nonlinear_refresh=True`
- [x] 3.3 Default `switch_fn` to all-CLOSED mask (more useful than all-OFF; switches need to start in a defined state)
- [x] 3.4 Lift common SimulationOptions kwargs (`max_newton_iterations`, `max_event_iterations`)
- [x] 3.5 Return SimulationResult directly
- [ ] 3.6 Optional: `result.plot(node="vout")` — **deferred** (matplotlib dependency on hot path is iffy; users plot themselves)

## 4. Tests

- [x] 4.1 C++ test: body-diode helper adds correct branch (test_mosfet_level1.cpp)
- [x] 4.2 C++ test: DC OP for sine/pwm/pulse/current/vcvs sources (test_dc_assemble.cpp)
- [x] 4.3 Python test: `simulate(...)` works on a simple RC
- [x] 4.4 Python test: `simulate(...)` on a MOSFET CS amp auto-uses Newton refresh

## 5. Documentation + commit

- [x] 5.1 Update CircuitBuilder docstrings (with_body_diode docstring added)
- [x] 5.2 Update `pulsim/v2.py` module docstring with `simulate(...)` example
- [x] 5.3 `openspec validate add-pulsim-v2-ergonomics-quickwins --strict`
- [x] 5.4 Commit and push (commit `dd330d5`)
