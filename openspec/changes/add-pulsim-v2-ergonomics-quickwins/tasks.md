## 1. Body diode auto-helper

- [ ] 1.1 Modify `CircuitBuilder::add_mosfet_level1` signature with optional `with_body_diode = false`
- [ ] 1.2 When true, auto-add `add_diode(source, drain, g_on=1e3, g_off=1e-9, V_th=0.5)` after the MOSFET
- [ ] 1.3 Same change for `add_igbt_level1` (collector-emitter body diode is rarer in real IGBTs but useful sometimes)
- [ ] 1.4 YAML support: `with_body_diode: true`
- [ ] 1.5 Python binding default arg
- [ ] 1.6 Test: MOSFET w/ body diode produces 2 branches (mosfet + diode)

## 2. DC operating point for V11/V12

- [ ] 2.1 Extend `dc_assemble.hpp` to evaluate `PWMVoltageSource::value_at(opts.t_start)`, `SineVoltageSource::value_at(opts.t_start)`, `PulseVoltageSource::value_at(opts.t_start)` and use those as the source values
- [ ] 2.2 Test: run a circuit with a sine source w/ `start_from_dc_op = true`; verify no exception + result matches direct evaluation
- [ ] 2.3 Test: same for pulse source (before vs after t_start)

## 3. Python `simulate` wrapper

- [ ] 3.1 Write `pulsim.v2.simulate(builder, t_end, dt, **kwargs)` in `python/pulsim/v2.py`
- [ ] 3.2 Auto-detect nonlinear branches → set `enable_nonlinear_refresh=True`
- [ ] 3.3 Default `switch_fn` to all-OFF mask
- [ ] 3.4 Lift common SimulationOptions kwargs (`max_newton_iterations`, `tol_newton_dx`, etc.)
- [ ] 3.5 Return SimulationResult directly
- [ ] 3.6 Optional: `result.plot(node="vout")` using lazy matplotlib import

## 4. Tests

- [ ] 4.1 C++ test: body-diode helper adds correct branch
- [ ] 4.2 C++ test: DC OP for sine source matches manual eval
- [ ] 4.3 Python test: `simulate(...)` works on a simple RC
- [ ] 4.4 Python test: `simulate(...)` on a MOSFET CS amp auto-uses Newton refresh

## 5. Documentation + commit

- [ ] 5.1 Update CircuitBuilder docstrings
- [ ] 5.2 Update `pulsim/v2.py` module docstring with `simulate(...)` example
- [ ] 5.3 `openspec validate add-pulsim-v2-ergonomics-quickwins --strict`
- [ ] 5.4 Commit and push
