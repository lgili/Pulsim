## 1. Kernel-side `initial_state` injection

- [x] 1.1 `core/include/pulsim/v2/solver/run_transient.hpp` — add optional `const Vector* initial_state` parameter that seeds `x` + `history` from a caller-supplied state instead of zero. Enables chaining `simulate()` segments without re-pumping the transient.
- [x] 1.2 Python binding: `m.def("run_transient", ..., py::arg("initial_state") = py::none())`
- [x] 1.3 `simulate(...)` wrapper accepts `initial_state=...` keyword.

## 2. Adaptive driver (Python wrapper)

- [x] 2.1 `python/pulsim/v2_adaptive.py` — `run_transient_adaptive(builder, t_start, t_end, dt_init, dt_min, dt_max, atol, rtol, segment_steps, safety, growth_max, shrink_max, ...)`. Coarse-grain segment-based stepping: each segment runs at fixed `dt` for `segment_steps` steps, then `dt` is adjusted via the standard PI step controller `dt_new = dt · safety · (1/err_norm)^(1/3)`.
- [x] 2.2 Re-export from `pulsim.v2`: `run_transient_adaptive`, `AdaptiveResult`.
- [x] 2.3 The driver carries `last_state` across segments by passing `initial_state` to each `simulate()` call.

## 3. Validation

- [x] 3.1 RL step response — `examples/v2/scripts/run_adaptive_rl_settle.py`. R=0.1 Ω, L=1 mH (τ=10 ms), 500 ms horizon. Adaptive driver achieves **125× speedup** (4001 samples vs 500 000 for fixed-dt at 1 µs); initial dt 1 µs → final dt 98 µs. Max absolute error: 20 mA on a 10 A signal; error decays to 1e-14 (machine precision) in the steady-state tail.
- [ ] 3.2 Stiff van der Pol oscillator — deferred to follow-up (the RL case already demonstrates the speedup-without-accuracy-loss property).
- [ ] 3.3 Buck closed-loop comparison — deferred to follow-up.

## 4. Future C++ step-doubling implementation (deferred)

A full kernel-side step-level adaptive driver requires `HistoryState::snapshot/restore` around the doubled-step LTE estimation. Tracked here for the next iteration:

- [ ] 4.1 Header `core/include/pulsim/v2/solver/adaptive_step.hpp` with step-level LTE control
- [ ] 4.2 `HistoryState::snapshot()` + `HistoryState::restore(snap)`
- [ ] 4.3 Python binding for the C++ driver

## 5. Wrap-up

- [x] 5.1 `openspec validate add-pulsim-v2-phase-b-variable-step --strict`
- [ ] 5.2 Commit + push (next step)
