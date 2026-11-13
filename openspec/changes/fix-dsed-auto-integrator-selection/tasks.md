## 1. Failing tests first
Circuit: 100 V, 10 Ω, switch (`g_on = 1e3`, `g_off = 1e-9`) to ground with 1 nF across it, gate off for the first half of each 10 µs period, 4 periods (`p8_caploop.py` / `p8c.py`). Exact value in the open interval: `v = 100 − (100 − 0.01)·exp(−(t − T)/1e-8)`.
- [x] 1.1 Python: `engine='dsed'` (default integrator): `max|v − exact|` in the open interval ≤ 1e-5 V, `0 ≤ v ≤ 100.001` everywhere, `n_exact_steps > 0`, `n_bdf2_steps == 0`. Fails today (96.07 V, range −99.98 … 196.08 V)
- [x] 1.2 Python: `integrator='bdf2'`: first sample after each turn-off = `100·(h/τ)/(1 + h/τ)` ± 0.01 V with `h` read from the run's own time grid (90.909 V at the 0.1 µs step it takes today), `max v ≤ 102.8 V`, `min v ≥ 0`. Fails today (166.67 V, −99.98 V)
- [x] 1.3 C++ `bdf2_step` bootstrap on `y' = (100 − y)/τ`, `y₀ = 0`: `h/τ = 100` → `y₁ = 99.0099`, `y₂ = 100.4731`; `h/τ = 10` → `y₁ = 90.9091`; relative 1e-9. Fails today (`y₁ = 196.0784`)
- [x] 1.4 C++ order check: on a non-stiff mode (`h/τ = 0.01`) the global error of BDF2 with the new bootstrap still falls by 4× (±10 %) when `h` is halved
- [x] 1.5 Python: a circuit whose mode has no exact stepper (sine source the stepper does not support, or a defective eigenbasis fixture) still runs on `'auto'` through DOPRI5/BDF2 and matches `engine='pwl'` at fine `dt` within 1e-3 of span

## 2. Fix
- [x] 2.1 `scheduler_auto.hpp`: before the detector's choice, take the exact step when `HasExactStep<System>` and `system_.has_exact_step()`; cap at `dt_max` and the next gate edge; count `n_exact_steps`
- [x] 2.2 `bdf2_integrator.hpp`: backward-Euler bootstrap; update the header comment (order, L-stability)
- [x] 2.3 Stats dict and `_PEDSimulationResult.n_exact_steps`
- [x] 2.4 Update tests that pin BDF2-path numbers (`core/tests/dsed/`, `test_dsed*.py`); each new value from the recurrence or the closed form

## 3. Docs
- [x] 3.1 `CHANGELOG.md` [Unreleased]: the BREAKING note with the table of proposal.md
- [x] 3.2 `simulate` / `_dsed_dispatch` docstrings: what `'auto'` selects, in order
- [x] 3.3 `python/pulsim/AI_GUIDE.md` and `docs/solvers.md`: remove any advice to pass `integrator='rk45'` to get the exact path
- [x] 3.4 Stubs via `scripts/generate_stubs.py` if affected

## 4. Verification
- [x] 4.1 `ruff check`
- [x] 4.2 `pytest python/tests`
- [x] 4.3 `ctest -j2`
- [x] 4.4 `openspec validate --all --strict`
- [x] 4.5 Stub check
- [x] 4.6 Re-run `p8c.py`; the `auto` line must read as the `rk45` line
