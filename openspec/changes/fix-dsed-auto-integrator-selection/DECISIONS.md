# fix-dsed-auto-integrator-selection — decisions (2026-11-12)

The proposal has no Open Questions. Choices made where it left room:

| Question | Option taken | Why | How to reverse |
|---|---|---|---|
| Where the exact step sits in the auto scheduler | Before the detector's integrator step, in its own block (`2a`) that counts, fires a due gate edge, re-selects the detector's choice and records the sample, then `continue`s | Mirrors the exact path of the RK45 scheduler; leaves the RK45/BDF2 code untouched | Delete block `2a` in `scheduler_auto.hpp` |
| Step cap | `min(dt_max, t_gate − t)` | Task 2.1; the same cap as the RK45 scheduler, so sample density is unchanged | — |
| Should `integrator='rk45'` also count `n_exact_steps`? | Yes, in `PEDResult` and its stats dict | One name for one quantity across both schedulers | Drop the field from `PEDResult` |
| BDF2 bootstrap | Backward Euler with `b(t + h)` | Proposal; L-stable; first order for one step, global order 2 kept (test 1.4) | Restore the Crank–Nicolson block |
| Fixture for task 1.5 (a mode with no exact stepper) | A trapezoidal pulse source on an RC | A sine source has a closed-form propagator (the oscillator is augmented into the state), so it does not exercise the numeric path; PWM/pulse sources do not | — |

## Tests that moved

- `test_dsed_end_to_end.py::test_dsed_integrator_auto_picks_rk45_on_non_stiff_buck`
  asserted `n_rk45_steps > 0` on `'auto'`. The buck's modes are LTI
  with a constant source, so `'auto'` now takes 1000 exact steps and no
  RK45 step; renamed `..._takes_the_exact_stepper_on_buck`, asserts
  `n_exact_steps > 0`, `n_bdf2_steps == 0`; final `v_C` 12.0000 V (the
  5 % bound is unchanged).
- `core/tests/dsed/test_bdf2_scheduler.cpp`: comments only (they named
  the Crank–Nicolson start); its 10 % RMSE bound still passes.
- `docs/how-pulsim-works/14-dsed-bdf2-and-dispatch.md` described the
  Crank–Nicolson start and "auto behaves as rk45 on a buck"; rewritten.

## p8c.py after the fix

```
dsed integrator=auto: n=42  max|v-exact| in off interval=1.000e-06 V; v range overall [0.000,100.000]
dsed integrator=rk45: n=42  max|v-exact| in off interval=1.000e-06 V; v range overall [0.010,100.000]
dsed integrator=bdf2: n=402 max|v-exact| in off interval=9.085e+00 V; v range overall [0.000,102.767]
```

`auto` reads as `rk45` (before: 96.07 V, range −99.98 … 196.08 V).
`bdf2` keeps a 9.09 V error on the first sample after each opening:
backward Euler at `h/τ = 10` gives 90.91 V where the exact value is
≈ 100 V; it no longer overshoots or goes negative.
