## Why

Pulsim has three periodic-steady-state routines. None of them works in
discontinuous conduction, and two answer a different question from the
one asked. Measured on the audit's DCM buck (24 V, 100 µH, 47 µF, 50 Ω,
D = 0.3, 100 kHz; exact `v_C` at the rising edge
8.99624708728943 V), audit `raw/numerics.md` item 8, probes
`numerics/p11_pss.py`, `p11b.py`, `p11c.py`:

| Routine | Where | Method | DCM buck |
|---|---|---|---|
| `steady_state` | `python/pulsim/steady_state.py:134` | one affine solve `(I − Φ)h* = c` on the trapezoidal companion history; Φ by finite differences of `n` one-period fixed-step runs | raises "no periodic orbit found" (`:338-349`), also when warm-started from 200 periods |
| `run_periodic_shooting` | `python/pulsim/periodic.py:94` | Newton with a finite-difference Jacobian on the MNA vector passed as `initial_state` (`:176-190`) | converges in 5 iterations to 8.8998 V, **−1.07 %** |
| `run_harmonic_balance` | `python/pulsim/harmonic.py:246` | `run_periodic_shooting`, then an FFT | inherits the −1.07 % |

- `steady_state` is not iterative: it linearises once around its start
  and requires every probe run to keep the base run's commutation
  pattern (`steady_state.py:249-296`), with commutation instants rounded
  to the step grid (`:253-255`). In DCM the pattern at the start is not
  the pattern at the orbit, and the instant of the diode turn-off moves
  with the state, so no single affine map exists. It works in CCM
  (27–114 ms).
- `run_periodic_shooting` restarts each period from an MNA vector, which
  resets the companion history, so it finds the fixed point of a
  different dynamical system. `steady_state.py:13-19` records the same
  effect as 0.14 % on a CCM buck. It also costs `N + 1` one-period runs
  per iteration.
- `frequency_response` (`python/pulsim/sampled_ac.py:97, 122`) is built
  on `steady_state`, so the fast Bode path is CCM-only as well (by
  reading; not run).
- All three need a fixed `dt`, and `steady_state` refuses every stateful
  device (`steady_state.py:225-242`).

Prototype (`probes/shooting_dcm.py`, numpy): Newton shooting on the
true state `(i_L, v_C)` with exact per-mode propagators and located
events converges on that DCM buck from the zero state in 4 Newton steps
(5 evaluations of the period map), passing through a CCM pattern on the
way, to `v_C = 8.9962470872 V`; 4000 periods of brute force agree to
5e-11. The monodromy matrix equals a finite difference of the map to
1e-8 and gives the Floquet multipliers 0 and 0.98898.

Two findings from the prototype shape the proposal:

- **In DCM with today's diode the event-time correction is the
  identity.** With resistor-pair switches and `V_th = 0` the vector
  field is continuous at the diode turn-off (relative jump 2e-10), so
  the saltation matrix is `I` and the plain product of propagators is
  already the monodromy. What makes DCM work is Newton on the true state
  with events located in time, not saltation.
- **Saltation is essential where the field jumps at a state event.** On
  a peak-current-mode buck (`probes/peak_current_mode.py`; D = 0.559)
  the multipliers are −1.26729 and 0.99895 with saltation, against the
  analytic `−D/(1 − D) = −1.2674` and −1.26729 by finite difference;
  without it they are 0.9993 ± 0.0146j, which reports a subharmonically
  unstable loop as stable. The same holds for a diode with a hysteresis
  band (`V_th > V_f`), for parameter-dependent gate instants, and for the
  ideal switch of `add-branch-form-ideal-switch`.

## What Changes

- **One solver**, `periodic_steady_state(circuit, period, …)`: Newton on
  `G(x₀) = P(x₀) − x₀`, where `P` is the one-period map on the true
  state (capacitor voltages, inductor currents, device and controller
  states) and the Jacobian is `M − I`, `M` the monodromy matrix.
- **Monodromy as a product**: `M = Φ_K·S_K·…·S_1·Φ_0`, `Φ_k` the flow
  Jacobian of segment `k` and `S_k` the saltation matrix of event `k`.
  On the exact propagator `Φ_k = e^{A_k h_k}`; on the implicit
  integrators `Φ_k` is the derivative of the discrete step, obtained
  with the cached factor. `M` is the derivative of the map as computed,
  so Newton converges quadratically.
- **Globalisation**: backtracking on `‖G‖`, a bounded number of
  transient periods as a fallback, a bounded total budget, and a
  residual history in every failure.
- **Outputs**: the orbit (snapshot and one period of waveforms), `M`,
  the Floquet multipliers with a stability verdict, the commutation
  pattern and instants, the residual, the number of period runs, and the
  sampled-data model `x[k+1] = M·x[k] + B·u[k]`, `y = C·x + D·u` that
  `frequency_response` uses.
- **Refusals by name**: a source or gate that is not `T`-periodic, a
  marginal direction (`I − M` singular, with the states involved), a
  device or controller whose state the solver cannot read or set.
- **Consolidation**: `steady_state`, `run_periodic_shooting`,
  `run_harmonic_balance` and `frequency_response` call the one solver.
- **BREAKING:** `run_periodic_shooting` and `run_harmonic_balance`
  results move (the bias from the history reset disappears: −1.07 % on
  the DCM buck, 0.14 % on the CCM buck of the `steady_state` docstring).
- **BREAKING:** `SteadyStateResult.history` (the companion history
  vector) is replaced by the true state; `monodromy` is expressed on the
  true state. `run_harmonic_balance` keeps its name for one release with
  a deprecation note, since it is not harmonic balance.

## Impact

- Affected specs: `periodic-steady-state` (new).
- Affected code: new `python/pulsim/periodic_steady_state.py` (slice 1)
  and, after Phase 3, a kernel `analysis/periodic.hpp`;
  `python/pulsim/{steady_state,periodic,harmonic,sampled_ac}.py` reduced
  to wrappers; `dsed/builder_adapter.hpp` and
  `python/dsed_bindings.cpp` (per-mask `A`, `b`, event log with the
  predicate gradient); docs (steady-state and Bode guides), stubs.
- Depends on:
  - `add-discontinuity-restart-rule` (Phase 2): on the fixed engine the
    true state is sufficient only when a period starts with the restart
    rule instead of a carried derivative history.
  - `add-fixed-step-event-splitting` (Phase 2): the fixed engine's map
    is differentiable in the event instant only once events are located.
  - `add-exact-propagator-strategy`: the product-of-propagators form.
  - `add-unified-device-interface` (Phase 3): stateful devices, through
    the flat device-state vector.
  - `add-unified-event-queue` (Phase 3): state-dependent gate edges
    (comparators), which are the events that need saltation.
  - `add-compiled-control-graph`: controller states in the periodic
    state.
  - `add-parameter-sensitivities` shares the variational machinery; the
    steady-state gradients are specified there.
- Ships on today's structure (tasks 1–4): the solver for circuits the
  current `dsed` exact stepper covers (linear elements, switches, PWL
  diodes, DC and sine sources, C++ schedules), in Python over the
  existing adapter, with `steady_state` and `frequency_response`
  delegating to it for those circuits.
- Must wait: the fixed and TR-BDF2 engines (Phase 2), Newton and
  stateful devices and comparator-driven gates (Phase 3), closed loops
  (control graph).
- Whether PLECS or PSIM use the same formulation was not checked; the
  statement that steady-state and loop-gain analyses are the most
  requested analyses in power electronics is the audit's judgement.
