## Context

A converter's periodic steady state is the fixed point of its one-period
map. The three existing routines approximate that map on the wrong
state (`run_periodic_shooting`) or linearise it once on a grid-quantised
pattern (`steady_state`). The owner's note on `steady_state.py` already
identified the state problem: the fixed engine's map is a map on the
companion history `(v_prev, i_prev)`, not on the MNA vector.

Constraints: the fixed engine keeps its trapezoidal semantics; no
"trapezoid at 2h" start (owner decision); refusals instead of silent
fallbacks; time-weighted averages on adaptive grids (owner note
*time-weight-averages-on-adaptive-grids*) for anything computed from the
orbit.

## Goals / Non-Goals

- Goals: one solver that converges in CCM and DCM; a monodromy matrix
  that is the derivative of the computed map; Floquet multipliers that
  are right when gates depend on the state; a sampled-data small-signal
  model from the same matrices; three legacy entry points reduced to
  wrappers.
- Non-Goals: unknown period (self-oscillating and hysteretic
  converters, frequency-controlled resonant converters with the
  frequency as an unknown); quasi-periodic regimes; harmonic balance in
  the frequency domain; continuous-time (harmonic transfer function)
  small-signal models above the Nyquist frequency of the period.

## Decisions

### D1. The unknown is the true state at a section

`x₀` collects, at `t₀` (by default a rising gate edge): capacitor
voltages, inductor currents, device states (the flat device-state
vector of `add-unified-device-interface`), controller states. The
period map is `P(x₀) = x(t₀ + T; x₀)`.

On the fixed trapezoidal engine a step also carries a derivative
history. With `add-discontinuity-restart-rule` the first step after
`t₀` is taken by the restart rule (two backward-Euler half-steps), which
needs no derivative history, so `P` is a function of the true state
alone. `t₀` must therefore be an instant where the restart rule
applies (a gate edge or a source breakpoint); for a circuit with none,
the solver inserts a restart at `t₀` in every period, the transient it
is compared with included.

- Alternatives: shoot on the companion history (today's
  `steady_state`) — twice the unknowns, engine-specific, and
  meaningless on the exact and TR-BDF2 engines; shoot on the MNA vector
  (today's `run_periodic_shooting`) — wrong fixed point, measured
  −1.07 %.

### D2. Newton with the monodromy matrix

```
G(x₀) = P(x₀) − x₀ = 0
(M − I)·δ = −G(x₀),      x₀ ← x₀ + λ·δ,      M = ∂P/∂x₀
```

`M` is a product over the `K` events of the period:

```
M = Φ_K · S_K · Φ_{K−1} · … · S_1 · Φ_0
Φ_k = ∂x(t_{k+1}⁻)/∂x(t_k⁺)      flow Jacobian of segment k
```

Saltation matrix of a state event `g(x, t) = 0` with a reset
`x⁺ = R(x⁻, t)` (Aizerman & Gantmakher 1958; Leine & Nijmeijer 2004;
for converters, Giaouris et al., IEEE TCAS-I 2008):

```
S = R_x + (f⁺ − R_x·f⁻ − R_t)·∇gᵀ / (∇gᵀ·f⁻ + g_t)
continuous state (R = identity):   S = I + (f⁺ − f⁻)·∇gᵀ / (∇gᵀ·f⁻ + g_t)
time event (instant independent of x):   S = R_x
```

`f⁻`, `f⁺` are the vector fields before and after the event, at the
event state. `∇gᵀ·f⁻ + g_t = 0` is a grazing event: the map is not
differentiable there and the solver reports it.

Measured consequences (`probes/`):

| Circuit | Event | `‖f⁺ − f⁻‖/‖f⁻‖` | Multipliers with `S` | Without `S` |
|---|---|---|---|---|
| DCM buck, resistor-pair diode, `V_th = 0` | diode turn-off (state) | 2e-10 | 0, 0.98898 | identical |
| Peak-current-mode buck, D = 0.559 | switch turn-off at `i_L = I_ref` (state) | O(1) | −1.26729, 0.99895 | 0.9993 ± 0.0146j |

So `S` is always formed, and is numerically `I` for continuous
piecewise-linear laws.

### D3. Flow Jacobians

- Exact propagator: `Φ_k = e^{A_k h_k}`, from the modal form already
  cached per mask; no extra period runs.
- Implicit integrators: the derivative of the discrete step with
  respect to the state at its start. For one implicit step
  `F(x₁, x₀) = 0`: `J·(∂x₁/∂x₀) = −∂F/∂x₀`, with `J` the converged
  Newton matrix already factored; `n` back-substitutions per step for
  `n` state directions. This is the forward-sensitivity recurrence of
  `add-parameter-sensitivities` with the initial state as the
  parameter.
- Until that recurrence exists in the kernel, the implicit path uses
  finite differences of `P` (`n` period runs for `M`), refreshed only
  when the convergence rate degrades (Broyden rank-one updates in
  between).

`M` must be the derivative of the map *as computed*. In the prototype,
replacing the eigen-form propagator by a Padé exponential (whose error
is a staircase in the segment duration) left the fixed point within
3.3e-8 V but turned quadratic convergence into linear convergence at
rate 0.3.

### D4. Globalisation and stopping

- Scaled norm: `‖G‖ = max_i |G_i| / (atol_i + rtol·|x_i|)`, `rtol =
  1e-9`, `atol` from the state scale of the base run.
- Backtracking: `λ ∈ {1, ½, …, 2⁻⁸}`, accept the first with
  `‖G(x₀ + λδ)‖ ≤ (1 − 1e-4·λ)·‖G(x₀)‖`.
- If no `λ` is accepted: advance `n_relax` periods (default 5) by plain
  transient and retry; at most three such rounds.
- Stop at `‖G‖ ≤ 1`, then verify with one more period run.
- Budget: `max_period_runs` (default 60); exceeding it raises with the
  residual history, the last pattern and the multipliers of the last
  `M`.

### D5. Outputs

| Field | Content |
|---|---|
| `snapshot`, `x0` | the orbit's state at `t₀`, resumable |
| `orbit` | one period of waveforms (the result type of `simulate`) |
| `monodromy` | `M` on the true state, with the state labels |
| `multipliers`, `stable` | eigenvalues of `M`; `stable = max|μ| < 1` |
| `pattern` | `(event name, instant/T)` list at convergence |
| `residual`, `n_period_runs`, `residual_history` | convergence record |
| `sampled_model(inputs, outputs)` | `A_d = M`, `B_d = ∂P/∂u` for an input held over the period, `C`, `D` |

`H(z) = C·(zI − M)⁻¹·B + D`, `z = e^{j2πfT}`, valid to `1/(2T)`.
Averages and RMS over the orbit use the time-weighted measures.

A multiplier outside the unit circle does not fail the solve: Newton
finds unstable orbits, and that is the information a designer needs
(`stable = False`, with the multiplier and its type: −1 flip, +1 fold,
complex pair).

### D6. Singular `I − M`

A multiplier equal to 1 means the orbit is not isolated: a lossless LC
loop, a floating capacitor, an integrator with no path to its input.
The solver computes the null vector and names the states with the
largest components. No pseudo-inverse.

### D7. Legacy entry points

| Function | After this change |
|---|---|
| `steady_state(builder, period=, dt=, …)` | wrapper; `dt` selects the fixed engine; returns the new result type with the old field names kept where they still have a meaning |
| `run_periodic_shooting` | wrapper; `x_steady_state` becomes the true state |
| `run_harmonic_balance` | wrapper: solver, then `analyze_harmonics` on the orbit; deprecated name |
| `frequency_response` | uses `sampled_model` |

## Applicability rules

Solved:

1. `period` given; every gate schedule and source has a period that
   divides it (integer ratio within 1e-12); an inverter with an integer
   carrier ratio is solved over its fundamental period.
2. Every state is readable and settable: linear elements always;
   devices through the flat device-state vector; controllers through
   the control graph.

Refused, by name:

- a source or schedule whose period does not divide `period`;
- an opaque Python `switch_fn` on the exact path (no `next_edge_after`),
  because its edges are not reproducible to rounding between runs;
- a Python `ClosedLoop` or `step_observer` with hidden state;
- a stateful device before `add-unified-device-interface` (as today);
- `I − M` singular (D6); a grazing event at the orbit (D2).

Engine choice: the exact strategy when its applicability rule holds;
otherwise TR-BDF2 with tolerances tightened to `rtol ≤ 1e-8` for the
map evaluations; the fixed engine only when the caller passes `dt`.

## Risks / Trade-offs

- Newton leaves the basin on a strongly nonlinear start → backtracking
  and the transient fallback; a warm start (`start_from=`) is kept.
- The pattern changes between iterations (CCM to DCM in the prototype)
  → nothing in the method assumes a fixed pattern; each map evaluation
  locates its own events.
- An adaptive engine makes `P` noisy at the tolerance level, which
  stalls Newton above `rtol` → tightened tolerances for the map, and a
  stop rule that recognises stagnation and reports the achieved
  residual.
- Finite-difference `M` on the implicit path costs `n` period runs →
  Broyden updates, and the analytic recurrence when the sensitivities
  land.
- Results of two public functions move → release note with the measured
  before/after on the DCM and CCM bucks.

## Migration Plan

1. Reference tests (prototype numbers as constants); characterise
   today's three routines on CCM and DCM.
2. Solver in Python on the current `dsed` adapter: map evaluation,
   monodromy from per-mask `(A, b)` and the event log, Newton with
   backtracking.
3. Outputs: multipliers, pattern, `sampled_model`.
4. `steady_state` and `frequency_response` delegate for circuits the
   exact path covers; the old code stays for the rest.
5. After Phase 2: map on the fixed engine and on TR-BDF2 through the
   true state; `run_periodic_shooting` and `run_harmonic_balance`
   delegate; the old implementations are removed.
6. After Phase 3: stateful devices, comparator-driven gates with
   saltation, controller states; kernel implementation with the
   variational recurrence.

Rollback: steps 2–4 add a function and switch two wrappers behind a
check; reverting the wrappers restores today's behaviour.

## Open Questions

1. Name of the public function: `periodic_steady_state`, or keep
   `steady_state` as the one name and change its result type
   (**BREAKING** either way for `history`)?
2. Remove `run_harmonic_balance` after one release, or rename it
   (`steady_state_spectrum`)?
3. An unstable orbit: return it with `stable = False` (proposed), or
   raise unless `allow_unstable=True`?
4. Should a `period_multiple=n` option (subharmonic orbits of period
   `nT`) be part of this change?
5. Unknown period (LLC with frequency control, hysteretic control): a
   later change with a phase condition, or out of scope for 2.x?
6. Default section `t₀`: the first rising gate edge (proposed) or
   `t = 0`?
