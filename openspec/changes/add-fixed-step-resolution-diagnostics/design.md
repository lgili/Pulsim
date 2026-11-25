## Context

The fixed engine takes the `dt` it is given. Its honest records today
are `dt_retries`, `event_iteration_breaches` and
`inductor_guard_actions`; none says "this step size does not resolve
this circuit". The trapezoidal rule fails in two recognisable ways: its
local error grows as `h³·y‴`, and for `h > 2τ` its amplification
factor `(1 − h/2τ)/(1 + h/2τ)` is negative, which shows as an
alternation that the rule never damps.

Evidence: `raw/numerics.md` item 6, `p13.py` (engine), and the numpy
prototype `p24_resolution_proto.py` (the diagnostics computed on a
re-implementation of the step, on today's engine and with the restart
rule of `add-discontinuity-restart-rule`).

## Goals / Non-Goals

- Goals: an under-resolved fixed-step run says so, names the variable
  and the time, and costs no second run; a well-resolved converter run
  stays silent; no numerical result changes.
- Non-Goals: step control (that is the variable-step engine); a bound
  on the global error; an estimate over the stateful devices' states;
  diagnostics for the variable-step and exact engines; changing `dt`
  automatically.

## Decisions

### D1. Truncation estimate from three derivative slots

The companion already stores, per capacitor, `i_n` and, per inductor,
`v_n`; `f = i_n/C` and `f = v_n/L` are `dy/dt` at the committed point.
For the trapezoidal rule the local error is `−(h³/12)·y‴`, and
`y‴ ≈ (f_{k+1} − 2·f_k + f_{k−1})/h²`, so

`e_k = (h/12)·|f_{k+1} − 2·f_k + f_{k−1}|`.

It needs two vectors of length "number of dynamic branches" kept in
the loop (not in `HistoryState`, so snapshots and resume are
unchanged) and no solve.

It is reported relative to the running peak of the variable,
`r_k = max_i e_{i,k} / max_{j ≤ k}|y_{i,j}|`: the fraction of a
variable's own magnitude lost to truncation in one step. A running
peak needs no second pass and no user tolerance.

The stencil must not straddle a discontinuity, where `y‴` does not
exist: an estimate is formed only when its three slots were produced
under one stamped mask, with no restart step and no source jump among
them, and not on the first two steps of a run.

Prototype, RC with a 10 V step, today's engine:

| `dt` | true max error / 10 V | `r_max` | alternating run |
|---|---|---|---|
| 10 τ | 1.7e-1 | 3.5e-1 | 6 |
| 4 τ | 3.2e-1 | 1.8e-1 | 3 |
| τ | 3.0e-1 | 2.7e-2 | 0 |
| τ/2 | 1.9e-1 | 7.5e-3 | 0 |
| τ/10 | 4.8e-2 | 3.3e-4 | 0 |
| τ/100 | 5.0e-3 | 3.3e-6 | 0 |

(The true error at fine steps is dominated by the first-order start,
which this estimate does not see; with the restart rule the true error
at τ/10 is 2.2e-3 and `r_max` is 2.7e-4.)

- Alternatives considered:
  - *Trapezoidal against a backward-Euler predictor*
    (`raw/numerics.md`'s wording). It needs a second solve per step,
    or an explicit predictor that is unstable on stiff rows, and it
    measures the predictor's O(h²) error, not the rule's.
  - *Normalising by a tolerance* (`atol + rtol·|y|`, as the
    variable-step engine does). A fixed-step user has set no
    tolerance; a default one would make the number mean "what another
    engine would accept".
  - *Normalising by the variable's range over the run.* Needs the end
    of the run; the per-sample trace could not be written as it goes.
  - *An accumulated (summed) estimate as a global error bound.* The
    sum over-states a decaying system's error by a factor that depends
    on the circuit; not offered as a bound. The trace is there for
    users who want to sum it.

### D2. Ringing detector

For every state entry (node voltages and branch currents) the loop
keeps the previous difference and a counter. A step extends the run
when `Δ_k·Δ_{k−1} < 0` and
`|Δ_k| > max(0.01·max(|x_k|, |x_{k−1}|), 1e-9·peak)`. A run of four or
more is recorded once, with the entry's name, the time it started, its
length so far and its largest `|Δ|`. The local-magnitude test matters:
a node that is 100 V when a switch is off and 10 mV when it is on would
hide a 4 % alternation of its on-state value behind its peak.

Prototype (largest run of alternating steps):

| Circuit | today's engine | with the restart rule |
|---|---|---|
| buck CCM, T/20 … T/400 | ≤ 1 | ≤ 1 |
| buck DCM, T/100 and T/400 | 49 and 195 | 1 and 1 |
| hard-switched 1 nF / 1 mΩ, dt = 1e-8 | 499 | 1 |
| same, dt = 1e-9 | 2071 | 517 |
| same, dt = 1e-10 | 263 | 165 |

- Alternatives considered: an FFT bin at Nyquist over a window (a
  transform per window and a window length to choose); a threshold
  against the running peak only (misses the case above).

### D3. When to warn

One `ResolutionWarning` per run when `r_max > 1e-2` or at least one
ringing record exists. On the prototype this warns for the RC at
`dt ≥ τ` and not at `τ/2` or below; for the DCM buck and the capacitor
loop on today's engine; and, with the restart rule, for the capacitor
loop only (`r_max` = 2.7e-2 at dt = 1e-8, where `dt` equals the 10 ns
charging time constant, and ringing at 1e-9 and 1e-10). Both bucks
stay silent with the restart rule from T/20 to T/400.

The message gives the variable, the time, the two numbers, and the
two actions: reduce `dt`, or omit `dt` to let the variable-step engine
choose. It does not say by how much to reduce `dt`.

### D4. Cost and switch

Two passes per step: one over the dynamic branches, one over the state
vector, both a handful of flops per entry, plus one `Real` per stored
sample. `SimulationOptions::resolution_diagnostics` (default true)
turns both off. The states are bit-identical either way.

### D5. Scope of the estimate

Linear capacitors and inductors only, the set the variable-step
engine's own estimate covers. The nonlinear capacitor, the saturable
inductor, the Lauritzen diode, the IGBT tail and the PMSM are covered
by the ringing detector through their node voltages and branch
currents, not by the truncation estimate. The result says how many
dynamic branches the estimate covered.

## Risks / Trade-offs

- Warning fatigue. A snubber with `τ < dt` is under-resolved by
  design in most converter runs, and will warn. Mitigation: one
  warning per run, a dedicated category that `warnings.filterwarnings`
  can silence, the option, and thresholds chosen so that the
  prototype's converters are silent.
- On today's engine the detector reports the post-event alternation
  the engine itself creates (DCM idle interval). That report is true:
  those samples are wrong. It disappears with the restart rule.
- The thresholds (1e-2, 1 %, four steps) come from one prototype and
  five circuits.
- Per-step overhead on the loop that
  `update-fixed-step-loop-bookkeeping` is making cheaper. Mitigation:
  the measurement gate in tasks; the option.
- The estimate is not a bound, and the docs must not present it as
  one.

## Migration Plan

Ship on by default: no result changes. Projects that turn warnings
into errors get a CHANGELOG note with the filter line. Rollback:
`resolution_diagnostics=False`.

## Open Questions

1. Default thresholds: warn at `r_max > 1e-2` and a 1 % ringing
   amplitude over four steps, or stricter?
2. Should a run of the examples and the benchmark circuits be used to
   tune them before release, with the list of circuits that warn
   attached to the pull request?
3. Should the warning be off by default for the first release and
   only the record be populated?
