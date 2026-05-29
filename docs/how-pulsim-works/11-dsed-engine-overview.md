# 11. The Path-Based Event-Driven (PED / DSED) engine

Chapters 2-9 covered Pulsim's default simulation paradigm — **fixed-step
trapezoidal companion + PWL state-space cache**. This chapter starts
Part II of the book: a **second simulation engine** that landed in
v1.6.0 under `pulsim.simulate(..., engine='dsed')`. Where the PWL
engine takes a constant $\Delta t$ and stamps the same MNA system
50,001 times to cover a 5 ms / 100 kHz buck CCM, the **Path-Based
Event-Driven (PED, aka DSED)** engine:

1. Predicts the next event time analytically (PWM gate edges,
   body-diode commutation, voltage thresholds — no aliasing).
2. Integrates between events with an adaptive-step ODE solver
   (Dormand-Prince RK45 or implicit BDF2, dispatched per-mode).
3. Handles mask transitions instantaneously at the predicted time.

On the same buck CCM benchmark, DSED takes **1007 steps in 2.21 ms**
— 50× fewer steps and **24× faster wall-clock** than PWL. The
geometric-mean speedup across 6 converter topologies (buck, boost,
buck-boost, half-bridge + AC input, floating-cap RLC, NPC split-bus)
is **14.5×**.

The next five chapters walk through the math + architecture top-down.
This chapter just sets up the *why*.

---

## 11.1 Why fixed-step trap struggles with PE switching

The PWL engine wins by precomputing the trapezoidal companion
factor $(J = G_{\text{static}} + \tfrac{2}{\Delta t} M_{\text{dyn}})$
per switch mask and reusing the LU at every step — chapter 4. The
trade-off it accepts: $\Delta t$ must be **small enough that no PWM
gate edge is missed by more than $\Delta t/2$**.

For a 100 kHz switching converter ($T_{\text{sw}} = 10$ µs) the
canonical heuristic is $\Delta t \le T_{\text{sw}}/100 = 100$ ns —
otherwise edge aliasing produces spurious sub-harmonics in the
output. That's 50,001 steps for a 5 ms transient. Of those:

| Where the time goes | Fraction |
|---|---:|
| Triangular solve through cached LU       | ~95% |
| `b_extra` overlay (PWM/sine source) | ~3% |
| `step_observer` callbacks | ~1% |
| Misc (loop overhead, history update) | ~1% |

The LU solve cost per step is already minimal (~1 µs for $n \le 10$
states). **You can't make a fixed-step engine faster by making each
step faster** — you'd need to take fewer steps. And the fixed-step
constraint says no.

There's a structural observation hiding here: **between consecutive
PWM gate edges, the circuit is a constant-coefficient LTI system**.
On `[t_k, t_{k+1}]` where $t_k$ and $t_{k+1}$ are consecutive gate
times, the dynamics are exactly

$$
\dot{\mathbf{x}}(t) = A_{m(t_k)} \mathbf{x}(t) + \mathbf{b}_{m(t_k)}(t)
$$

with $A_{m}, \mathbf{b}_m$ frozen — only the time-varying part of
$\mathbf{b}$ (sine source) moves smoothly. There's no reason to
re-evaluate the dynamics every 100 ns when they don't change.
A high-order adaptive RK can cover the entire $(t_k, t_{k+1})$
segment in **just 1-2 steps** while keeping accuracy at $10^{-7}$.

That observation is the entire DSED win.

---

## 11.2 The PED scheduler in one figure

The outer loop, written in pseudocode:

```text
t  = 0
x  = x0
m  = switch_fn(0)            # initial mode (mask)
A, b = extractor(m)           # per-mask LTI state-space
h  = h_init

while t < t_end:
    # 1. PREDICT next event time analytically.
    t_gate  = switch_fn.next_edge_after(t)
    t_pred  = min(t_gate, t_end)

    # 2. INTEGRATE between events with adaptive step.
    while t < t_pred:
        h = min(h, t_pred - t)               # never overshoot the event
        x_new, err = rk45_step(A, b, x, h)
        if PI_controller.accept(err):
            t += h
            x = x_new
            scan_for_predicate_events()     # body diode, V_thresh, etc.
            h = PI_controller.new_step(err) # grow if accurate
        else:
            h = PI_controller.shrink(err)   # try again with smaller h

    # 3. FIRE event: swap mask, INVALIDATE FSAL caches, restart h.
    if t_pred == t_gate:
        m_new = switch_fn(t + ε)
        if m_new != m:
            m = m_new
            A, b = extractor(m)              # cached after first hit
            rk45_state.invalidate()
            PI_controller.reset()
```

Three things hide the entire complexity:

| Symbol | What it is | Chapter |
|---|---|---|
| `extractor(m)` | MNA $\to$ continuous-time $(A_m, \mathbf{b}_m)$, with floating-cap handling | [12](12-dsed-state-space-extraction.md) |
| `rk45_step + PI_controller + scan_for_predicate_events` | DOPRI5 with FSAL, Söderlind PI step controller, cubic Hermite interp + Illinois root | [13](13-dsed-dopri5-adaptive.md) |
| Per-mode dispatch RK45↔BDF2 | Stiffness detector + auto-dispatcher (`PEDSimulatorAuto`) | [14](14-dsed-bdf2-and-dispatch.md) |

The native C++ implementation of the loop + the
`NativePwm2Switch` fast path that gives DSED its 24× story are
covered in [chapter 15](15-dsed-native-bindings.md). The end-to-end
speedup tables are in [chapter 16](16-dsed-benchmarks.md).

---

## 11.3 Why "path-based"

The "**P**" in PED comes from a structural insight that distinguishes
this scheduler from a textbook adaptive RK with event detection:

> **The full simulation $[0, t_{\text{end}}]$ is partitioned into
> a sequence of *mode segments* $(t_k, t_{k+1})$ separated by
> events (PWM edges, body-diode commutations). Each mode segment
> is an LTI ODE with constant $(A_m, \mathbf{b}_m)$. The scheduler
> walks the segment graph instead of the time axis.**

This is the same structural exploitation that the PWL engine uses
to cache LU factors per mask (chapter 4) — applied at a different
level. Where PWL says "different mask $\Rightarrow$ different LU,
reuse aggressively", PED says "**different mask $\Rightarrow$
different ODE, predict the boundary, integrate freely between**".

The two engines are complementary:

| | PWL (chapter 4) | PED (this chapter) |
|---|---|---|
| Step size | Fixed $\Delta t$ | Adaptive $h$ between events |
| Per-mask cost | One LU + many solves | One state-space extract + many RK45 steps |
| Hot loop | Triangular solve | RK45 stage (`A·x` + stages) |
| Gate-edge handling | Edge aliasing within $\Delta t/2$ | Exact analytical placement |
| Best for | Many modes, short transients, nonlinear devices | Few modes, smooth between events, LTI per mode |
| Worst case | High switching freq → many wasted steps | Very stiff system + many events → small $h$ |
| Failure mode | Numerical noise from misaligned edges | Extractor rejects nonlinear devices |

In practice the two **share the same** `PwlStateSpaceCache` —
DSED's `compute_lti_state_space(mask)` is an extra method on the
PWL cache (chapter 12). So users get both engines from the same
`pulsim.CircuitBuilder`; the `engine='dsed'` vs `engine='pwl'`
kwarg switches the consumer of the cached factors.

---

## 11.4 Scope today

DSED handles any circuit Pulsim can express that satisfies:

1. **LTI per mask** — every switch-state combination must give a
   constant-coefficient ODE. This excludes Shockley diodes, MOSFET
   $V_{th}$ models, saturable inductors. For those, users go through
   the explicit `pulsim.dsed.run_user_lti(system, switch_fn, x0, t_end, ...)`
   API with a hand-rolled LTI system (validated on buck DCM with
   body-diode commutation in
   [Gate 3](https://github.com/lgili/Pulsim/blob/main/notes/GATE3_PROGRESS.md)).
2. **No parallel capacitors** — two caps sharing both terminals
   create a singular $M_{\text{dyn}}$ block. Rejected with an
   error pointing to the merge-equivalent workaround.
3. **No inductor loops** — same problem in the dual. Rejected
   with a "merge into $L_{\text{eq}}$" message.

What it does support:

- Buck, boost, buck-boost, flyback, forward (caps to ground)
- Half-bridge, full-bridge (caps to ground)
- **NPC split DC bus** — floating cap stack, handled by
  $T^\top M T$ congruence (chapter 12 §12.5)
- **MMC submodule stacks** — floating cap chain, same congruence
- **PFC with AC input** — sine-driven via the time-varying
  source overlay (chapter 12 §12.6)
- **Grid-tied inverter** — sine $V_{\text{grid}}$ same way

---

## 11.5 Takeaways

- **PWL wins on robust, no-surprises throughput** at small $\Delta t$.
  DSED wins when the per-cycle PWM rhythm is predictable and the
  between-event dynamics are smooth.
- The DSED engine is **opt-in** via `engine='dsed'`. PWL stays
  default for backward compatibility.
- The 24× buck CCM number is wall-clock on a 5 ms / 100 kHz
  simulation — not a microbench. Across 6 converter topologies
  the geo-mean speedup is 14.5× ([chapter 16](16-dsed-benchmarks.md)).
- The architecture composes with the existing
  `PwlStateSpaceCache` (chapter 4) — no fork of the kernel.

## Further reading

- [Chapter 12 → MNA $\to$ state-space extraction](12-dsed-state-space-extraction.md)
- [Chapter 13 → DOPRI5 + adaptive PI + event scan](13-dsed-dopri5-adaptive.md)
- [Chapter 14 → BDF2 + stiffness detector + auto-dispatch](14-dsed-bdf2-and-dispatch.md)
- [Chapter 15 → Native C++ bindings (the 24× story)](15-dsed-native-bindings.md)
- [Chapter 16 → DSED benchmarks](16-dsed-benchmarks.md)

External references:

- **Hairer, Wanner**, *Solving ODEs II: Stiff and DAE Problems*,
  Springer 1996 — DOPRI5 + BDF families, PI step control.
- **Söderlind, G.**, *Automatic control and adaptive
  time-stepping*, Numerical Algorithms 31, 2002 — the PI controller
  Pulsim uses.
- **Najm, F.**, *Circuit Simulation*, Wiley 2010 §3.4.2 —
  MNA $\to$ state-space reduction via Schur complement.
- **Allmeling & Hammer**, *Discrete State Event-Driven simulation*,
  PLECS technical reports — the modern industrial state-event
  family DSED descends from.
