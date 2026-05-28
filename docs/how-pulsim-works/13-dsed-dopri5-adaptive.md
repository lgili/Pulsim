# 13. DOPRI5 + adaptive PI + event scan

Chapter 12 reduced the per-mask MNA system to an LTI ODE
$\dot{\mathbf{x}}_s = A_m \mathbf{x}_s + \mathbf{b}_m(t)$. This
chapter explains **how the PED scheduler integrates that ODE
between events**:

1. **DOPRI5** (Dormand-Prince 5(4)) — a 7-stage explicit RK with
   FSAL (First-Same-As-Last) optimisation. §13.1.
2. **Söderlind PI step controller** — picks $h_{\text{new}}$
   from the local error estimate using both proportional and
   integral feedback. §13.2.
3. **Cubic Hermite interpolation** + **Illinois root finder** —
   for sub-step predicate events (body diodes, voltage thresholds)
   that don't align with gate edges. §13.3.

The DOPRI5 + PI combination is the textbook choice for non-stiff
adaptive RK (Hairer-Wanner Vol. I); we don't claim novelty there.
What's worth documenting is **how it's wired to the event-driven
outer loop** — that's where the PED architecture pays off.

---

## 13.1 DOPRI5 — Dormand-Prince 5(4) with FSAL

DOPRI5 is a 7-stage explicit Runge-Kutta method with embedded
4th-order error estimator. For a system
$\dot{\mathbf{x}} = f(t, \mathbf{x})$, one step from $(t_n, \mathbf{x}_n)$ to
$(t_n + h, \mathbf{x}_{n+1})$ evaluates:

$$
\begin{aligned}
\mathbf{k}_1 &= f(t_n,\, \mathbf{x}_n) \\
\mathbf{k}_i &= f\Bigl(t_n + c_i h,\; \mathbf{x}_n + h\sum_{j=1}^{i-1} a_{ij}\, \mathbf{k}_j\Bigr), \quad i=2,\ldots,7 \\
\mathbf{x}_{n+1}^{(5)} &= \mathbf{x}_n + h \sum_{i=1}^{7} b_i\, \mathbf{k}_i \quad\text{(5th-order)} \\
\mathbf{x}_{n+1}^{(4)} &= \mathbf{x}_n + h \sum_{i=1}^{7} \hat{b}_i\, \mathbf{k}_i \quad\text{(embedded 4th)}
\end{aligned}
$$

The local error estimate is the 5th-vs-4th difference:

$$
\mathbf{e}_{n+1} = \mathbf{x}_{n+1}^{(5)} - \mathbf{x}_{n+1}^{(4)}
                = h \sum_{i=1}^{7} (b_i - \hat{b}_i)\, \mathbf{k}_i
$$

The full Butcher tableau Pulsim uses is the standard
Dormand-Prince (1980):

$$
\begin{array}{c|ccccccc}
0       & & & & & & & \\
1/5     & 1/5 & & & & & & \\
3/10    & 3/40 & 9/40 & & & & & \\
4/5     & 44/45 & -56/15 & 32/9 & & & & \\
8/9     & 19372/6561 & -25360/2187 & 64448/6561 & -212/729 & & & \\
1       & 9017/3168 & -355/33 & 46732/5247 & 49/176 & -5103/18656 & & \\
1       & 35/384 & 0 & 500/1113 & 125/192 & -2187/6784 & 11/84 & 0 \\ \hline
b_i^{(5)} & 35/384 & 0 & 500/1113 & 125/192 & -2187/6784 & 11/84 & 0 \\
\hat{b}_i^{(4)} & 5179/57600 & 0 & 7571/16695 & 393/640 & -92097/339200 & 187/2100 & 1/40 \\
\end{array}
$$

### FSAL — First-Same-As-Last

Notice the last row of $a_{ij}$ equals $b_i^{(5)}$: the 7th stage
is evaluated **at** $(t_n + h, \mathbf{x}_{n+1}^{(5)})$. If the
step is accepted, that same evaluation is $\mathbf{k}_1$ for the
**next** step. The FSAL trick saves one $f$-evaluation per step
(reducing the cost from 7 to 6 RHS calls).

The C++ implementation `pulsim::dsed::rk45::step` keeps an
`RK45State::k7_cached` between steps; on `invalidate()` (called at
gate events or when the controller rejects), the cache is dropped
and the next step pays the full 7 evaluations.

---

## 13.2 Söderlind PI step controller

Given the per-component scaled error norm

$$
\text{err} = \sqrt{\frac{1}{n_s} \sum_{i=1}^{n_s}
                    \left(\frac{e_{n+1,i}}{\text{atol} + \text{rtol}\cdot\max(|x_{n,i}|, |x_{n+1,i}|)}\right)^2}
$$

the controller decides whether to **accept** the step and what
$h_{\text{new}}$ to try next.

**Acceptance**: $\text{err} \le 1$ accepts. Otherwise reject, shrink
$h$, retry.

### The PI law

Söderlind 2002 (*Automatic control and adaptive time-stepping*)
proves that classical I-only step control — $h_{\text{new}} = h \cdot
\text{err}^{-1/(p+1)}$ — has poles on the imaginary axis and
oscillates around the optimal $h^\ast$. A **PI controller** stabilises
the loop:

$$
\boxed{
h_{\text{new}} = \text{safety} \cdot h \cdot
                  \underbrace{\left(\frac{1}{\text{err}}\right)^{k_P}}_{\text{proportional}}
                  \cdot
                  \underbrace{\left(\frac{\text{err}_{\text{prev}}}{\text{err}}\right)^{k_I}}_{\text{integral}}
}
$$

For DOPRI5 ($p = 5$) Pulsim uses the standard Söderlind constants:

| Constant | Value | Role |
|---|---:|---|
| $k_P$ | $0.7 / p = 0.14$ | Proportional gain |
| $k_I$ | $0.4 / p = 0.08$ | Integral gain |
| safety | 0.9 | Reduces $h$ slightly below ideal to allow margin |
| $h_{\max} / h_{\min}$ | 5 / 0.2 | Per-step growth/shrink caps |

### Accept-path vs reject-path

```cpp
// In pulsim::dsed::PIController::accept(...)
const Real factor = std::pow(Real{1} / err, kP)
                  * std::pow(err_prev / err, kI);
const Real h_proposed = h * std::clamp(safety * factor,
                                        h_shrink_min,
                                        h_grow_max);

if (err <= Real{1}) {
    // Accept the step. Store err for next iteration's I term.
    err_prev = std::max(err, Real{1e-2});  // protect against zero
    return {true, h_proposed};
}
// Reject — back off, retry. NO `err_prev` update (avoid windup).
return {false, h * std::clamp(safety * std::pow(Real{1}/err, kP),
                                h_shrink_min, h_grow_max)};
```

The reject path uses **only** the proportional term (no integral
accumulation across failed attempts) and the same safety factor.

### A subtle bug we fixed (Gate 4.D.3)

The original implementation had the rejection branch as

```cpp
factor = std::pow(Real{1} / err, kP) / safety;  // BUG
```

For `err` just over 1.0, `(1/err)^kP` was ~0.99 and dividing by
0.9 (safety) gave ratio ~1.1 → step grew on reject. The PI
controller would oscillate forever. Fix: `safety * (1/err)^kP`
per Hairer-Wanner Vol. II §IV.2. After the fix the controller
converges in 1-2 retries on every test scenario.

---

## 13.3 Sub-step predicate events: Hermite + Illinois

PWM gate edges are **scheduled** events — the scheduler knows
exactly when they fire (`switch_fn.next_edge_after(t)` returns the
analytical time). The DSED scheduler simply caps the step at the
predicted gate-edge time, integrates up to it exactly, swaps the
mask, restarts.

**Predicate events** are different. Examples:

- Body-diode commutation: triggered when $i_L$ crosses zero in
  DCM, time unknown a priori.
- Voltage-threshold protection: triggered when $v_{\text{out}} > V_{\text{max}}$.
- Custom user predicates: anything `g(t, x) = 0`.

These can fire **inside** a successful RK step at some
$t^\ast \in (t_n, t_n + h)$. The scheduler needs to:

1. **Detect** that a predicate sign-change happened in the step.
2. **Locate** $t^\ast$ to high precision without re-integrating.
3. **Apply** the state projection (e.g., zero $i_L$ at commutation)
   and reset $h$.

### Step 1: detection via Hermite interpolation

After accepting the DOPRI5 step we have $(t_n, \mathbf{x}_n,
\mathbf{k}_1)$ at the left endpoint and $(t_n+h, \mathbf{x}_{n+1},
\mathbf{k}_7)$ at the right (both FSAL). The cubic Hermite
interpolant fits an order-4-accurate trajectory through both
endpoints:

$$
\mathbf{x}(t_n + \theta h) = h_{00}(\theta)\mathbf{x}_n
                              + h_{10}(\theta)\, h\, \mathbf{k}_1
                              + h_{01}(\theta)\mathbf{x}_{n+1}
                              + h_{11}(\theta)\, h\, \mathbf{k}_7
$$

with the Hermite basis functions

$$
h_{00}(\theta) = 2\theta^3 - 3\theta^2 + 1, \quad
h_{10}(\theta) = \theta^3 - 2\theta^2 + \theta, \quad
h_{01}(\theta) = -2\theta^3 + 3\theta^2, \quad
h_{11}(\theta) = \theta^3 - \theta^2
$$

for $\theta \in [0, 1]$. Evaluate each registered predicate
$g_k(t, \mathbf{x})$ at $\theta = 0$ and $\theta = 1$; a **sign
change** means a root in $(0, 1)$.

The interpolation is FREE — the FSAL slots $\mathbf{k}_1$ and
$\mathbf{k}_7$ are already in memory.

### Step 2: localisation via Illinois

For each predicate that flipped sign, the Illinois (modified
regula falsi) root finder finds $t^\ast$ inside the step:

```text
g_left, g_right are the predicate values at θ = 0 and θ = 1.
Both have opposite signs (we wouldn't be here otherwise).

while abs(t_right - t_left) > tol:
    t_mid = t_left - g_left * (t_right - t_left) / (g_right - g_left)
    g_mid = g(t_mid)                       # interpolate via Hermite

    if g_left * g_mid < 0:
        # Root in [t_left, t_mid]
        if g_right > 0 was just halved: g_right /= 2  # Illinois rule
        t_right = t_mid
        g_right = g_mid
    else:
        if g_left > 0 was just halved: g_left /= 2
        t_left  = t_mid
        g_left  = g_mid
```

The Illinois variant halves the *non-bracketing* endpoint's
$g$-value to break the slow-convergence pathology of plain regula
falsi (where one endpoint stays stuck). Convergence is
**superlinear** (order ~1.65 — golden ratio).

**Fallback**: if Illinois fails to converge in 20 iterations
(possible for non-monotone predicates near a root), the scheduler
switches to **Brent's method** which guarantees convergence in
$O(\log(1/\epsilon))$ via bracket halving + inverse quadratic
interpolation.

### Step 3: state projection at $t^\ast$

For body-diode commutation, the user supplies a
`state_projection_fn(t, x)` callback that zero-clamps the inductor
current at the commutation:

```python
def project_at_zcd(t, x):
    x[I_L_INDEX] = 0.0   # snap to zero — DCM transition
    return x
```

The scheduler:

1. Evaluates `x_hermite = hermite_interp(t_star)` to get the
   pre-projection state at $t^\ast$.
2. Calls `x = state_projection(t_star, x_hermite)`.
3. Logs the event in `result.event_log` with type=`Predicate`.
4. **Invalidates** the RK45 FSAL cache and resets the PI controller
   (the projected state has a different first derivative — old
   $\mathbf{k}_7$ is stale).
5. Continues integration from $(t^\ast, x_{\text{projected}})$.

Validated end-to-end in Gate 3 on a buck DCM scenario: the
scheduler placed the body-diode commutation within $10^{-9}$ s of
the analytical zero-crossing, matching a PWL fixed-step reference
to 5 significant figures on the output ripple.

---

## 13.4 Worked example: 1 RK step in the buck CCM hot loop

Concrete numbers from `test_dsed_returns_fewer_steps_than_pwl` at
$t = 1.5 \times 10^{-5}$ s (well into transient, $\mathbf{x} =
[6.2, 4.7]$ for $[v_C, i_L]$):

1. `switch_fn.next_edge_after(1.5e-5)` returns `2.0e-5` (next
   HS-off transition).
2. Controller proposes $h = 2.7 \times 10^{-7}$ s (PI says "previous
   step was clean, grow 1.2×"). Capped at `t_gate - t = 5e-6` —
   plenty of room.
3. `rk45_step(f, t, x, h, state)` runs 6 evaluations of $f =
   A\mathbf{x} + \mathbf{b}$ (FSAL inherits $\mathbf{k}_1$ from the
   previous step), returns
   $\mathbf{x}_{\text{new}} = [6.21, 4.92]$ and $\mathbf{e} =
   [1.4 \times 10^{-8}, 2.9 \times 10^{-8}]$.
4. Error norm: $\sqrt{\tfrac{1}{2}((1.4 \times 10^{-8}/(10^{-9} +
   10^{-6}\cdot 6.21))^2 + \ldots)} \approx 0.014$.
5. Controller: $\text{err} = 0.014 \ll 1$, accept; propose $h_{\text{new}}
   = 0.9 \cdot 2.7\times 10^{-7} \cdot (1/0.014)^{0.14} \cdot
   (\text{err}_{\text{prev}}/0.014)^{0.08} \approx 4.1 \times 10^{-7}$ s.
6. No predicate sign change ($i_L > 0$ throughout — buck CCM has no
   body diode). Commit, advance to $t = 1.527 \times 10^{-5}$ s.

Total per-step C++ cost: ~3.8 µs on a M1 (chapter 16 §16.2). Of
that:

| Stage | Cost | Where |
|---|---:|---|
| 6× RHS (A·x + b) | ~1.6 µs | dense gemv at $n_s = 2$, vectorised |
| PI controller decision | ~0.1 µs | scalar arithmetic |
| `compute_*_b_extra` walk | ~0.4 µs | pool iteration (could be cached, future Bridge.14) |
| Bookkeeping (state copy, history) | ~0.1 µs | small allocations |
| Per-step total | **~3.8 µs** | (vs 60 µs in pure Python — chapter 15) |

---

## 13.5 Takeaways

- DOPRI5 is the right choice for non-stiff PE between events:
  high order (5) on smooth segments, FSAL halves cost, error
  estimator is "free".
- The PI step controller is what keeps the adaptive size **near
  optimal** without oscillation. The 2002 Söderlind paper is
  the literature anchor.
- Cubic Hermite + Illinois lets us locate sub-step predicate
  events in $O(\log(1/\epsilon))$ without re-integrating —
  critical for body-diode commutation in DCM converters.
- Between gate edges the scheduler grows $h$ to whatever
  accuracy allows. On buck CCM the grown $h$ approaches
  $\sim T_{\text{sw}}/2 \approx 5$ µs (capped by the next gate
  edge, not by accuracy).

## Further reading

- C++ implementation: `core/include/pulsim/dsed/rk45_dormand_prince.hpp`,
  `step_controller.hpp`, `event_predictor.hpp`.
- Python prototype (Gate 1): `prototype/dsed/rk45_dormand_prince.py`.
- [Chapter 14 → BDF2 + stiffness detector + auto-dispatch](14-dsed-bdf2-and-dispatch.md)

External references:

- **Dormand & Prince**, *A family of embedded Runge-Kutta formulae*,
  J. Comp. Appl. Math. 6 (1980).
- **Söderlind, G.**, *Automatic control and adaptive
  time-stepping*, Numerical Algorithms 31 (2002) — the PI law +
  pole analysis.
- **Hairer, Norsett, Wanner**, *Solving Ordinary Differential
  Equations I*, Springer 1993 — DOPRI5 + Hermite continuous
  extension, the canonical reference.
- **Brent, R.**, *Algorithms for Minimization Without Derivatives*,
  Prentice-Hall 1973 — Brent's method root finder.
