# Design — `add-path-based-dsed-engine`

Technical decisions and architectural rationale. Read after
`proposal.md` (motivation) and before `tasks.md` (implementation
checklist).

---

## 1. Architectural placement

DSED is a new **outer loop / scheduler** that consumes the existing
Pulsim infrastructure as a pure dependency:

```
┌─────────────────────────────────────────────────────────────────┐
│  Python facade: pulsim.simulate(b, t_end, engine='dsed', ...)   │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  C++23 pulsim::dsed::Scheduler (NEW, ~500 lines)                 │
│    while t < t_end:                                              │
│      1. predict_next_event(...) → t*                             │
│      2. dt = min(t* - t, controller.next_dt())                   │
│      3. (x_new, err) = integrator.step(A(m), x, dt)              │
│      4. if err > tol: reduce dt, retry                           │
│      5. t += dt; x = x_new                                       │
│      6. if event at t: handle(event); partial_refactor; m = m'   │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼  USES (no modifications)
┌─────────────────────────────────────────────────────────────────┐
│  pulsim::sparse::PulsimSparseLuSolverT  (UNCHANGED, v1.4.0)     │
│  pulsim::pwl::PwlStateSpaceCache         (UNCHANGED, v1.4.0)    │
│  pulsim::pwl::assemble_segment            (UNCHANGED, v1.4.0)   │
└─────────────────────────────────────────────────────────────────┘
```

This separation is deliberate:
- Easier review (no changes to v1.4.0 code paths).
- Easier rollback (the `engine='pwl'` path is untouched).
- Easier benchmarking (clean A/B comparison between engines).

---

## 2. Event prediction algorithm

The event predictor maintains a list of armed predicates
$\{g_i(\vx, t)\}$, computes their values at $t$, and predicts
when any of them will first cross zero.

**Mathematical formulation.** Given:
- State $\vx(t) \in \mathbb{R}^{\nstate}$
- Linear dynamics $\matE \dot{\vx} = \matA(\mathbf{m}) \vx + \matB \vu(t)$
  within the current mask
- Predicates $\{g_i: \mathbb{R}^{\nstate} \times \mathbb{R} \to \mathbb{R}\}$
- Currently armed predicates $A(t) \subset \{1, \ldots, N_g\}$
  (i.e. only $i$ such that $g_i(\vx(t), t) > 0$ AND $\dot g_i < 0$,
  or vice versa)

Predict $t^* = \min_{i \in A(t)} \{\tau > t : g_i(\vx(\tau), \tau) = 0\}$.

**For each armed $i$**, use Hermite cubic interpolation between
$g_i(t)$ and $g_i(t + \Delta t_{\text{nominal}})$ to find the
zero-crossing time analytically. Refine via Newton-Raphson on
$g_i(\tau) = 0$ if needed (typically converges in 2–3 iterations).

**Cost.** $O(|A(t)|)$ per call. For typical SMPS topologies
$|A(t)| \le 10$, so this is negligible compared to the linear
solve. Predicates with $g_i$ and $\dot g_i$ both same-sign are
not armed and incur zero cost.

**Predicate types built-in.**

| Type | Predicate $g_i$ | Triggers on |
|------|-----------------|--------------|
| `GateEdge` | $t - t_{\text{next switch}}$ | scheduled gate edges |
| `DiodeForwardThreshold` | $V_{\text{anode}} - V_{\text{cathode}} - V_{\text{th}}$ | diode turn-on |
| `DiodeReverseBlock` | $I_{\text{diode}}$ | diode turn-off |
| `CurrentZeroCross` | $I_{\text{branch}}$ | DCM detection, body-diode commutation |
| `VoltageThreshold` | $V_{\text{node}_a} - V_{\text{node}_b} - V_{\text{th}}$ | comparator events |
| `Custom` | user lambda | user-defined |

---

## 3. Adaptive integrator selection

Three schemes coexist behind a unified `Step(A, x, dt) → (x_new, err)`
interface:

| Scheme | When | Implementation |
|--------|------|----------------|
| **Trapezoidal** | fallback / debug | reuses existing v1.4.0 code |
| **RK45 (Dormand-Prince)** | non-stiff continuous, $\nstate \le 50$ | classic Butcher tableau, embedded error |
| **BDF2** | stiff (eigenvalue ratio > 10) | implicit, calls `solver.solve(...)` per step |
| **Krylov-Φ** | $\nstate \ge 50$, well-conditioned | rational-Krylov $e^{A \Delta t}$ via Saad's algorithm |

**Selection rule.**

```
def pick_integrator(mask):
    A = J(mask)
    eigval_ratio = compute_eigval_ratio_cached(A)  # once per mask
    if eigval_ratio > 10:           # stiff
        return BDF2
    if nstate(A) >= 50:             # large state
        return KrylovPhi
    return RK45                     # default
```

`compute_eigval_ratio_cached(A)` runs power iteration for the largest
eigenvalue and inverse iteration for the smallest, both with $O(\nstate^2)$
cost. Cached per mask in the `PwlSegment` struct.

**Override.** User can force `integrator='rk45'|'bdf2'|'krylov_phi'|'trap'`
to bypass selection for debugging.

---

## 4. PI-controlled step adaptation

Step size after step $n$ with local-truncation-error estimate
$\text{err}_n$:

\[
\Delta t_{n+1} = \Delta t_n \cdot \min\left(\rho_{\max},
    \left(\frac{\text{tol}_n}{\text{err}_n}\right)^{k_P}
    \cdot \left(\frac{\text{err}_{n-1}}{\text{err}_n}\right)^{k_I}
\right)
\]

with $\text{tol}_n = r_\text{tol} \cdot \|\vx_n\| + a_\text{tol}$,
default $r_\text{tol} = 10^{-6}$, $a_\text{tol} = 10^{-9}$,
$k_P = 0.7$, $k_I = 0.3$, $\rho_{\max} = 5$. The PI form is from
arXiv 2503.09898 (March 2025).

**Rejection rule.** If $\text{err}_n > \text{tol}_n$, reject the
step, halve $\Delta t$, retry. Cap consecutive rejections at 5
before raising `DSEDError` (likely a stiffness regime not handled).

---

## 5. Event handling state machine

On event firing at $t^*$:

```
1. Resolve event:
   - Read predicate type (GateEdge / DiodeOn / etc.)
   - Determine new mask m' (toggle the appropriate bit)
2. Compute exact x(t*) via the chosen integrator's interpolant
3. Update mask: m → m'
4. changed_cols = bits_diff(m, m')          // 1 or more bits
5. if changed_cols.size() == 0:              // no change (spurious)
       continue
6. if cache.has_segment(m'):                 // already factored
       L, U, P = cache.get(m')
       if path_len(changed_cols) <= MAX_PATH_LENGTH_RATIO * n:
           solver.partial_refactor(J(m'), changed_cols)
       // else cached factor already current; no work
7. else:                                     // new mask
       segment = cache.add(m')
       solver.analyze(segment.J)             // pay analyze + factorize
       solver.factorize(segment.J)
8. Re-arm predicates for new mask m'
9. Continue scheduler loop at t* with x(t*) and m'
```

**Telemetry.** Each event records:
- `(t, event_type, old_mask, new_mask)` → `result.event_log`
- Path-based hit rate: `cache.metrics.rank1_hits` etc.

---

## 6. Backward DSED for parasitic stiffness

When `eigval_ratio > 100` (severe stiffness, e.g. parasitic
ringing after switching), the forward integrator either fails or
chokes. The Wang 2021 backward DSED adds:

1. Backward integration from $t^*$ slightly into the past to
   damp the parasitic mode before the event.
2. Forward integration with BDF2 + tight tolerance through the
   event itself.
3. Standard scheduler resumes after parasitic transient dies.

Implementation: 2 extra BDF2 calls per stiff event. Cost overhead
$\sim 5\%$ wall-clock; worth it for the stability gain.

---

## 7. Output sampling

DSED produces an irregular time grid. For user convenience two
output modes:

- **`output='native'`** (default): `result.times` and
  `result.states` are the actual DSED event grid + integrator
  steps. Useful for waveform inspection, FFT (after resampling).
- **`output='fixed_dt'`**: results interpolated to a uniform
  grid (`result.times = arange(0, t_end, output_dt)`). Cost: one
  Hermite interpolation per output sample. Easier downstream
  consumption.

Both modes share the same internal scheduler; only the post-
processing differs.

---

## 8. Threading model

Single-threaded by design. Parallelism opportunities:
- **Event predictor**: predicates are independent → trivially
  parallelisable, but typical $|A(t)| \le 10$ makes this
  pointless.
- **Krylov-Φ**: matrix-vector products are sparse → parallelism
  possible at very large $\nstate$, but Pulsim's target range
  doesn't motivate this.
- **Per-segment parallelism**: distinct masks could solve on
  distinct threads in batch sweep workloads, but that's a
  separate OpenSpec (sweep-orchestrator).

Deferring to v3.0.0 (not in this proposal).

---

## 9. Backwards compatibility

`simulate(...)` without `engine=` defaults to `engine='pwl'`
(v1.4.0 behaviour). All existing notebooks, projects, tests
continue to work without changes.

Breaking changes introduced by v2.0.0:
- **None at the C++ API.** All new code is in `pulsim::dsed::*`
  namespace.
- **None at the Python API for default usage.** `engine='pwl'`
  is the default.
- **Result struct extended** with `event_log: list[Event]`
  (empty list when engine='pwl' for forward compatibility).
- **CMake target extended** (`pulsim::core` gains the dsed
  module unconditionally; can be excluded via
  `PULSIM_BUILD_DSED=OFF` for very minimal builds).

---

## 10. Why this composes elegantly with v1.4.0

| v1.4.0 contribution | How DSED uses it |
|---------------------|--------------------|
| `PulsimSparseLuSolverT::partial_refactor` | Called at every event handler (single- and multi-bit) |
| `PwlStateSpaceCache::solve_rank1` | Backing store for per-segment factors |
| `refactor_parametric` (parametric mode) | Used by DSED when scheduler triggers a parameter change (rare; for adaptive load studies) |
| Path-based event-driven cache invalidation | Already invalidated on mask change; reused as-is |
| Pivot-fault recovery | DSED inherits the same fallback path |
| Templated `PulsimSparseLuSolverT<Scalar>` | DSED uses real-scalar version; complex is for AC sweep (separate path) |
| Captured benchmark infrastructure | DSED adds `[dsed][microbench]` Catch2 tag |

The fit is so clean that DSED essentially *just* swaps the outer
loop. Every algorithmic improvement Pulsim makes to the LU kernel
(future reachability-based fast path, adaptive threshold, etc.)
benefits both engines automatically.

---

## 11. Open technical questions

These need to be resolved during Gate 0 (literature deep dive)
before writing any code:

1. **What is the DSIM event-priority resolver?** Public DSIM docs
   are vague. We may need to invent our own deterministic
   priority rule for simultaneous events.
2. **Newton convergence rate on multi-event predicates?**
   Hermite cubic may not be enough; might need root-bracketing
   first to ensure unique root in the interval.
3. **How does PLECS handle the "almost simultaneous" event case
   (events within machine epsilon of each other)?** Our
   conservative choice: fire in priority order, never collide.
4. **What's the right cache-eviction policy for the segment
   dictionary under DSED?** Fixed-dt PWL cache assumes long
   simulations re-visit masks; DSED's variable step may visit
   masks more episodically. May need LRU eviction.

---

## 12. References (informal — corrected after Gate 0 lit-dive)

- Tsinghua DSED line:
  - **Zhu, Zhao, Shi, Yu**, *Discrete State Event-Driven Framework
    with a Flexible Adaptive Algorithm for Simulation of Power
    Electronic Systems*, IEEE TPE 34(12), 11692–11705, 2019.
    [IEEE 8675318] — note original 2019 paper attribution
    corrected from the OpenSpec draft (was Yan/Wang/Zhao).
  - **Wang/Li/Zhao**, *Backward Discrete State Event-Driven
    Approach for Simulation of Stiff Power Electronic Systems*,
    IEEE TPE, 2021. [IEEE 9351911] — note "backward" =
    implicit/quantized, NOT BDF (per `notes/DSED_FOUNDATIONS.md` §3).
  - **Liu, Shi, Zhao**, *Hybrid Time and Event Co-simulation
    Framework for Power Electronics Systems*, IEEE PEDG, 2023.
  - **US Patent US10970432B2** (Zhu/Zhao/Shi/Yu, Tsinghua) —
    full FA-DSED algorithm disclosure.
  - **US Patent US10747918B2** (Wang/Li/Zhao) — BDSED disclosure.
- DSIM commercial: <https://www.dsimtechnology.com/>
- PLECS variable-step: <https://docs.plexim.com/plecs/5.0/>;
  *PLECS Nanostep* article in *Bodo's Power Systems*, Feb 2024.
- Cardiff matrix-exp for PE: Wang & Niu, EPSR 2020.
- Adaptive ODE solvers references:
  - **Hairer, Norsett, Wanner**, *Solving ODE I: Nonstiff
    Problems*, Springer 1993/2008, §II.5 (DOPRI5).
  - **Hairer, Wanner**, *Solving ODE II: Stiff and DAE Problems*,
    Springer 1996, §V.1 (BDF).
  - **Söderlind**, *Automatic Control and Adaptive Time-Stepping*,
    Numerical Algorithms 31, 281–310, 2002.
  - **Saad**, *Analysis of Some Krylov Subspace Approximations
    to the Matrix Exponential Operator*, SIAM JNA 29, 209–228,
    1992.
  - **Sidje**, *EXPOKIT*, ACM TOMS 24, 130–156, 1998 (φ-functions).
- **NOT a DSED paper**: arXiv:2503.09898 (Huang/Liu/Sun/Qiu) is on
  power-grid transient stability with adaptive-order DTM. We
  borrow its **PI-controller formula** (which is portable, see
  §13.3 below) but do not cite it as DSED prior art.

---

## 13. Concrete formulas — ready for C++ translation

This section was added during Gate 0 to fix the algorithms before
Gate 1 prototype work begins. Each subsection ends with the C++23
implementation sketch the developer can lift directly into
`core/include/pulsim/dsed/`.

### 13.1 Event prediction — Illinois root-finder

We replace the secant method used by Tsinghua (patent
US10970432B2 claim 1) with the **Illinois algorithm** (a
modified regula-falsi that prevents stagnation on near-tangent
crossings — the failure mode at critically-damped diode
commutation).

Given predicate $g(\tau)$ and interval $[\tau_a, \tau_b]$ with
$g(\tau_a) \cdot g(\tau_b) < 0$:

```
def illinois(g, ta, tb, ga=None, gb=None, tol=1e-9, max_iter=20):
    if ga is None: ga = g(ta)
    if gb is None: gb = g(tb)
    side = 0
    for _ in range(max_iter):
        tc = (ta*gb - tb*ga) / (gb - ga)
        gc = g(tc)
        if abs(gc) < tol or abs(tb - ta) < tol:
            return tc
        if gc*gb < 0:                       # root in [tc, tb]
            ta, ga = tc, gc
            if side == -1: gb *= 0.5        # Illinois weight
            side = -1
        else:                                # root in [ta, tc]
            tb, gb = tc, gc
            if side == +1: ga *= 0.5
            side = +1
    raise EventPredictorError("Illinois did not converge")
```

**Cost.** $\sim$10–15 predicate evaluations per event on typical
diode-commutation curves. Each predicate eval is one
linear-system query + arithmetic, $O(\nstate)$. Use Hermite
cubic interpolation of $g$ between $\tau_a$ and $\tau_b$ for
intermediate evaluations to avoid re-integrating the state ODE.

Fallback to **Brent's method** if Illinois reports stagnation
twice on the same event (3× slower per iteration but bulletproof).

### 13.2 Dormand-Prince RK45 (DOPRI5) — default integrator

7-stage, order 5(4), FSAL (First-Same-As-Last). Reference:
Hairer/Norsett/Wanner 1993, §II.5.

**Butcher tableau:**

```
c    | A
-----+----------------------------------------------
0    |
1/5  | 1/5
3/10 | 3/40        9/40
4/5  | 44/45       -56/15       32/9
8/9  | 19372/6561  -25360/2187  64448/6561  -212/729
1    | 9017/3168   -355/33      46732/5247   49/176     -5103/18656
1    | 35/384      0            500/1113     125/192    -2187/6784      11/84
-----+----------------------------------------------
b    | 35/384      0            500/1113     125/192    -2187/6784      11/84       0     (order 5)
b̂    | 5179/57600  0            7571/16695   393/640    -92097/339200   187/2100    1/40  (order 4)
```

**Update + embedded error:** with
$k_i = f(t_n + c_i h, \, \vx_n + h \sum_j A_{ij} k_j)$,

$$
\vx_{n+1} = \vx_n + h \sum_i b_i k_i,
\qquad
\mathbf{err} = h \sum_i (b_i - \hat b_i) k_i
$$

**FSAL:** $k_7 = f(t_{n+1}, \vx_{n+1})$ becomes $k_1$ of the next
accepted step → 6 RHS evals per accepted step.

### 13.3 BDF2 — fallback for stiff intervals

Reference: Hairer & Wanner 1996, §V.1.

**Fixed-step:**
$$
\vx_{n+1} = \tfrac{4}{3} \vx_n - \tfrac{1}{3} \vx_{n-1}
           + \tfrac{2}{3} h \cdot f(t_{n+1}, \vx_{n+1})
$$

**Linear case** (our case between events, $f = \matA \vx + \vb$):
$$
\Bigl(\Eye - \tfrac{2}{3} h \matA\Bigr) \vx_{n+1}
   = \tfrac{4}{3} \vx_n - \tfrac{1}{3} \vx_{n-1}
     + \tfrac{2}{3} h \cdot \vb(t_{n+1})
$$
**One sparse linear solve per step**, reusing the LU of
$(\Eye - \tfrac{2}{3} h \matA)$ until $h$ or topology changes.

**Variable-step** (ratio $\rho = h_n / h_{n-1}$):
$$
\alpha_0 = \tfrac{(1+\rho)^2}{1+2\rho}, \quad
\alpha_{-1} = -\tfrac{\rho^2}{1+2\rho}, \quad
\beta = \tfrac{1+\rho}{1+2\rho}
$$
$$
\vx_{n+1} = \alpha_0 \vx_n + \alpha_{-1} \vx_{n-1}
           + \beta h_n f(t_{n+1}, \vx_{n+1})
$$
Stability requires $\rho \le 1.91$ — PI controller must clamp.

**Stability.** BDF2 is A-stable but not L-stable
($R(\infty) = 1/3$). For very stiff transients consider
**RADAU IIA** (Hairer/Wanner §IV.8) instead: order-5, 3-stage,
fully implicit, L-stable. Per-step cost ~3× BDF2 but converges
faster on the stiffest intervals. Decision deferred to Gate 4
empirics on LLC fixture.

### 13.4 PI-controlled step adaptation

Reference: Söderlind 2002, §4 (gains for order-5 methods).

$$
\Delta t_{n+1} = \Delta t_n \cdot \min\Bigl(\rho_{\max},
   (\text{tol}/\text{err}_n)^{k_P}
   \cdot (\text{err}_{n-1}/\text{err}_n)^{k_I}\Bigr)
$$

**Defaults:** $k_P = 0.7$, $k_I = 0.3$ for DOPRI5; $k_P = 0.4$,
$k_I = 0.3$ for BDF2 (per Söderlind Table 5 H211b); $\rho_{\max}
= 5$, $\rho_{\min} = 0.1$, safety factor $0.9$.

**Norm.** Per-component scale:
$$
\text{sc}_i = a_{\text{tol}} + r_{\text{tol}} \cdot \max(|x_n^i|, |x_{n+1}^i|), \qquad
\|\mathbf{err}\| = \sqrt{\frac{1}{\nstate} \sum_i \frac{(\text{err}_i)^2}{\text{sc}_i^2}}
$$
Defaults $r_{\text{tol}} = 10^{-6}$, $a_{\text{tol}} = 10^{-9}$.

**Acceptance:** $\|\mathbf{err}\| \le 1$. Otherwise: $h \leftarrow
\max(0.1 \cdot h, h \cdot (1/\text{err})^{k_P} / \text{safety})$;
cap at 5 consecutive rejections.

**Wind-up protection:** clamp
$\text{err}_{n-1} \in [0.1, 1.0]$ between steps; reset to $1.0$
on any topology change.

### 13.5 Krylov-Φ matrix exponential — large $\nstate$ option

References: Saad 1992, Sidje 1998 (EXPOKIT), Niesen/Wright 2012.

For LTI step $\vx_{n+1} = e^{\matA h} \vx_n + h \cdot \varphi_1(\matA h) \cdot \vb_n$
with $\varphi_1(z) = (e^z - 1)/z$, build $m$-step Arnoldi:
$$
\matA \matV_m = \matV_{m+1} \matH_{m+1, m}, \quad
\|\matV_m^T \matV_m - \Eye\| < 10^{-12}
$$
Then $e^{\matA h} \vv \approx \beta \matV_m e^{\matH_m h} \vec{e}_1$
with $\beta = \|\vv\|$. Compute small-dense $e^{\matH_m h}$ via
scaling-and-squaring + Padé[6/6] (Higham 2005). Use augmented
matrix trick for φ-functions (Sidje 1998 Eq. 2.7).

**A-posteriori error** (Saad Thm 4.5):
$\text{err}_m \approx \beta \cdot H_{m+1, m} \cdot |\vec{e}_m^T e^{\matH_m h} \vec{e}_1|$.

**Adaptive $m$.** Start $m = 10$, double until $\text{err}_m \le \text{tol}$,
cap $m_{\max} = 40$.

**Cost.** Arnoldi: $m \cdot (\nnz(\matA) + 2 m \nstate)$ flops.
Dense expm: $O(m^3)$. Storage: $(m+1) \cdot \nstate$ doubles.

**Use only when:** $\nstate \ge 50$ AND topology stable for $\ge
10$ consecutive steps (to amortise Arnoldi setup) AND
$\kappa(\Eye - \tfrac{2}{3} h \matA)$ makes LU expensive (i.e.
fill-in $\gg \nnz(\matA)$).

### 13.6 Integrator selector — Gate 4 deliverable

```python
def select(n_state, lambda_max, lambda_min_stiff,
           event_density_per_sec, h_target):
    stiffness = abs(lambda_max) / max(abs(lambda_min_stiff), 1e-30)
    events_per_step = event_density_per_sec * h_target

    if events_per_step > 0.2:
        return "RK45"             # explicit, dense-output for event location

    if stiffness < 50:
        return "RK45"             # FSAL beats anything implicit

    if n_state < 50 or stiffness < 1e4:
        return "BDF2"             # 1 cached LU per topology, A-stable

    if n_state >= 50 and stiffness >= 1e4 and events_per_step < 0.05:
        return "Krylov-phi"

    return "trap"                  # safe fallback
```

### 13.7 Open implementation questions (resolved at Gate 1)

The lit-dive surfaced 5 questions that need empirical answers
before Gate 2 starts. They are tracked in `notes/DSED_FOUNDATIONS.md`
§7 and copied here for visibility:

1. Default $m_{\max}$ for Krylov-Φ (target: $m \in [10, 30]$,
   test on LLC at Gate 4).
2. Illinois-vs-Brent fallback policy (test on bouncy critically-
   damped commutation at Gate 3).
3. Default hysteresis band for chatter prevention (start $\pm 1\%$
   of threshold magnitude, tune per-circuit at Gate 1).
4. Path-cache eviction policy (FIFO at Gate 1; LRU only if profiler
   demands at Gate 5).
5. Event-priority resolution convention: **gate edges > diode
   events > custom predicates**, fired in that order on
   simultaneous timestamps. Document this convention in
   user-facing docs + tests.

---

## 14. Patent-safety summary (Gate 0 finding)

The Tsinghua FA-DSED algorithm is patented (US10970432B2). To
operate safely outside its broadest claims:

| Patent claim | Our deliberate choice |
|--------------|-----------------------|
| Variable-order Taylor recursion | DOPRI5 + BDF2 + Krylov-Φ — none of which is Taylor |
| 4 event classes (control/external/passive/state) | **5-tier resolver** (gate / diode-on / diode-off / current-ZC / voltage-threshold + user custom) — different taxonomy |
| Secant method on polynomial threshold | **Illinois (with Brent fallback)** — different root-finder |

The BDSED patent (US10747918B2) covers state-quantization;
we do not use quantization, so we are clearly outside.

**Public-facing language.** Our paper and docs refer to the
*"DSED paradigm"* (a class of algorithms predating Tsinghua) but
describe our specific implementation as **"Path-Based
Event-Driven (PED)"** to avoid any branding collision with
DSIM. The OpenSpec change-id retains `dsed` for SEO/discovery
purposes only.

A lawyer review of these choices is scheduled at Gate 5 before
v2.0.0 release.
