# DSED Foundations — Gate 0 Deliverable

**Project:** `add-path-based-dsed-engine` (OpenSpec proposal,
target Pulsim v2.0.0).
**Gate:** 0 — Literature deep dive + design memo.
**Author:** Luiz Carlos Gili (synthesised by AI research agents).
**Date:** 2026-05-26.
**Status:** 📋 Draft awaiting user approval before Gate 1 begins.

This memo synthesises three deep research surveys covering the
Tsinghua DSED line, DSIM/PLECS commercial documentation, and the
numerical-methods foundations Pulsim's path-based event-driven
engine will rest on. The goal is a single shared mental model for
the engine before any code lands.

---

## 1. Naming clarification

Three terms get conflated; we distinguish them sharply:

- **DSED paradigm** — *"the simulator advances continuously between
  events and handles topology changes instantaneously at predicted
  event times"*. A class of algorithms. PLECS variable-step is a
  DSED-paradigm simulator; so is Spice's adaptive-step engine; so
  is what we're proposing.
- **Tsinghua DSED** (Zhu/Zhao/Shi/Yu, IEEE TPE 2019, patent
  **US10970432B2**) — a *specific* DSED instance built on Taylor
  series recursion + 4-class event taxonomy + secant root-find.
- **DSIM** (Powersys / DSIM Technology, commercial spin-off of
  the Tsinghua line) — the proprietary implementation of Tsinghua
  DSED.

Our implementation is a **DSED-paradigm engine** that
**deliberately differs from Tsinghua DSED** in every
patent-relevant detail (see §6 below). To avoid any branding
collision, our internal name for the engine is **PED**
(Path-Based Event-Driven). The OpenSpec change-id keeps the
`dsed` token for SEO/discovery, but every prose mention in the
paper uses "PED" + "DSED paradigm" rather than "DSED" as a
shorthand for our specific implementation.

---

## 2. The DSED paradigm in one paragraph

A DSED-paradigm simulator treats the circuit as a **hybrid
system** with two coexisting modes. In the **continuous mode**,
the switch mask is fixed and the state evolves under the linear
ODE $\matE \dot{\vx} = \matA(\mathbf{m})\vx + \matB \vu(t)$. In
the **discrete mode**, an event (gate edge, diode commutation,
threshold crossing) fires instantaneously and the mask transitions
$\mathbf{m} \to \mathbf{m}'$. Instead of sampling time at fixed
$\Delta t$ and *checking* if an event occurred (Pulsim v1.4.0's
model), the scheduler **predicts** the next event time
analytically from the current state derivative and the
event-trigger predicates, **advances to that time exactly**, then
handles the discrete transition. The integration between events
uses any standard ODE method; the event detection requires
zero-finding on each predicate. The combination delivers two wins
over fixed-$\Delta t$: no wasted steps in slow regions, and no
aliasing of events between samples.

---

## 3. What the Tsinghua line built (and what we will NOT copy)

The Tsinghua DSED line has four published variants:

| Variant | Year | Patent | What it added |
|---|---|---|---|
| FA-DSED | 2019 (Zhu/Zhao/Shi/Yu, IEEE TPE 34(12)) | US10970432B2 | Taylor-series variable-order integrator + 4-class event taxonomy (control / external / passive / state) + secant root-find on polynomial threshold |
| BDSED | 2021 (Wang/Li/Zhao, IEEE TPE) | US10747918B2, CN107290977A | Implicit state-quantization for parasitics-induced stiffness |
| PAT | 2019 | CN110633523A | Piecewise analytical switching-transient sub-models nested in one outer step |
| Hybrid-time | 2023 (Liu/Shi/Zhao, PEDG) | — | Co-simulation harness for DSED + fixed-step environments via SVID Taylor-coefficient interfaces |

**Performance claims (Tsinghua + DSIM, public).** Tsinghua reports
~10× over PLECS on multileg-bridge audio amp and a 50-kVA SST;
DSIM marketing reports **>1300× on a tiny LLC at very stiff
settings** but only **~2× on a 576-switch SST** (the
manual's own data, Section 3). The 1000× headline is a
cherry-pick at small scale; the asymptotic regime collapses.

### What's patent-encumbered

The 2019 patent's broadest claims cover:
1. **Variable-order Taylor recursion** for LTI state advance.
2. **Four event classes** (control / external / passive / state).
3. **Secant root-find on polynomial threshold function** $g_{\mathrm{th}}(\Delta t)$.

For Pulsim to operate safely outside these claims, we use:
1. **Dormand-Prince RK45** (Hairer/Norsett/Wanner 1993) as
   default integrator; BDF2 or RADAU IIA when stiff; Krylov-Φ
   matrix exponential when $\nstate \ge 50$ and event-density
   is low. *Not Taylor.*
2. **Five-tier event classifier** (gate edges, diode forward/reverse
   commutations, current zero-crossings, voltage thresholds,
   user-defined custom) with explicit priority resolver. *Not
   four-class.*
3. **Illinois (regula-falsi-with-anti-stagnation) or Brent
   root-finder** instead of pure secant, since secant stalls on
   near-tangent crossings (critically-damped diode commutation is
   the canonical failure case). *Not secant.*

These choices are independent of the patent, *and* arguably
technically better — RK45 is widely-implemented and well-studied,
Brent is more robust than secant on the corner cases that matter
for power-electronic event detection.

---

## 4. What's publicly documented in PLECS / DSIM (and what we'll
   document better)

| Feature | DSIM | PLECS variable-step | Pulsim PED (proposed) |
|---------|------|---------------------|------------------------|
| Integrator family | Variable-order Taylor | DOPRI + RADAU + auto-switch | DOPRI + BDF2 + Krylov-Φ + auto-select |
| Step controller | Heuristic + event-time cap | Hairer-Wanner standard | PI controller (Söderlind 2002) |
| Event taxonomy | 4 classes (closed) | Zero-crossing functions (no formal classes) | 5-tier priority resolver, public-doc |
| Root-find | Secant | Bisection | Illinois with Brent fallback |
| Linear solve at events | Cached full state-space per $\mathbf{m}$ | Cached full state-space per $\mathbf{m}$ | **Path-based partial refactor** |
| Discrete switch support | **Forbidden** (modules only) | Allowed | Allowed (with hysteresis policy) |
| Chatter handling | Not disclosed | $\sigma$-cycle repeat abort | Hysteresis band + blanking time, documented |
| Open source | No | No | **Yes (MIT)** |
| Pricing | Opaque, sales-only | Public ($5.5–11k/yr commercial) | Free |

The cells in **bold** are where Pulsim PED genuinely differentiates
from the closed-source incumbents. Path-based partial refactor at
the event handler is the central algorithmic novelty.

---

## 5. The path-based partial-refactor integration (the novelty)

Both DSIM and PLECS *cache the full $(\matJ, \matL, \matU,
\matP)$ tuple per switch mask* and pay a full
$\texttt{analyze} + \texttt{factorize}$ on the first encounter of
any new mask. Their documentation is silent on incremental
factor-update. The state of the art has not made the connection
that **DSED's event handler is exactly the workload that
Chan-Brandwajn-Tinney rank-1 update was designed for**.

In Pulsim PED, every event handler call executes:

1. Read predicate type (gate edge / diode on / diode off /
   custom).
2. Compute new mask $\mathbf{m}'$.
3. Compute the changed-column set
   $C = \{c : \matJ_{\mathrm{new}}[:, c] \neq \matJ_{\mathrm{old}}[:, c]\}$.
4. Query the v1.4.0 `partial_refactor_count_path(C)` and compare
   to $\maxratio$.
5. If below threshold: call `partial_refactor(J_new, C)` —
   $O(\sqrt{\nstate})$ cost.
6. Otherwise: fall back to fresh `analyze + factorize` —
   the v1.4.0 baseline path.

This is the v1.4.0 `solve_rank1` dispatch, exactly as is, just
called from the PED scheduler instead of a fixed-$\Delta t$ loop.
**Zero changes** to the v1.4.0 LU code path.

Expected hit-rate per the v1.4.0 multi-bit microbench
(captured in `artigos/02_tpel_methods/benchmarks/results/
multi_bit_microbench.csv`):

- Single-bit transitions (most gate edges): **always engages**.
- Multi-bit transitions $\delta = 2$ (gate + simultaneous diode):
  $\sim 45\%$ engage at $\nstate = 14$.
- Multi-bit transitions $\delta = 3$ (multi-leg simultaneous):
  $\sim 25\%$ engage.

So PED's expected path-based engagement rate is high enough to
make the integration worthwhile. The 30% hit-rate threshold in
the OpenSpec Gate 5 validation is conservative.

---

## 6. The three target benchmark scenarios for Gates 1–4

Per the tasks.md gates, we need three concrete scenarios that
exercise different aspects of PED:

### Scenario A — Buck CCM (Gate 1 validation)

Simple, smooth, no events except gate edges. Goal: prove
correctness + no regression vs v1.4.0 trapezoidal.

| Parameter | Value |
|-----------|-------|
| Topology | Synchronous buck, $V_{\mathrm{in}} = 24$~V, $V_{\mathrm{out}} = 12$~V |
| Switching | $f_{\mathrm{sw}} = 100$~kHz, $D = 0.5$ |
| Passives | $L = 100~\mu$H, $C = 100~\mu$F, $R_{\mathrm{load}} = 2.4~\Omega$ |
| Window | 5~ms |
| Expected events | 1000 gate edges (no diode commutation in CCM) |
| Baseline | v1.4.0 with $\Delta t = 100$~ns (50 001 steps) |
| Target | RMSE ≤ 0.1 % on $V_{\mathrm{out}}$; wall-clock ≤ 2× baseline |

### Scenario B — Buck DCM (Gate 3 validation)

Adds zero-current-detection events. Goal: prove PED's event
prediction reaches PSIM-level accuracy on body-diode commutation,
and starts paying back wall-clock.

| Parameter | Value |
|-----------|-------|
| Same as A, but $R_{\mathrm{load}} = 24~\Omega$ (DCM at $D = 0.5$) |
| Window | 5~ms |
| Expected events | 1000 gate edges + ~500 ZCD events |
| Target | Output ripple within 1 % of PSIM; ≥ 5× v1.4.0 wall-clock |

### Scenario C — LLC resonant (Gate 4 validation)

Stiff during transitions, smooth during resonance. Tests BDF2
backward-DSED auto-selection and the Krylov-Φ when $\nstate$
grows. Goal: 50% wall-clock at minimum.

| Parameter | Value |
|-----------|-------|
| Topology | Half-bridge LLC with magnetizing + leakage + resonant tank |
| Switching | $f_{\mathrm{sw}} = 100$~kHz, soft-switching mode |
| Stiffness | $\lambda_{\max}/\lambda_{\min} \approx 10^4$ during dead-band |
| Window | 10~ms |
| Expected events | Many simultaneous gate + ZCD + ZVD per cycle |
| Target | Complete in ≤ 50 % v1.4.0 wall-clock |

These three scenarios cover the regime-transition story (non-stiff
smooth → stiff with events → mixed). Successful Gates 1–4
validation against all three is what the paper-bound benchmark in
Gate 5 will then expand to the full ten-converter reference suite.

---

## 7. Open questions that survived the lit-dive

These need resolution at Gate 1 (prototype) before scaling to
Gates 2+:

1. **Default $q_{\max}$ for Krylov-Φ?** The Cardiff PE literature
   suggests $m = 10$–$30$; Saad 1992 has no a-priori formula.
   Test empirically on LLC at Gate 4.
2. **Illinois vs Brent vs polynomial-secant for diode-commutation
   root-finding?** Brent is bulletproof but ~3× more work per
   iteration than Illinois. Test on a bouncy critically-damped
   commutation at Gate 3.
3. **Hysteresis band default for chatter prevention?** DSIM's
   patent says nothing; PLECS aborts on $\sigma$-cycle. Default
   guess: $\pm 1\%$ of the threshold's nominal magnitude; tune
   per-circuit at Gate 1.
4. **Path-cache LRU eviction policy under variable-step?**
   v1.4.0's cache assumes long simulations re-visit masks
   uniformly; PED's variable step may visit masks episodically.
   Start with FIFO; profile + switch to LRU at Gate 5 if needed.
5. **Event-priority resolution for exact ties?** Default: gate
   edges > diode events > custom predicates, fired in that order
   on simultaneous timestamps. Document this convention in
   user-facing docs + tests.

---

## 8. Recommendation — proceed to Gate 1

The lit-dive confirms:

1. The DSED paradigm is the right next algorithmic investment
   beyond v1.4.0 PWL cache.
2. The Tsinghua/DSIM patent risk can be sidestepped with concrete
   algorithm choices (DOPRI / Illinois-Brent / 5-tier resolver /
   PI controller) that are independently better-documented and
   better-supported.
3. **Path-based partial refactor at the event handler is
   genuinely unclaimed in the public literature.** This is the
   central novel contribution.
4. The 5–6 month / 7-gate plan in the OpenSpec proposal is
   compatible with the technical risks identified here.

**Proposed Gate 0 sign-off:** approve the OpenSpec proposal as
written, with the patent-safety algorithm choices clarified in
the extended `design.md` (Section 13 added in this round). Once
approved, Gate 1 (Python prototype on buck CCM) starts.

---

## Sources cited in this memo

- Tsinghua DSED line:
  - Zhu/Zhao/Shi/Yu, *Discrete State Event-Driven Framework
    with a Flexible Adaptive Algorithm…*, IEEE TPE 34(12),
    2019. [IEEE 8675318]
  - Wang/Li/Zhao, *Backward Discrete State Event-Driven
    Approach for Simulation of Stiff Power Electronic
    Systems*, IEEE TPE, 2021. [IEEE 9351911]
  - US Patent 10970432B2 (FA-DSED).
  - US Patent 10747918B2 (BDSED).
  - Liu/Shi/Zhao, *Hybrid Time and Event Co-simulation
    Framework for Power Electronics Systems*, PEDG, 2023.
- DSIM commercial: <https://www.dsimtechnology.com/>
- PLECS: <https://docs.plexim.com/plecs/5.0/>; *Nanostep*
  article, *Bodo's Power Systems*, Feb 2024.
- Numerical foundations:
  - Hairer/Norsett/Wanner, *Solving ODE I: Nonstiff Problems*,
    Springer 1993/2008, §II.5 (DOPRI5 Butcher tableau).
  - Hairer/Wanner, *Solving ODE II: Stiff and DAE Problems*,
    Springer 1996, §V.1 (BDF), §IV.7 (Rosenbrock-W).
  - Söderlind, *Automatic Control and Adaptive Time-Stepping*,
    Numerical Algorithms 31, 2002, §4 (PI controller).
  - Saad, *Analysis of Some Krylov Subspace Approximations to
    the Matrix Exponential Operator*, SIAM JNA 29, 1992.
  - Sidje, EXPOKIT, ACM TOMS 24, 1998 (φ-functions).
- Patent-safety + commercial-comparison sources:
  US10970432B2, US10747918B2, USPTO/EPO patent searches as of
  2026-05-26.

**This memo is also the seed of the §I "Background and related
work" of the future TPEL methods paper #2**, with citation
numbering already laid out.
