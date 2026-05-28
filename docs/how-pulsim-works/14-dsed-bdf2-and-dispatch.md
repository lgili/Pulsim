# 14. BDF2 + stiffness detector + per-mode dispatch

Chapter 13 covered DOPRI5, the **explicit** integrator the PED
scheduler uses on non-stiff segments. On **stiff** segments (small
snubber RC, body-diode reverse-recovery, lossy magnetics) the
explicit step size shrinks faster from stability than from accuracy
— DOPRI5 is wasted. The PED scheduler responds by switching to
**BDF2** (Backward Differentiation Formula, order 2) for that
mode segment.

This chapter covers:

1. **BDF2** integration + Crank-Nicolson bootstrap (§14.1).
2. **Stiffness detector** — eigenvalue-based, evaluated once per
   mode segment (§14.2).
3. **`PEDSimulatorAuto`** — the per-mode-segment dispatcher
   (§14.3).
4. When BDF2 wins, when it doesn't (§14.4).

If you only care about LTI buck/boost/half-bridge type circuits,
the auto-dispatcher correctly classifies them as non-stiff and
sticks to DOPRI5. The BDF2 path matters for snubber-heavy
designs, multi-level converters with cell-mismatch dynamics, and
motors with skin-effect models.

---

## 14.1 BDF2 + Crank-Nicolson bootstrap

BDF2 is an **implicit** linear multistep method. For
$\dot{\mathbf{x}} = f(t, \mathbf{x})$ at step $n+1$:

$$
\boxed{
\frac{3}{2}\mathbf{x}_{n+1} - 2\mathbf{x}_n + \frac{1}{2}\mathbf{x}_{n-1}
= h\, f(t_{n+1}, \mathbf{x}_{n+1})
}
$$

For an LTI system $f(t, \mathbf{x}) = A\mathbf{x} + \mathbf{b}(t)$ the
update is a single linear solve:

$$
\Bigl(\tfrac{3}{2} I - h A\Bigr)\, \mathbf{x}_{n+1}
= 2 \mathbf{x}_n - \tfrac{1}{2}\mathbf{x}_{n-1} + h\, \mathbf{b}(t_{n+1})
$$

Pulsim caches the LU of $J = \tfrac{3}{2} I - h A$ between steps
(since $A$ and $h$ are constant per mode segment). The hot loop is
1 LU solve per step — same cost class as DOPRI5's matmul but with
**unconditional stability**: BDF2 is A-stable, so the step size is
limited by **accuracy only**, not by stability.

### The bootstrap problem

BDF2 needs $\mathbf{x}_{n-1}$ on the first step (after a mask
change or at $t = 0$). No prior step exists. Pulsim uses
**Crank-Nicolson** (trapezoidal rule) for the first step — it's
A-stable, order 2 (matches BDF2 — same local truncation error
class), and self-starting:

$$
\Bigl(I - \tfrac{h}{2} A\Bigr)\, \mathbf{x}_{n+1}
= \Bigl(I + \tfrac{h}{2} A\Bigr)\, \mathbf{x}_n
  + \tfrac{h}{2}(\mathbf{b}(t_n) + \mathbf{b}(t_{n+1}))
$$

After the bootstrap step the BDF2 hot loop has $\mathbf{x}_{n-1}
= \mathbf{x}_n$ and $\mathbf{x}_n = \mathbf{x}_{n+1}^{CN}$ for the
second step. From then on it's pure BDF2.

On every gate event the scheduler invalidates the BDF2 state
(both history $\mathbf{x}_{n-1}$ and cached LU of $J$) — the new
mask has a new $A$, and the discontinuity in $f$ would corrupt the
multi-step history. The next mode segment starts fresh with another
CN bootstrap.

### Step size in BDF2

Unlike DOPRI5, Pulsim's BDF2 implementation uses a **fixed**
$h_{\text{BDF2}}$ (default $10^{-6}$ s). The justification:

1. **Implicit stability** means $h$ can be much larger than
   DOPRI5 on stiff problems — accuracy is the binding constraint.
2. Variable-step BDF2 requires bookkeeping for the previous step
   size (the coefficients $-2, +\tfrac{1}{2}$ in the recurrence
   change to $-(1+\omega)/\omega, +1/\omega/(1+\omega)$ where
   $\omega = h_n / h_{n-1}$). Doable but adds risk.
3. PED's main use of BDF2 is **getting past a stiff segment**
   in a reasonable wall-clock. Fixed $h$ is robust + matches the
   3-state stiff RLC benchmark validation.

A future variant could add PI step control on BDF2's local error
(estimated via the difference $\|x_{n+1}^{BDF2} - x_{n+1}^{CN}\|$).
Not implemented today.

---

## 14.2 Stiffness detector

Switching between explicit and implicit costs needs a **cheap,
reliable** stiffness diagnostic. Pulsim uses the
**eigenvalue-based ratio**:

$$
\boxed{
\text{stiffness}(A, h) = \max_k |\lambda_k(A)|\, \cdot\, h
}
$$

Compare against a threshold (default 10.0):

| Ratio | Interpretation | Choice |
|---|---|---|
| $\le 10$ | Non-stiff at this $h$ | **DOPRI5** |
| $> 10$ | Stiff at this $h$ — explicit would have to shrink $h$ for stability | **BDF2** |

### Why eigenvalues?

For an explicit RK of order $p$, the stability region
$\{h\lambda : |R(h\lambda)| \le 1\}$ extends roughly out to
$|h\lambda| \le 2.7$ along the negative real axis for DOPRI5 (5th
order). So a stiff $\lambda$ (large negative real part) forces
$h \le 2.7/|\lambda|$ for stability **regardless** of accuracy
needs. When accuracy would have permitted, say, $h = 10^{-5}$ but
stability forces $h = 10^{-7}$, the explicit method is doing
100× the work it needs to.

BDF2 is A-stable: its stability region is the entire left half
plane $\Re(h\lambda) < 0$. No matter how stiff, BDF2's $h$ is set
by accuracy.

The threshold of 10 is conservative: empirically, DOPRI5 with PI
control handles $|h\lambda|$ up to ~5 without rejection cascades
(the PI loop shrinks $h$ automatically). The 10× margin ensures
we don't churn rejections.

### Implementation cost

`StiffnessDetector::select(mode_id, A, h_max)` is called **once
per mode segment** (when the scheduler enters a new mask). It:

1. Looks up `mode_id` in a per-mode integrator cache. If
   classified already, return the cached choice.
2. Computes $|\lambda_{\max}|$ via `Eigen::EigenSolver`. Cost:
   $O(n_s^3)$ but $n_s$ is small (2-10 for typical PE).
3. Multiplies by $h_{\max}$, compares against threshold.
4. Caches the result, returns `DOPRI5` or `BDF2`.

For a buck CCM with 4 masks: 4 eigensolves total over the whole
simulation, each ~1 µs at $n_s = 2$. Negligible.

---

## 14.3 `PEDSimulatorAuto` — the per-mode dispatcher

The auto-dispatcher is the union of the two scheduler loops with
a top-of-iteration branch on the stiffness verdict:

```text
while t < t_end:
    t_gate  = next_gate_edge(t, t_end)

    # Each iteration starts by querying the detector for the
    # current mode + a tentative step size.
    choice = detector.select(mode_id_of(current_mask), A_cur, dt_max)
    integrator_used[mode_segment_id] = choice

    while t < t_gate:
        if choice == DOPRI5:
            h_use = min(h_rk45, dt_max, t_gate - t)
            x_new, err = rk45_step(A_cur, b_fn, t, x, h_use, rk_state)
            (accepted, h_next) = PI.accept(err, x, x_new, h_use)
            if not accepted: continue  # back off, retry
            h_rk45 = h_next
            t += h_use
        else:  # BDF2
            h_use = min(h_bdf2, t_gate - t)
            x_new = bdf2_step(A_cur, b_fn, t, x, h_use, bdf2_state)
            t += h_use

        x = x_new

    # At t_gate: swap mask, invalidate BOTH integrator states,
    # re-query stiffness for the new mode segment.
    if t_gate < t_end:
        fire_gate_event(t_gate)
        rk_state.invalidate()
        bdf2_state.invalidate()
        PI.reset()
        # mode_id(current_mask()) is now different
        # → next loop iteration queries detector again
        # → h_rk45 = dt_init (avoid carrying stale step size across mask)
```

The key invariant: **stiffness is queried per-mode-segment, not
per-step**. Within a segment the chosen integrator is committed —
no mid-segment switching. This keeps the bookkeeping simple and
avoids pathological flip-flop loops.

### Why per-segment, not per-step?

A naïve implementation would re-query stiffness every step. That
would:

1. Pay an $O(n_s^3)$ eigensolve on every step ($n_s = 2$ → 1 µs,
   $n_s = 10$ → 50 µs, blowing past the integrator cost).
2. Risk thrashing — BDF2's bootstrap is stateful (it's order 1 on
   the CN step, order 2 from step 2 onward). Restarting it every
   step would lose accuracy.
3. Add no value — within a mode segment $A$ is constant, so the
   eigenvalues don't change. The verdict at the start is the
   verdict for the whole segment.

The implementation caches the verdict by `mode_id` (hash of the
SwitchStateMask). Re-visiting a mask uses the cached choice
without re-evaluating.

---

## 14.4 When BDF2 wins — and when it doesn't

The auto-dispatcher's selection is correct for:

| Scenario | $\lambda_{\max}$ | $h_{\text{stable, DOPRI5}}$ | Verdict |
|---|---|---|---|
| Buck CCM, $L=100$µH, $C=100$µF, $R=2.4$ | $\sim 5000$ rad/s | $\sim 540$ µs | DOPRI5 |
| Buck with small snubber, $RC = 10$ ns | $\sim 10^8$ rad/s | $\sim 27$ ns | **BDF2** |
| Boost CCM at 100 kHz | $\sim 1000$ rad/s | $\sim 2.7$ ms | DOPRI5 |
| Multi-level NPC with cell-cap mismatch | $\sim 10^5 - 10^7$ rad/s | $\sim 10$ µs – 270 ns | **BDF2** |
| Motor with skin-effect $R(\omega)$ | $\sim 10^6$ rad/s | $\sim 2.7$ µs | **BDF2** |

For standard CCM converters (buck/boost/buck-boost without
snubbers), DOPRI5 always wins. The end-to-end test
`test_dsed_integrator_auto_picks_rk45_on_non_stiff_buck`
verifies that `n_bdf2_steps == 0` and `n_rk45_steps > 0` on a
plain buck — the dispatcher correctly stays explicit.

### Honest accounting

Across 13 benchmark scenarios (Gate 5 sweep), BDF2 fires on **5
of 13** when small snubbers or stiff loads are added. On the **8
non-stiff** scenarios DOPRI5 carries the whole simulation, and
the auto-dispatch's geo-mean speedup over pure-DOPRI5 is **1.03×**
— essentially break-even. The auto-dispatcher's value is the
**worst-case** speedup on stiff problems (up to **90×** on the
3-state stiff RLC, where DOPRI5-only chokes on $h \sim 10^{-9}$).

For the buck-family converters the user cares about most, the
auto-dispatcher's behaviour is identical to forcing
`integrator='rk45'`. The dispatcher is a safety net for stiff
edge cases, not the main performance lever.

---

## 14.5 Takeaways

- BDF2 + CN bootstrap is the right choice for stiff PE segments:
  A-stable, order 2, one LU solve per step (same hot-loop cost as
  DOPRI5).
- The eigenvalue-based stiffness detector (`max|λ|·h vs 10`) is
  cheap (one $O(n_s^3)$ per mode segment, $n_s$ small) and
  correct: it routes plain buck to DOPRI5 and snubbered/multi-
  level to BDF2.
- The per-mode dispatcher avoids per-step re-queries. The
  per-mode integrator cache is keyed by `mode_id_of(mask)` — a
  hash of the SwitchStateMask.
- For non-stiff converters (buck/boost/half-bridge), the
  auto-dispatcher's behaviour matches `integrator='rk45'`. The
  value of auto is in **stiff edge cases** (up to 90× win).

## Further reading

- C++ implementation: `core/include/pulsim/dsed/bdf2_integrator.hpp`,
  `stiffness_detector.hpp`, `scheduler_bdf2.hpp`, `scheduler_auto.hpp`.
- Python prototype: `prototype/dsed/bdf2_integrator.py`.
- Gate 4 / Gate 5 progress notes for the stiff-fraction sweep
  validation: `notes/GATE4_PROGRESS.md`, `notes/GATE5_PROGRESS.md`.
- [Chapter 13 → DOPRI5 + adaptive PI](13-dsed-dopri5-adaptive.md)
- [Chapter 15 → Native bindings architecture](15-dsed-native-bindings.md)

External references:

- **Curtiss & Hirschfelder**, *Integration of stiff equations*,
  Proc. Natl. Acad. Sci. 38 (1952) — the original BDF derivation.
- **Hairer & Wanner**, *Solving ODEs II: Stiff and DAE Problems*,
  Springer 1996 — the canonical reference for BDF + A-stability.
- **Söderlind, G.**, *Numerical Algorithms* 31 (2002) — PI step
  control for variable-step BDF (the future variant).
