# 12. MNA → continuous-time state-space extraction

The PED scheduler (chapter 11) integrates the **continuous-time
state equation**

$$
\dot{\mathbf{x}}_s(t) = A_m \mathbf{x}_s(t) + \mathbf{b}_m(t)
$$

between events on every mode $m$. The PWL engine (chapters 2-4)
works with the **discrete-time trap-companion matrix**

$$
J_m(\Delta t) = G_{\text{static}} + \frac{2}{\Delta t} M_{\text{dyn}}
$$

that's already cached per mask. This chapter walks the derivation
that recovers the continuous-time $(A_m, \mathbf{b}_m)$ from
$J_m(\Delta t)$ — the algorithmic core of
`PwlStateSpaceCache::compute_lti_state_space(mask)`.

Sections 12.1-12.4 cover the canonical case (capacitors-to-ground
only). §12.5 generalises to **floating capacitors** via a
$T^\top M T$ congruence transform. §12.6 adds the projection
matrix $B$ that lets time-varying sources (sine, PWM, pulse) flow
into the reduced state-space at runtime.

---

## 12.1 The MNA structure we start from

From chapter 2, every Pulsim mask $m$ has an MNA system

$$
\underbrace{(G_{\text{static}} + \tfrac{2}{\Delta t}\, M_{\text{dyn}})}_{J(\Delta t)} \mathbf{x}_{n+1}
\;=\; -(\mathbf{b}_{\text{const}} + \mathbf{b}_{\text{hist}}(\mathbf{x}_n))
$$

where $\mathbf{x} \in \mathbb{R}^{n_{\text{MNA}}}$ is a long vector
of *all* MNA unknowns — node voltages, source branch currents,
inductor branch currents. The trap-companion model packages capacitors
and inductors into:

$$
M_{\text{dyn}} = \begin{cases}
+C_k & \text{for cap } k \text{ on node } i,\ \text{at } (i, i)\\
-L_k & \text{for inductor } k,\ \text{at } (\text{branch}_k, \text{branch}_k)\\
0 & \text{elsewhere}
\end{cases}
$$

and $G_{\text{static}}$ holds resistors + the static parts of
voltage-source constraint rows + switch conductances.

The state-vector subset we care about for ODE integration is

$$
\mathbf{x}_s = \begin{bmatrix} v_{C_1} & \cdots & v_{C_{n_c}} &
                                i_{L_1} & \cdots & i_{L_{n_l}}
\end{bmatrix}^\top \in \mathbb{R}^{n_s}, \quad n_s = n_c + n_l
$$

— the rest of $\mathbf{x}$ is algebraic (node voltages, source
branch currents). Our goal is to project $J_m, M_{\text{dyn}},
\mathbf{b}$ onto $\mathbf{x}_s$ and end up with $\dot{\mathbf{x}}_s
= A_m \mathbf{x}_s + \mathbf{b}_m$.

---

## 12.2 Step 1 — recover $M_{\text{dyn}}$ and $G_{\text{static}}$

We have $J(\Delta t)$ cached but **not** the decomposition.
Recovering $(M_{\text{dyn}}, G_{\text{static}})$ would require
re-reading the stamping passes — invasive. Instead we exploit the
linear $\Delta t$-dependence: assemble $J$ at **two** $\Delta t$
values and linearly combine.

Let $h_a, h_b > 0$ be two recovery step sizes (Pulsim defaults:
$h_a = 10^{-6}$ s, $h_b = 5 \times 10^{-7}$ s — chosen for
numerical well-conditioning across typical PE scales). Then

$$
\begin{aligned}
J(h_a) &= G_{\text{static}} + \tfrac{2}{h_a} M_{\text{dyn}} \\
J(h_b) &= G_{\text{static}} + \tfrac{2}{h_b} M_{\text{dyn}}
\end{aligned}
$$

Solving the $2 \times 2$ block-system for the two unknown matrices:

$$
\boxed{
M_{\text{dyn}} = \frac{h_a\, h_b}{2\,(h_b - h_a)}\bigl(J(h_a) - J(h_b)\bigr)
}
$$

$$
G_{\text{static}} = J(h_a) - \frac{2}{h_a} M_{\text{dyn}}
$$

This is computed by the C++ extractor as:

```cpp
DenseMatrix M_dyn = (Jd_a - Jd_b) * ((h_a * h_b) / (2 * (h_b - h_a)));
DenseMatrix G_st  = Jd_a - (2.0 / h_a) * M_dyn;
```

Both matrices are $n_{\text{MNA}} \times n_{\text{MNA}}$ — full
MNA size. The reduction to $n_s$ happens in §12.4 via Schur
complement.

---

## 12.3 Step 2 — identify state rows

Walking the device pool gives us the MNA row index for each state
variable:

- **Capacitor $k$ to ground**: state $v_{C_k}$ lives at the
  row of the non-ground node. Index: `branch.from` if it's not
  ground, else `branch.to`.
- **Inductor $k$**: state $i_{L_k}$ lives at the inductor's
  branch-current row (added during MNA augmentation).
  Index: `pool.branch_var_id_for_inductor(branch.id, graph)`.

Concatenating: `state_rows = [cap_pos_1, ..., cap_pos_nc,
ind_branch_1, ..., ind_branch_nl]`. The remaining rows are
**algebraic** (node KCL rows for non-state nodes + source branch
KVL rows).

Floating caps (where neither terminal is ground) break this
direct mapping — they're handled by the §12.5 congruence transform.

---

## 12.4 Step 3 — Schur complement: project onto $\mathbf{x}_s$

The full MNA system in continuous time is a DAE:

$$
M_{\text{dyn}} \dot{\mathbf{x}} = -G_{\text{static}} \mathbf{x} - \mathbf{b}(t)
$$

with $M_{\text{dyn}}$ block-singular: only state rows have non-zero
diagonal ($+C_k$ or $-L_k$); algebraic rows are zero. Reorder
$\mathbf{x}$ into $[\mathbf{x}_s; \mathbf{x}_a]$ and partition:

$$
M_{\text{dyn}} = \begin{bmatrix} M_{ss} & 0 \\ 0 & 0 \end{bmatrix}, \quad
G_{\text{static}} = \begin{bmatrix} G_{ss} & G_{sa} \\ G_{as} & G_{aa} \end{bmatrix}, \quad
\mathbf{b} = \begin{bmatrix} \mathbf{b}_s \\ \mathbf{b}_a \end{bmatrix}
$$

The state and algebraic rows give:

$$
\begin{aligned}
M_{ss}\, \dot{\mathbf{x}}_s &= -G_{ss}\, \mathbf{x}_s - G_{sa}\, \mathbf{x}_a - \mathbf{b}_s \\
0 &= -G_{as}\, \mathbf{x}_s - G_{aa}\, \mathbf{x}_a - \mathbf{b}_a
\end{aligned}
$$

Solve the algebraic block for $\mathbf{x}_a$ (requires
$G_{aa}$ invertible — a topology condition equivalent to "the
non-state-bearing subnetwork has a unique algebraic solution per
state"):

$$
\mathbf{x}_a = -G_{aa}^{-1}(G_{as}\, \mathbf{x}_s + \mathbf{b}_a)
$$

Substitute back into the state equation:

$$
M_{ss}\, \dot{\mathbf{x}}_s = (G_{sa} G_{aa}^{-1} G_{as} - G_{ss})\, \mathbf{x}_s
                           + (G_{sa} G_{aa}^{-1} \mathbf{b}_a - \mathbf{b}_s)
$$

Multiply by $M_{ss}^{-1}$ (diagonal: $+1/C_k$ for caps, $-1/L_k$
for inductors — note the minus sign on inductors comes from the
KVL convention $v_{a} - v_{b} - L\dot{i} = 0$):

$$
\boxed{
\dot{\mathbf{x}}_s = \underbrace{M_{ss}^{-1}(G_{sa} G_{aa}^{-1} G_{as} - G_{ss})}_{A_m}\, \mathbf{x}_s
                  + \underbrace{M_{ss}^{-1}(G_{sa} G_{aa}^{-1} \mathbf{b}_a - \mathbf{b}_s)}_{\mathbf{b}_m}
}
$$

That's the per-mask state-space $(A_m, \mathbf{b}_m)$ the PED
scheduler integrates. Eigen's `FullPivLU<DenseMatrix>` on $G_{aa}$
does the $O(n_a^3)$ work once per mask, then the Schur complement
gives $A_m$ in another $O(n_s^2 n_a + n_s^3)$.

The validation for buck CCM ($V_{\text{in}} = 24$ V, $L = 100$ µH,
$C = 100$ µF, $R = 2.4$ Ω) recovers the textbook Erickson §7.4
matrices to 6 decimal places (`test_lti_state_space.cpp`):

$$
A_{HS\_on} = \begin{bmatrix} -1/(RC) & 1/C \\ -1/L & 0 \end{bmatrix}
           = \begin{bmatrix} -4166.67 & 10000 \\ -10000 & 0 \end{bmatrix},
\quad \mathbf{b}_{HS\_on} = \begin{bmatrix} 0 \\ V_{\text{in}}/L \end{bmatrix}
                          = \begin{bmatrix} 0 \\ 240000 \end{bmatrix}
$$

---

## 12.5 Floating capacitors: the $T^\top M T$ congruence

The §12.3-12.4 derivation assumes every capacitor has a terminal
on ground. For an output cap on `out --- gnd` the state row is
exactly the row of `out`. But for **NPC split DC bus**, **MMC
SubModule stacks**, and **half/full-bridge differential output
caps**, the cap lives between two non-ground nodes:

```text
v_dc_pos ●━━━━━━━━━━━┳━━━━━━━━ ● v_mid
                     ┃
                     ╪ C_top   (floating: neither terminal is ground)
                     ┃
v_mid ●━━━━━━━━━━━━━━┻━━━━━━━━ ● v_dc_neg
                     ┃
                     ╪ C_bot
                     ┃
                  (gnd?)
```

The MNA stamp for a floating cap between nodes $(a, b)$:

$$
M_{\text{dyn}}\bigl|_{(a,b)\text{ block}} = \begin{bmatrix} +C & -C \\ -C & +C \end{bmatrix}
$$

This is a **rank-1 matrix** — eigenvalues $(2C, 0)$, eigenvectors
$\bigl(\tfrac{1}{\sqrt{2}}(1, -1)\bigr)$ for $2C$ (the differential
mode = cap voltage) and $\bigl(\tfrac{1}{\sqrt{2}}(1, 1)\bigr)$ for
$0$ (the common mode, which has no derivative). The Schur step in
§12.4 can't proceed because the partition of $M_{\text{dyn}}$
across state and algebraic rows isn't block-diagonal anymore —
both $v_a$ and $v_b$ have derivative terms, but they're
constrained.

### The fix: a congruence transform $T$

Pick a coordinate change $\mathbf{x} = T\, \tilde{\mathbf{x}}$
such that, in the new coordinates, every cap has its dynamic term
on a **single** row (the cap's "pos" row), and the other terminal
becomes purely algebraic. The transformed MNA equation

$$
M_{\text{dyn}}\, \dot{\mathbf{x}} + G_{\text{static}}\, \mathbf{x} = \mathbf{b}
$$

becomes (substituting $\mathbf{x} = T\tilde{\mathbf{x}}$ and
left-multiplying by $T^\top$ to preserve symmetry of structure):

$$
\underbrace{T^\top M_{\text{dyn}} T}_{\tilde{M}}\, \dot{\tilde{\mathbf{x}}}
+ \underbrace{T^\top G_{\text{static}} T}_{\tilde{G}}\, \tilde{\mathbf{x}}
= \underbrace{T^\top \mathbf{b}}_{\tilde{\mathbf{b}}}
$$

A symmetric congruence: the dynamic block diagonalises if $T$ is
chosen as the eigenbasis of the rank-1 cap stamps.

### Constructing $T$ from a BFS over the cap-edge graph

Model capacitors as edges in an undirected graph over MNA nodes
(plus a synthetic "ground" node sentinel). Each connected
component is a "cap cluster":

```text
COMPONENT                  ANCHOR (algebraic / non-state in this comp)
─────────────────────      ───────────────────────────────────────────
Touches ground             ground (synthetic) → ALL real nodes carry a state
Floating (no ground)       lowest-index MNA node → other nodes carry states
```

Within a component, BFS from the anchor orients each cap edge:
`pos` = farther-from-anchor, `neg` = closer-to-anchor. Edges that
don't fit the spanning tree (cycles → parallel caps) are rejected
with a clear error pointing to the merge-equivalent workaround.

The transform matrix is constructed iteratively:

$$
T = I, \quad \text{then for each tree edge } (\text{parent}, \text{child})\
\text{in REVERSE BFS order:}\
T_{:, \text{parent}} \mathrel{+}= T_{:, \text{child}}
$$

**Why this works** — for a chain of 3 caps $C_1, C_2, C_3$ with anchor at the bottom:

```text
node 0 ●━━╪━━ node 1 ●━━╪━━ node 2 ●━━╪━━ node 3 (anchor)
        C1            C2            C3
```

The reverse-BFS edge order is $C_1, C_2, C_3$, and successive
$T_{:, \text{parent}} \mathrel{+}= T_{:, \text{child}}$ builds

$$
T = \begin{bmatrix}
1 & 1 & 1 & 1 \\
0 & 1 & 1 & 1 \\
0 & 0 & 1 & 1 \\
0 & 0 & 0 & 1
\end{bmatrix}
$$

— upper-triangular with 1's accumulating toward the anchor column. The
new states are exactly the cap voltages:
$\tilde{x}_0 = v_{C_1} = v_0 - v_1$,
$\tilde{x}_1 = v_{C_2} = v_1 - v_2$,
$\tilde{x}_2 = v_{C_3} = v_2 - v_3$,
$\tilde{x}_3 = v_3$ (the algebraic anchor).

After the congruence $\tilde{M} = T^\top M T$ on the cap block:

$$
\tilde{M}\bigl|_{4 \times 4} = \begin{bmatrix}
C_1 & 0 & 0 & 0 \\
0 & C_2 & 0 & 0 \\
0 & 0 & C_3 & 0 \\
0 & 0 & 0 & 0
\end{bmatrix}
$$

— exactly the diagonal $M_{ss}$ shape §12.4 needs. The anchor row
(last) is zero → it joins the algebraic block, the Schur step
proceeds as before.

For grounded-component caps the same algorithm runs but every
"parent" walks toward the synthetic ground (which has no real
MNA row), so the update loop skips those edges and $T$ stays
identity in their parts. That's why DSED falls back to a fast
path for grounded-only circuits (the buck regression check
benchmarks identically with/without the congruence code).

### Validation

| Test scenario | Expected | Recovered |
|---|---|---|
| Buck CCM (grounded) | Erickson eq. 7.45 | 6-digit match |
| Single floating R-C-R | $A = -1/(R_{\text{tot}} C)$, $\|b\| = V/(R_{\text{tot}} C)$ | exact |
| NPC 2-cap stack 100 V | $v_{C,\text{ss}} \approx \pm 50$ V each | -49.998 V each |
| MMC-style 3-cap chain | sum $\approx V_{\text{in}}$, balanced | -100, -100, -100 V |
| Parallel caps | throws (cycle) | ✓ rejected with merge pointer |

---

## 12.6 Time-varying sources: the projection matrix $B$

Sine, PWM and pulse voltage sources stamp $V_{\text{baseline}} = 0$
at MNA assembly time and apply the actual value via
`b_extra(t) = -V_{\text{source}}(t)` on the source's branch-var
row (chapter 3 §3.4). For DSED we need the same overlay in
**state-space coordinates** — i.e., we want

$$
\mathbf{b}_m(t) = \mathbf{b}_{\text{const}} + B_m\, \mathbf{b}_{\text{extra}}(t)
$$

where $\mathbf{b}_{\text{extra}}(t)$ is the full MNA-size vector
that `run_transient` would have built. Tracing through §12.4 +
§12.5, the linear map from $\mathbf{b}_{\text{extra}}$
(original-coords, full MNA) to $\mathbf{b}_m$ (state-coords,
reduced) is

$$
\boxed{
B_m \;=\; M_{ss}^{-1}\,\bigl(G_{sa}\, G_{aa}^{-1}\, S_a - S_s\bigr)\, T^\top
}
$$

where $S_s \in \{0,1\}^{n_s \times n_{\text{MNA}}}$ and
$S_a \in \{0,1\}^{n_a \times n_{\text{MNA}}}$ are **row-selector
matrices** ($S_s[i, \text{state\_rows}[i]] = 1$, similarly for
$S_a$). The transpose $T^\top$ accounts for the §12.5
coordinate change applied to $\mathbf{b}$.

The extractor returns $B_m$ as the fifth field of `ContinuousLTI`.
The Python adapter caches it per mask and computes
`b_state(t) = b_constant + B @ b_extra(t)` per scheduler step,
where `b_extra(t)` is built natively from
`builder.compute_time_varying_b_extra(t)` (the C++ helper that
sums the sine/PWM/pulse `b_extra` walks).

### Validation: corner-frequency check

A sine source driving an RC filter at $\omega RC = 1$ should give
output amplitude $V_{\text{amp}} / \sqrt{2}$ and phase lag
$\pi/4$. DSED measured:

| Quantity | Expected | DSED |
|---|---|---|
| $|V_{\text{out, peak}}|$ at corner | $V_{\text{amp}}/\sqrt{2} = 7.071$ V | 7.070 V |
| Relative error | — | **0.01 %** |

---

## 12.7 What this chapter gave us

By the end of this chapter, we have for any mask $m$:

- $A_m \in \mathbb{R}^{n_s \times n_s}$ — the per-mode dynamics
- $\mathbf{b}_{\text{const}, m} \in \mathbb{R}^{n_s}$ — the DC part
- $B_m \in \mathbb{R}^{n_s \times n_{\text{MNA}}}$ — the time-varying overlay
- `state_rows: list[int]` + `state_is_cap: list[bool]` — index plumbing

All cached lazily on the first scheduler `set_mask(m)` call. The
**PED scheduler** (chapters 13-14) just calls `A_matrix() /
b_vector(t) / rhs(t, x)` from C++.

## Further reading

- C++ extractor: `core/include/pulsim/pwl/cache.hpp::compute_lti_state_space`
- Validation tests: `core/tests/layer4_v3/test_lti_state_space.cpp`
- Design doc: `notes/DSED_BRIDGE_DESIGN.md` §§ 2, 5.1b, 12
- [Chapter 11 → DSED motivation](11-dsed-engine-overview.md)
- [Chapter 13 → DOPRI5 + adaptive PI](13-dsed-dopri5-adaptive.md)

External references:

- **Najm, F.**, *Circuit Simulation*, Wiley 2010, §3.4.2 — MNA →
  state-space reduction via Schur complement.
- **Brenan, Campbell, Petzold**, *Numerical Solution of Initial-
  Value Problems in Differential-Algebraic Equations*, SIAM 1996 —
  index-1 DAE theory, the regularity condition on $G_{aa}$.
- **Erickson & Maksimovic**, *Fundamentals of Power Electronics*,
  Springer 2020, eq. 7.45 — the analytical reference matrices for
  the buck CCM validation.
