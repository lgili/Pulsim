# MMC Arm Block — design doc

**Status:** L0–L3 implemented, both Half-Bridge and Full-Bridge
submodules supported across every layer. Per-step math lives in
``core/include/pulsim/mmc/arm.hpp`` (header-only C++23); the
Python module ``python/pulsim/mmc.py`` is a thin orchestrator
(dataclasses, builder helpers, observers, time-series drivers).
Integration tests under
``python/tests/test_mmc_arm_{average,multilevel,equivalent,detailed}.py``.

**Reference:** Sousa (2022), *Sistemas de Controle para a Operação
Eficiente de Conversores Modulares Multiníveis em Acionamentos
Elétricos.* PhD thesis, UFSC.

This doc captures the model equations, the layered block hierarchy, the
parameter / API contract, and the staged validation plan for the
Modular Multilevel Converter (MMC) support in pulsim. The goal is a
simulator that is

* **fast enough** to run an MMC with 30+ submodules per arm at switching
  frequencies of tens of kHz on commodity hardware,
* **faithful enough** to reproduce dead-time, minimum-pulse-width, and
  intra-arm capacitor balancing effects when those matter,

by giving the user a *small ladder of models* (L0 → L3) and letting them
pick the right point on the speed-vs-fidelity curve.

User-facing tutorial: see [Modular Multilevel Converters](../mmc.md).

---

## 1. MMC anatomy refresher

An MMC arm `Bⱼ,ₖ` is a series chain of N submodules (SMs) plus an arm
inductor `L_b`. Each SM is one capacitor `C_SM` plus a switching cell
that decides what fraction of `v_C_SM` appears across the SM's terminals.

The two canonical SM topologies:

| SM type | Switches | v_SM values | sₙ range |
|---|---|---|---|
| Half-Bridge (HB) | 2 IGBT + 2 diodes | `{0, +v_C_SM}` | `{0, 1}` |
| Full-Bridge (FB) | 4 IGBT + 4 diodes | `{−v_C_SM, 0, +v_C_SM}` | `{−1, 0, 1}` |

The three-phase DC/AC MMC (thesis fig 2.7) ships 6 arms (P/N × {a,b,c})
plus 6 arm inductors. The DC bus sits across (P↔N); each phase's
midpoint feeds the AC port. This is the topology we target first.

---

## 2. Arm equations

Notation follows the thesis. Capital `(¯·)` means local average over the
arm's switching period `T_eq`.

### 2.1 Switched model — eqs (2.7), (2.9)

Per-arm switching function `s_b = Σ sₙ`:

* HB: `s_b ∈ {0, 1, …, N}`
* FB: `s_b ∈ {−N, …, N}`

With the simplification `v_C_SMₙ ≈ v_C / N` (intra-arm balanced):

```
v_b  = (s_b / N) · v_C                 (2.7)
dv_C / dt = (1 / C_SM) · s_b · i_b     (2.9)
```

**One state per arm** (`v_C`, the *sum* of the SM capacitor voltages).

### 2.2 Averaged model — eqs (2.13), (2.14)

Define the modulation index `m_b = s_b / N`:

* HB: `m_b ∈ [0, 1]`
* FB: `m_b ∈ [−1, 1]`

and the arm-equivalent capacitance `C = C_SM / N`:

```
v̄_b = m_b · v̄_C                       (2.13)
dv̄_C / dt = (1 / C) · m_b · ī_b        (2.14)
```

These are *exactly* the equations of an ideal transformer with turns
ratio `m_b : 1` feeding the equivalent capacitor `C` (thesis fig 2.6).
That is the abstraction we instantiate in pulsim.

### 2.3 Arm-current dynamics — eq (2.30)

For arm `(j, k)` of the generalized MMC with port-1 voltage `v_{j,1}`,
port-2 voltage `v_{k,2}`, and common-mode `v_cm`:

```
L_b · di_{b,j,k} / dt = v_{j,1} − v_{k,2} − v_cm − v_{b,j,k}     (2.30)
```

`L_b` is the physical arm inductor — pulsim already supports it as a
regular inductor branch; we don't have to bake it into the block.

### 2.4 Current decomposition — eq (2.22)

Each arm current decomposes into the port-1 share, the port-2 share,
and a circulating component:

```
i_{c,j,k} = i_{b,j,k} − i_{j,1}/H − i_{k,2}/F     (2.22)
```

For DC/AC three-phase: `F = 2`, `H = 3` ⇒ `N_ic = 2` independent
circulating currents (degrees of freedom for energy-balance control).

### 2.5 SM-equivalent simulation model (thesis 3.4.2) — eqs (3.95)–(3.99)

This is the model we want at L2. Still one state per arm, but
faithfully reproduces dead-time and minimum-pulse-width effects.

Two simultaneous level counts:

```
s_w = Σ s_{d1,n}            (3.97)  → "defined" SMs (one switch on)
s_u = Σ (1 − s_{d1,n} − s_{d2,n})    (3.98)  → "free-wheel" SMs (both off)
```

Voltages (with `v_C_SMₙ ≈ v_C / N`):

```
v_w = (s_w / N) · v_C        (3.95)
v_u = (s_u / N) · v_C        (3.96)
```

Two anti-parallel ideal diodes `D_p`, `D_n` route the arm current
between `v_w` and `v_u` according to the sign of `i_b` (thesis
fig 3.10).

Capacitor state:

```
v_C = (1/C) · ∫ [ (s_w/N) · i_b + (s_u/N) · i_d − i_aux ] dt    (3.99)
```

where `i_d` is the current that flows through the free-wheel diodes
(determined by the routing block) and `i_aux` is an optional auxiliary
power-supply load.

**Dead-time** is injected via two monostables per arm:

* `Me_p` fires on every rising edge of `s` (the arm reference). It
  generates a pulse of width `T_d` (the dead-time) and amplitude `Δs`.
  This pulse is *subtracted* from `s_w`.
* `Me_n` does the same on falling edges; its pulse is *added* to `s_u`.

**Minimum-pulse-width** (`T_m` — the IGBT spec) is enforced by a
saturator in front of `s` (thesis fig 3.11) that limits how many SM
states can flip per simulation step, plus additional monostables
`Me_lp`, `Me_ln` that keep recently-toggled SMs disabled for `T_m`.

---

## 3. Layered block hierarchy

We ship four layers. Each is one self-contained block in
`pulsim.mmc` (Python wrapper) backed by a header-only C++23 kernel in
`core/include/pulsim/mmc/arm.hpp`. The Python side owns dataclasses,
builder integration, observers, and the time-series drivers; the kernel
owns the per-step math.

| Layer | Python class | Builder helper | States/arm | Captures | Use case |
|---|---|---|---|---|---|
| **L0** | `MmcArmAverage`     | `add_mmc_arm_average`     | 1 | continuous `m_b`, eq (2.13/14) | small-signal Bode, fast trade studies |
| **L1** | `MmcArmMultilevel`  | `add_mmc_arm_multilevel`  | 1 | discrete `s_b ∈ {0,…,N}` (HB) or `{-N,…,N}` (FB) | modulation strategy (PS-PWM, IPD) |
| **L2** | `MmcArmEquivalent`  | `add_mmc_arm_equivalent`  | 1 | + dead-time + min pulse (thesis 3.4.2) | loss estimation, fast dynamics |
| **L3** | `MmcArmDetailed`    | `add_mmc_arm_detailed`    | N | + intra-arm `v_C_SM` spread + sort-and-select | balancing algorithms, fault injection |

The blocks share the same external port shape (two electrical
terminals A/B, a control-input port for the modulating signal, an
optional measurement port for `v_C`), so a user can swap layers
without changing topology.

Both submodule topologies (`sm_type="half_bridge"` and
`sm_type="full_bridge"`) are supported across every layer. FB encoding
at L2/L3 uses a signed insertion count `s_b ∈ {-N..N}`: positive states
add `+v_C` per SM, negative states add `−v_C` per SM, and zero states
free-wheel. The L2 dead-time monostable routes by `sign(i_b)` exactly as
the thesis derives; L3 sort-and-select picks SMs by polarity of the
desired contribution, then by capacitor voltage relative to the charging
direction.

---

## 4. Parameters

Every layer takes the same minimum set:

| Parameter | Symbol | Units | Notes |
|---|---|---|---|
| `n_sm`     | N            | int   | submodules per arm (default 30 for med-volt) |
| `c_sm`     | C_SM         | F     | one SM's capacitance |
| `sm_type`  | —            | enum  | `"half_bridge"` / `"full_bridge"` |
| `v_c0`     | v_C(0)       | V     | initial sum-of-cap voltage |

L2 / L3 additionally take:

| Parameter | Symbol | Units | Notes |
|---|---|---|---|
| `t_dead`   | T_d  | s | dead-time between complementary switches |
| `t_min`    | T_m  | s | IGBT minimum on/off pulse width |
| `r_p`      | R_p  | Ω | parallel resistance modelling SM losses |
| `p_aux`    | P_aux | W | per-SM auxiliary-supply load (constant power) |

L3 additionally exposes per-SM `c_sm[n]` and balancing-algorithm
parameters; that's a later concern.

---

## 5. API surface (proposed)

### 5.1 Python (`pulsim` flat namespace)

```python
import pulsim as p

b = p.CircuitBuilder()
b.add_voltage_source("Vdc", "p", "n", 11.5e3)

# One arm — L0 average model
b.add_mmc_arm_average(
    name="B_a_p",
    node_a="p",            # arm top
    node_b="mid_a",        # arm midpoint (before L_b)
    n_sm=7,
    c_sm=100e-6,
    sm_type="half_bridge",
    v_c0=11.5e3,
    m=p.const_signal(0.5),  # or a switch_fn returning m_b(t) ∈ [0,1]
)
b.add_inductor("L_a_p", "mid_a", "phase_a", 750e-6)

# Symmetric negative arm + b, c phases ...
# (pulsim helper p.add_mmc_three_phase_dc_ac() will do this in one call)
```

A topology helper builds the full 3-phase DC/AC MMC in one line:

```python
b = p.CircuitBuilder()
p.add_mmc_three_phase_dc_ac(
    b,
    dc_pos="p", dc_neg="n",
    ac_nodes=("phase_a", "phase_b", "phase_c"),
    n_sm=7, c_sm=100e-6, l_b=750e-6,
    sm_type="half_bridge",
    layer="average",        # "average" | "multilevel" | "equivalent" | "detailed"
    modulator=...,          # callable t -> (m_a, m_b, m_c) for each phase × pos/neg
)
```

### 5.2 C++ (`core/include/pulsim/mmc/arm.hpp`)

All per-step math is `constexpr`-friendly C++23 living in
`core/include/pulsim/mmc/arm.hpp` (header-only). The public entry
points exposed via pybind11 (`python/bindings.cpp`):

* `mmc_arm_average_step`     — `(v_C, m_b, i_b, dt, params) → (v_C_next, v_b)`
* `mmc_arm_multilevel_step`  — `(v_C, s_b, i_b, dt, params) → (v_C_next, v_b)`
* `mmc_arm_equivalent_step`  — `(state, s_ref, i_b, dt, params, sm_type) → (state_next, v_b, s_w, s_u, s_n)`
* `mmc_arm_detailed_step`    — `(state, s_ref, i_b, dt, params) → (state_next, v_b, insertion_mask)`
* `ps_pwm_switching_function`, `ipd_switching_function` — quantizers
  used by the Python multilevel driver.

The Python side (``python/pulsim/mmc.py``) is a thin orchestrator:
it owns dataclasses, builder integration (the `add_mmc_arm_*` family
and the three-phase topology helper), per-step observers, and the
time-series drivers (`simulate_mmc_arm_*`). There is no pure-Python
fallback by design — a single implementation is the only contract
worth maintaining.

---

## 6. Validation plan

### L0 — sinusoidal steady-state vs. analytical

Set up a single arm fed by `i_b(t) = Î cos(ωt)` with `m_b(t) = M cos(ωt + φ)`.
The thesis (apêndice B) gives the closed-form components of the
instantaneous arm power `p_b(t)`. We verify:

1. **Average power** absorbed by the arm equals (V̂·Î / 2) · cos(φ).
2. **Capacitor voltage ripple** matches the integral of the harmonic
   components in eq (2.43) within ≤ 2 % amplitude.
3. **DC component** of `v_C` settles to `V_dc / N · M̂` (steady state).

### L1 — multilevel waveform vs. PS-PWM expectation

Drive the arm with phase-shifted PWM on `N` carriers, `s_b ∈ {0,…,N}`,
sinusoidal modulating wave. The output `v_b(t)` should show `N+1`
distinct levels, switching frequency `N·f_carrier` on the equivalent
output. We compare the THD against an analytical estimate for PS-PWM.

### L2 — vs. L0 with dead-time stress

Inject a fast modulation step from `m_b = 0.4` to `m_b = 0.8`. L0 gives
the idealised response; L2 should show:

1. Transient delay of order `T_d`.
2. Output level cap that respects `T_m` (no pulses narrower than the
   IGBT spec).
3. Loss spike consistent with the per-edge switching energies.

This is the experiment Sousa runs in fig 3.12.

### L3 — intra-arm balancing

Disable the balancing algorithm and watch capacitor voltages diverge.
Then turn it on (sort-and-select per Tu, Hu, Xu 2011 — the standard
algorithm) and confirm the spread stays bounded.

---

## 7. Implementation order — done

The roadmap below was followed; every step has shipped and is covered
by integration tests.

1. ✅ **L0 builder helper** — `p.add_mmc_arm_average(...)` composing
   pulsim primitives (controlled VS + cap + controlled CS).
2. ✅ **3-phase topology helper** — `p.add_mmc_three_phase_dc_ac(...)`
   wraps 6 arms + 6 arm inductors + DC bus connections.
3. ✅ **L0 showcase YAML** — `examples/mmc_three_phase_dc_ac.yaml`
   exercising the full converter with open-loop sinusoidal modulation.
4. ✅ **L1 multilevel** — quantises continuous `m_b` into `s_b` with
   PS-PWM or IPD carrier distribution. Both HB and FB.
5. ✅ **L2 SM equivalent** — dead-time monostables + min-pulse-width
   saturator. HB at first, FB added as a follow-up using a signed
   insertion count.
6. ✅ **L3 detailed** — per-SM state with sort-and-select balancing.
   HB + FB; balancing tracks the polarity of the desired contribution
   and the sign of `i_b`.

---

## 8. Resolved design choices

The original "open questions" section is preserved here for historical
context; each was settled during implementation.

* **`m_b` as callable vs state.** L0 ships as a callable
  (`m_b: float | Callable[[float], float]` on `add_mmc_arm_average`);
  L1–L3 also take a per-step reference signal that the user updates
  inside `step_observer` or via a BlockChain executor. Control loops
  remain a user concern — the arm exposes the knobs and observers.
* **L2 monostable discretisation.** We follow the thesis: monostables
  are evaluated on the simulation step `T_s` (cheap, small phase
  error). Sub-step grids were not needed to match the validation
  baselines.
* **L3 balancing.** Sort-and-select per Tu/Hu/Xu (2011) is the default,
  extended to signed insertion counts for FB. Variants can be plugged
  in by the user (the kernel exposes the per-SM voltages each step).

---

*This document grows as the implementation evolves. Section 9+ will be
filled in with measured benchmarks and observed gotchas.*
