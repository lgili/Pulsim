# Modular Multilevel Converters (MMC)

Pulsim ships a four-layer block hierarchy for **Modular Multilevel
Converters** — from a single-state averaged arm useful for small-signal
trade studies all the way to a per-submodule detailed model with
sort-and-select balancing. Every layer supports both **half-bridge (HB)**
and **full-bridge (FB)** submodules, and the same external port shape
(two electrical terminals + a control-input signal), so you can swap
fidelity without rewiring topology.

The per-step math is C++23 header-only (`core/include/pulsim/mmc/arm.hpp`);
the Python module `pulsim.mmc` is a thin orchestrator (dataclasses,
builder integration, observers, time-series drivers).

For the underlying equations, validation plan, and design rationale see
the internal design doc [MMC Arm Block](internals/mmc-arm-block.md).

## The four fidelity layers

| Layer | Builder helper | Driver | States/arm | What it captures | When to use |
|---|---|---|---|---|---|
| **L0** | `add_mmc_arm_average`     | `simulate_mmc_arm_average`     | 1 | Continuous `m_b`, ideal averaged dynamics | Small-signal Bode, fast control-loop trade studies |
| **L1** | `add_mmc_arm_multilevel`  | `simulate_mmc_arm_multilevel`  | 1 | Discrete `s_b ∈ {0..N}` (HB) or `{-N..N}` (FB) with PS-PWM / IPD | Modulation strategy, multilevel waveform THD |
| **L2** | `add_mmc_arm_equivalent`  | `simulate_mmc_arm_equivalent`  | 1 | + dead-time + minimum pulse-width (thesis 3.4.2) | Loss estimation, fast dynamics with realistic switching |
| **L3** | `add_mmc_arm_detailed`    | `simulate_mmc_arm_detailed`    | N | + intra-arm `v_C_SM` spread + sort-and-select balancing | Capacitor-balancing algorithms, fault injection |

A general rule of thumb: start at L0, move up only when the questions
demand it. L0 → L1 catches harmonics; L1 → L2 catches dead-time / loss;
L2 → L3 catches per-SM voltage spread.

## Submodule topologies

| `sm_type`        | Switches per SM | `s_b` range | `v_b` per SM | `m_b` range |
|---|---|---|---|---|
| `"half_bridge"`  | 2 IGBT + 2 diodes | `{0..N}` | `{0, +v_C_SM}` | `[0, 1]` |
| `"full_bridge"`  | 4 IGBT + 4 diodes | `{-N..N}` | `{-v_C_SM, 0, +v_C_SM}` | `[-1, 1]` |

FB encoding at L2/L3 uses a signed insertion count: positive states add
`+v_C` per SM, negative states add `-v_C` per SM, and zero states
free-wheel. The L2 dead-time monostable routes by `sign(i_b)` and the
L3 sort-and-select picks SMs by polarity of the desired contribution,
then by capacitor voltage relative to the charging direction.

## Modulation schemes

L1/L2/L3 share two carrier-distribution strategies via the
`modulation_scheme` parameter:

* `"ps_pwm"` — **Phase-Shifted PWM**. N carriers spaced `2π/N`. Output
  switching frequency is `N · f_carrier`. Default.
* `"ipd"` — **In-Phase Disposition**. N carriers vertically stacked,
  same phase. Smaller per-cycle ripple, simpler harmonic structure.

Both helpers `ps_pwm_switching_function(t, m_ref, n_sm, f_carrier, sm_type)`
and `ipd_switching_function(t, m_ref, n_sm, f_carrier, sm_type)` are
exposed as standalone callables in case you want to drive your own
arm with the quantiser.

## L0 — average-value arm

The L0 arm is mathematically an *ideal transformer with turns ratio
`m_b : 1`* feeding the arm-equivalent capacitor `C = c_sm / n_sm`:

$$
\begin{aligned}
v_b &= m_b \cdot v_C \\
\frac{d v_C}{d t} &= \frac{1}{C} \cdot m_b \cdot i_b
\end{aligned}
$$

```python
import math
import pulsim as p

dt = 5e-6
params = p.MmcArmAverageParams(
    n_sm=8, c_sm=4.7e-3,
    sm_type="half_bridge",
    v_c0=800.0,                 # initial v_C(0)
)

# Standalone driver — feed your own m_b(t), i_b(t)
omega = 2 * math.pi * 60
res = p.simulate_mmc_arm_average(
    duration=0.05, dt=dt,
    m_b=lambda t: 0.5 + 0.45 * math.sin(omega * t),
    i_b=lambda t: 50.0 * math.sin(omega * t - 0.1),
    params=params,
)
print(f"v_C(end) = {res.v_C[-1]:.2f} V  (drift = {res.v_C[-1]-res.v_C[0]:+.2f})")
```

To embed an L0 arm into a larger circuit, use the builder integration
plus the matching observer/`b_extra_fn`:

```python
b = p.CircuitBuilder()
b.add_voltage_source("Vdc", "p", "n", 1500.0)

arm = p.add_mmc_arm_average(
    b, name="arm_p",
    node_a="p", node_b="mid",
    params=params,
    m_b=lambda t: 0.5 + 0.45 * math.sin(omega * t),
)
b.add_inductor("L_arm", "mid", "load", 750e-6)
b.add_resistor("R_load", "load", "n", 5.0)

obs, bex = p.make_mmc_arm_observer(b, arm, dt=dt)
res = p.simulate(b, t_end=0.05, dt=dt,
                  step_observer=obs, b_extra_fn=bex,
                  start_from_dc_op=True)
print(f"final v_C = {arm.v_C:.2f} V")
```

## L1 — multilevel quantised switching

L1 inherits L0's averaged capacitor dynamics but replaces the
continuous `m_b` with a discrete insertion count `s_b` produced by a
carrier-based modulator. The arm voltage becomes `v_b = (s_b / N) · v_C`.

```python
params = p.MmcArmMultilevelParams(
    n_sm=8, c_sm=4.7e-3,
    sm_type="full_bridge",
    v_c0=800.0,
    f_carrier=2_000.0,            # per-SM carrier frequency
    modulation_scheme="ps_pwm",   # or "ipd"
)

res = p.simulate_mmc_arm_multilevel(
    duration=0.05, dt=1e-6,
    m_ref=lambda t: 0.9 * math.sin(2 * math.pi * 60 * t),
    i_b=lambda t: 100.0 * math.sin(2 * math.pi * 60 * t - 0.1),
    params=params,
)
# res.s_b is the integer insertion count; res.v_b is (s_b/N)·v_C
```

## L2 — SM-equivalent with dead-time and min-pulse-width

L2 keeps a single capacitor state per arm but adds the two effects
that dominate loss and small-pulse behaviour in real IGBT MMCs:

* **dead-time** between complementary switches (`t_dead`),
* **minimum-pulse-width** enforcement (`t_min`).

The implementation follows Sousa (2022) §3.4.2: a per-SM monostable
fires on each toggle, and the arm output during dead-time is routed
through anti-parallel diodes according to the sign of `i_b`. Two
simultaneous level counts are produced: `s_w` (defined-inserted) and
`s_u` (free-wheeling).

```python
params = p.MmcArmEquivalentParams(
    n_sm=8, c_sm=4.7e-3,
    sm_type="full_bridge",        # HB or FB
    v_c0=800.0,
    f_carrier=2_000.0,
    t_dead=2e-6,                  # 2 µs dead-time
    t_min=5e-7,                   # 0.5 µs min-pulse-width
)

res = p.simulate_mmc_arm_equivalent(
    duration=0.05, dt=1e-7,        # dt must be ≤ t_dead
    m_ref=lambda t: 0.9 * math.sin(2 * math.pi * 60 * t),
    i_b=lambda t: 100.0 * math.sin(2 * math.pi * 60 * t - 0.1),
    params=params,
)
```

!!! note "Step-size requirement"
    L2 needs `dt ≤ min(t_dead, 1/(N·f_carrier)/10)` to resolve
    dead-time events. Coarser steps collapse the dead-time effect
    silently — the simulation will run, but the output drifts toward
    the L1 ideal.

## L3 — per-SM detailed with sort-and-select balancing

L3 carries one capacitor state *per submodule* (length-`n_sm` array
`v_C_per_sm`). Each step a balancing algorithm picks which `|s_b|` SMs
to insert based on:

1. **Polarity** of the desired contribution (`sign(s_b)`).
2. **Capacitor voltage** vs. **charging direction** — lowest `v_C` is
   selected when the arm is charging, highest when discharging. The
   default algorithm is sort-and-select per Tu/Hu/Xu (2011), extended
   to signed insertion counts for FB.

```python
params = p.MmcArmDetailedParams(
    n_sm=8, c_sm=4.7e-3,
    sm_type="full_bridge",
    v_c0=800.0,
    f_carrier=2_000.0,
    t_dead=2e-6,
    t_min=5e-7,
)

res = p.simulate_mmc_arm_detailed(
    duration=0.05, dt=1e-7,
    m_ref=lambda t: 0.9 * math.sin(2 * math.pi * 60 * t),
    i_b=lambda t: 100.0 * math.sin(2 * math.pi * 60 * t - 0.1),
    params=params,
)

# res.v_C_per_sm has shape (n_steps + 1, n_sm)
import numpy as np
spread = res.v_C_per_sm.max(axis=1) - res.v_C_per_sm.min(axis=1)
print(f"max intra-arm spread: {spread.max():.2f} V")
```

## Three-phase DC/AC topology helper

`add_mmc_three_phase_dc_ac` composes a full three-phase MMC (6 arms +
6 arm inductors) using L0 arms in a single call:

```python
b = p.CircuitBuilder()
b.add_voltage_source("Vdc",   "dc_p", "dc_n", 1500.0)
b.add_resistor("R_load_a", "ac_a", "ac_n", 10.0)
b.add_resistor("R_load_b", "ac_b", "ac_n", 10.0)
b.add_resistor("R_load_c", "ac_c", "ac_n", 10.0)
b.add_resistor("R_n",      "ac_n", "gnd",  1e6)   # neutral bleed

omega = 2 * math.pi * 60
mmc = p.add_mmc_three_phase_dc_ac(
    b,
    dc_pos="dc_p", dc_neg="dc_n",
    ac_nodes=("ac_a", "ac_b", "ac_c"),
    n_sm=8, c_sm=4.7e-3, l_b=750e-6,
    sm_type="half_bridge",
    v_c0=800.0,
    m_signals=(
        lambda t: 0.5 + 0.45 * math.sin(omega * t),
        lambda t: 0.5 + 0.45 * math.sin(omega * t - 2*math.pi/3),
        lambda t: 0.5 + 0.45 * math.sin(omega * t + 2*math.pi/3),
    ),
)

dt = 5e-6
obs, bex = p.make_mmc_arms_observer(b, mmc.all_arms, dt=dt)
res = p.simulate(b, t_end=0.05, dt=dt,
                  step_observer=obs, b_extra_fn=bex,
                  start_from_dc_op=True)

print(f"v_C upper-a end: {mmc.arm_a_p.v_C:.2f} V")
print(f"v_C lower-a end: {mmc.arm_a_n.v_C:.2f} V")
```

When `m_signals` is a 3-tuple, the lower arms get the complement
(`1 - m_X` for HB, `-m_X` for FB) so the arm sum matches the DC bus
by construction. Pass a 6-tuple for independent control of each arm
(useful for circulating-current suppression and per-arm energy
balancing).

## Where to go next

* [MMC Arm Block — design doc](internals/mmc-arm-block.md) — model
  equations, validation plan, layer-by-layer details.
* `examples/mmc_three_phase_dc_ac.yaml` — YAML showcase exercising the
  three-phase helper with open-loop sinusoidal modulation.
* `python/tests/test_mmc_arm_average.py`,
  `test_mmc_arm_multilevel.py`,
  `test_mmc_arm_equivalent.py`,
  `test_mmc_arm_detailed.py` — integration tests cover both HB and FB
  for every layer.
* Reference: Sousa (2022), *Sistemas de Controle para a Operação
  Eficiente de Conversores Modulares Multiníveis em Acionamentos
  Elétricos.* PhD thesis, UFSC.
