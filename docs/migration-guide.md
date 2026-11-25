# Migration Guide

* [From 2.0 to 2.1](#from-20-to-21)
* [Pre-1.0 to 1.0](#pre-10-to-10) and the 1.5 → 1.6 notes below it.

## From 2.0 to 2.1

The existing `simulate()` options keep their 2.0 defaults (a few defaults
below the Python API moved: see the changelog's *Changed*). A 2.0 script
still needs attention in three ways: a few calls now raise a named error instead of running, some
calls print a `DeprecationWarning` / `FutureWarning`, and some results move
because a silent wrong answer was corrected. The complete list is the
2.1.0 entry of the changelog (sections *Changed*, *Deprecated*, *Fixed*).

### Calls that now raise

An option the engine that runs cannot honour is refused instead of being
ignored with a `UserWarning`. The most common case: `rtol` with a `dt`
(an explicit `dt` routes `engine='auto'` to the fixed step, which has no
error control).

```python
import numpy as np
import pulsim as p

b = p.CircuitBuilder()
b.add_voltage_source("V1", "in", "gnd", 1.0)
b.add_resistor("R1", "in", "out", 1e3)
b.add_capacitor("C1", "out", "gnd", 1e-6)
try:
    p.simulate(b, t_end=1e-3, dt=1e-6, rtol=1e-6)
except p.UnsupportedError:
    print("rtol with dt: refused")
res = p.simulate(b, t_end=1e-3, dt=1e-6)          # drop rtol, or drop dt
print(type(res.times).__name__, res.engine_used)
```

```text
rtol with dt: refused
ndarray pwl
```

Others, each with a message that names the fix: `integrator=` on `pwl`,
`trbdf2` or `auto`; `controller_period=` without a `step_observer`; a
switch mask of the wrong width; `store_every < 1`; `thermal_specs` on an
element without a loss model; `run_mna_sweep` on a circuit with switches
and no `mask=`; `periodic_steady_state` on latching devices, loss ports
or closed loops. `add_nonlinear_diode` is retired:

```python
try:
    p.CircuitBuilder().add_nonlinear_diode("D1", "a", "gnd", {})
except p.RetiredDeviceError:
    print("use add_diode(..., V_f=...) or add_shockley_diode(...)")
```

```text
use add_diode(..., V_f=...) or add_shockley_diode(...)
```

### `res.times` is a numpy array

`SimulationResult.times` is a read-only, zero-copy `numpy.ndarray`
(it was a list). `len`, indexing, slicing and `np.asarray(res.times)`
are unchanged; list-only idioms (`res.times + [...]`, `.index`,
`if res.times:`) need `list(res.times)`. `np.asarray(res.times)` is now
the read-only view itself: use `np.array(res.times)` or `.copy()` before
writing into it.

### What the new warnings mean

| Warning (category) | Raised by | Replace with |
|---|---|---|
| `DeprecationWarning` | `simulate(max_newton_iterations=0)` | `None` (the default, 50); 0 will be refused |
| `DeprecationWarning` | `simulate(enable_nonlinear_refresh=...)` | omit it |
| `DeprecationWarning` | `builder.update_resistor_R` / `update_inductor_L` / `update_capacitor_C` / `update_voltage_source_V` | `builder.set_value("R1.R", v)` or `circuit.with_value("R1.R", v)` |
| `DeprecationWarning` | `add_pmsm`, `add_bldc`, `add_induction_motor`, `add_dc_motor` and their `make_*_observer` | `add_pmsm_mna`, `add_bldc_mna`, `add_induction_machine_mna`, `add_dc_machine` |
| `DeprecationWarning` | `make_thyristor_switch_fn`, `BlockChain.add_thyristor` | `CircuitBuilder.add_thyristor` |
| `DeprecationWarning` | `run_periodic_shooting`, `run_harmonic_balance` | `p.periodic_steady_state(..., engine='pwl', dt=dt)` |
| `DeprecationWarning` | a Python callable as `measured=` / `setpoint=` of a control helper, `add_c_block(fn=)` | a node name (`measured="vout"`), `code=` or `lib=` |
| `FutureWarning` | `add_igbt(...)` without `reverse_blocking` | pass `reverse_blocking=True` or `False` explicitly; the default becomes `True` in a later release |

```python
import warnings

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    p.simulate(b, t_end=1e-4, dt=1e-6, max_newton_iterations=0)
print(caught[0].category.__name__)
```

```text
DeprecationWarning
```

### Results that move

Seen 2026-11-25 on
this branch against a build of tag v2.0.0 (macOS arm64, same compiler). Two sets,
each run unchanged on both versions with `simulate()`'s default options:

* every benchmark netlist of v2.0.0's `benchmarks/circuits` that both versions
  load — 104 of 126, read by `benchmarks/tools/v1_reader.py` (2.0 shipped no
  reader), with the netlist's `dt` (so `engine='auto'` lands on `pwl`);
  quantity: the time-weighted mean of the benchmark's declared observable;
* every v2.0.0 example script (`examples/scripts`), stdout compared.

Of 99 observables, 47 differ by more than 1e-6 relative (36 by more than 1e-3,
26 by more than 1 %). Where the benchmark has its own baseline (26 rows), 2.1
is nearer it in 23; in the other 3 both versions are 4 V or more away, so that
baseline does not describe the netlist as read (no regression signal). Rows
marked shoot-through: 2.0 drew 1 kA or more through a switch (the switch masks
were checked identical on `synchronous_buck`); not yet traced to one entry.

The rows above 1 % (the full table is in the changelog's 2.1.0 *Changed*):

| Circuit (`benchmarks/circuits/`) | Quantity | 2.0.0 | 2.1.0 | Rel. | Against the benchmark's baseline | Explained by |
|---|---|---|---|---|---|---|
| thermal/long_run_drift_buck.yaml | mean V(out) | 4.75094 | 11.9869 | 6.0e-01 | 2.1 nearer ref 11.9868; error 7.24 → 0.000142 | 2.0 drew 2.4e+03 A through a switch (shoot-through); 2.1 does not — not traced to one entry |
| switching/isolated/dual_active_bridge.yaml | mean V(out) | -6.20319 | -10.3262 | 4.0e-01 | 2.0 nearer ref 5.31576; error 11.5 → 15.6 | 2.0 drew 4.8e+03 A through a switch (shoot-through); 2.1 does not — not traced to one entry |
| switching/boost/cascaded_boost_buck.yaml | mean V(out) | 5.8282 | 9.44043 | 3.8e-01 | 2.1 nearer ref 11.3487; error 5.52 → 1.91 | not traced to one entry |
| magnetics/half_bridge_transformer_dc.yaml | mean V(out) | 5.42476 | 4.3543 | 2.0e-01 | 2.1 nearer ref 4.22729; error 1.2 → 0.127 | 2.0 drew 2.4e+04 A through a switch (shoot-through); 2.1 does not — not traced to one entry |
| switching/multilevel/npc_three_level_halfbridge.yaml | mean V(n_lo) | 41.4404 | 35.5616 | 1.4e-01 | 2.1 nearer ref 34.1238; error 7.32 → 1.44 | 2.0 drew 5.6e+04 A through a switch (shoot-through); 2.1 does not — not traced to one entry |
| switching/boost/synchronous_boost.yaml | mean V(out) | 15.5875 | 17.8563 | 1.3e-01 | 2.1 nearer ref 17.8546; error 2.27 → 0.00173 | not traced to one entry |
| switching/isolated/synchronous_flyback.yaml | mean V(out) | -14.4634 | -15.543 | 7.0e-02 | no baseline | not traced to one entry |
| stress/stiff_rc_high_freq_switching.yaml | mean V(out) | 11.3398 | 11.9731 | 5.3e-02 | 2.1 nearer ref 11.9786; error 0.639 → 0.00557 | 2.0 drew 2.4e+03 A through a switch (shoot-through); 2.1 does not — not traced to one entry |
| switching/buck/cascaded_buck_buck.yaml | mean V(out) | 6.07937 | 6.39914 | 5.0e-02 | no baseline | 2.0 drew 2.4e+03 A through a switch (shoot-through); 2.1 does not — not traced to one entry |
| stress/periodic_rc_pwm.yaml | mean V(out) | 1.39954 | 1.46493 | 4.5e-02 | 2.1 nearer ref 2.46191; error 1.06 → 0.997 | not traced to one entry |
| switching/boost/boost_pfc_open_loop.yaml | mean V(out) | 46.3807 | 48.3155 | 4.0e-02 | 2.1 nearer ref 47.3531; error 0.972 → 0.962 | not traced to one entry |
| switching/isolated/forward_converter.yaml | mean V(out) | 11.2004 | 11.607 | 3.5e-02 | no baseline | not traced to one entry |
| switching/inverter/half_bridge_inverter_lc.yaml | mean V(load) | 12.2919 | 11.8652 | 3.5e-02 | 2.1 nearer ref 11.8664; error 0.426 → 0.00113 | 2.0 drew 1.2e+04 A through a switch (shoot-through); 2.1 does not — not traced to one entry |
| switching/multilevel/flying_capacitor_3level.yaml | mean V(out) | 12.3075 | 11.9824 | 2.6e-02 | 2.1 nearer ref 11.9888; error 0.319 → 0.00645 | 2.0 drew 1.2e+03 A through a switch (shoot-through); 2.1 does not — not traced to one entry |
| three_phase/interleaved_buck_3ph.yaml | mean V(out) | 6.47245 | 6.62523 | 2.3e-02 | 2.1 nearer ref 43.0579; error 36.6 → 36.4 | not traced to one entry |
| switching/buck/triple_sync_buck.yaml | mean V(out_b) | 11.3455 | 11.0925 | 2.2e-02 | no baseline | 2.0 drew 4.8e+03 A through a switch (shoot-through); 2.1 does not — not traced to one entry |
| switching/buck/synchronous_buck.yaml | mean V(out) | 11.3455 | 11.0925 | 2.2e-02 | no baseline | 2.0 drew 2.4e+03 A through a switch (shoot-through); 2.1 does not — not traced to one entry |
| switching/buck_boost/bidirectional_buck_boost.yaml | mean V(out) | 11.3404 | 11.0935 | 2.2e-02 | no baseline | 2.0 drew 2.4e+03 A through a switch (shoot-through); 2.1 does not — not traced to one entry |
| switching/buck/buck_with_current_sense.yaml | mean V(out) | 11.3347 | 11.0938 | 2.1e-02 | no baseline | 2.0 drew 2.4e+03 A through a switch (shoot-through); 2.1 does not — not traced to one entry |
| switching/buck/ll06_buck_vcswitch.yaml | mean V(out) | 23.2926 | 23.7919 | 2.1e-02 | no baseline | not traced to one entry |
| switching/inverter/half_bridge_inverter.yaml | mean V(load) | 11.7357 | 11.5307 | 1.8e-02 | no baseline | 2.0 drew 2.4e+03 A through a switch (shoot-through); 2.1 does not — not traced to one entry |
| switching/inverter/half_bridge_lc_filter.yaml | mean V(out) | 11.1898 | 11.3845 | 1.7e-02 | 2.1 nearer ref 11.3842; error 0.194 → 0.000279 | 2.0 drew 2.4e+03 A through a switch (shoot-through); 2.1 does not — not traced to one entry |
| switching/basic/pwm_voltage_rc.yaml | mean V(out) | 3.26242 | 3.20856 | 1.7e-02 | no baseline | not traced to one entry |
| switching/buck/ll08_buck_mosfet.yaml | mean V(out) | 5.07002 | 5.14011 | 1.4e-02 | no baseline | Level-1 MOSFET: diode `V_F` 0.5 → 0.7 V as a drop; no channel current at V_DS = 0 (Changed) |
| switching/buck/buck_mosfet_nonlinear.yaml | mean V(out) | 5.07002 | 5.14011 | 1.4e-02 | 2.1 nearer ref 20.0711; error 15 → 14.9 | Level-1 MOSFET: diode `V_F` 0.5 → 0.7 V as a drop; no channel current at V_DS = 0 (Changed) |
| switching/buck/high_freq_gan_buck.yaml | mean V(out) | 11.8483 | 11.9901 | 1.2e-02 | 2.1 nearer ref 11.9903; error 0.142 → 0.00019 | 2.0 drew 2.4e+03 A through a switch (shoot-through); 2.1 does not — not traced to one entry |

| 2.0.0 example script (run unchanged) | Quantity | 2.0.0 | 2.1.0 | Explained by |
|---|---|---|---|---|
| `run_buck_closed_loop_chain.py` | exit | runs | `ControlGraphBuildError`: `SetpointBlock` has no kernel implementation | Python evaluator of chains removed (Changed) |
| `run_foc_motor_drive.py` | exit | runs | `ControlGraphBuildError`: `RotorAngle` has no kernel implementation | same |
| `run_dc_motor_speed.py`, `run_pmsm_foc.py` | exit | runs | `AttributeError`: `MixedDomainBlockChain` has no `channels` | same |
| `run_buck.py`, `run_flyback.py`, `run_half_wave_rectifier.py`, `run_rlc_step_response.py`, `run_common_source_amplifier.py` | exit | `TypeError: 'module' object is not callable` | runs | 2.0 example breakage fixed |
| `run_thermal_buck.py` | T_j rise | 13.79 K | 15.19 K | `add_foster_network` was a Cauer ladder (Fixed) |
| `run_adaptive_rl_settle.py` | max \|i − i_an\| | 2.044e-02 A | 5.000e-04 A | TR-BDF2 error control (toward the closed form) |
| `run_ac_sweep_rc.py` | max error vs analytic | 0.005 dB / 0.694° | 0.003 dB / 0.338° | toward the analytic answer; not traced to one entry |
| `run_im_direct_online_start.py` | \|ψ_r\| | 0.9841 Wb | 0.9551 Wb | induction machine model (deprecated `add_induction_motor`); not traced |
| `run_im_ifoc_closed_loop.py` | i_d / i_q (targets 8 / 6 A) | 8.294 / 8.690 A | 9.005 / 7.507 A | same |
| `run_im_vf_open_loop.py` | ω_final, \|ψ_r\| | 157.07 rad/s, 0.9839 Wb | 157.13 rad/s, 0.9548 Wb | same |
| `run_psfb_ja_loss_analysis.py` | i_peak; permalloy loss | 7.493 A; 2692.6 W | 7.872 A; 24.43 W | J-A model; not traced (the 2.1 script uses another core) |
| `run_psfb_hysteretic_inloop.py` | i_peak; final M | 2.481 A; −306.2 A/m | 2.466 A; −185.2 A/m | not traced |
| `run_phase_shift_full_bridge.py` | V_out DC | 15.04 V | 15.22 V | not traced |
| `run_buck_with_helper.py` | V_out; duty | 4.9980 V; 0.4001 | 4.9997 V; 0.3818 | control chain order / kernel block laws (Changed) |
| `run_buck_chain_logged.py` | v_out; duty | 12.017 V; 0.506 | 12.005 V; 0.496 | same |
| `run_device_loss_to_thermal.py` | R_line loss | 1129.301 W | 1114.244 W | not traced |
| `run_boost_saturable_closed_loop.py` | mean V_out; i_L max | 22.98 V; 125.712 A | 23.00 V; 127.573 A | not traced |
| `run_buck_closed_loop.py` | max V_out in transient | 19.060 V | 19.096 V | not traced |
| `run_ac_sweep_buck.py` | \|H\| at 100 Hz | 22.9586 | 22.9253 | not traced |
| `run_auto_tune_buck.py` | \|G\| at 2 kHz; Kp | 57.928; 0.00725 | 57.854; 0.00728 | not traced |
| `run_pwm_chopper_realistic_mosfet.py` | V_drain mean | 5.34 V | 5.32 V | not traced |
| `run_boost_closed_loop.py`, `run_pfc_boost_closed_loop.py`, `run_adversarial_sweep.py` | printed values | — | last-digit changes (≤ 0.1 %) | — |

### PWL diode forward drop (PSIM `V_f`)

`add_diode` gained `V_f` (default 0): the PSIM "Diode Voltage Drop",
`v = V_f + i/g_on` while ON, so the diode dissipates `V_f·i + i²/g_on`.
Omitting it changes nothing. `V_th` stays the turn-on threshold.

**Breaking:** `add_three_phase_vsi(V_f=0.7)` now passes `V_f` as that
drop (it was the turn-on threshold `V_th` and added no loss). The
freewheel diodes' conduction loss rises by `V_f·I_avg`, they no longer
share current with a gated-on switch below `R_on·|i| = V_f`, and the
leg-output voltage during dead time shifts by `V_f`. For the old
behaviour pass `V_f=0.0, V_th=0.7`. `add_bridge_rectifier` gained a
`V_f=0.0` drop; its `V_th=0.7` threshold is unchanged. The composite MOSFET / IGBT
diodes' `V_F` became a conduction drop too (below).

### MOSFET body diode and IGBT freewheel diode: `V_F` is a drop

`add_mosfet`, `add_mosfet_with_body_diode`, `add_mosfet_level1` (body
diode) and `add_igbt_level1(with_fwd=True)` build their diode as
`add_diode(V_th=0, V_f=V_F)`: while it conducts, `v = V_F + i/g_on`. In 2.0
`V_F` was only the turn-on threshold. With the default `V_F=0.7`, 10 A
through the body diode of an off MOSFET drops 0.710 V instead of 0.010 V,
and a 24 V buck freewheeling through a gate-off `add_mosfet` reads
V_out 11.640 V instead of 11.990 V. The Level-1 diodes turned on at a fixed
0.5 V; they now take `V_F` (default 0.7 V). `add_igbt` has no diode and is
unchanged. To get the 2.0 diode back (a 0.7 V threshold, no drop), build the
switch and the diode yourself:

```python
def body_diode_drop(V_th, V_f):
    b = p.CircuitBuilder()
    b.add_voltage_source("Vd", "d", "gnd", 0.0)
    b.add_current_source("Is", "s", "gnd", 10.0)    # 10 A forced through the diode
    b.add_switch("Q", "d", "s", 1 / 1e-3, 1 / 1e9)  # the channel
    b.add_diode("Q_body", "s", "d", 1e3, 1e-9, V_th=V_th, V_f=V_f)
    off = p.SwitchStateMask(b.graph.num_switches)    # every bit off: channel open
    r = p.simulate(b, t_end=1e-6, dt=1e-7, switch_fn=lambda t: off)
    return r.v("s")[-1] - r.v("d")[-1]

print(f"2.0 diode (V_th=0.7, V_f=0): v_SD = {body_diode_drop(0.7, 0.0):.3f} V")
print(f"2.1 diode (V_th=0, V_f=0.7): v_SD = {body_diode_drop(0.0, 0.7):.3f} V")
```

```text
2.0 diode (V_th=0.7, V_f=0): v_SD = 0.010 V
2.1 diode (V_th=0, V_f=0.7): v_SD = 0.710 V
```

### Python callbacks in control chains

`MixedDomainBlockChain.make_step_observer(use_kernel=False)` raises
`DeprecationWarning` as an exception (a warnings filter does not silence
it): the Python evaluator of a chain is gone. A chain is compiled to the
kernel: `chain.channels` no longer exists, and a block with no kernel law
(`SetpointBlock`, `RotorAngle`, ...) raises `ControlGraphBuildError` when
the run starts. Wrap custom Python code in `pulsim.PythonBlock(fn)`.

### Default-path results to re-check

* A MOSFET with a nonlinear output capacitance on the variable-step engine
  (`engine='trbdf2'`, or `auto` without `dt`): the conduction energy after
  a gate edge reads +3.91 % against its closed form (a known limitation).
* Loss summaries: a switch's `i_rms` now counts the step where it flips
  (+0.56 % on a 30 % PWM at 100 steps per period, toward the closed form).

## Pre-1.0 to 1.0
Pulsim 1.0.0 shipped a brand-new kernel. The pre-1.0 API surface
(``Circuit``, ``Simulator``, ``YamlParser``, ``Preset``, ``codegen``,
``fmu``, ``schematic``, ``templates``, ``robust=True``, …) was
retired during the 1.0 cycle.

This guide maps every legacy idiom to its 1.0 equivalent. If you're
hitting an ``AttributeError: pulsim.Foo was a v1 symbol`` at runtime,
look up ``Foo`` below.

## At a glance

| Aspect | Pre-1.0 (`v1`) | 1.0 |
|---|---|---|
| Top-level import | `import pulsim as ps` | `import pulsim as p` (same alias works) |
| Build a circuit | `ps.Circuit()` + `ps.Resistor()` + `ps.add_component(...)` | `p.CircuitBuilder()` + `b.add_resistor(...)` |
| Run a transient | `ps.Simulator(ckt, opts).run_transient(x0)` | `p.simulate(b, t_end=..., dt=...)` |
| Load from YAML | `ps.YamlParser(opts).load("foo.yaml")` | `p.load_yaml_file("foo.yaml")` |
| AC sweep / Bode | `Simulator.run_ac_sweep(AcSweepOptions(...))` | `p.run_ac_sweep(b, …)` (MNA, fast) |
| Swept-sine FRA | `Simulator.run_fra(...)` | `p.run_fra(b, …)` (closed-loop friendly) |
| Closed-loop control | Hand-rolled callbacks / `signal_evaluator.py` | `p.MixedDomainBlockChain` + `PIController`, `PIDController`, … |
| Plot | matplotlib by hand | `p.scope(b, res, signals=[...])`, `p.plot_bode(...)` |
| Discovery | n/a | `p.catalog()`, `p.example("buck_open_loop")` |

The 1.0 surface is **flat** — there's no `pulsim.v2` namespace. All
device builders, analyses, and helpers live directly on the
`pulsim` module.

## Top-level objects

### `pulsim.Circuit` → `pulsim.CircuitBuilder`

<!-- doctest: skip reason="shows a pre-2.1 API from an older migration note; kept for readers porting old code" -->
```python
# Pre-1.0
ckt = ps.Circuit()
r1 = ckt.add_node("n1")          # node names were integer IDs returned by add_node
gnd = ckt.add_node("gnd")
ckt.add_component(ps.Resistor("R1", r1, gnd, 100.0))

# 1.0 — string node names; helpers per device kind
b = p.CircuitBuilder()
b.add_resistor("R1", "n1", "gnd", 100.0)
```

### `pulsim.Simulator` → `pulsim.simulate(...)`

<!-- doctest: skip reason="shows a pre-2.1 API from an older migration note; kept for readers porting old code" -->
```python
# Pre-1.0
opts = ps.SimulationOptions(t_end=1e-3, dt=1e-5)
sim = ps.Simulator(ckt, opts)
res = sim.run_transient(ckt.initial_state())

# 1.0
res = p.simulate(b, t_end=1e-3, dt=1e-5)
```

For full control over the cache/solver/options pipeline, the
explicit form is still available:

<!-- doctest: skip reason="shows a pre-2.1 API from an older migration note; kept for readers porting old code" -->
```python
cache = p.PwlStateSpaceCache(b.graph, b.pool)
cache.build(dt=1e-5)
opts = p.SimulationOptions(t_start=0.0, t_end=1e-3, dt=1e-5)
res  = p.run_transient(cache, b.graph, b.pool, opts,
                       switch_fn=lambda t: p.SwitchStateMask(0))
```

### `pulsim.YamlParser` → `pulsim.load_yaml_file(...)`

<!-- doctest: skip reason="shows a pre-2.1 API from an older migration note; kept for readers porting old code" -->
```python
# Pre-1.0
parser = ps.YamlParser(ps.YamlParserOptions())
ckt, opts = parser.load("circuit.yaml")

# 1.0
loaded = p.load_yaml_file("circuit.yaml")
b = loaded.builder
res = p.simulate(b, t_end=..., dt=...)
```

### `pulsim.Preset` / `pulsim.AdvancedOptions` → explicit `SimulationOptions`

No global presets in 1.0. Set what you need on `SimulationOptions`
directly. Solver kind, Newton tolerance, line-search globalization,
event-bisection thresholds — all explicit knobs.

## Device models

| Pre-1.0 | 1.0 |
|---|---|
| `ps.Resistor("R", a, b, R)` | `b.add_resistor("R", a, b, R)` |
| `ps.Capacitor("C", a, b, C)` | `b.add_capacitor("C", a, b, C)` |
| `ps.Inductor("L", a, b, L)` | `b.add_inductor("L", a, b, L)` |
| `ps.VoltageSource("V", a, b, V)` | `b.add_voltage_source("V", a, b, V)` |
| `ps.CurrentSource("I", a, b, I)` | `b.add_current_source("I", a, b, I)` |
| `ps.Diode("D", a, k, ...)` | `b.add_diode("D", anode, cathode, g_on=..., g_off=..., V_th=...)` |
| `ps.MOSFET(...)` | `b.add_mosfet_level1(...)` (Shichman-Hodges) or `b.add_mosfet_with_body_diode(...)` |
| `ps.IGBT(...)` | `b.add_igbt_level1(...)` |
| `ps.VoltageControlledSwitch(...)` | gate via `p.SwitchStateMask` + `switch_fn(t)` |
| `ps.Transformer(...)` | `b.add_two_winding_transformer(...)` or `b.add_multi_winding_transformer(...)` |
| `ps.SineParams(...)` | `b.add_sine_voltage_source(...)` |
| `ps.PulseParams(...)` | `b.add_pulse_voltage_source(...)` |
| `ps.PWMVoltageSource(...)` | `b.add_pwm_voltage_source(...)` |
| `ps.SaturableInductor(...)` | `b.add_saturable_inductor(...)` |

## Control surfaces

The hand-rolled pre-1.0 step-callback / `signal_evaluator.py` flows
are replaced by the `MixedDomainBlockChain` executor (runs in C++,
no Python interpreter cost per step):

<!-- doctest: skip reason="shows a pre-2.1 API from an older migration note; kept for readers porting old code" -->
```python
chain = p.MixedDomainBlockChain()
pi    = chain.add_block(p.PIController(kp=0.5, ki=1200.0))
chain.wire(source="vout", to=pi.input)
chain.wire(source=pi.output, to="duty")

res = p.run_transient_with_chain(
    cache, b.graph, b.pool, opts,
    switch_fn, chain, chain_dt=1e-5,
)
```

Available blocks: `PIController`, `PIDController`, `Comparator`,
`RateLimiter`, `FirstOrderLowPass`, `Clarke`/`Park`/inverse,
`ThyristorBlock`, `FuseBlock`, `FosterThermalNetwork`, ...

## Analyses

| Pre-1.0 | 1.0 |
|---|---|
| `Simulator.run_ac_sweep(...)` | `p.run_ac_sweep(b, freqs=..., excite_fn=..., output_idx=...)` (swept-sine) or `p.run_mna_sweep(b, ...)` (linear circuits and switches at a fixed `mask=`; refuses Newton devices) |
| `Simulator.run_fra(...)` | `p.run_fra(...)` — swept-sine FRA (closed-loop and nonlinear-friendly) |
| `Simulator.run_periodic_steady_state(...)` | not yet on 1.0 — file an issue if you need it |
| `pulsim.sweep.run(...)` (Monte-Carlo / LHS / Cartesian) | `p.sweep(builder_factory, ...)` and `p.monte_carlo(...)` |

## Features that did NOT migrate

| Feature | Status on 1.0.0 |
|---|---|
| `pulsim.codegen` (C99 codegen) | not on 1.0 |
| `pulsim.fmu` (FMI 2.0 CS export) | not on 1.0 |
| `pulsim.schematic` (ELK + netlistsvg auto-layout) | not on 1.0 |
| `pulsim.templates.{buck,boost,buckboost,…}` (converter auto-design) | not on 1.0 |
| `pulsim.Preset` / `AdvancedOptions` global presets | replaced by explicit `SimulationOptions` knobs |
| `compressor_load` + R600a/R134a refrigerants | not on 1.0 |
| MMC arm templates | not on 1.0 (single-arm helper deferred) |
| `RobustnessProfile` retry-loop | not on 1.0 (use DC-OP strategies) |
| Single-phase induction motor (PSC) | not on 1.0 (DC motor, BLDC, PMSM, three-phase induction are) |

If you need any of these on 1.0, pin pulsim `0.10.x` or open an
issue on the tracker — the 1.0 architecture supports each of them
as a future increment, none was retired for technical reasons.

## A note on the `pulsim.v2` namespace

Versions `0.10.x` exposed a parallel `pulsim.v2` surface while the
new kernel was being built next to the legacy one. With the legacy
kernel retired and the new kernel becoming the default in 1.0.0,
the `v2` prefix was dropped: `pulsim.v2.CircuitBuilder` is now
`pulsim.CircuitBuilder`, `pulsim.v2.simulate` is `pulsim.simulate`,
and so on. Code that used the `v2` prefix during the transition
just needs:

<!-- doctest: skip reason="shows a pre-2.1 API from an older migration note; kept for readers porting old code" -->
```python
# Before
import pulsim.v2 as p
from pulsim.v2 import CircuitBuilder, simulate

# After
import pulsim as p
from pulsim import CircuitBuilder, simulate
```

The C++ side mirrors this: `pulsim::v2::Foo` is now `pulsim::Foo`,
headers moved from `pulsim/v2/<sub>/<file>.hpp` to
`pulsim/<sub>/<file>.hpp`, and the CMake alias renamed from
`pulsim::v2` to `pulsim::core`.

## 1.5 → 1.6 — API stability notes

Three call-site shapes changed between 1.5 and 1.6.x. Each one
broke working GUI integrations on the bump; flagged here so future
binders don't trip on the same shape.

### `pulsim.sweep` — package → single function

In 1.5, `pulsim.sweep` was a *subpackage* with helpers:

<!-- doctest: skip reason="shows a pre-2.1 API from an older migration note; kept for readers porting old code" -->
```python
# Pre-1.6 — no longer works
from pulsim.sweep import Distribution, Cartesian, metrics

dist = Distribution.normal(mean=10.0, std=1.0)
grid = Cartesian({"R": [10, 20, 50]})
kpi = metrics.rms_voltage("vout")
```

In 1.6, `pulsim.sweep` is a single *function* (Cartesian grid by
default) and `pulsim.monte_carlo` is its random-draw sibling. The
distribution / metric helper classes were retired — callers supply
their own lambdas:

<!-- doctest: skip reason="shows a pre-2.1 API from an older migration note; kept for readers porting old code" -->
```python
# 1.6 — grid sweep (drop-in for the old Cartesian pattern)
import pulsim as p

def build_buck(R: float):
    b = p.CircuitBuilder()
    ...   # use R inside
    return b

def kpi(res, params):
    import numpy as np
    return {"V_out": float(np.asarray(res.v("vout"))[-1])}

res = p.sweep(
    build_buck,
    params={"R": [10.0, 20.0, 50.0]},   # Cartesian grid
    kpi_fn=kpi,
    t_end=10e-3, dt=1e-6,
)
print(res.to_dataframe())
```

<!-- doctest: skip reason="shows a pre-2.1 API from an older migration note; kept for readers porting old code" -->
```python
# 1.6 — Monte Carlo (drop-in for the old Distribution pattern)
res = p.monte_carlo(
    build_buck,
    distributions={
        "R": lambda rng: rng.normal(10.0, 1.0),    # ← was Distribution.normal
    },
    kpi_fn=kpi,
    n_samples=500,
    seed=42,                                         # reproducible
    t_end=10e-3, dt=1e-6,
)
```

Each `distributions[name]` is a callable that receives a
`numpy.random.Generator` (the default RNG seeded by `seed=`) and
returns a single float. Use `rng.normal`, `rng.uniform`,
`rng.lognormal`, etc. directly — no helper class needed.

| 1.5 helper | 1.6 replacement |
|---|---|
| `pulsim.sweep.Distribution.normal(μ, σ)` | `lambda rng: rng.normal(μ, σ)` |
| `pulsim.sweep.Distribution.uniform(a, b)` | `lambda rng: rng.uniform(a, b)` |
| `pulsim.sweep.Distribution.lognormal(μ, σ)` | `lambda rng: rng.lognormal(μ, σ)` |
| `pulsim.sweep.Cartesian({...})` | `params={...}` kwarg on `p.sweep` |
| `pulsim.sweep.metrics.rms_voltage(name)` | KPI lambda — extract the trace, compute RMS yourself |
| `pulsim.sweep.metrics.peak_current(name)` | KPI lambda — `np.max(np.abs(res.i(name)))` |
| `pulsim.sweep.metrics.settling_time(...)` | KPI lambda — your own first-time-below-2 %-band logic |

The 1.6 `kpi_fn` returns a `dict[str, float]`. Combine multiple
KPIs in one pass:

<!-- doctest: skip reason="shows a pre-2.1 API from an older migration note; kept for readers porting old code" -->
```python
def kpi(res, params):
    import numpy as np
    vout = np.asarray(res.v("vout"))
    iL = np.asarray(res.i("L1"))
    return {
        "V_out_ss": float(vout[-1]),
        "V_out_rms": float(np.sqrt(np.mean(vout ** 2))),
        "I_L_peak": float(np.max(np.abs(iL))),
    }
```

This is a deliberate API simplification — the 1.5 helper classes
duplicated what callers can express in three lines of numpy.

### `add_rc_snubber` — positional → keyword-only

<!-- doctest: skip reason="shows a pre-2.1 API from an older migration note; kept for readers porting old code" -->
```python
# Pre-1.6 — used to accept positional args
p.add_rc_snubber(builder, R, C, from_node, to_node)   # TypeError in 1.6

# 1.6 — keyword-only
p.add_rc_snubber(
    builder,
    R=1.0, C=470e-9,
    from_node="vdc", to_node="sw",
    name_prefix="Snub",     # optional, defaults to "Snub"
)
```

The full 1.6 signature:

<!-- doctest: skip reason="shows a pre-2.1 API from an older migration note; kept for readers porting old code" -->
```python
def add_rc_snubber(
    builder, *,
    R: float, C: float,
    from_node: str, to_node: str,
    name_prefix: str = "Snub",
) -> None
```

Positional calls now raise `TypeError: add_rc_snubber() takes 1
positional argument but 5 were given` — the `*,` after `builder`
forces every parameter to be named.

### PMSM / VSI / motors — module-level functions, no `*Params` struct

In 1.5 some helpers had a Pre-1.0-style POD params struct (e.g.
`PmsmParams(R_s=..., L_s=...)` → `circuit.add_pmsm(params)`).
1.6 standardised on module-level functions that take the parameters
as **direct kwargs**. There's no `PmsmParams` / `ThreePhaseVsiParams`
struct — pass the parameters directly:

<!-- doctest: skip reason="shows a pre-2.1 API from an older migration note; kept for readers porting old code" -->
```python
# 1.6
motor = p.add_pmsm(
    builder,
    name="M1",
    phase_nodes=("ua", "ub", "uc"),
    neutral_node="nn",
    R_s=0.5, Ld=1.8e-3, Lq=2.2e-3,   # T2.1: per-axis Ld/Lq for IPM
    psi_pm=0.05, pole_pairs=4,
    J=1e-3, B=1e-4,
    i_d_init=0.0, i_q_init=0.0,       # T2.1: per-axis ICs
    theta_init=0.0,
)
```

Same shape for `add_three_phase_vsi`, `add_bldc`,
`add_induction_motor`, etc. — pass the parameters as keyword args
on the module-level function; the helper returns a small dataclass
(e.g. `PMSM`, `BLDC`) carrying the live state. There is no separate
params struct.

If a GUI converter built against the old `Params` shape needs to
construct kwargs from a settings dict, just splat it:

<!-- doctest: skip reason="shows a pre-2.1 API from an older migration note; kept for readers porting old code" -->
```python
pmsm_kwargs = {
    "name": "M1",
    "phase_nodes": ("ua", "ub", "uc"),
    "neutral_node": "nn",
    "R_s": settings["Rs"],
    "Ld": settings["Ld"],
    "Lq": settings["Lq"],
    "psi_pm": settings["psi_pm"],
    "pole_pairs": settings["pp"],
    "J": settings["J"],
    "B": settings.get("B", 0.0),
}
motor = p.add_pmsm(builder, **pmsm_kwargs)
```

This is intentional — no parallel `*Params` data structure to keep
in sync with the kwarg list.

## See also

- `examples/scripts/` — 20 runnable reference scripts.
- `docs/tutorials/` — six narrative tutorials.
- `docs/api-reference.md` — the full surface in one page.
- `docs/gotchas.md` — every footgun we've hit so far.
