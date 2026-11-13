# Pulsim 2.x usage guide (for AI coding assistants)

This file ships inside the `pulsim` package and describes its version.
`python -m pulsim.guide` prints it with the installed version on the
first line; `pulsim.ai_guide()` returns the same text. When this guide
and your memory of Pulsim disagree, the guide wins: models are trained
on Pulsim 1.x and other simulators, and Pulsim 2 changed behaviour but
kept most names. Every `python` block below is executed by the test
suite, and every printed number is the one it prints.

Short version:

- Build with `p.CircuitBuilder()`, run with `p.simulate(b, t_end, dt=None)`,
  read with `res.v("node")`, `res.i("branch")`, `res.times`.
- Pass `dt` for PWM-driven converters (fixed-step `pwl`), with every
  gate edge on the `dt` grid (3.2). Without `dt` the default
  `engine='auto'` usually picks the variable-step TR-BDF2.
- Never `.mean()` a waveform. Use `p.steady_state_stats`, `p.time_average`,
  `p.time_rms` or `np.trapezoid`. RMS of a switched (pulsed) current on
  a fixed grid: `np.sqrt(p.time_average(t, i**2))` (trap 14).
- Switch bits come from `b.switch_index_of(name)`; mask size from
  `b.graph.num_switches`. `p.SwitchStateMask(n)` is a SIZE.
- PWL diode `V_th` is a turn-on threshold; `V_f` is the forward drop.
  Without `dt`, rerun with `dt` on any `simulate` warning (trap 13).
- Loss and thermal summaries average the WHOLE run, start-up included
  (3.4).
- `run_mna_sweep(freqs=)` covers linear parts and switches at `mask=`;
  it refuses Newton devices.
- Check every result physically (section 6) before reporting it.
- A Pulsim exception is a refusal with a reason. Read it; do not catch
  it, and do not loosen `tol_newton_dx` / `tol_newton_res` to get past it.

## 1. Ask the installed package first

- `python -m pulsim.guide` (this text) and `python -m pulsim.guide --path`.
- The native classes are pybind11 builtins: `inspect.signature` fails on
  them. Read the first line of `__doc__`, or the shipped type stubs
  (`pulsim/_pulsim.pyi`, PEP 561) that IDEs and mypy read.
- `p.catalog()` lists every builder method and helper by category;
  `p.example("buck")` prints a runnable snippet (`"rc"`, `"boost"`,
  `"yaml"`, `"bode"`, ...).
- `res.v(name)` / `res.i(name)` with a misspelled name raise
  `p.NameNotFoundError` with close matches. An attribute that does not
  exist on `pulsim` raises `AttributeError`, with a migration hint for
  retired 1.x names.

```python
import pulsim as p

print(p.CircuitBuilder.add_current_source.__doc__.splitlines()[0])
```

```text
add_current_source(self: pulsim._pulsim.CircuitBuilder, name: str, n_pos: str, n_neg: str, I: typing.SupportsFloat | typing.SupportsIndex) -> pulsim._pulsim.CircuitBuilder
```

That union is pybind11's `float` (the stub writes `float`).

## 2. Mental model

- **Netlist by name.** Nodes are strings; `"gnd"`, `"GND"` and `"0"` are
  ground. Every `b.add_*` call adds a branch between two nodes; the
  first argument is the branch name used later by `res.i(name)`.
- **Piecewise-linear switching.** `add_switch`, `add_mosfet`, `add_igbt`
  and `add_diode` are two-state branches: a conductance `g_on` (or
  `1/R_on`) when on, `g_off` when off; only `add_diode(V_f=)` adds a
  series drop, while on. Each one owns a bit in the switch-state mask. Controlled
  switches follow `switch_fn(t) -> SwitchStateMask`; diodes commutate
  on their own. Smooth devices (`add_shockley_diode`, `add_mosfet_level1`,
  saturable magnetics) are solved with Newton; `add_nonlinear_diode` is retired.
- **Engines** (`p.simulate(..., engine=...)`):

| engine | step | use it for |
|---|---|---|
| `'auto'` (default) | routes: `dt` given -> `pwl`; no `dt` -> TR-BDF2 when the circuit qualifies | most runs |
| `'pwl'` | fixed `dt` (required), trapezoidal, cached factorisation per switch state | PWM converters, bit-exact reruns |
| `'trbdf2'` | variable, error-controlled; `dt` is a ceiling | accuracy on smooth or diode-commutated circuits |
| `'dsed'` | variable, event-driven; refuses Newton devices and floating subnets (reference a star neutral with 1 MOhm) | cross-checks of PWL circuits |

- `res.engine_used` says which engine ran; `res.engine_route_reason`
  says why, when `auto` did not take the variable-step engine.
- **Pass `dt`** when a `switch_fn` drives the circuit, when you need a
  uniform grid, or when you compare runs bit for bit. Every on-time
  must be a whole number of steps, and every edge on the grid (3.2).
- **Omit `dt`** for passive, source-driven and diode-commutated
  circuits (trap 13), and as the second engine in a cross-check.

## 3. Canonical examples

### 3.1 RC step: the minimal script

```python
import numpy as np
import pulsim as p

b = p.CircuitBuilder()
b.add_voltage_source("Vin", "n0", "gnd", 5.0)
b.add_resistor("R", "n0", "vc", 1e3)
b.add_capacitor("C", "vc", "gnd", 1e-6)          # tau = 1 ms

res = p.simulate(b, t_end=5e-3)                  # no dt: engine='auto'
v_tau = np.interp(1e-3, res.times, res.v("vc"))  # grid is non-uniform
print(f"v_C(tau) = {v_tau:.3f} V [{res.engine_used}], "
      f"exact {5 * (1 - np.exp(-1)):.3f} V")
```

```text
v_C(tau) = 3.161 V [trbdf2], exact 3.161 V
```

### 3.2 Buck converter with a PWM `switch_fn`

A `switch_fn` maps `t` to a `SwitchStateMask`. On the fixed-step engine
gate edges can only fall on the `dt` grid, so make the on-time a whole
number of steps (trap 10):

```python
DT = 1e-7

b = p.CircuitBuilder()
b.add_voltage_source("Vin", "vin", "gnd", 24.0)
b.add_mosfet("Q1", "vin", "sw")              # also adds diode "Q1_body"
b.add_diode("D1", "gnd", "sw", 1e3, 1e-9)    # g_on, g_off; V_th = 0
b.add_inductor("L", "sw", "vout", 100e-6)
b.add_capacitor("C", "vout", "gnd", 47e-6)
b.add_resistor("RL", "vout", "gnd", 10.0)

q1 = b.switch_index_of("Q1")                 # never count bits by hand
n_sw = b.graph.num_switches                  # Q1, Q1_body, D1 -> 3
pwm = p.make_pwm_switch_fn(100e3, 0.5, q1, n_sw)   # f, D, bit, size

res = p.simulate(b, t_end=10e-3, dt=DT, switch_fn=pwm)
s = p.steady_state_stats(res, "vout", t_from=8e-3)
il = p.steady_state_stats(res, "L", t_from=8e-3, kind="i")
print(f"bits Q1={q1} of {n_sw}; V_out = {s.mean:.3f} V, ripple "
      f"{s.ripple_pp * 1e3:.1f} mV pp, I_L {il.ripple_pp:.3f} A pp [{res.engine_used}]")
print(f"analytic: dI = {12 * 5e-6 / 100e-6:.3f} A, dV = dI/(8 f C) = "
      f"{0.6 / (8 * 100e3 * 47e-6) * 1e3:.1f} mV")
```

```text
bits Q1=0 of 3; V_out = 11.999 V, ripple 16.8 mV pp, I_L 0.589 A pp [pwl]
analytic: dI = 0.600 A, dV = dI/(8 f C) = 16.0 mV
```

`steady_state_stats` returns `.mean .rms .min .max .ripple_pp .t_from
.t_to`. An edge of the built-in helpers that falls on a `dt` grid point
takes effect at that point, so the on-time is the same in every period:

```python
on = {sum(pwm(k * DT).get(q1) for k in range(j * 100, j * 100 + 100))
      for j in range(600)}
print("on-steps per period:", sorted(on))    # 50 asked
```

```text
on-steps per period: [50]
```

For a half bridge, `p.make_dead_time_pwm_pair_fn` drives the high- and
low-side bits with a dead time; for a three-phase bridge see 3.7. The
helpers also name their edges (`pwm.next_edge_after(t)`), so the
variable-step engines land on them. A plain callable has no edges:
`trbdf2` can step over a short pulse or dead time silently (D = 0.03:
0.17 V vs 1.20 V) and `dsed` bisects only changes it sees at its
`dt_max / 10` polls (and warns). Use a helper or pass `dt`. Both call
it out of time order: keep it a pure function of `t`.

### 3.3 Bridge rectifier from `pulsim.topology`

Topology helpers are module-level functions that take the builder first
and keyword-only node names. The diodes are named `<name>__D1` ...
`<name>__D4`. Ground one DC rail (trap 8).

```python
b = p.CircuitBuilder()
b.add_sine_voltage_source("Vac", "ac1", "ac2", 0.0, 100.0, 50.0)
br = p.add_bridge_rectifier(b, "BR", ac_a="ac1", ac_b="ac2",
                            dc_pos="dc", dc_neg="gnd", V_th=0.7)
b.add_capacitor("C", "dc", "gnd", 1e-3)
b.add_resistor("RL", "dc", "gnd", 50.0)

rect = p.simulate(b, t_end=0.06)
t = np.asarray(rect.times)
v_dc = p.time_average(t, rect.v("dc"), t_from=0.04)
print(f"{br.diode_names}; V_dc = {v_dc:.2f} V [{rect.engine_used}]")
```

```text
['BR__D1', 'BR__D2', 'BR__D3', 'BR__D4']; V_dc = 92.83 V [trbdf2]
```

It may warn of a step accepted above tolerance at an event; run the
balance of 6.1 before using any diode current from it.

### 3.4 Losses and junction temperature

`p.device_loss_summary` returns one dict per branch. Switching
energies come from `switch_specs`: each gate edge costs
`E_ref * (|V_blk| / V_ref) * (|I| / I_ref)`, with `V_blk` measured across
the open switch (before a turn-on, after a turn-off) and `I` through the
closed one (a soft edge is cheap, 3.7).
An offset-plus-slope conduction model `V_f0*|i| + r_on*i^2` comes from
`conduction_specs`. `p.device_thermal_summary` adds a Foster network per
device (`p.FosterStage(R_th_K_per_W, tau_s)`). Both cover the WHOLE
record unless you pass `t_from=` / `t_to=` (seconds; bounds interpolated
on any grid, edges counted when they start inside): on the bridge of
trap 13 (`t_end = 0.1 s`, `V_f0 = 0.7`, `r_on = 0.02`) the run's diode
conduction loss is 17x its 80-100 ms value. `T_j_trace` starts at
`T_ambient_C` at t = 0 and settles only after several of the largest
`tau`; `T_j_avg` / `T_j_peak` cover the window.

```python
b = p.CircuitBuilder()
b.add_voltage_source("Vin", "vin", "gnd", 24.0)
b.add_mosfet("Q1", "vin", "sw", R_on=0.05)
b.add_diode("D1", "gnd", "sw", 1e3, 1e-9)
b.add_inductor("L", "sw", "vout", 100e-6)
b.add_capacitor("C", "vout", "gnd", 47e-6)
b.add_resistor("RL", "vout", "gnd", 2.0)
pwm = p.make_pwm_switch_fn(100e3, 0.5, b.switch_index_of("Q1"), b.graph.num_switches)
res = p.simulate(b, t_end=5e-3, dt=DT, switch_fn=pwm)
t = np.asarray(res.times)

sw = {"Q1": {"E_on_ref": 20e-6, "E_off_ref": 30e-6, "V_ref": 24.0,
             "I_ref": 6.0}}
cond = {"D1": {"V_f0": 0.7, "r_on": 0.02}}
for w in (None, 3e-3):
    rows = {r["name"]: r for r in p.device_loss_summary(
        b, res, switch_specs=sw, conduction_specs=cond, t_from=w)}
    q, d = rows["Q1"], rows["D1"]
    print(f"{'3-5 ms' if w else 'run'}: Q1 cond {q['P_avg']:.3f} W, "
          f"sw {q['P_sw_avg']:.3f} W; D1 cond {d['P_cond_model_avg']:.3f} W")

th = p.device_thermal_summary(
    b, res, switch_specs=sw, T_ambient_C=40.0, t_from=3e-3,
    thermal_specs={"Q1": {"stages": [p.FosterStage(0.5, 1e-3),
                                     p.FosterStage(1.0, 2e-2)]}})
P = th[0]["P_total_avg"]
print(f"Q1: P = {P:.2f} W, T_j peak = {th[0]['T_j_peak']:.1f} C at 5 ms; "
      f"settles near 40 + 1.5 K/W * P = {40 + 1.5 * P:.1f} C")
```

```text
run: Q1 cond 0.905 W, sw 5.034 W; D1 cond 2.453 W
3-5 ms: Q1 cond 0.878 W, sw 4.987 W; D1 cond 2.422 W
Q1: P = 5.86 W, T_j peak = 44.2 C at 5 ms; settles near 40 + 1.5 K/W * P = 48.8 C
```

`P_avg` of a diode is `v^2 * g_on` only: the PWL diode has no forward
drop (trap 1). Diode losses are only as good as the diode current;
check it with the charge balance of section 6.1 first.

`diode_specs` (reverse recovery) charges `E_rr_ref * V_R / V_R_ref` per
turn-off, with `V_R` read one `t_rr` (default 100 ns) after it:

```python
rr = {r["name"]: r for r in p.device_loss_summary(
    b, res, diode_specs={"D1": {"E_rr_ref": 10e-6, "V_R_ref": 24.0}})}
print(f"P_rr = {rr['D1']['P_sw_avg']:.3f} W, {rr['D1']['n_turn_off_events']} "
      "turn-offs (10 uJ * 100 kHz = 1 W at V_R = 24 V)")
```

```text
P_rr = 0.988 W, 500 turn-offs (10 uJ * 100 kHz = 1 W at V_R = 24 V)
```

### 3.5 YAML netlist

`p.load_yaml_*` return a `LoadedCircuit` (`.builder`, `.options`,
`.switch_names`). `simulation:` must be top-level (nested: refused);
`p.simulate(lc)` takes t_end/dt from it. The gate drive is not in the
YAML: pass a `switch_fn`.

```python
lc = p.load_yaml_string("""
circuit:
  devices:
    - {type: voltage_source, name: V1, from: in, to: gnd, V: 10.0}
    - {type: resistor, name: R1, from: in, to: out, R: 1.0e3}
    - {type: capacitor, name: C1, from: out, to: gnd, C: 1.0e-6}
simulation:
  t_end: 5.0e-3
  dt: 1.0e-6
""")
res = p.simulate(lc)
print(f"v_out(end) = {res.v('out')[-1]:.3f} V")
```

```text
v_out(end) = 9.933 V
```

### 3.6 Parameter sweep

`p.sweep` calls a builder factory per point (Cartesian product of
`params`), forwards `**simulate_kwargs` to every run, and calls
`kpi_fn(res, params)`.

```python
def make_buck(R_load):
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", 24.0)
    b.add_mosfet("Q1", "vin", "sw")
    b.add_diode("D1", "gnd", "sw", 1e3, 1e-9)
    b.add_inductor("L", "sw", "vout", 100e-6)
    b.add_capacitor("C", "vout", "gnd", 47e-6)
    b.add_resistor("RL", "vout", "gnd", R_load)
    return b

ref = make_buck(10.0)                        # same topology -> same bits
pwm = p.make_pwm_switch_fn(100e3, 0.5, ref.switch_index_of("Q1"), ref.graph.num_switches)
out = p.sweep(make_buck, params={"R_load": [5.0, 10.0, 100.0]},
              kpi_fn=lambda r, prm: {"vout": p.steady_state_stats(
                  r, "vout", t_from=8e-3).mean},
              t_end=10e-3, dt=DT, switch_fn=pwm, verbose=False)
assert not out.failed, out.failed
for R, v in zip(out.params["R_load"], out.kpis["vout"]):
    print(f"R_load = {R:5.1f} ohm -> V_out = {v:.2f} V")
```

```text
R_load =   5.0 ohm -> V_out = 12.00 V
R_load =  10.0 ohm -> V_out = 12.00 V
R_load = 100.0 ohm -> V_out = 15.79 V
```

At 100 ohm the inductor current reaches zero each cycle (discontinuous
conduction), so `V_out` rises above `D * V_in`: physics, not an error. `p.sweep_path_aware` reuses the factorisation for R/L/C-only
sweeps; `parallel=True` runs on threads when the step loop is native.

### 3.7 Three-phase inverter with sinusoidal PWM

`add_three_phase_vsi` returns `switch_indices` in the order
`p.ThreePhaseLegIndices` takes them (HSa, LSa, HSb, LSb, HSc, LSc);
`p.make_three_phase_spwm_fn` drives all six with a dead time.

```python
b = p.CircuitBuilder()
b.add_voltage_source("Vdc", "dcp", "gnd", 600.0)
vsi = p.add_three_phase_vsi(b, "INV", vdc_pos="dcp", vdc_neg="gnd",
                            out_a="a", out_b="b", out_c="c")
for ph in "abc":                             # star load, neutral "n" floats
    b.add_resistor(f"R{ph}", ph, f"m{ph}", 10.0)
    b.add_inductor(f"L{ph}", f"m{ph}", "n", 5e-3)
legs = p.ThreePhaseLegIndices(*vsi.switch_indices)
for dead in (0.0, 1e-6):                     # f_carrier, f_1, m, legs, size, dead time
    spwm = p.make_three_phase_spwm_fn(10e3, 50.0, 0.8, legs,
                                      b.graph.num_switches, dead)
    inv = p.simulate(b, t_end=40e-3, dt=1e-7, switch_fn=spwm)
    t = np.asarray(inv.times)
    w = t >= 20e-3                           # one whole 50 Hz period
    c1 = 2 * np.trapezoid((inv.v("a") - inv.v("b"))[w]
                          * np.exp(-2j * np.pi * 50 * t[w]), t[w]) / 20e-3
    print(f"dead {dead:.0e} s: V_LL1 = {abs(c1) / np.sqrt(2):.1f} V rms, "
          f"I_a = {p.time_rms(t, inv.i('La'), t_from=20e-3):.2f} A rms, "
          f"P_dc = {p.time_average(t, 600 * inv.i('Vdc'), t_from=20e-3):.0f} W")
    on = (inv.i("INV__Dha") > 1e-3)[w]       # HSa is switch_indices[0]
    hs = np.asarray(inv.switch_states)[w, vsi.switch_indices[0]]
    print(f"  INV__Dha on {100 * on.mean():.1f} %, {100 * (on & hs).mean():.1f} % with HSa on")
print(f"ideal V_LL1 = sqrt(3/8) * m * V_dc = {np.sqrt(3 / 8) * 0.8 * 600:.1f} V")
```

```text
dead 0e+00 s: V_LL1 = 293.9 V rms, I_a = 16.77 A rms, P_dc = -8435 W
  INV__Dha on 0.0 %, 0.0 % with HSa on
dead 1e-06 s: V_LL1 = 284.6 V rms, I_a = 16.24 A rms, P_dc = -7912 W
  INV__Dha on 1.0 %, 0.0 % with HSa on
ideal V_LL1 = sqrt(3/8) * m * V_dc = 293.9 V
```

The dead time costs `4/pi * V_dc * t_d * f_sw * sqrt(3/2)` = 9.4 V of
`V_LL1`; `P_dc < 0` is the source sign convention (section 4). In the
dead time the freewheel diode carries the current (drop `V_f + i / g_on`);
once `HSa` is on, the switch takes it all and the diode never shares
(printed above). `switch_specs` on `HSa` (4 / 3.5 mJ at 600 V, 50 A)
give 11.3 W with the dead time (soft edges cost about 0 J) and 22.6 W
without (every edge hard). Model the dead time.

## 4. Exact signatures of the most used calls

`b` is a `CircuitBuilder`, `res` a `SimulationResult`; the test suite
checks these lines. `...` marks more keyword-only options (`help()`).

```text
p.CircuitBuilder()
b.add_resistor(name, n_pos, n_neg, R_ohms)
b.add_capacitor(name, n_pos, n_neg, C_farads, c0=None, ...)
b.add_inductor(name, n_pos, n_neg, L_henries, i0=None, ...)
b.add_voltage_source(name, n_pos, n_neg, V)
b.add_current_source(name, n_pos, n_neg, I)
b.add_sine_voltage_source(name, n_pos, n_neg, v_dc, v_amplitude, frequency, phase=0.0)
b.add_pulse_voltage_source(name, n_pos, n_neg, v_initial, v_pulsed, t_start, pulse_width, period=0.0, rise_time=0.0, fall_time=0.0)
b.add_pwm_voltage_source(name, n_pos, n_neg, v_high, v_low, frequency, duty, phase=0.0)
b.add_pwl_current_source(name, n_pos, n_neg, t, i, periodic=False)
b.add_vccs(name, in_pos, in_neg, n_pos, n_neg, gm)
b.add_cccs(name, control, n_pos, n_neg, gain)
b.add_ccvs(name, control, out_pos, out_neg, rm)
b.add_gyrator(name, p1_pos, p1_neg, p2_pos, p2_neg, g)
b.add_diode(name, anode, cathode, g_on, g_off, V_th=0.0, V_f=0.0)
b.add_shockley_diode(name, anode, cathode, I_S=1e-12, n=1.0, V_T=0.025852, G_min=1e-12, BV=0.0)
b.add_switch(name, n_pos, n_neg, g_on, g_off)
b.add_mosfet(name, drain, source, R_on=0.001, R_off=1000000000.0, body_diode=True, V_F=0.7)
b.add_igbt(name, collector, emitter, R_on=0.01, R_off=1000000000.0)
b.add_transformer(name, p_from, p_to, s_from, s_to, L_p, L_s, k=1.0)
b.node_id_of(name)
b.switch_index_of(name)
p.SwitchStateMask(num_switches=0)
p.make_pwm_switch_fn(frequency, duty, switch_idx, num_switches, phase=0.0)
p.make_dead_time_pwm_pair_fn(frequency, duty, hs_switch_idx, ls_switch_idx, num_switches, dead_time, phase=0.0)
p.make_three_phase_spwm_fn(carrier_frequency, modulation_frequency, modulation_index, legs, num_switches, dead_time, modulation_phase=0.0, carrier_phase=0.0)
p.ThreePhaseLegIndices(hs_a, ls_a, hs_b, ls_b, hs_c, ls_c)
p.simulate(builder, t_end=None, dt=None, *, engine='auto', rtol=None, atol=None, ...)
res.v(name, t=None)
res.i(name, t=None, *, port=None)
p.steady_state_stats(result, what, *, t_from=None, t_to=None, last=None, kind=None)
p.periodic_steady_state(builder, *, period, switch_fn=None, start_from=None, ...)
p.time_average(t, y, t_from=None, t_to=None)
p.time_rms(t, y, t_from=None, t_to=None)
p.add_bridge_rectifier(builder, name, *, ac_a, ac_b, dc_pos, dc_neg, g_on=1000.0, g_off=1e-09, V_th=0.7, V_f=0.0)
p.add_three_phase_vsi(builder, name, *, vdc_pos, vdc_neg, out_a, out_b, out_c, R_on=0.001, R_off=1000000000.0, with_freewheel_diodes=True, V_f=0.7, R_d=0.05, V_th=0.0)
p.load_yaml_string(yaml_text)
p.load_yaml_file(path)
p.sweep(builder_factory, *, params, kpi_fn, parallel=False, n_workers=None, backend='auto', verbose=True, **simulate_kwargs)
p.device_loss_summary(builder, result, *, switch_fn=None, core_loss_specs=None, diode_specs=None, switch_specs=None, conduction_specs=None, t_from=None, t_to=None)
p.device_thermal_summary(builder, result, *, thermal_specs, T_ambient_C=25.0, switch_fn=None, switch_specs=None, diode_specs=None, core_loss_specs=None, conduction_specs=None, t_from=None, t_to=None)
p.scope(builder, result, *, signals=None, title=None, save=None, stack=True, show=False, tight_lw=False)
```

Notes: `res.i(name)` is positive from a device's first node to its
second, through it: `add_mosfet` drain to source, `<name>_body` source
to drain, VSI diode `INV__Dha` from `out_a` to `vdc_pos`. So
`res.i(<voltage source>)` is negative when it delivers power (1 V into
1 ohm reads -1 A). `add_sine_current_source` / `add_pulse_current_source`
take the voltage versions' arguments, `v_` renamed `i_`. Current sources
(also pwl, `vccs`, `cccs`) drive OUT of `n_pos`; `res.i` reads that.
`control` names an earlier branch with a current unknown.
Gyrator: `i1 = g*v2`, `i2 = -g*v1` into `p1_pos`/`p2_pos`;
`port=1` reads `i1`. `t` in `res.v(name, t)` is a
sample INDEX; at a time, use `np.interp(t, res.times, res.v(name))`.
A ground name gives zeros. `phase` of the two PWM helpers and `carrier_phase`
are time shifts in SECONDS (`t - phase`); `modulation_phase` and the
`phase` of `add_sine_voltage_source` are radians.
`add_capacitor(c0=, esr=, esl=)` / `add_inductor(i0=, esr=, c_par=)`
set initial conditions and series parasitics.
Other `simulate` keywords: `switch_fn`, `t_start`, `store_every`,
`auto_regularize`, `progress`.

## 5. Traps

Each row was measured on this version. The fix column is what to write.

| # | Wrong assumption | What actually happens | Do this instead |
|---|---|---|---|
| 1 | `V_th` of `add_diode` / `add_bridge_rectifier`, or `add_mosfet(V_F=)` (body diode), is a forward drop | It is the turn-on threshold. On-state is `V_f + I/g_on`: 10 A into `g_on = 20 S` gives 0.500 V for `V_th = 0` and `0.7` alike, 1.200 V with `V_f = 0.7`. `add_three_phase_vsi(V_f=)` IS a drop | Pass `V_f` (PSIM's diode drop; losses include `V_f*I`), or `b.add_shockley_diode` for an exponential law. Never also a `conduction_specs` `V_f0` |
| 2 | `v.mean()` over a result is the average | The default grid is variable-step and clusters at events. On the rectifier of 6.2 `.mean()` reads 5.007 V against 6.330 V (-21 %) | `p.steady_state_stats`, `p.time_average`, `p.time_rms`, `np.trapezoid(y, t) / (t[-1] - t[0])` |
| 3 | `p.SwitchStateMask(1)` sets bit 0 | It is a mask of SIZE 1, all OFF | `m = p.SwitchStateMask(b.graph.num_switches); m.set(i, True)` |
| 4 | MOSFET k is switch bit k | `add_mosfet` adds `<name>_body`, so bits interleave with diodes (below) | `b.switch_index_of(name)` |
| 5 | Loosening `tol_newton_dx` / `tol_newton_res` fixes a Newton abort | Both are absolute on purpose. On a snubbed flyback, `tol_newton_dx=1e-5, tol_newton_res=1e-4` let `auto` finish at 3.53 V; the fixed engine gives 15.31 V. No error, no warning | Keep the defaults; read the abort; run `engine='pwl'` with a `dt`; fix gate edges, body diodes, initial state |
| 6 | `p.scope` is a module | `p.scope(b, res, signals=[...])` is the plotting function (needs matplotlib) | `from pulsim.scope import LiveScope` (needs pyqtgraph); `import pulsim.scope as m` gives you the function |
| 7 | Tie an isolated secondary or a floating neutral to ground with a tiny resistor | A milliohm tie is a galvanic bond. On the VSI of Tutorial 04 a 1 mOhm neutral-to-rail tie drove 58.8 A of DC through every phase (0.0 A floating or with 1 MOhm) | Leave it to `auto_regularize` (default on): it ties each floating subnet with 1 GOhm and warns. Your own tie: 1 MOhm or larger |
| 8 | `res.v` of a floating DC rail is the DC voltage | It is to ground. A bridge with `ac_b="gnd"`, load across `dp`/`dn`: `v("dp")` spans -0.7..325.0 V, `v("dn")` -325.0..0.7 V; `dp - dn` averages 321.85 V on both engines. Without `dt` it warns the rails "reach the rest of the circuit only through switches/diodes"; with 10 mF: 323.94 V, 324.51 V with `dt` | `res.v("dp") - res.v("dn")`, or ground one DC rail (3.3) |
| 9 | A fixed-step run is second order from t = 0 | It starts from an inconsistent state and is first order globally, ICs included: RL decay with `i0=1`, error at 1 ms 1.85e-3 at `dt=1e-5`, 0.92e-3 at `5e-6`; sample 0 of node voltages reads 0 V | Halve `dt` and compare (6.3); discard the start-up; or omit `dt` |
| 10 | Any `dt` works with PWM | Gate edges land on the `dt` grid. Buck D = 0.25 at 100 kHz with `make_pwm_switch_fn`: 6.24 V with `dt=2e-7` (on-time 12.5 steps, run as 13), 6.00 V with `dt=1e-7` | Make `D/f_sw` a whole number of steps; the helpers then hold it exactly (3.2) |
| 11 | Omitting `switch_fn` leaves switches open | With `dt`, undriven controlled switches are ALL CLOSED and `simulate` only warns: the buck of 3.2 runs at D = 1 and settles at 24.0 V, not 12 V; without `dt`, `auto` refuses | Always pass `switch_fn` when the circuit has controlled switches |
| 12 | `except Exception:` around `simulate` keeps a script robust | A broad except turns a refusal into a silent wrong answer | Let it raise, or catch `pulsim.UnsupportedError` (`.engine`, `.feature`); never `Exception` |
| 13 | A warning from a run without `dt` is noise | It names what TR-BDF2 could not guarantee: a diode that "rode its conduction boundary" or was held in a state "its commutation rule forbids" (a missed commutation, also named by `res.i`), a `switch_fn` pulse narrower than its sampling. No warning is no proof: a one-shot `switch_fn` pulse under a quarter of the step ceiling (`dt`, else at most `t_end/1000`) can be skipped silently (2 us in 20 ms: 11 of 20 phases) | Rerun with `dt` and compare (6.2); `dt` at most 4x the narrowest pulse |
| 14 | `p.time_rms` (and `.rms`) of a switched current is exact on a fixed grid | It joins samples with lines, so each hard edge is a one-step ramp: low by about `h / (3 t_on)` in `i^2`. Buck MOSFET (48 to 12 V, D = 0.25, 100 kHz): 1.4942 / 1.4993 / 1.5019 A at `dt` = 1e-7 / 5e-8 / 2.5e-8, analytic 1.5044 A. Same for `device_loss_summary`'s `i_rms` (not its `P_avg`) | `np.sqrt(p.time_average(t, i**2, t_from=...))`: 1.5044 A at all three. Keep `time_rms` for smooth or variable-grid traces; halve `dt` to check |
| 15 | Controllers run each step | A source ticks at `t_start + k*T`, `ceil(span/T)` times | `step_observer` needs `controller_period=`; see `res.control_tick_counts` |

Traps 1, 3 and 4, executed:

```python
for vth, vf in ((0.0, 0.0), (0.7, 0.0), (0.0, 0.7)):
    b = p.CircuitBuilder()
    b.add_current_source("I1", "a", "gnd", 10.0)   # 10 A out of "a"
    b.add_diode("D1", "a", "gnd", 20.0, 1e-9, V_th=vth, V_f=vf)
    r = p.simulate(b, t_end=1e-5, dt=1e-6)
    print(f"V_th = {vth}, V_f = {vf}: V(a) = {r.v('a')[-1]:.3f} V")

m = p.SwitchStateMask(1)
print("SwitchStateMask(1):", m.size, "bit, bit 0 on:", m.get(0))

b = p.CircuitBuilder()
b.add_voltage_source("V", "in", "gnd", 10.0)
b.add_mosfet("Q1", "in", "sw")
b.add_diode("D1", "gnd", "sw", 1e3, 1e-9)
b.add_mosfet("Q2", "sw", "gnd")
b.add_resistor("R", "sw", "gnd", 1.0)
print({n: b.switch_index_of(n) for n in ("Q1", "Q1_body", "D1", "Q2", "Q2_body")})
```

```text
V_th = 0.0, V_f = 0.0: V(a) = 0.500 V
V_th = 0.7, V_f = 0.0: V(a) = 0.500 V
V_th = 0.0, V_f = 0.7: V(a) = 1.200 V
SwitchStateMask(1): 1 bit, bit 0 on: False
{'Q1': 0, 'Q1_body': 1, 'D1': 2, 'Q2': 3, 'Q2_body': 4}
```

## 6. How to verify a result

A number is not a result until one of these checks passes. Treat 0 A,
+-1e7 V or more, and a perfectly flat trace as a setup error until proven
otherwise (a reversed source, an open switch, a floating node).

### 6.1 Charge and energy balance from independent branches

KCL holds over time: the charge a source delivers must reappear in the
branches it feeds, each computed independently. On the rectifier of 3.3
every coulomb from the AC source flows through the diodes into `RL` or
`C`:

```python
t = np.asarray(rect.times)
q_src = np.trapezoid(np.abs(rect.i("Vac")), t)
q_load = (np.trapezoid(rect.i("RL"), t)
          + 1e-3 * (rect.v("dc")[-1] - rect.v("dc")[0]))   # C * dV
print(f"source {q_src * 1e3:.1f} mC, load {q_load * 1e3:.1f} mC")
assert abs(q_load - q_src) < 1e-3 * q_src
```

```text
source 199.4 mC, load 199.4 mC
```

The diode branch currents must close the same balance (`D1`, `D2` feed
the DC node):

```python
q_diodes = np.trapezoid(rect.i("BR__D1") + rect.i("BR__D2"), t)
print(f"diodes {q_diodes * 1e3:.1f} mC")
assert abs(q_diodes - q_src) < 1e-3 * q_src
```

```text
diodes 199.4 mC
```

If a balance does not close, do not use any switch or diode branch
current from that run, nor any loss or temperature derived from one:
cross-check with a `dt` (6.2), then update pulsim or report the circuit.

For a converter, check the energy too: input energy
`np.trapezoid(v_in * i_in, t)` equals output energy plus losses plus the
change of stored energy (`C*v**2/2`, `L*i**2/2`). Over a window
`[t0, t1]` on a variable grid, take the end values with
`np.interp(t0, t, v)`, not the first sample after `t0`.

### 6.2 Two engines agree

Run the same circuit with `dt` (fixed step) and without (variable step)
and compare time-weighted statistics, never sample means:

```python
def ldrc():
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("V", "a", "gnd", 0.0, 20.0, 1e3)
    b.add_inductor("Lc", "a", "m", 111e-6)
    b.add_diode("D", "m", "o", 1e3, 1e-9, V_th=0.7)
    b.add_resistor("R", "o", "gnd", 2.0)
    b.add_capacitor("C", "o", "gnd", 10e-6)
    return b

fixed = p.simulate(ldrc(), t_end=3e-3, dt=2e-7)
var = p.simulate(ldrc(), t_end=3e-3)
for r in (fixed, var):
    t = np.asarray(r.times)
    v = r.v("o")
    print(f"{r.engine_used:6s} time-weighted {p.time_average(t, v, t_from=2e-3):.3f} V, "
          f"sample mean {v[t >= 2e-3].mean():.3f} V")
```

```text
pwl    time-weighted 6.331 V, sample mean 6.330 V
trbdf2 time-weighted 6.330 V, sample mean 5.007 V
```

The engines agree; the sample mean on the variable grid does not.
When two engines disagree, check the metric before blaming an engine.
Compare ripple too: the buck of 3.2 reads 11.999 V on both engines but
16.8 mV pp with `dt` and 15.4 mV without (analytic 16.0 mV): `min`,
`max` and `ripple_pp` are sample extremes, which the variable grid
(about 20 samples per period) misses. Read ripple from a fixed-`dt`
run, or `np.interp` onto a uniform grid.

### 6.3 Step-size convergence

Halve `dt` and rerun. A figure that moves by more than you need is not
converged. The fixed engine is first order from the start (trap 9):

```python
def rl():
    b = p.CircuitBuilder()
    b.add_inductor("L", "a", "gnd", 1e-3, i0=1.0)
    b.add_resistor("R", "a", "gnd", 1.0)            # tau = 1 ms
    return b

for dt in (1e-5, 5e-6):
    r = p.simulate(rl(), t_end=2e-3, dt=dt)
    err = np.interp(1e-3, r.times, r.i("L")) - np.exp(-1)
    print(f"dt = {dt:.0e}: i_L(tau) error {err:.2e} A")
```

```text
dt = 1e-05: i_L(tau) error 1.85e-03 A
dt = 5e-06: i_L(tau) error 9.21e-04 A
```

### 6.4 When `simulate` raises

The message names the device, node or option involved (trap 12). If
you change engine to get past a refusal, confirm the answer with 6.2.
Typical refusals: a singular matrix (a floating node), a Newton
abort (trap 5), a part file without provenance, a mismatched snapshot.

## 7. Coming from Pulsim 1.x or pre-1.0

Retired names still raise `AttributeError` with a hint (e.g.
`Circuit`, `Simulator`, `YamlParser`, `Preset`, `PulseParams` from
pre-1.0; the `v2` sub-namespace of 0.10). Changes that keep
names but change results:

| 1.x habit | 2.x |
|---|---|
| `Circuit()` + device objects | `p.CircuitBuilder()` + `b.add_*` methods |
| `Simulator(...).run_transient(...)` | `p.simulate(b, t_end, dt=None, ...)` |
| `YamlParser(...).load(path)` | `p.load_yaml_file(path)` / `p.load_yaml_string(text)` |
| `simulate` implied the fixed-step engine | `engine='auto'`: `dt` gives a fixed step; no `dt` usually gives TR-BDF2 |
| `add_mosfet` was one switch bit | it is two (`<name>_body`); use `b.switch_index_of` |
| an IGBT conducted in reverse | `b.add_igbt_level1` blocks it (`with_fwd=True` adds the freewheel diode). The PWL `b.add_igbt` is still a bidirectional switch: gated on, -10 A gives -0.1 V |
| `add_three_phase_vsi` built six bare switches | it adds six freewheel diodes (`with_freewheel_diodes=True`) |
| `p.add_hysteretic_inductor` + `p.make_hysteretic_inductor_observer` | the inductor is solved inside Newton; drop the observer, whose factory now raises |
| `p.add_pmsm` (observer, one average L) | still there; `b.add_pmsm_mna` stamps the machine with full saliency |
| `p.add_dc_motor` (observer) | deprecated; `b.add_dc_machine` solves it in the matrix |
| parts in `pulsim.lib` resolved by part number | they are synthetic; pass `allow_synthetic=True` to use them |

## 8. Paste this into your project's assistant config

For `CLAUDE.md`, `AGENTS.md`, `.github/copilot-instructions.md` or
`.cursorrules`:

```text
Before writing or changing code that uses pulsim, run
`python -m pulsim.guide` and follow it. Verify every simulated number
with the checks in its section 6 before reporting it.
```

## 9. Where to read more

Paths are from the repository root (the site mirrors `docs/`).

- `docs/getting-started.md`; `docs/tutorials/01-rc-charging.md` ...
  `07-dc-machine.md` (executed, end to end).
- `docs/api-reference.md`, `docs/gotchas.md`, `docs/troubleshooting.md`,
  `docs/losses-and-thermal.md`,
  `docs/loss-thermal.md`, `docs/mental-model.md`, `docs/how-pulsim-works/`.
- `CHANGELOG.md`: what changed and why.
