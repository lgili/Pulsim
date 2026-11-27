# Ideal switches, branch form, and complementarity diode selection

Pulsim models switched devices in two forms:

- **Conductance form** (`form="conductance"`, the current default): a switch or
  diode stamps a finite conductance `g_on` when closed and `g_off` when open.
  `g_on` must be finite and positive (`g_on = inf` is refused), and opening
  `g_off = 0` across an inductor with no freewheel path causes large numerical
  spikes.
- **Branch form** (`add_ideal_switch`, or `form="branch"` on `add_switch` /
  `add_diode`): allocates a dedicated branch-current unknown $i_s$ in the MNA
  system (`I(<name>)` in `b.state_var_names()`, read directly via `res.i(name)`).
  Because voltage and current sit in separate equations, `R_on = 0` and
  `g_off = 0` are well-conditioned and exact.

Pair branch-form switches and diodes with **complementarity mode selection**
(`mode_selection="lcp"` or `"auto"`), which resolves multi-diode commutations
through a linear complementarity problem (LCP) rather than iterative settle
loops—so artificial anti-chatter `V_th > V_f` bands are no longer required.

---

## 1. Branch-form API and YAML

Use `add_ideal_switch` or pass `form="branch"` to `add_switch` / `add_diode`:

| Builder call | Defaults (`form="branch"`) | Notes |
|---|---|---|
| `b.add_ideal_switch(name, n_pos, n_neg, R_on=0.0, g_off=0.0)` | `R_on=0.0`, `g_off=0.0` | Branch current unknown $i_s$; `R_on = 0` is an ideal 0 V short when closed |
| `b.add_switch(name, n_pos, n_neg, g_on=None, g_off=None, *, form="conductance", R_on=None)` | `R_on=0.0`, `g_off=0.0` | With `form="branch"`, omit `g_on`/`R_on` for ideal `R_on = 0`, or set `R_on` (or `g_on`, stored as `R_on = 1/g_on`) |
| `b.add_diode(name, anode, cathode, g_on=None, g_off=None, V_th=0.0, V_f=0.0, *, form="conductance", R_on=None)` | `R_on=0.0`, `g_off=0.0`, `V_th=0.0`, `V_f=0.0` | While ON, $v_{ak} = V_f + R_{\text{on}} i_d$; `V_th` is the turn-on threshold |

In YAML netlists, add `form: branch` directly on `type: switch` or
`type: diode`:

```yaml
circuit:
  devices:
    - {type: voltage_source, name: V1, from: a, to: gnd, V: 10.0}
    - {type: switch, name: S1, from: a, to: sw, form: branch}
    - {type: diode, name: D1, anode: gnd, cathode: sw, form: branch, R_on: 0.0, g_off: 1e-9, V_f: 0.7}
    - {type: inductor, name: L1, from: sw, to: out, L: 100.0e-6}
    - {type: capacitor, name: C1, from: out, to: gnd, C: 47.0e-6}
    - {type: resistor, name: R1, from: out, to: gnd, R: 10.0}
```

The branch form runs on `pwl`, `trbdf2`, `dsed` (alias `exact`), and the
kernel AC sweep (`run_mna_sweep`); the low-level retired binding
`pulsim._pulsim.run_transient_bdf1(...)` refuses it with
`UnsupportedError(engine="bdf1", feature="branch_form")` (while
`simulate(..., engine="bdf1")` rejects `"bdf1"` with `ValueError`).

An ideal RL circuit reaches the exact closed-form current
$i(t) = \frac{V}{R}(1 - e^{-t/\tau})$, and `res.i("S")` reads the solved
switch current directly:

```python
import math
import pulsim as p

b = p.CircuitBuilder()
b.add_voltage_source("V", "vin", "gnd", 10.0)
b.add_ideal_switch("S", "vin", "a", R_on=0.0, g_off=1e-9)
b.add_resistor("R", "a", "b", 1.0)
b.add_inductor("L", "b", "gnd", 1e-3)

m_on = p.SwitchStateMask(b.graph.num_switches)
m_on.set(b.switch_index_of("S"), True)

res = p.simulate(b, t_end=10e-3, switch_fn=lambda t: m_on,
                 engine="trbdf2", rtol=1e-9, atol=1e-12)
exact = 10.0 * (1.0 - math.exp(-10.0))
print(f"I(S) in state_var_names: {'I(S)' in b.state_var_names()}")
print(f"i_L(10 ms) = {res.i('L')[-1]:.6f} A, i_S = {res.i('S')[-1]:.6f} A, exact = {exact:.6f} A")
```

Expected output:

```text
I(S) in state_var_names: True
i_L(10 ms) = 9.999546 A, i_S = 9.999546 A, exact = 9.999546 A
```

---

## 2. Structural refusals (`UnsupportedError`)

On the first visit to each switch mask, the kernel runs a per-mode structural
pass whenever any branch-form element has `R_on = 0` or `g_off = 0`. When a
mode is structurally singular or unsupported, `simulate()` raises
`pulsim.UnsupportedError` with `.engine`, `.feature`, `.device`, and `.kind`:

| `feature` | Meaning | Typical cause | Fix |
|---|---|---|---|
| `floating_node` | An island of nodes has no path to ground because every boundary element is open with `g_off = 0`. | Two open ideal switches in series, or an ungrounded island bounded solely by OFF `g_off = 0` diodes. | Give at least one boundary element `g_off > 0` (e.g. `1e-9`), or tie the island to ground with a high-value resistor (`1e9` Ω). |
| `ideal_voltage_loop` | Closed `R_on = 0` elements form a zero-impedance loop with voltage sources (or inductors at DC). | Shoot-through across a voltage rail, parallel closed ideal switches, or an ideal diode forward-biased across a source. | Fix the gate pattern, or add finite `R_on > 0` (or series impedance). |
| `dependent_states` | A mode closes a capacitor loop (`R_on = 0`) or opens an inductor cut-set (`g_off = 0`), constraining state variables. | `engine="pwl"` run without `discontinuity_restart=True`, or a circuit combining `R_on=0`/`g_off=0` with a nonlinear stateful device (e.g. `SaturableInductor`). | Pass `discontinuity_restart=True` on `pwl` (automatic on `trbdf2`/`dsed`), or use `R_on > 0` and `g_off > 0` with nonlinear stateful models. |

Note on loops with ideal diodes: if an ON ideal diode closes a loop with a
voltage source and closed ideal switches that *reverse-biases* the diode, the
solver releases that diode OFF rather than refusing; only a diode
forward-biased into a zero-impedance loop raises `ideal_voltage_loop`.

```python
import pulsim as p

# 1) floating_node: two open g_off=0 switches leave "mid" isolated
b_float = p.CircuitBuilder()
b_float.add_voltage_source("V", "vin", "gnd", 10.0)
b_float.add_ideal_switch("S1", "vin", "mid", R_on=0.0, g_off=0.0)
b_float.add_ideal_switch("S2", "mid", "gnd", R_on=0.0, g_off=0.0)
b_float.add_resistor("R", "vin", "gnd", 10.0)
try:
    p.simulate(b_float, t_end=1e-5, dt=1e-6, engine="pwl",
               switch_fn=lambda t: p.SwitchStateMask(2))
except p.UnsupportedError as exc:
    print("1:", exc.engine, exc.feature, exc.device, exc.kind)

# 2) ideal_voltage_loop: two closed R_on=0 switches in parallel across a source
b_loop = p.CircuitBuilder()
b_loop.add_voltage_source("V", "vin", "gnd", 10.0)
b_loop.add_ideal_switch("S1", "vin", "gnd")
b_loop.add_ideal_switch("S2", "vin", "gnd")
m_both = p.SwitchStateMask(2)
m_both.set(0, True)
m_both.set(1, True)
try:
    p.simulate(b_loop, t_end=1e-5, dt=1e-6, engine="pwl",
               switch_fn=lambda t: m_both)
except p.UnsupportedError as exc:
    print("2:", exc.engine, exc.feature, exc.device, exc.kind)

# 3) dependent_states: opening an inductor cut-set on pwl without restart
b_cut = p.CircuitBuilder()
b_cut.add_voltage_source("V", "vin", "gnd", 10.0)
b_cut.add_ideal_switch("S", "vin", "a", R_on=0.0, g_off=0.0)
b_cut.add_resistor("R", "a", "b", 1.0)
b_cut.add_inductor("L", "b", "gnd", 1e-3)
try:
    p.simulate(b_cut, t_end=1e-4, dt=5e-5, engine="pwl",
               switch_fn=lambda t: p.SwitchStateMask(1))
except p.UnsupportedError as exc:
    print("3:", exc.engine, exc.feature, exc.device, exc.kind)
```

Expected output:

```text
1: pwl floating_node S1, S2 Switch
2: pwl ideal_voltage_loop S1 Switch
3: pwl dependent_states S Switch
```

---

## 3. Dependent states, `reinit_records`, `on_impulse`, and `impulse_losses`

When an ideal switch closes a capacitor loop with unequal voltages or opens an
inductor cut-set carrying non-zero current, the state must jump instantaneously
onto the constraint manifold:

- **Inductor cut-set** (`kind == "cut_set"`): $i_L$ jumps to $0$ (conserving
  total flux across any remaining loop, dissipating $\Delta W = \frac{1}{2} L i_L(t^-)^2$
  with impulse $\Lambda = L \, i_L(t^-)$ [$\text{V}\cdot\text{s}$]).
- **Capacitor loop** (`kind == "capacitor_loop"`): $v_{C1}$ and $v_{C2}$
  equalize while conserving total charge $Q = C_1 v_1 + C_2 v_2$, transferring
  charge $\Lambda$ [$\text{C}$] and removing
  $\Delta W = \frac{1}{2} \frac{C_1 C_2}{C_1 + C_2} (v_1 - v_2)^2$ [$\text{J}$].

On `pwl` (with `discontinuity_restart=True`) and `trbdf2`, three SDIRK(2,2)
steps execute the projection cleanly without trapezoidal ringing, and each
non-zero jump is appended to `res.reinit_records` (`ReinitRecord`):

- `.t`: event time [s]
- `.kind`: `"cut_set"` or `"capacitor_loop"`
- `.devices`: switching device names that formed the structure
- `.states`: constrained `L` or `C` branch names
- `.cause`: switch name the energy is booked to
- `.lambda_`: transferred flux [$\text{V}\cdot\text{s}$] or charge [$\text{C}$]
- `.dW`: energy removed by the impulse [J]
- `.projection_gap`: discrepancy between the explicit projection and the restart end state
- `.sample`: index of the sample carrying the Dirac pulse (`-1` if not stored)

Control how impulses are handled with `simulate(..., on_impulse=...)`:

- `"record"` (default): project, continue, and record in `res.reinit_records`.
- `"warn"`: record and emit a `UserWarning` naming the first impulse.
- `"error"`: raise `UnsupportedError(feature="impulse")` naming the switch and constrained states.

A **consistent** commutation (for example, an ideal diode turning off at
$i_L = 0$ in DCM, or closing a switch across equal-voltage capacitors) produces
no impulse and no record (`len(res.reinit_records) == 0`).

Because $v \cdot i$ is zero both before and after an ideal impulsive event,
normal conduction power `P_avg` does not include $\Delta W$. Instead,
`pulsim.losses.impulse_losses(b, res)` and
`pulsim.device_loss_summary(b, res)` report impulsive energy separately under
`E_impulse_total`, `n_impulses`, and `P_impulse_avg`:

```python
import numpy as np
import pulsim as p
from pulsim.losses import impulse_losses

# 1) Inductor cut-set: 10 V / 1 ohm / 1 mH opened at 5 ms
b_cs = p.CircuitBuilder()
b_cs.add_voltage_source("V", "vin", "gnd", 10.0)
b_cs.add_ideal_switch("S", "vin", "a", R_on=0.0, g_off=0.0)
b_cs.add_resistor("R", "a", "b", 1.0)
b_cs.add_inductor("L", "b", "gnd", 1e-3)
k = b_cs.switch_index_of("S")

def open_at_5ms(t):
    m = p.SwitchStateMask(1)
    m.set(k, t < 5e-3)
    return m

r_cs = p.simulate(b_cs, t_end=5.3e-3, dt=5e-5, engine="pwl",
                  switch_fn=open_at_5ms, discontinuity_restart=True)
rec = r_cs.reinit_records[0]
print(f"cut_set: t={rec.t*1e3:.2f} ms, |lambda|={abs(rec.lambda_)*1e3:.3f} mV*s, "
      f"dW={rec.dW*1e3:.3f} mJ, i_L_after={r_cs.i('L')[-1]:.1f} A")

# 2) Capacitor loop: 1 uF at 10 V closed onto 3 uF at 2 V -> 4 V, 6 uC, 24 uJ
b_cl = p.CircuitBuilder()
b_cl.add_capacitor("C1", "x", "gnd", 1e-6, c0=10.0)
b_cl.add_capacitor("C2", "y", "gnd", 3e-6, c0=2.0)
b_cl.add_ideal_switch("S", "x", "y", R_on=0.0, g_off=1e-9)
m_cl = p.SwitchStateMask(1)
m_cl.set(0, True)

r_cl = p.simulate(b_cl, t_end=5e-7, dt=1e-7, engine="pwl",
                  switch_fn=lambda t: m_cl, auto_regularize=False,
                  discontinuity_restart=True)
imp = impulse_losses(b_cl, r_cl)["S"]
summary = {row["name"]: row for row in p.device_loss_summary(b_cl, r_cl)}
print(f"cap_loop: v_C1={r_cl.v('x')[-1]:.6f} V, v_C2={r_cl.v('y')[-1]:.6f} V, "
      f"E_imp={imp['E_impulse_total']*1e6:.1f} uJ, "
      f"summary_n={summary['S']['n_impulses']}")

# 3) on_impulse="error" refuses impulsive projections
try:
    p.simulate(b_cs, t_end=5.3e-3, dt=5e-5, engine="pwl",
               switch_fn=open_at_5ms, discontinuity_restart=True,
               on_impulse="error")
except p.UnsupportedError as exc:
    print("on_impulse='error':", exc.feature, exc.device)
```

Expected output:

```text
cut_set: t=5.00 ms, |lambda|=9.933 mV*s, dW=49.329 mJ, i_L_after=0.0 A
cap_loop: v_C1=4.000000 V, v_C2=4.000000 V, E_imp=24.0 uJ, summary_n=1
on_impulse='error': impulse S
```

---

## 4. Complementarity diode law, `V_th` vs `V_f`, and `mode_selection`

Every PWL diode obeys the complementarity law between its ideal conduction
current $i_d$ and blocking voltage gap $w$:

$$
0 \le i_d \;\perp\; w = V_f + R_{\text{on}} \, i_d - v_{ak} \ge 0,
\qquad
i_{\text{term}} = i_d + g_{\text{off}} \, v_{ak}.
$$

- **`V_f`** ($\ge 0$): the forward voltage drop while conducting ($v_{ak} = V_f + R_{\text{on}} i_d$).
- **`V_th`**: the turn-on threshold (effective turn-on is $\max(V_{\text{th}}, V_f)$).
  Setting $V_{\text{th}} > V_f$ creates a hysteresis band (turns ON at $v_{ak} \ge V_{\text{th}}$, turns OFF at $i_d \le 0$).

In earlier versions, users added an artificial anti-chatter band
(`V_th = 1e-2`, `V_f = 0`) to prevent multi-diode iterative settle loops from
oscillating near zero crossings. With `mode_selection="lcp"` or
`mode_selection="auto"`, the selector solves one $k$-port LCP (Lemke's method
over the companion network, with circuit-scale tie widening and fallback
enumeration on non-passive circuits) so all diodes commutate consistently in
one step:

- **`mode_selection=None` / `"legacy"`** (default): traditional settle with the
  default-mode guard where active (`dsed` populates `res.diagnostics["mode_guard"]`
  on diode circuits; `pwl` populates `res.diagnostics["mode_guard"]` when its
  guard is active on conductance-form diode circuits and `{}` otherwise;
  `trbdf2` leaves `res.diagnostics` unset).
- **`mode_selection="lcp"`**: the LCP decides diode modes (`res.diagnostics["mode_selection"]`
  always reports `lcp_count`, `lcp_pivots_total`, `max_pivots_per_event`, `ties`,
  `widenings`, `enumerations`, `settle_rounds_total`, and `passivity`; `pwl` and
  `trbdf2` also report `mask_cycles`, `shorted_held`, `shorted_released`, and
  `sites`, while `dsed` reports `diodes` and `h_e`).
- **`mode_selection="auto"`**: enables `"lcp"` wherever supported (`pwl` and
  `dsed` on circuits with PWL diodes and no **gated diodes** such as
  thyristors/TRIACs/GTOs or reverse-blocking IGBTs, no Newton devices, and no
  `sensitivities=`—ordinary controlled switches from `add_switch` /
  `add_ideal_switch` are supported) and falls back to `"legacy"` otherwise,
  recording the choice and reason in `res.diagnostics["mode_selection_auto"]`.
  (On `trbdf2`, explicit `mode_selection="lcp"` runs; `mode_selection="auto"`
  currently logs fallback to `"legacy"`.)
- **`mode_selection="shadow"`**: runs the legacy settle untouched and evaluates
  the LCP alongside it (`res.diagnostics["mode_shadow"]`).

**Anti-chatter `V_th` bands are no longer needed** under `mode_selection="lcp"`
or `"auto"`: leave `V_th=0.0` (unless modelling an actual turn-on hysteresis),
and pass `V_f` for the real forward drop:

```python
import numpy as np
import pulsim as p

# Ideal DCM buck (V_th=0, R_on=0, g_off=0): zero settle rounds, zero impulses
b = p.CircuitBuilder()
b.add_voltage_source("Vin", "vin", "gnd", 24.0)
b.add_ideal_switch("S", "vin", "sw", R_on=0.0, g_off=0.0)
b.add_diode("D1", "gnd", "sw", form="branch", R_on=0.0, g_off=0.0, V_th=0.0, V_f=0.0)
b.add_inductor("L", "sw", "vout", 100e-6, i0=0.0)
b.add_capacitor("C", "vout", "gnd", 47e-6, c0=8.996397)
b.add_resistor("R", "vout", "gnd", 50.0)

T = 1e-5
pwm = p.make_pwm_switch_fn(1.0 / T, 0.3, b.switch_index_of("S"), b.graph.num_switches)
res = p.simulate(
    b, t_end=20 * T, dt=T / 100, engine="pwl", switch_fn=pwm,
    discontinuity_restart=True, enable_substep_state_correction=True,
    mode_selection="auto",
)
ms = res.diagnostics["mode_selection"]
auto_info = res.diagnostics["mode_selection_auto"]
idle = ~np.asarray(res.switch_states).any(axis=1)
print(f"auto -> {auto_info['chosen']}, settle_rounds={ms['settle_rounds_total']}, "
      f"reinit_records={len(res.reinit_records)}")
print(f"vout={res.v('vout')[-1]:.4f} V, idle i_L_max={np.max(np.abs(np.asarray(res.i('L'))[idle])):.1f} A")
```

Expected output:

```text
auto -> lcp, settle_rounds=0, reinit_records=0
vout=8.9964 V, idle i_L_max=0.0 A
```

`vout` is a measured value, not a closed form: the sample at `t = 20 T` of
the periodic orbit (the run starts on it, `c0=8.996397`). The small-ripple
DCM average `M = 2 / (1 + sqrt(1 + 4K/D^2)) = 0.375` (`K = 2L/(RT) = 0.4`)
gives 9.0 V, the level it ripples around. `pytest
python/tests/test_doc_snippets.py -k ideal-switch` re-runs this block and
fails when a printed digit moves. `idle i_L_max=0.0 A` is exact: the inductor
current is zero while both devices are off in DCM.

---

## 5. Gauge convention (floating islands bounded by OFF diodes)

In an ungrounded diode bridge (such as a single-phase or three-phase rectifier
feeding an isolated DC bus `p` / `n`), when every diode blocks simultaneously
the entire DC bus becomes a **floating island** whose common-mode potential is
undetermined (a **gauge**):

- **Always read differential voltages** (`res.v("p") - res.v("n")`) or ground
  one rail (`dc_neg="gnd"`). The differential voltage across the bus and the
  diode complementarity variables are gauge-invariant; the individual
  node-to-ground potentials `res.v("p")` and `res.v("n")` depend on tiny
  boundary conductances (`g_off` or a high-value tie resistor such as `1e9` Ω).
- **Current status of `g_off = 0` diode islands:** on `pwl` and `trbdf2`, an
  island bounded *exclusively* by open diodes with `g_off = 0` (and no path to
  ground) is currently refused by name with
  `UnsupportedError(feature="floating_node", kind="Diode")`. Give those diodes
  a small `g_off > 0` (e.g. `1e-9` S) or keep a `1e9` Ω reference tie from one
  bus node to ground (`auto_regularize=True` inserts one automatically at
  build time when the DC bus is statically isolated).

```python
import numpy as np
import pulsim as p

# Bridge with g_off=0 and no tie: refused with floating_node / Diode
b_iso = p.CircuitBuilder()
b_iso.add_sine_voltage_source("Vac", "ac", "gnd", 0.0, 100.0, 50.0)
for nm, a, c in (("D1", "ac", "p"), ("D2", "gnd", "p"),
                 ("D3", "n", "ac"), ("D4", "n", "gnd")):
    b_iso.add_diode(nm, a, c, form="branch", R_on=0.0, g_off=0.0)
b_iso.add_capacitor("C", "p", "n", 100e-6)
b_iso.add_resistor("R", "p", "n", 100.0)

try:
    p.simulate(b_iso, t_end=1e-3, dt=1e-5, engine="pwl",
               auto_regularize=False)
except p.UnsupportedError as exc:
    print("isolated g_off=0 bridge:", exc.feature, exc.kind)

# With g_off=1e-9 (and/or a 1 GOhm tie), differential bus p - n is gauge-invariant
b_ref = p.CircuitBuilder()
b_ref.add_sine_voltage_source("Vac", "ac", "gnd", 0.0, 100.0, 50.0)
for nm, a, c in (("D1", "ac", "p"), ("D2", "gnd", "p"),
                 ("D3", "n", "ac"), ("D4", "n", "gnd")):
    b_ref.add_diode(nm, a, c, form="branch", R_on=0.0, g_off=1e-9)
b_ref.add_capacitor("C", "p", "n", 100e-6)
b_ref.add_resistor("R", "p", "n", 100.0)
r_ref = p.simulate(b_ref, t_end=40e-3, dt=2e-5, engine="pwl",
                   discontinuity_restart=True, mode_selection="lcp")
v_bus = np.asarray(r_ref.v("p")) - np.asarray(r_ref.v("n"))
# Closed form: the bus follows the 100 V sine to the peak and past it until
# the capacitor current reverses (tan(theta) = -w*R*C), then decays with R*C
# to the zero crossing at 40 ms.
w, RC = 2 * np.pi * 50.0, 100.0 * 100e-6
theta = np.pi - np.arctan(w * RC)
v_cf = 100.0 * np.sin(theta) * np.exp(-(np.pi - theta) / (w * RC))
print(f"V(p - n) at 40 ms = {v_bus[-1]:.2f} V, closed form {v_cf:.2f} V")
```

Expected output:

```text
isolated g_off=0 bridge: floating_node Diode
V(p - n) at 40 ms = 63.75 V, closed form 63.75 V
```

---

## 6. Porting from PSIM and PLECS

### Porting from PSIM

- **Switches and diodes:** PSIM's ideal switch and diode default to zero
  on-resistance and a forward drop `V_f` (with zero hysteresis band). Map them
  to `b.add_ideal_switch(name, n_pos, n_neg)` (or `add_switch(..., form="branch")`)
  and `b.add_diode(name, anode, cathode, form="branch", R_on=0.0, V_f=...)`.
- **Forward drop vs threshold:** in Pulsim, pass the PSIM diode voltage drop as
  `V_f`, leaving `V_th=0.0` (remember: `V_th` is only a turn-on threshold, not a
  forward drop).
- **Fixed-step discontinuities:** PSIM runs with a fixed `dt` and silently
  absorbs switching impulses across capacitors or inductors. In Pulsim with
  `dt` passed (`engine="pwl"`), pass `discontinuity_restart=True` whenever
  ideal elements can form capacitor loops or inductor cut-sets—otherwise `pwl`
  raises `UnsupportedError(feature="dependent_states")`. Check
  `res.reinit_records` (or `on_impulse="warn"`) to audit any impulsive loss.

### Porting from PLECS

- **Ideal piecewise-linear elements:** PLECS models switches and diodes with
  $R_{\text{on}} = 0$, $g_{\text{off}} = 0$, and exact state-space projection at
  mode changes. Map them to `form="branch"` (`R_on=0.0, g_off=0.0`) with
  `V_f` for the diode forward drop.
- **Variable-step vs fixed-step:** omit `dt` to use `engine="trbdf2"` (or pass
  `engine="dsed"` for exact event propagation); both run SDIRK/exact
  projections automatically on dependent modes. With fixed `dt`, pass
  `discontinuity_restart=True` and `enable_substep_state_correction=True`.
- **Multi-diode commutation:** pass `mode_selection="lcp"` (or `"auto"`) so
  coupled diodes resolve via the LCP without artificial `V_th` bands.
- **Strict structural checks and loss accounting:** unlike PLECS, Pulsim
  refuses modes that leave a node bounded solely by `g_off = 0` switches/diodes
  (`floating_node`) or short voltage sources (`ideal_voltage_loop`), and books
  any impulsive projection energy separately in `res.reinit_records` and
  `impulse_losses(b, res)` (`E_impulse_total` in `device_loss_summary`).
