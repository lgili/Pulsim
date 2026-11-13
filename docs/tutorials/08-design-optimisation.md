# Tutorial 08 — Using Pulsim inside an optimiser

An optimiser calls `f(design)` thousands of times. It needs the same
answer for the same input, a finite answer or an explicit failure, and
a cost per call it can budget. `pulsim.design.Evaluator` is that
contract around your own `build(design)` function.

## 1. The contract

`build(design)` returns the circuit and its gate schedule. The
evaluator reaches the periodic steady state, takes every KPI on ONE
steady period with time-weighted integrals (never a plain `.mean()` of
samples), and returns an immutable `EvaluationRecord`.

```python
import pulsim as p
from pulsim.design import Evaluator, SteadySpec, as_objectives, as_constraints

T = 1e-5          # 100 kHz

def build(d):
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", 24.0)
    b.add_switch("S", "vin", "sw", 1e3, 1e-9)
    b.add_diode("D1", "gnd", "sw", 1e3, 1e-9)
    b.add_inductor("L", "sw", "x", d["L"])
    b.add_resistor("RLs", "x", "vout", 0.01)
    b.add_capacitor("C", "vout", "gnd", d["C"])
    b.add_resistor("RL", "vout", "gnd", 1.0)
    pwm = p.make_pwm_switch_fn(1 / T, 0.5, b.switch_index_of("S"),
                               b.graph.num_switches)
    return b, pwm

ev = Evaluator(build, period=T, loads=("RL",), output="vout",
               constraints={"ripple_v": (None, 0.12)})
rec = ev.evaluate({"L": 100e-6, "C": 47e-6})
print(rec.status, rec.counters["steady_method"])
print(f"efficiency {rec.kpis['efficiency']:.6f}")
print(f"ripple     {rec.kpis['ripple_v'] * 1e3:.4f} mV")
print(f"losses     {sorted(rec.losses)}")
```

```text
ok pss
efficiency 0.989117
ripple     15.9588 mV
losses     ['D1', 'RLs', 'S']
```

The efficiency and the ripple agree with a matrix-exponential
reference of the same orbit to 2e-12 and 6e-11 relative, and the total
loss to 8e-15 (`python/tests/test_design_evaluation.py`). The record also carries the
design, a SHA-256 `design_id`, the per-device losses by category, the
route and solver counters, and the provenance (version, extension
hash, options digest).

## 2. Four statuses, never a default number

| status | when | what the record carries |
|---|---|---|
| `ok` | steady period reached, every bound met | KPIs, violations all 0 |
| `infeasible` | a bound exceeded, or a thermal runaway | KPIs and `violations[name]` in the KPI's unit (runaway: no KPIs) |
| `refused` | `pulsim.UnsupportedError` anywhere | `refusal = (engine, feature, device, kind)`, no KPIs |
| `failed` | an engine's non-convergence, `not_settled`, a non-finite KPI | `reason`, counters, no KPIs |

Any other exception propagates: it is a bug, not a design outcome. Do
not wrap `evaluate` in a broad `except`; that is how refusals became
silent wrong answers in a GUI.

```python
small_c = ev.evaluate({"L": 100e-6, "C": 4.8e-6})
print(small_c.status, f"{small_c.violations['ripple_v'] * 1e3:.2f} mV over")

def shockley(d):
    b, pwm = build(d)
    b.add_shockley_diode("D2", "gnd", "sw")
    return b, pwm

refused = Evaluator(shockley, period=T, loads=("RL",),
                    engine="dsed").evaluate({"L": 100e-6, "C": 47e-6})
print(refused.status, refused.refusal)
```

```text
infeasible 30.40 mV over
refused ('dsed', '', 'D2', 'ShockleyDiode')
```

For an optimiser, map records to vectors. `refused` and `failed` get
`penalty` in every component; count them, never read the penalty as a
value:

```python
for r in (rec, small_c, refused):
    print(r.status, as_objectives(r, ["efficiency", "ripple_v"],
                                  maximise=["efficiency"], penalty=1e30),
          as_constraints(r, ["ripple_v"], penalty=1e30))
```

```text
ok [-0.98911739  0.01595878] [0.]
infeasible [-0.98911759  0.15040419] [0.03040419]
refused [1.e+30 1.e+30] [1.e+30]
```

## 3. Steady state, junction temperature, batches

* `SteadySpec(method="pss")` is Newton shooting on the exact stepper
  (`pulsim.periodic_steady_state`); `"transient"` simulates until the
  estimated distance to the orbit is below `tol` (else `failed`,
  `not_settled`); `"auto"` (default) takes `pss` when its applicability
  check accepts the circuit and the transient otherwise, and says so in
  `counters["steady_method"]`. A `pss` failure is `failed`, never a
  silent transient.
* `ThermalSpec(r_th={"S": 2.0}, t_amb=40)` closes `T_j` by a damped
  fixed point (0.1 K); switching tables are read at each device's
  `T_j`; it stops when the estimated distance to the fixed point (not
  the last step) is below `tol`; the circuit is re-simulated only when
  `temperature_dependent=True` (then `build` reads `design["_T_j"]`).
* `ev.evaluate_many(designs, workers=8)` keeps the input order and gives
  the same records on 1 or 8 threads. On the `pss` route the period
  runs and the KPI integrals are one kernel call with the GIL released
  (`counters["kpi_path"] == "kernel"`, about 0.4 ms per buck design),
  but the Newton loop is Python, so threads do not speed it up;
  `backend="process"` helps for large batches of picklable evaluators.

## 4. Why derivative-free

Central finite differences of `dv_C(40 T)/dL` on a DCM buck, relative
error at relative steps 1e-6 … 1e-2 (`add-parameter-sensitivities`):

| engine | relative error |
|---|---|
| `pwl`, dt = T/100 | −9.3, −9.5, −12.9, +2.0, −0.46 |
| `pwl`, dt = T/400 | +9.1, +9.1, +2.6, −0.040, +0.041 |
| `trbdf2`, default tolerances | −1.1e-4 … +2.9e-3 |

The fixed engine puts each diode commutation on its grid, so an
event-dependent KPI is a staircase in the parameters and its gradient
is noise. The evaluator offers no finite differences; use a
derivative-free optimiser (NSGA-II, CMA-ES, a Latin-hypercube sample).

## 5. The worked example and its parts

`examples/optimization/buck_pareto.py` sweeps a synchronous 48 V →
12 V buck over `f_sw`, `L`, `C` and a choice of two MOSFETs, and
prints the efficiency-against-volume front (pymoo's NSGA-II when it is
installed, `pip install pymoo`; otherwise a Latin-hypercube sample and
a numpy non-dominated filter). **Its two MOSFETs are synthetic
illustrations and its volume law is illustrative**: the front is an
observation of that model, not a design result. Pulsim's shipped parts
are synthetic too (`pulsim.lib.part` refuses them unless
`allow_synthetic=True`); for real numbers import the vendor's PLECS
thermal file with `pulsim.lib.import_plecs_xml`.
