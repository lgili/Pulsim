# Parameter sensitivities

Exact derivatives of a simulated waveform, and of KPIs computed from it,
with respect to circuit parameters. They are available on the `dsed`
exact stepper only. Every other route refuses with a typed
`pulsim.UnsupportedError`. No gradient is ever a finite difference
presented as exact.

```python
import pulsim as p

b = p.CircuitBuilder()
b.add_voltage_source("Vin", "vin", "gnd", 24.0)
b.add_switch("S", "vin", "sw", 1e3, 1e-9)
b.add_diode("D1", "gnd", "sw", 1e3, 1e-9)
b.add_inductor("L", "sw", "vout", 100e-6)
b.add_capacitor("C", "vout", "gnd", 47e-6)
b.add_resistor("RL", "vout", "gnd", 50.0)
pwm = p.make_pwm_switch_fn(100e3, 0.3, b.switch_index_of("S"),
                           b.graph.num_switches)

res = p.simulate(b, 400e-6, engine="dsed", switch_fn=pwm,
                 sensitivities=["L.L", "RL.R", "S.duty"])
res.sens("vout", "L.L")               # d v_out / d L at every sample
v, g = res.kpi("vout", "avg", t_from=300e-6)   # value and {name: grad}

orbit = p.periodic_steady_state(b, period=1e-5, switch_fn=pwm,
                                sensitivities=["L.L", "S.duty"])
orbit.dx0["S.duty"]                   # d x* / d D at the section
orbit.kpi("vout", "ripple")           # orbit KPI and its gradient
```

## What is differentiated

The map is the `dsed` exact stepper with its events located
(`rtol = 1e-11`) and **without** the event projection. This is the map
that `periodic_steady_state` uses. The sensitivity run can therefore
differ from a plain `dsed` run where the projection would act.

Between events, every step is the exact LTI propagator `exp(A h)`. Its
derivative is formed in one of two ways:

- in the eigenbasis (Daleckii-Krein with `phi1` divided differences);
- by the Van Loan block exponential on a near-defective mode.

The pass re-derives every step and compares it with the engine's
sample. A step it cannot reproduce refuses the run.

The derivatives `dA/dp`, `db/dp` and the node-voltage recovery come
from assemblies of the circuit, and they are exact for R, L, C and DC
source values. The duty is differentiable on a fixed-period
trailing-edge PWM: `dtau/dD = T` on each falling edge.

The steady-state gradient is `dx*/dp = (I - M)^-1 dP/dp`. It costs one
more period run, which `n_period_runs` counts.

## Names

A parameter name has the form `"<component>.<field>"`, where the field
is one of:

- `R`, `L`, `C` or `V`;
- the `components()` key, such as `R_ohms`;
- `"<switch>.duty"`.

The following are refused, and the refusal names the device and the
field:

- switch conductances;
- diode thresholds;
- Newton-device parameters;
- integer, topology and table parameters.

## KPIs

`res.kpi(signal, kind, t_from=, t_to=)` takes the following kinds:

| kind | value |
| --- | --- |
| `final` | value at `t_to` |
| `avg` | time-weighted average |
| `rms` | RMS |
| `energy` | the integral |
| `peak` | maximum |
| `valley` | minimum |
| `ripple` | peak - valley |

Integrals are closed forms of the exact trajectory. A signal that jumps
at an event, such as a switch node, adds the term
`(y- - y+) dtau/dp`. Value and gradient are of the same quantity, so a
central difference of the value agrees to rounding.

`non_unique` marks a peak or valley that is attained at two separate
instants. There the function is not differentiable, and the gradient
given is that of the first instant.

RMS, peak and ripple over a near-defective mode are refused in this
release.

## The event rule

A gradient exists only while the event pattern is locally constant.

- **Event time.** At a diode event,
  `dtau/dp = -(grad g . S- + g_p) / (grad g . f-)`, and the
  sensitivity jumps by `(f- - f+) dtau/dp`.
- **Grazing is refused.** An event that is tangent to its switching
  surface (`|grad g . f-| <= 1e-9 |grad g| |f-|`) is refused with
  feature `grazing_event`, naming the diode: one side of the parameter
  removes the event.
- **Near misses are flagged.** An armed predicate can come within 1e-6
  of its range of zero without crossing, for example at the CCM/DCM
  boundary. The gradient is then computed, because it is exact for the
  pattern taken. It is also flagged with a `UserWarning`, and the
  details go in `res.sensitivity.flags`.

## Discrete against continuous gradient

The gradient is that of the exact, continuous-time trajectory: the
limit of the simulation as the step and event tolerances go to zero.
It is not the derivative of a fixed-step engine's discrete output.

At one step per `T/100`, the `pwl` engine's finite differences of
`v_out` are wrong by factors up to 13. The reason is that its edges
snap to the grid. Use the exact path to optimise. Then verify the
optimum on any engine.

## Optimisation

`p.value_and_grad(objective, names)` wraps an objective that returns
`(value, {name: grad})`, or a `KpiValue`, into the
`f(x) -> (value, grad)` form that `scipy.optimize.minimize(jac=True)`
expects. A missing gradient raises an error.

`pulsim.design.Evaluator(..., gradients={"L": "L.L"})` adds exact
gradients of `v_out`, `ripple_v` and `ripple_i.<L>` to `record.grads`
on the `pss` route. The following are refused with feature
`gradients`:

- the transient route;
- the thermal loop;
- a design variable that is not the circuit value itself.

## Cost

One engine run, plus one propagation per parameter per step and four
assemblies per element parameter. Eight parameters cost one run,
against sixteen by central differences.
`res.sensitivity.counters` reports the counts.
