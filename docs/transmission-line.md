# Transmission line (Bergeron)

`b.add_transmission_line(name, a, a_ref, b, b_ref, Z0=..., delay=..., model="bergeron")`
adds a lossless line of surge impedance `Z0` and one-way travel time
`delay` (τ). Each port follows the travelling-wave law

    v − Z0·i = E(t),   E_a(t) = w_b(t − τ),   E_b(t) = w_a(t − τ),

where `w = v + Z0·i` is the wave a port sends and `i` is the current into
the line. The line has no lumped approximation and no section count. A
front leaves one end and arrives at the other exactly τ later. The older
`model="pi"` (a lumped ladder that overshoots a step by 27 %) is still the
default.

`R=` adds a series resistance as `R/4`, two half-lines, `R/2` and `R/4`
(sub-elements `{name}.ra`, `{name}.rm`, `{name}.rb`). The loss sits in
those resistors. `R > Z0` is refused. A lossless line stores energy in
flight and dissipates none.

## Worked example: the lattice diagram

A 1 V step behind 10 Ω drives a 50 Ω, 100 ns line with an open far end.
The incident wave is `V+ = 50/60 V`. The source reflects with
`Γs = (10 − 50)/60 = −2/3` and the open end with `ΓL = +1`. After
(2n+1)·τ the far end therefore steps to `2·V+·Σ_{j≤n} Γs^j`, that is
5/3 V, 5/9 V and 35/27 V.

```python
import pulsim as p
b = p.CircuitBuilder()
b.add_pulse_voltage_source("V", "s", "gnd", 0.0, 1.0, 10e-9, 1.0, 0.0, 1e-12, 1e-12)
b.add_resistor("Rs", "s", "k", 10.0)
b.add_transmission_line("T", "k", "gnd", "m", "gnd", Z0=50.0, delay=100e-9,
                        model="bergeron")
r = p.simulate(b, t_end=700e-9, dt=1e-9, engine="pwl")
```

The far-end plateaus come out as 1.666666666667, 0.555555555556 and
1.296296296296 V, relative error ≤ 2.2e-16. The script is
`examples/scripts/run_transmission_line.py`. It also drives a quarter-wave
transformer: at `f = 1/(4τ)` a 10 Ω load appears as `Z0²/R_L` = 250 Ω.

## Engines

| Engine | Line |
|---|---|
| `pwl` (fixed step) | Yes. `dt > τ` is refused by name (the far end would be in the future). With τ an integer multiple of `dt` the far end reads a stored sample: exact to rounding. Otherwise it is interpolated linearly: second order in `dt`, never amplifying, with fronts smeared over one `dt` and plateaus exact. |
| `trbdf2` | Yes. The step is capped at τ (`_trbdf2_stats["n_line_bound_steps"]` counts the steps the line set). A step lands where a wave the run resolved at one end arrives at the other: at a source corner + τ, and at `t_k + τ` for every recorded sample where a wave changes slope by more than `h_max` could interpolate within `atol + rtol·|w|`. Fronts, corners and reflections arrive on a step boundary, and the effect stops once the waves are smooth to the tolerance. On a matched line the rejects equal those of the same circuit without the line. |
| `trbdf2`, `integrator="esdirk43"` | Refused by name: linear interpolation of the line's past would cut its order to 2 and make its error estimate lie. |
| `exact` / `dsed`, `bdf1` | Refused by name (the capability table). |
| AC sweep | Yes, exact: the line's two-port (ABCD) rows at each frequency, `θ = ωτ`. A quarter wave turns `R_L` into `Z0²/R_L`. |
| DC operating point | A wire (`v_b = v_a`, `i_a = −i_b`). A transient from `start_from_dc_op=True` starts with the DC waves, so it starts without a wave. |
| `periodic_steady_state` | Refused by name: the line's state is its last τ of waves, which is not in the node vector. |

## What the error control does not see

On `trbdf2` the step is chosen by the circuit's reactive states, capped
at τ, plus the landings above. The line's interpolation error is not part
of the error estimate, and a wave is only as fine as the steps that first
recorded it. On a circuit whose only dynamics is the line (resistors and
lines) driven by a smooth source, nothing refines the step below τ,
whatever `rtol` is. A 3.3 MHz sine through a 100 ns line is then off by
6.6 mV of 0.5 V at any `rtol`. Set the step ceiling (`dt=` on `trbdf2`) to resolve
the waves you care about, or use `pwl` with τ a multiple of `dt`.

## Results and snapshots

`res.i("T")` is the current into port a and `res.i("T.b")` the current
into port b. A resume (`resume_from=`) restores the line's recorded past
(snapshot field `line_state`), so a resumed run equals the uninterrupted
one.
