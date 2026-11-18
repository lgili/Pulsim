# Parameters and value updates

A `pulsim.Circuit` (`builder.build()`) is immutable. To change a value
you derive a new circuit; the original is never touched, so a circuit
can be shared between threads, cached, or compared by hash.

## One field: `with_value`

```python
import pulsim as p

b = p.CircuitBuilder()
b.add_voltage_source("Vs", "a", "gnd", 12.0)
b.add_inductor("L1", "a", "o", 1e-3)
b.add_capacitor("C1", "o", "gnd", 10e-6)
b.add_resistor("RL", "o", "gnd", 5.0)
c = b.build()

c2 = c.with_value("RL.R", 8.0)       # a new circuit; c is unchanged
c2.get_value("RL.R")                  # 8.0
```

Every value field of every device kind works. `c.devices()` lists each
device's fields under `"fields"`, by the names `with_value` takes; the
argument names of `add_*` (`R_ohms`, `L_henries`, `C_farads`, ...) are
accepted as aliases. The result equals rebuilding the circuit with the
new literal, bit for bit: same `value_hash()`, same waveforms.

A value out of the field's bound, an unknown device or an unknown
field raises `ValueError` naming the device and the field:

```text
with_value("RL.R"): field R of device "RL" must be a finite number > 0 (got -1)
with_value("RL.Q"): device "RL" (Resistor) has no value field "Q"; its fields: R, G
```

Beyond the field's own bound, the device is checked as its `add_*`
checks it: a nonlinear capacitor's grading `m` stays below 1, an IGBT's
`tau_tail` and `k_tail` are set together, and a field the device's law
does not read (a junction field of a table-law capacitor, `L_0` of a
table saturable inductor) is refused instead of silently ignored.

`builder.set_value("RL.R", 8.0)` does the same on a builder, in place.

## Named parameters: `param` and `with_params`

Declare a parameter on the builder and pass it to any `add_*` in place
of a number. The field takes the parameter's value, and
`with_params` re-values every field it drives:

```python
b = p.CircuitBuilder()
L = b.param("Lbase", 1e-3, "H", lo=1e-9)   # lo / hi: optional bounds
R = b.param("Rload", 5.0, "Ohm")
b.add_voltage_source("Vs", "a", "gnd", 12.0)
b.add_inductor("L1", "a", "o", L)
b.add_capacitor("C1", "o", "gnd", 10e-6)
b.add_resistor("RL", "o", "gnd", R)
c = b.build()

c3 = c.with_params(Lbase=2.2e-3, Rload=8.0)   # or with_params({"Lbase": 2.2e-3})
c3.structure_hash() == c.structure_hash()     # True: same topology
```

`with_params` checks the parameter's own bounds and the bound of every
field it drives before changing anything; an error names the
parameter, the device and the field. Which fields a parameter drives
is part of the structure (`structure_hash`); its value is not, so
every `with_params` derivation shares one symbolic analysis.
`with_value` on a driven field turns it back into a literal.

A parameter binds to a device's value field. An argument that is not
a value field (an initial condition such as `i0`, a composite's
argument such as `add_transformer`'s `L_p`) is refused, naming the
argument. `add_nonlinear_capacitor` (an overloaded method) does not
take a `Parameter` (`TypeError`); bind its field after adding it:
`b.bind_param("C1.C0", "Cj")`.

## Expressions

A `str` value makes a derived parameter: an expression over the
parameters declared before it, re-evaluated whenever one of them is set.
A derived parameter cannot be set itself.

```python
Lb = b.param("Lbase", 1e-3, "H")
L2 = b.param("L2", "2*Lbase", "H")       # 2e-3, bit for bit
c.with_params(Lbase=2e-3)                # L2 -> 4e-3
```

Grammar: `+ - * /`, `^` (right-associative, binds tighter than unary
minus: `-2^2` is -4), parentheses, C-locale numbers (`1e-3`, no unit
suffixes), `pi`, and the functions `abs sqrt exp log log10 sin cos tan
atan sinh cosh tanh` (one argument) and `min max pow atan2` (two). The
arithmetic runs in the order written, so the result equals the same
arithmetic typed out. Errors name the parameter (or device and field),
the text and the column; a non-finite result is refused.

In a YAML netlist, `circuit.parameters: [{name, value, unit}]` declares
parameters (a string `value` is an expression; any order, a cycle is
refused naming it), and a string in a value field is an expression:
`{type: inductor, name: L1, from: a, to: b, L: "2*Lbase"}` drives
`L1.L` through a derived parameter `L1:L`. Enumerated keys (`law`,
`model`, `form`) are never read as expressions. `spice_to_builder`
keeps top-level `.param` lines as parameters, evaluates `{expression}`
values, and drives a top-level R / L / C whose value names a parameter.
`Circuit.to_yaml()` writes the expression text, not its value.

## Sweeping a circuit

`pulsim.sweep(circuit, params={"Lbase": [...], "RL.R": [...]}, ...)`
(and `monte_carlo(circuit, ...)`) derives every point with
`with_params` / `with_value`. On the pwl engine, with no Python hook in
the step loop, the points run in the kernel: native threads
(`parallel=True`, `n_workers`), no GIL, one symbolic analysis per sweep
and one topology preflight per structure. Point 0 runs through
`simulate()` and every other point is finished by it, so warnings and
`result.plan` / `result.circuit` are those of a direct call, and the
results are bit-identical to `simulate(circuit.with_value(...))`.
`result.backend == "kernel"`; `result.counters` reports
`kernel_points`, `symbolic_analyses` and `preflights`. A Python
`step_observer`, `b_extra_fn` or closed loop, another engine, or
`backend="process"` keeps the per-point `simulate()` path.

## Deprecated: `update_*`

`update_resistor_R`, `update_inductor_L`, `update_capacitor_C` and
`update_voltage_source_V` still work and warn (`DeprecationWarning`).
Replace

| old | new (immutable) | new (in place) |
|---|---|---|
| `b.update_resistor_R("RL", 8.0)` | `c.with_value("RL.R", 8.0)` | `b.set_value("RL.R", 8.0)` |
| `b.update_inductor_L("L1", 2e-3)` | `c.with_value("L1.L", 2e-3)` | `b.set_value("L1.L", 2e-3)` |
| `b.update_capacitor_C("C1", 2e-6)` | `c.with_value("C1.C", 2e-6)` | `b.set_value("C1.C", 2e-6)` |
| `b.update_voltage_source_V("Vs", 7.0)` | `c.with_value("Vs.V", 7.0)` | `b.set_value("Vs.V", 7.0)` |

`set_value` checks the field's bound, which `update_*` did not.
