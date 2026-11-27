# Error reference

Pulsim refuses rather than returning a wrong answer. Every error it raises
on purpose is a `pulsim.PulsimError`, and each class keeps the standard
base its raise sites used before, so code written against `ValueError`,
`RuntimeError` or `KeyError` keeps working. All of them live in
`pulsim.errors` and are also `pulsim.<Class>`.

| Class | Bases | Means | Attributes |
|---|---|---|---|
| `PulsimError` | `Exception` | any of the below | |
| `OptionsError` | `PulsimError`, `ValueError` | a keyword value no engine can take | `option` |
| `UnsupportedError` | `PulsimError`, `ValueError` | an engine refuses a device kind or a feature | `engine`, `feature`, `device`, `kind` |
| `NetlistError` | `PulsimError`, `ValueError` | the circuit itself is wrong (a pin on the wrong domain, a magnetic network with no reference) | `device`, `node` |
| `NameNotFoundError` | `PulsimError`, `KeyError` | `res.v` / `res.i` / a lookup on a name that is not registered | `name`, `kind`, `suggestions` |
| `ConvergenceError` | `PulsimError`, `RuntimeError` | the engine gave up | `kind`, `t_failed`, `device` on the exact engine |
| `SimulationAborted` | `ConvergenceError` | a transient died part-way | `partial` (the run up to the failure), `t_failed` |
| `SingularError` | `SimulationAborted` | the step failed on a singular matrix; the message names the rows (usually a floating node) | as `SimulationAborted` |
| `NewtonRoundoffBudgetExceeded` | `SimulationAborted` | the Newton roundoff-acceptance budget ran out | as `SimulationAborted` |
| `Cancelled` | `PulsimError`, `RuntimeError` | `should_continue` returned False | `where`, `point_index` |
| `InternalError` | `PulsimError`, `RuntimeError` | an engine broke its own invariant: a pulsim bug, please report it | |

`NameNotFoundError` is deliberately not a `ValueError`: a name lookup
caught by an `except ValueError` fallback would be swallowed.

## Catching

```python
import pulsim as p

try:
    res = p.simulate(b, t_end=1e-3, dt=1e-7, switch_fn=sw)
except p.UnsupportedError as e:      # a refusal: change the input
    print(e.engine, e.feature, e.device)
except p.SimulationAborted as e:     # the run died: inspect what it did
    res = e.partial
```

Catch by class, never with `except Exception`: a broad handler turns a
refusal into a silent wrong answer. A GUI that reports refusals can catch
`p.PulsimError` and show `str(e)`; everything else is a bug to surface.

A `SingularError` is raised when a fixed-step transient step meets a
singular matrix, also after its dt-halving retries (the variable-step
engine still reports a plain `SimulationAborted` with the same kind of
message). The same failure outside a transient (a DC operating point on its
own) stays a plain `RuntimeError` with the same message, so a
`ConvergenceError` handler does not start catching topology defects.

## Run diagnostics

The warnings a `simulate()` call emits are also kept on its result, in
order, as `result.diagnostics["messages"]`: a tuple of records with
`code` (a stable name such as `"preflight_ties"`, `"dt_retries"`,
`"implausible_voltage"`), `message` (the warning's text) and `category`
(its warning class). The warnings themselves are unchanged: same
category, same text, same line attributed to your script. A tool that
shows them (a GUI's message panel) reads the tuple instead of capturing
warnings.

```python
for d in res.diagnostics["messages"]:
    print(d.code, d.category.__name__, d.message.splitlines()[0])
```
