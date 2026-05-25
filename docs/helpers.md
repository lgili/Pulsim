# UX helpers — discover, plot, monitor

These are the helpers that turn a 20-line matplotlib + state-vector
indexing dance into a one-line call. They live in
[`python/pulsim/v2_discovery.py`](https://github.com/lgili/Pulsim/blob/main/python/pulsim/v2_discovery.py)
and [`python/pulsim/v2_plot.py`](https://github.com/lgili/Pulsim/blob/main/python/pulsim/v2_plot.py)
and are re-exported from `pulsim`, so the import surface stays flat:

```python
import pulsim as p
```

## 1. Discovery — never leave the Python REPL

### `p.catalog([category])`
List every device, helper, and entry point grouped by category
(passives, sources, switches, nonlinear actives, magnetics,
controlled sources, PWM helpers, control blocks, AC analysis,
top-level entry points). Pass a substring to filter:

```python
>>> p.catalog()                  # everything
>>> p.catalog("sources")         # only the source category
>>> p.catalog("control blocks")  # only PIController etc.
```

Each entry has:

- **One-line description** — what it is in a sentence
- **"when:"** — when to reach for this vs. an alternative

### `p.example(name)`
Print a runnable code snippet:

```python
>>> p.example("buck")           # open-loop buck topology
>>> p.example("PIController")   # closed-loop PI pattern
>>> p.example("bode")           # AC sweep recipe
>>> p.example("tune")           # auto-tune from Bode
>>> p.example("rc")             # simplest 3-line RC charge
>>> p.example("yaml")           # load a YAML netlist
>>> p.example("boost")          # open-loop boost
```

Calling with an unknown name prints the list of available snippets.

### `p.tour()`
One-page walking tour of pulsim — six steps from "build a circuit" to
"auto-tune from a measured Bode". Read top to bottom for the
end-to-end story.

### `p.list_topologies()`
Catalog of every runnable script in `examples/scripts/` (open
loop / closed loop / AC analysis / diagnostic). Use to find the
example that matches your topology.

## 2. Plot helpers — no matplotlib boilerplate

### `p.scope(builder, result, signals=[...])`
The one you'll reach for most. Signals are node names; each is
plotted on its own y-axis (stacked subplots), time axis auto-scaled
(ns / µs / ms / s), optional save to PNG.

```python
res = p.simulate(b, t_end=5e-3, dt=1e-7, switch_fn=pwm)
p.scope(b, res, signals=["sw", "vout"], save="output/buck.png")
```

Options:
- `tight_lw=True` for dense PWM traces
- `stack=False` to put all signals on one axes
- `title=...`, `show=True`, etc.

### `p.scope_fft(builder, result, signal=..., f_fundamental=...)`
FFT magnitude spectrum with optional THD annotation. Defaults to
skipping the first 30 % of samples (initial settling) and applies a
Hann window with amplitude correction.

```python
p.scope_fft(b, res, signal="vout",
              f_max=600.0, f_fundamental=60.0,
              save="output/rectifier_fft.png")
# Title gets " THD = X.XX %" appended automatically.
```

Returns `(fig, freqs, mag)` if you want to do further analysis.

### `p.scope_grid(builder, result, panels=[...])`
Multi-panel waveform with per-panel customisation:

```python
p.scope_grid(b, res, panels=[
    dict(signals=["vout"], ylabel="V_out [V]"),
    dict(signals=["sw"],   ylabel="V_sw [V]", lw=0.4),
])
```

### `p.plot_currents(builder, result, branch_ids=[1, 5])`
Plot inductor or source branch currents. Resolves the state-vector
index via `branch_var_id_for_inductor` (falls back to
`branch_var_id_for_source`).

### `p.plot_bode(sweep_result, compare_analytical=...)`
Magnitude + phase Bode for an `AcSweepResult` from
`p.run_ac_sweep(...)`. Pass `compare_analytical=lambda f: H(jω)`
to overlay an analytical reference.

### `p.compare(builder, {label: result}, signal="vout")`
Overlay one signal from multiple simulation runs on the same axes —
the parameter-sweep helper:

```python
runs = {}
for kp in [0.01, 0.05, 0.1]:
    runs[f"Kp={kp}"] = run_my_sim(kp)
p.compare(b, runs, signal="vout", title="Buck — Kp sensitivity")
```

## 3. Progress monitor — know your simulation is alive

The `simulate(..., progress=...)` kwarg controls feedback during
long runs:

| Value         | Behaviour                                       |
| ------------- | ----------------------------------------------- |
| `False`       | (default) silent                                |
| `True`        | print "progress: X%" every 10 %                 |
| `<int>`       | print every N % (e.g. `progress=20`)            |
| `"bar"`       | animated ASCII bar (one line, updates in place) |

```python
res = p.simulate(b, t_end=10e-3, dt=1e-7, switch_fn=pwm,
                 progress="bar")
#   [████████████████░░░░░░░░░░░░░░░░░░░░░░] 41 %   t=4.10 ms (0.8 s)
```

Implementation: reuses the `step_observer` callback. If you
also pass your own `step_observer=...` (e.g. for a closed-loop PI),
the progress observer wraps it transparently.

For programmatic monitoring (UI integration), pass `step_observer`
yourself and emit your own progress events — the `progress=`
kwarg is just sugar for the common case.

## 4. End-to-end — what it looks like now

Before (manual matplotlib):

```python
res = p.run_transient(cache, b.graph, b.pool, opts, switch_fn=pwm)
times = np.asarray(res.times) * 1e3
vout_idx = b.node_id_of("vout")
v_out = np.array([s[vout_idx] for s in res.states])
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(times, v_out, lw=0.8)
ax.set_xlabel("time [ms]"); ax.set_ylabel("V_out [V]")
ax.grid(alpha=0.3); ax.set_title(...)
plt.tight_layout()
plt.savefig("output/buck.png", dpi=120)
```

After (helpers):

```python
res = p.simulate(b, t_end=5e-3, dt=1e-7, switch_fn=pwm,
                 progress="bar")
p.scope(b, res, signals=["sw", "vout"],
          title="Buck converter", save="output/buck.png")
```

Same plot, one quarter the code, time units auto-picked, no
manual `node_id_of` calls.

---

## 5. v1.5 — named lookups + closed-loop + ergonomics

The v1.5 surface lands three additive helpers — every existing
script keeps working unchanged.

### 5.1 Named result accessors (`add-python-named-lookups`)

`result.v(name)` / `result.i(name)` / `result.power(name)` resolve
node / branch / device names without raw column arithmetic:

```python
res = p.simulate(b, t_end=20e-3, dt=2e-6, switch_fn=pwm)

# Instead of:
states = np.asarray(res.states)
v_out  = states[:, b.node_id_of("vout")]
i_L1   = states[:, b.graph.num_nodes + b.branch_index_of("L1")]

# Just:
v_out = res.v("vout")          # full series
v_end = res.v("vout", t=-1)    # scalar
i_L1  = res.i("L1")            # branch current (inductor / source only)
p_R   = res.power("R_L")       # avg dissipation
```

Builder introspection:

```python
b.switch_index_of("Q1")   # bit position in SwitchStateMask
b.branch_index_of("L1")   # branch_id offset (alias of branch_id_of)
b.devices()               # [DeviceInfo(name, kind, terminals), …]
```

Unknown names raise `pulsim.NameNotFoundError(KeyError)` with
`suggestions` from `difflib.get_close_matches`.

### 5.2 PI + PWM closed-loop helper (`add-python-closed-loop-helper`)

Replaces the 30-line closure-with-mutable-list pattern with one
call:

```python
pi = p.PIController(Kp=0.08, Ki=40.0, output_min=0.05, output_max=0.95)
loop = p.bind_pi_to_switch(
    b, pi=pi,
    measured=lambda x: x[b.node_id_of("vout")],
    setpoint=5.0,
    switch="Q1",        # name or int index
    freq=10e3,
)
res = p.simulate(b, t_end=20e-3, dt=2e-6, closed_loops=[loop])

# Inspect the loop's history post-run:
ts, ds = np.asarray(loop.duty_history).T  # (N,) (N,)
```

For circuits with multiple independent loops, pass the list:
`p.simulate(..., closed_loops=[loop_a, loop_b])`. The helper
composes their `switch_fn`s via `make_combined_switch_fn` and
fans out the `step_observer`s automatically.

For one PI driving multiple switches (e.g. half-bridge
complementary pair), use `p.bind_pi_to_duty_callable(...)` instead:
it returns `(duty_get, step_observer, history)` so the caller
composes the switch_fn manually with both bits.

A complete buck closed-loop in ~25 lines lives at
[`examples/scripts/run_buck_with_helper.py`](../examples/scripts/run_buck_with_helper.py).

### 5.3 Builder ergonomics (`add-python-builder-ergonomics`)

**Initial conditions** as `add_*` kwargs:

```python
b.add_capacitor("Cout", "vout", "gnd", 220e-6, c0=12.0)  # IC=12 V
b.add_inductor("L1", "sw", "vout", 220e-6, i0=0.5)       # IC=0.5 A
# Or post-hoc:
b.set_initial("Cout", 12.0)
# Synthesised automatically when simulate(initial_state=None):
res = p.simulate(b, t_end=20e-3, dt=2e-6)  # Cout starts at 12 V.
```

**GUI-friendly aliases** for human-readable names:

```python
b.set_alias("vin", node="net_42")
b.aliases()                # {"vin": ("node", "net_42")}
res.v("vin") == res.v("net_42")   # alias resolves transparently
```

**Cancellation** via `should_continue=` on every long-running entry
point:

```python
cancel_event = threading.Event()

# GUI Run button reuses one builder for many runs; Cancel button:
res = p.simulate(b, t_end=20e-3, dt=2e-6,
                 should_continue=lambda: not cancel_event.is_set())

T_j = p.compute_temperature(t, p_loss, stages,
                              should_continue=lambda: not cancel_event.is_set())

x_dc = p.compute_dc_op(b,
                         should_continue=lambda: not cancel_event.is_set())
```

All raise `pulsim.Cancelled(RuntimeError)` with a typed
`.where` and `.point_index` attribute pinpointing where the
cancellation fired.
