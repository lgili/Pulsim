# YAML composite devices + `chain:` wiring

The YAML loader (`pulsim.load_yaml_string` /
`pulsim.load_yaml_file`) understands two layers:

1. **`circuit:`** — the electrical topology. Devices are stamped
   into the `CircuitBuilder` as usual; built-in kinds (R/L/C, MOSFET,
   IGBT, diode, sources, transformer, VCVS, …) plus composite
   devices (induction motor, hysteretic inductor) all parse here.
2. **`chain:`** — the *control* layer. Block-chain wiring that runs
   each simulation step alongside the kernel: PMSM observer, IM
   observer, motor mechanical, hysteretic-core observer, etc. The
   loader doesn't process `chain:` itself; it returns the parsed
   builder and you call `pulsim.wire_chain_from_yaml(...)` to attach
   the chain to a runnable `CxxBlockChain`.

> **TL;DR**
>
> ```python
> loaded = pulsim.load_yaml_file("im_dol.yaml")
> chain  = pulsim.wire_chain_from_yaml(loaded, chain_spec_yaml)
> step_obs = chain.make_step_observer(dt)
> res = pulsim.simulate(loaded.builder, t_end=..., dt=dt,
>                          step_observer=step_obs,
>                          b_extra_fn=chain.make_b_extra_fn(N_state))
> ```

## Composite device types in `circuit:`

### `induction_motor` — squirrel-cage IM

```yaml
circuit:
  devices:
    - {type: voltage_source, name: Va, from: a, to: gnd, V: 0.0}
    - {type: voltage_source, name: Vb, from: b, to: gnd, V: 0.0}
    - {type: voltage_source, name: Vc, from: c, to: gnd, V: 0.0}
    - type: induction_motor
      name: IM
      phase_nodes: [a, b, c]
      neutral_node: n
      R_s: 2.4565
      L_s: 0.09475
      R_r: 1.22825
      L_r: 0.09475
      L_m: 0.087392
      pole_pairs: 2
    - {type: resistor, name: R_leak, from: n, to: gnd, R: 1.0e6}
```

The loader expands the IM into **6 named branches** behind the
scenes: `IM_Rs_{a,b,c}` (stator winding resistances), `IM_Lsig_{a,b,c}`
(leakage inductances) and `IM_E_{a,b,c}` (back-EMF dummy sources).
The chain block resolves them via `builder.branch_id_of(...)`.

### `hysteretic_inductor` — Jiles-Atherton core

```yaml
- type: hysteretic_inductor
  name: L_core
  from: n1
  to: gnd
  Ms: 4.0e5
  a: 50.0
  alpha: 5e-5
  c: 0.20
  k: 30.0
  N_turns: 100
  l_m: 0.05
  A_core: 1e-4
```

Expands into `L_core_L0` (air-core inductance) + `L_core_V_M`
(dummy voltage source carrying the magnetisation EMF). The JA
observer in the chain reads `L_core_L0`'s branch current and
writes `+N·A·µ₀·dM/dt` into the `L_core_V_M` source row.

## The `chain:` section

`wire_chain_from_yaml(loaded, chain_yaml)` is a separate call
because the kernel's `CxxBlockChain` is a different module than
the YAML circuit loader. Pass the chain spec as a YAML string OR
a Python list of dicts:

```python
chain = pulsim.wire_chain_from_yaml(loaded, """
- type: induction_motor
  device: IM
  J: 0.01
  B: 0.05
  T_load: 0.0
  R_s: 2.4565
  L_s: 0.09475
  R_r: 1.22825
  L_r: 0.09475
  L_m: 0.087392
  pole_pairs: 2
  omega_channel: omega
  theta_channel: theta
  torque_channel: torque

- type: hysteretic_inductor
  device: L_core
  Ms: 4.0e5
  a: 50.0
  alpha: 5e-5
  c: 0.20
  k: 30.0
  N_turns: 100
  l_m: 0.05
  A_core: 1e-4
  M_channel: M
  B_channel: B
""")
```

Each block must carry:

- `type:` — registered handler kind (see table below).
- `device:` — the *name* of the topology device this chain block
  attaches to. The handler resolves the deterministic branch names
  the loader created.

### Supported chain block types (v1.5)

| `type:` | Topology device kind | Chain output channels |
|---|---|---|
| `induction_motor` | `induction_motor` | `omega`, `theta`, `psi_alpha/beta` (opt), `torque` (opt), `slip` (opt) |
| `hysteretic_inductor` | `hysteretic_inductor` | `M`, `B`, `H`, `vm` (all optional) |
| `sliding_mode_observer` | *(pure channel — no topology device)* | `theta_hat`, `omega_hat`, optional `e_alpha/beta_hat`, `low_speed_flag` |
| `flux_mras_observer` | *(pure channel — no topology device)* | `omega_hat`, optional `psi_adj_alpha/beta`, `psi_ref_alpha/beta` |

The two sensorless observers (`sliding_mode_observer`,
`flux_mras_observer`) are *pure channel-based* — they read the
αβ voltages and currents from the chain channel bus and write
estimated angle/speed back, without needing topology-resolved
branch indices. Wire them as standalone blocks alongside the IM
or PMSM that produces the αβ inputs.

### Unknown block types

```python
pulsim.wire_chain_from_yaml(loaded, [
    {"type": "not_a_real_block", "device": "IM"}
])
# → KeyError: chain block #0 has unknown type 'not_a_real_block';
#   supported: ['flux_mras_observer', 'hysteretic_inductor',
#               'induction_motor', 'sliding_mode_observer']
```

Honest error — no silent skip.

### Empty spec

```python
chain = pulsim.wire_chain_from_yaml(loaded, [])
assert chain.size() == 0
```

Returns a valid empty chain (zero blocks). Useful for testing the
loader path without any observers attached.

## `simulation:` block

The same YAML file can carry a `simulation:` block with solver
options:

```yaml
simulation:
  t_start: 0.0
  t_end:   1.0e-3
  dt:      1.0e-5
  enable_newton_line_search: false
  enable_newton_lm:          false
  enable_substep_state_correction: false
  max_event_iterations:   16
  max_newton_iterations:  50
  tol_newton_dx:  1.0e-9
  tol_newton_res: 1.0e-9
  # Phase 2.4 schema — adaptive RK selector (v1.5).
  integrator:  kernel       # "kernel" | "dopri5" | "radau"
  rtol:        1.0e-5       # tolerances for adaptive paths
  atol:        1.0e-8
  dt_init:     0.0
```

The loader puts the parsed values on `loaded.options`
(a `SimulationOptions` instance). You can forward them into
`simulate()`:

```python
loaded = pulsim.load_yaml_file("buck.yaml")
res = pulsim.simulate(
    loaded.builder,
    t_end=loaded.options.t_end,
    dt=loaded.options.dt,
    integrator=loaded.options.integrator,
    rtol=loaded.options.rtol,
    atol=loaded.options.atol,
    dt_init=loaded.options.dt_init,
)
```

**Today** only `integrator="kernel"` (default) actually runs;
`"dopri5"` and `"radau"` raise a clear `NotImplementedError` with
a pointer to the v1.6 cache refactor. The schema is wired so YAML
files stay forward-compatible. See [solvers.md](solvers.md) for
the detailed rationale.

## End-to-end example

```python
# 1. Load the topology + simulation options.
loaded = pulsim.load_yaml_file("examples/im_dol.yaml")
b      = loaded.builder
opts   = loaded.options

# 2. Wire the chain.
chain = pulsim.wire_chain_from_yaml(loaded, """
- type: induction_motor
  device: IM
  J: 0.01
  B: 0.05
  T_load: 0.0
  R_s: 2.4565
  L_s: 0.09475
  R_r: 1.22825
  L_r: 0.09475
  L_m: 0.087392
  pole_pairs: 2
  omega_channel: omega
""")

# 3. Pull observer + b_extra wiring out of the chain.
step_obs   = chain.make_step_observer(opts.dt)
b_extra_fn = chain.make_b_extra_fn(b.pool.state_size(b.graph))

# 4. Run.
res = pulsim.simulate(
    b, t_end=opts.t_end, dt=opts.dt,
    switch_fn=lambda t: pulsim.SwitchStateMask(0),
    step_observer=step_obs,
    b_extra_fn=b_extra_fn,
)
print("ω =", chain.get_channel("omega"))
```

## Extending — add your own block type

Three steps:

1. Write a handler function in `python/pulsim/yaml_chain.py` that
   takes `(chain, builder, spec)` and calls `chain.add_*` for your
   block.
2. Register it in `_BLOCK_HANDLERS`:
   ```python
   _BLOCK_HANDLERS = {
       "induction_motor":      _add_induction_motor_block,
       "hysteretic_inductor":  _add_hysteretic_inductor_block,
       "sliding_mode_observer": _add_sliding_mode_observer_block,
       "flux_mras_observer":   _add_flux_mras_observer_block,
       "your_new_type":        _add_your_new_block,   # ← here
   }
   ```
3. Add a test in `python/tests/test_yaml_chain_*.py` exercising
   the new `type:` value end-to-end.
