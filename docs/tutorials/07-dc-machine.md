# Tutorial 07 — A DC machine as a circuit

`b.add_dc_machine` puts a permanent-magnet DC machine inside the
circuit matrix. The mechanics are a node: its voltage is the speed
`ω` [rad/s] and a current into it is a torque [N·m]. The inertia `J`
is a capacitor on that node, viscous friction `B` a resistor `1/B`, a
constant load torque a current source, and the coupling an ideal
transformer of ratio `K` between the speed node and the armature:

```text
a_pos ─ R_a ─ L_a ─ e ─[ secondary ]─ a_neg      v_e − v_a_neg = K·ω
omega ─[ primary ]─ gnd                          torque K·i into omega
omega ─ C = J ─ gnd,  omega ─ R = 1/B ─ gnd,  I = T_load out of omega
```

Nothing is taken from the previous step, so the machine is second
order on the fixed-step engine, runs on `trbdf2`, and has a DC
operating point. The parameters below are a declared example, not a
real motor.

## Steady state from the DC operating point

With the inductor shorted and the inertia open, the DC solution is the
loaded steady state: `i = (T_load + B·ω)/K`, `V = R_a·i + K·ω`, so
`ω = (V − R_a·T_load/K)/(K + R_a·B/K)` = 89.552 rad/s.

```python
import pulsim as p

def machine():
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "a", "gnd", 10.0)
    b.add_dc_machine("M", "a", "gnd", "w", R_a=0.5, L_a=1e-3, K=0.1,
                     J=1e-5, B=1e-4, T_load=0.2)
    return b

b = machine()
x = p.compute_dc_op(b, gmin=0.0)
i_a = x[b.pool.branch_var_id_for_inductor(b.branch_id_of("M.la"), b.graph)]
w = x[b.node_id_of("w")]
print(f"i = {i_a:.6f} A, w = {w:.6f} rad/s")
```

```text
i = 2.089552 A, w = 89.552239 rad/s
```

## Start-up

From rest the speed rises to the same value. The armature current is
`result.i("M.la")`; the electromagnetic torque is `K` times it.

```python
r = p.simulate(machine(), t_end=0.1, dt=10e-6, engine="pwl")
print(f"w(100 ms) = {r.v('w')[-1]:.4f} rad/s, i = {r.i('M.la')[-1]:.4f} A")
```

```text
w(100 ms) = 89.5522 rad/s, i = 2.0896 A
```

A speed-dependent load is another element on the speed node: a
resistor `1/B` from `w` to ground is a viscous load, a second machine
connected to the same node is a shaft coupling. `p.add_dc_motor` and
its observer are deprecated: they advance the speed by forward Euler
from the previous step and run on `pwl` only.
