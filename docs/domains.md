# Node domains

Pulsim solves electrical, thermal and mechanical networks in one MNA
system. A thermal node is a "voltage" in °C, a heat flow a "current" in
W; a shaft speed is a "voltage" in rad/s, a torque a "current" in N·m.
The numbers never change with the domain. The tag on a node says what
the number means, so that the circuit can refuse a wrong connection
and report the right unit.

## Declaring a domain

```python
b = pulsim.CircuitBuilder()
b.node("tj", "thermal")        # "electrical" (default), "thermal", "mechanical"
b.add_current_source("P", "gnd", "tj", 10.0)    # 10 W into the junction
b.add_resistor("Rjc", "tj", "ta", 0.5)          # 0.5 K/W
c = b.build()
c.node_domain("tj"), c.unit_of("tj")            # ("thermal", "degC")
```

YAML takes the same tag in `nodes:`:

```yaml
circuit:
  nodes: [a, {name: tj, domain: thermal}]
```

- A node never declared is electrical, unless a typed pin implies its
  domain: a PMSM's `omega_node` is mechanical (`rad/s`) and its
  `theta_node` mechanical (`rad`).
- Ground is the reference of every domain and takes no tag.
- `magnetic` is reserved (`add-permeance-magnetic-domain`) and refused.
- A node declared twice with two domains is refused.
- `add_foster_network`, `add_cauer_thermal_network` and
  `add_shared_heatsink` tag the nodes they create as thermal.

## What the tag checks

At `build()`, every device is checked against its nodes:

- Semiconductors (switch, diode, MOSFET, IGBT, Shockley and Lauritzen
  diodes), the saturable inductor, the nonlinear capacitor and a PMSM's
  phases are electrical only. A thermal node on a diode pin is refused:
  `device "D1": pin from (electrical) is connected to node "tj", which
  is thermal (declared)`.
- Every other device is domain-generic: a resistor, capacitor, source
  or controlled source works in any domain. Its pins come in pairs
  (`from`/`to`, then each control pair), and the two pins of one pair
  must share a domain. A VCVS sensing a temperature and driving an
  electrical node is legal; a resistor from a thermal node to an
  electrical one is refused.

The preflight labels each finding with the domain of its subnet. A
floating thermal network (no path to ambient) reads
`thermal-domain: ...` and carries `finding.domain == "thermal"`.

## What the tag does not change

- Simulation results: a tagged circuit and its untagged twin give the
  same samples, bit for bit.
- Hashes of an untagged circuit. A non-electrical tag is structure, so
  it changes `structure_hash` (and `value_hash`, which folds the
  structure in).
- Result traces keep their names; `Circuit.unit_of(node)` gives the
  unit. Result objects do not carry units yet.
