## Why
PLECS invented electro-thermal co-simulation and they sell it as their headline feature for the EV / inverter market: junction temperatures, thermal RC networks, and most importantly the **feedback** of temperature back into device parameters (R_DS_on(T_j), V_F(T_j), V_CE,sat(T_j)).

Pulsim's `electrical_scope` / `thermal_scope` blocks already hint at the infrastructure; the `update-electrothermal-component-observability` change in `openspec/changes/` extended it further. We need to close the loop: temperature-dependent device parameters, a thermal RC network, and benchmarks that prove the coupling works.

## What Changes
- Add a `thermal_node` primitive (or repurpose existing thermal-domain plumbing) that participates in a thermal RC network: each node has a heat capacity (J/K), and resistors between thermal nodes have units K/W.
- Add a `power_dissipation` source-style block that injects heat into a thermal node based on electrical losses computed from another component (V·I product of a referenced device).
- For temperature-dependent devices: extend `IdealSwitch` / `MOSFET` / `IGBT` parameter handling so `g_on` (or `r_on`) can be expressed as `r_on_25 · (1 + α·(T_j − 25 °C))`.
- New benchmarks:
  - `electrothermal_buck_steady` — a buck running at fixed load; junction temperature of the high-side MOSFET reaches a documented steady-state T_j.
  - `electrothermal_load_step` — buck with a load step; observe the thermal transient (T_j rises, settles).
  - `electrothermal_self_heating_resistor` — a power resistor with its own thermal RC; demonstrate self-heating and the temperature coefficient on resistance.

## Impact
- Affected specs: `device-models` (temperature-dependent parameters), `kernel-v1-core` (thermal-node primitive), `benchmark-suite` (new benchmarks).
- Affected code: C++ primitives for thermal nodes + thermal-feedback to electrical parameters; new YAML benchmarks.
- This is the biggest C++ change of any of the ten proposals — likely requires its own dedicated branch.
