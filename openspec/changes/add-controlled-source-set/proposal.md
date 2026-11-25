## Why

Pulsim's set of linear sources stops at one controlled source and one
constant current source.

Verified on this branch (review `raw/models.md`, inventory rows 2–3 and
ranked item 9; synthesis M7):

- The only controlled source is the VCVS
  (`core/include/pulsim/models/vcvs.hpp`,
  `stamping/stamp_vcvs.hpp`). There is no VCCS, CCVS, CCCS or gyrator.
- The only current source is DC. Its header says so: "Time-varying
  current sources are NOT supported by this V0 model — the user achieves
  time variation via `b_extra_fn(t)`"
  (`core/include/pulsim/models/current_source.hpp:30-33`). A
  `b_extra_fn` is a Python callback, which rules out the threaded sweep
  path and any engine that does not take it.
- Voltage sources already have sine and pulse waveforms in the kernel
  (`sources/sine_b_extra.hpp`, `sources/pulse_b_extra.hpp`); the
  current source cannot use them.
- The YAML loader has no type for any of these
  (`core/include/pulsim/yaml/loader.hpp`: `vcvs`, `op_amp_ideal` and
  `current_source` only).
- `dsed` rejects both the DC current source and the VCVS (inventory,
  measured in the review probe `models/p1_engine_matrix.py`).

Why it matters beyond completeness:

- The **gyrator** is the coupling element of the permeance-capacitor
  magnetic domain (`add-permeance-magnetic-domain`): a winding of N
  turns is a gyrator with `g = 1/N` between the electrical and the
  magnetic network.
- A **VCCS** integrates a node voltage into another node (`dθ/dt = ω`
  for the machines of `add-mna-native-machines`) and builds behavioural
  transconductance models.
- **Current-controlled sources** build current mirrors, current-sense
  amplifiers and CCCS-based averaged switch models.
- A **table current source** is how a measured load-current profile or
  a double-pulse stimulus enters a run without Python in the loop.

Other tools (general knowledge, not checked against current
documentation): every SPICE has the E, F, G and H elements and PWL,
pulse and sine independent current sources; PLECS and PSIM provide
controlled voltage and current sources and, in PLECS, the magnetic
interface that is a gyrator.

## What Changes

- **Stamps** (`stamping/stamp_controlled.hpp`), all linear, constant,
  mask-independent:
  - VCCS: `i = g_m·(v_cp − v_cn)` [S], no extra unknown.
  - CCVS: `v_out = r_m·i_c` [Ω], one branch unknown.
  - CCCS: `i = β·i_c`, no extra unknown.
  - Gyrator: `i_1 = g·v_2`, `i_2 = −g·v_1` [S], no extra unknown,
    lossless.
- **Builder**: `add_vccs(name, in_pos, in_neg, from, to, gm)`,
  `add_ccvs(name, control, out_pos, out_neg, rm)`,
  `add_cccs(name, control, from, to, gain)`,
  `add_gyrator(name, p1_pos, p1_neg, p2_pos, p2_neg, g)`. `control`
  names a branch that carries a current unknown (a voltage source of
  any kind, an inductor, a VCVS, a CCVS, an ideal-transformer
  secondary); any other branch is refused with the list of valid kinds.
- **Current-source waveforms**:
  `add_sine_current_source(name, from, to, i_dc, i_amplitude,
  frequency, phase=0)`,
  `add_pulse_current_source(name, from, to, i_initial, i_pulsed,
  t_start, pulse_width, period=0, rise_time=0, fall_time=0)`,
  `add_pwl_current_source(name, from, to, t, i, periodic=False)`. They
  reuse the voltage sources' waveform evaluators and declare their
  breakpoints.
- Terminal convention of every current output: that of
  `add_current_source` — a positive value leaves the source at `from`
  into the external circuit.
- **Bindings, stubs, `components()`**, and `res.i` for the new kinds.
- **YAML** types `vccs`, `ccvs`, `cccs`, `gyrator`,
  `sine_current_source`, `pulse_current_source`, `pwl_current_source`.
- **Engines**: `pwl`, `trbdf2`, the DC operating point and the AC
  sweep; on `dsed` each kind either runs or is refused by the
  conformance error.
- No **BREAKING** change.

## Impact

- Affected specs: `device-models`, `netlist-yaml`.
- Affected code: new `stamping/stamp_controlled.hpp`,
  `models/{controlled_sources,pwl_waveform}.hpp`;
  `models/current_source.hpp`, `stamping/stamp_current_source.hpp`,
  `sources/{sine_b_extra,pulse_b_extra}.hpp`,
  `pwl/{device_pool,assemble,dc_assemble}.hpp`,
  `analysis/mna_sweep.hpp`, `dsed/builder_adapter.hpp`,
  `builder/circuit_builder.hpp`, `yaml/loader.hpp`,
  `python/bindings.cpp`, `python/pulsim/__init__.py` (`res.i`), stubs,
  docs.
- Dependencies: none required (the roadmap lists none). Related:
  - `add-engine-conformance-matrix` — rows for the seven kinds.
  - `fix-mna-sweep-correctness` — the sweep these stamps must appear
    in.
  - `update-trbdf2-error-control` — treats source breakpoints as
    events; until it lands this change clamps the step to the next
    declared breakpoint itself.
  - `add-unified-event-queue` — will own the breakpoints.
- Consumers: `add-permeance-magnetic-domain` (gyrator),
  `add-mna-native-machines` (VCCS for the angle node).
- Can ship on today's structure in full: linear stamps and sources are
  not part of the stateful-device chain.
