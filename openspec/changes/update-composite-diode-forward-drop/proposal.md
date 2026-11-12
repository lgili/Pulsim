## Why

The owner's decision of 2026-11-06 is that Pulsim's PWL diode follows
PSIM: an ideal switch plus a series conduction drop `V_f` while ON.
`add-switched-diode-forward-drop` gave `add_diode`, the bridge rectifier
and the three-phase VSI that drop, and deliberately left the diodes
inside the composite devices as they were. They still conduct at 0 V:

- `add_mosfet(V_F=0.7)` and `add_mosfet_with_body_diode(V_F=0.7)` pass
  `V_F` as the turn-on **threshold** `V_th`
  (`core/include/pulsim/builder/circuit_builder.hpp:1232-1233`; the doc
  comment at `:1183-1185` says so).
- The body diode of `add_mosfet_level1` and the freewheel diode of
  `add_igbt_level1(with_fwd=True)` hard-code `g_on = 1e3 S`,
  `V_th = 0.5 V` and no drop (`circuit_builder.hpp:197-200` and
  `:579-582`).

Measured (probe
`/Users/lgili/pulsim_wt/core_review_2026-11/models/p6b.py`, and
`probes/d9b_forced.py`, `probes/d9_buck_emulated.py` in this folder):

| Case | Today | With a 0.7 V drop |
|---|---|---|
| 10 A through the body diode of `add_mosfet(V_F=0.7)`, gate off | v_SD = 0.010 V, **0.10 W** | 0.710 V, **7.10 W** |
| Same, gate on (channel conducts) | 0.010 V, 0.10 W | 0.010 V, 0.10 W (unchanged) |
| 10 A through the `_body` diode of `add_mosfet_level1`, gate off | 0.010 V | 0.710 V |
| Buck 24 V → 12 V, 10 A, 100 kHz, freewheeling through a gate-off MOSFET: V_out | 11.990 V | 11.640 V |
| Same buck: body-diode loss | 0.052 W | 3.479 W |

The buck rows agree on `pwl`, `trbdf2` and `dsed` to four digits. Every
dead-time and freewheel conduction loss computed with a composite device
is low by `V_F·I_avg`, and the diode's `V_F` — the one number users take
from a datasheet — has no effect on the answer once the diode is on.

## What Changes

- **BREAKING: `V_F` of `add_mosfet` and `add_mosfet_with_body_diode` is
  the body diode's conduction drop.** The body diode is built with
  `V_f = V_F`, `V_th = 0`: it turns on at `v_SD ≥ V_F` (the same voltage
  as today), conducts `i = g_on·(v_SD − V_F)`, and turns off at zero
  current. Default `V_F = 0.7 V` is unchanged, so every call that relies
  on the default changes too.
- **BREAKING: `add_mosfet_level1` and `add_igbt_level1` gain `V_F`**
  (keyword, default 0.7 V) for their `_body` / `_fwd` diode, with the
  same meaning. The hard-coded 0.5 V threshold is removed.
- YAML: `mosfet` and `mosfet_with_body_diode` `V_F` take the new
  meaning; `mosfet_level1` and `igbt_level1` accept `V_F`.
- `components()` reports `V_f` for the `_body` / `_fwd` branches (it
  already does for any PWL diode), so `res.i`, `device_loss_summary` and
  `device_thermal_summary` need no change: they read the law from the
  diode.
- `V_F = 0` reproduces a drop-free diode that turns on at 0 V.
- Docs: `AI_GUIDE.md` trap 1 and the `add_mosfet` signature note,
  `docs/gotchas.md`, the builder doc comments, the YAML reference.
- `add_igbt` is **not** changed: it is a bare controlled switch with no
  diode and no `V_F` (`circuit_builder.hpp:1244-1254`). An IGBT with
  `V_ce0` and reverse blocking is `add-pwl-igbt-and-thyristor`.

## Impact

- Affected specs: `device-models`, `python-bindings`.
- Affected code: `core/include/pulsim/builder/circuit_builder.hpp`
  (four builders), `core/include/pulsim/yaml/loader.hpp`,
  `python/bindings.cpp` (two new keywords, docstrings), stubs,
  `python/pulsim/AI_GUIDE.md`, docs, CHANGELOG.
- Affected callers, by
  `grep -rlE "add_mosfet\(|add_mosfet_with_body_diode\(|add_mosfet_level1\(|add_igbt_level1\(|type: *(mosfet|mosfet_with_body_diode|mosfet_level1|igbt_level1)\b"`:
  85 files — `examples/` 30 (8 YAML, 6 notebooks, 16 scripts),
  `python/tests/` 24, `docs/` 12, `projects/` 10, `core/tests/` 5,
  `python/pulsim/` 4. design.md lists them. Any of them whose diode
  conducts moves by `V_F·I`; each expected value that moves must be
  re-derived, not re-recorded.
- Dependencies: `add-switched-diode-forward-drop` (the `V_f` parameter;
  implemented on this branch). No dependency on other roadmap changes.
  `restore-reference-benchmark-harness` should freeze its fixed-engine
  checkpoints after this change or re-freeze with it.
- Users affected: every user of the four composite builders whose body
  or freewheel diode conducts — synchronous converters during dead
  time, asynchronous converters freewheeling through a gate-off device,
  inverters. Output voltages drop by roughly `V_F` times the diode duty
  and diode losses rise by `V_F·I_avg`. PulsimGUI's conduction-fidelity
  work (`fix-device-conduction-fidelity`) was blocked on this.
- Migration: to keep a diode with no drop, pass `V_F=0.0`. To keep the
  old "threshold only" diode exactly, build it explicitly:
  `add_mosfet(..., body_diode=False)` plus
  `add_diode(name + "_body", source, drain, 1e3, 1e-9, V_th=0.7)`.
