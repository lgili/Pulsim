# Decisions taken during implementation

The proposal has no design.md and no Open Questions. Choices made where
it left room:

| Point | Choice | Why | How to reverse |
|---|---|---|---|
| Refusal type | `std::invalid_argument` in C++ (pybind11 maps it to `ValueError`); Python-side mask refusal is `ValueError` | `add-engine-conformance-matrix` has not landed; the proposal says to use these until it does | Raise `UnsupportedError` once it exists |
| Kind name in the message | New `pwl::stored_kind_name(StoredKind)` next to the enum, spelled as the Python `StoredKind` members | The kernel had no kind-to-name map; `add-engine-conformance-matrix` adds a table that should become its source | Replace with the table's `name` column |
| Which switch names the mask refusal lists | `components()` entries of kind `switch` and `diode` (every branch that owns a mask bit) | Same set `graph.num_switches` counts | — |
| Time-domain routines named in the refusal | `pulsim.run_ac_sweep`, `pulsim.fra.run_fra`, `pulsim.frequency_response`, `run_mna_sweep(use_kernel=False, dt=, t_end=)` | The proposal names `run_fra`, which is not exported at top level (`pulsim.fra.run_fra`) | text only |
| Gate 2.3 ("RC, RL, RLC equal the pre-change values to 1e-12") | Checked against the closed form to 1e-12 relative (`test_mna_sweep_assembly.cpp`), plus the two pre-existing analytic tests in `test_mna_sweep.cpp` | The closed form is a stronger reference than a recorded pre-change run | — |
| Multi-winding check (3.3) | Three windings, `k = 0.95` pairwise, against the impulse-FFT path within 1e-3 | `k = 1` (the default) is refused by the builder as not realisable for three windings | — |

Measured after the change (`p3_ac.py`, 100 Hz / 1 kHz / 10 kHz): coupled
transformer `|H|` = 0.0617 / 0.3885 / 0.4949 (exact, was 0 / 0 / 0);
RC through a closed switch with `mask=` matches `Z/(R + 1/g_on + Z)` to
1e-9 (without `mask=` it raises naming `S`); Shockley diode, saturable
inductor and junction capacitor raise naming `'D'`/ShockleyDiode,
`'L'`/SaturableInductor, `'C'`/NonlinearCapacitor. All proposal numbers
reproduced.
