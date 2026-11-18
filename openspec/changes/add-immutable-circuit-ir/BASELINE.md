# Baseline records (task 0.2) and after (task 10.2)

Machine: the shared 16 GB Mac, load average ~20 from other agents'
builds; times are reported, never asserted. Counters decide.

## components() dumps

`probes/components_baseline.json`: one entry per shipped example
(`examples/*.yaml`, 30) and harness circuit (`benchmarks/circuits/**/*.yaml`,
122): the number of `components()` entries and a digest of the dump.
136 load; 16 are recorded as unloadable (`Unmapped` by the harness
reader, or a load error), as they are today. `python/tests/test_circuit_ir.py`
checks the builder and `build()` against this record for every one.
Regenerate with `python probes/sweep_probe.py components` after adding a
circuit.

## Builder private attributes set from Python

`python probes/sweep_probe.py private` at the slice's base (1aef1a52):

| attribute | set | read with `getattr` |
|---|---|---|
| `_c_blocks` | `c_block.py:436` | `__init__.py` ×3 (dsed refusal, pwl composition, trbdf2 blockers) |
| `_pulsim_ic` | `_builder_ergonomics.py:112` | `_builder_ergonomics.py:109` |
| `_pulsim_aliases` | `_builder_ergonomics.py:121` | `_builder_ergonomics.py:118` |
| `_shockley_params` | `_builder_ergonomics.py:445` | `_builder_ergonomics.py:441, 467` |

Four attributes set, seven `getattr` reads, as the proposal counted.

## 100-point sweeps (`probes/sweep_probe.py sweep`)

Serial, `engine='pwl'`; buck: 100 values of L (2000 steps each); ladder
n = 1502 (N = 500 RLC sections): 100 values of Rload (20 steps each, so
build + analyse + assemble dominate). Numeric factorisations = segment
builds + event builds + event refactors of the PWL cache.

Baseline (factory form, rebuilt per point; best of 3):

| sweep | wall | per point | builds | symbolic analyses | numeric factorisations |
|---|---|---|---|---|---|
| buck | 53.4 ms | 0.53 ms | 100 | 100 | 418 |
| ladder1502 | 415.7 ms | 4.16 ms | 100 | 100 | 100 |

After (task 10.2, this slice; interleaved with the factory form in the
same process, best of 3). `circuit` = one `build()`, then
`simulate(circuit.with_value(field, x))` per point; every derived circuit
shares the receiver's `SymbolicPlan`:

| sweep | form | wall | per point | builds | symbolic analyses | numeric factorisations | final states |
|---|---|---|---|---|---|---|---|
| buck | factory | 47.1 ms | 0.47 ms | 100 | 100 | 418 | |
| buck | circuit | 42.6 ms | 0.43 ms | 1 | **1** | 418 | bit-identical |
| ladder1502 | factory | 497.1 ms | 4.97 ms | 100 | 100 | 100 | |
| ladder1502 | circuit | 168.3 ms | 1.68 ms | 1 | **1** | 100 | bit-identical |

Numeric factorisations are unchanged: each point has new values, so each
refactors; only the symbolic analysis (pattern, ordering) is shared.

Threads (`pulsim.sweep(circuit, params=..., parallel=True, n_workers=8)`,
backend `thread`, best of 3, results bit-equal to serial): buck 100 points 41.5 ms serial → 17.6 ms
(2.4×); ladder1502 231 ms → 211 ms (1.1×: 20 steps per point, so the
per-point Python work — derive, clone, preflight — holds the GIL).
Short of audit row 23 (3.0–3.1× on four): task 5.2 stays open.
