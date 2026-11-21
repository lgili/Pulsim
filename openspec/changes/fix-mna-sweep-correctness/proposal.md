## Why

The kernel AC sweep (`pulsim.run_mna_sweep` with `freqs=` and
`v_in_branch_id=`, C++ `analysis::run_mna_sweep`) returns a transfer
function for circuits it does not model, with no error. Four causes, all
re-read on this branch:

1. The mass matrix `E` is assembled by a second, hand-written stamper
   that knows only `PassiveLinear` capacitors and inductors
   (`core/include/pulsim/analysis/mna_sweep.hpp:118-134`). The mutual
   terms of `pool.transformer_couplings()` are missing, although the
   transient assembler stamps them (`pwl/assemble.hpp:303-315`).
2. The static matrix is `dc_assemble` (`mna_sweep.hpp:183`), which skips
   every `BranchKind::Nonlinear` branch: a Newton device is an open
   circuit, a saturable inductor a short, a junction capacitor absent.
   Nothing is linearised at an operating point.
3. The Python wrapper passes an all-OFF switch mask whatever the circuit
   (`python/pulsim/mna_sweep.py:142`), so a converter is swept with
   every switch and PWL diode open.
4. A broad `except Exception` around the kernel call
   (`mna_sweep.py:161`) turns any kernel failure into a fall-through to
   the impulse-FFT path, or into an unrelated "pass `dt` and `t_end`"
   error.

Measured, probe
`/Users/lgili/pulsim_wt/core_review_2026-11/models/p3_ac.py`, `|H|` at
100 Hz / 1 kHz / 10 kHz:

| Circuit | Kernel today | Exact |
|---|---|---|
| Coupled transformer, k = 0.99, 10 Ω source and load | 0 / 0 / 0 | 0.0617 / 0.3885 / 0.4949 |
| RC low-pass through a switch meant closed | 0 / 0 / 0 | 0.998 / 0.8467 / 0.1572 |
| Shockley diode biased at 1 A, 1 Ω in series | 1 / 1 / 1 | 0.0252 |
| Saturable inductor (L0 = 1 mH), 10 Ω / 10 Ω | 0.5 / 0.5 / 0.5 | 0.4998 / 0.477 / 0.1517 |
| Junction capacitor (C0 = 1 µF), 100 Ω | 0.9999 (flat) | 0.998 / 0.8467 / 0.1572 |

## What Changes

- **`E` comes from the transient assembler.** The sweep takes `(G, C)`
  from `pwl::assemble_segment_split(graph, pool, mask)` and uses
  `E = C/2` (the split stores the trapezoidal `2/dt` coefficients). The
  private `detail::assemble_E_matrix` is deleted, so the sweep and the
  transient can no longer disagree on a linear dynamic element; mutual
  inductance follows.
- **The switch state is the caller's.** `run_mna_sweep(..., mask=)`
  accepts a `SwitchStateMask`. When the circuit has switch branches
  (controlled switches or PWL diodes) and no mask is given, the kernel
  path refuses and names them; it no longer assumes all-OFF.
- **Newton kinds are refused.** If any branch is `BranchKind::Nonlinear`
  the kernel sweep raises an error naming the device and its kind and
  pointing to the time-domain routines (`run_fra`, `frequency_response`,
  or the impulse-FFT path of this function with `use_kernel=False`).
- **No broad `except`.** Kernel errors propagate. The impulse-FFT path
  runs only when the caller did not supply `freqs` and
  `v_in_branch_id`, or passed `use_kernel=False`.
- **BREAKING:** calls that returned a wrong curve now raise (Newton
  device present; switches present and no `mask`). Circuits made of R,
  L, C, sources, VCVS and ideal transformers without coupling are
  unchanged to rounding.

## Impact

- Affected specs: `ac-analysis`.
- Affected code: `core/include/pulsim/analysis/mna_sweep.hpp`,
  `python/pulsim/mna_sweep.py`, `python/bindings.cpp`
  (`run_mna_sweep_kernel` unchanged in signature), new tests in
  `python/tests/test_mna_sweep_correctness.py` and
  `core/tests/analysis/`, `examples/scripts/run_mna_sweep_rc.py`
  (check), docs, CHANGELOG, stubs only if the Python signature is part
  of them.
- Dependencies: none. If `add-engine-conformance-matrix` has landed, the
  refusals are raised as `UnsupportedError` (a `ValueError` subclass);
  otherwise as `ValueError` / `std::invalid_argument`. The general
  solution — AC as `∂f/∂x + jω·∂q/∂x` at the operating point for every
  device — belongs to `add-unified-device-interface` and is out of
  scope.
- Users affected: anyone sweeping a transformer (results become
  correct), a converter (must now pass the operating-point `mask`), or a
  circuit with Newton devices (must use a time-domain routine until the
  device interface lands). In-repo callers: `test_cancellation.py` and
  one example script, both linear RC.
