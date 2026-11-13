## 1. Failing tests first
All tolerances are relative to the closed-form value at f ∈ {100 Hz, 1 kHz, 10 kHz}.
- [x] 1.1 Coupled transformer (`add_transformer`, `L_p = L_s = 1 mH`, `k = 0.99`, 10 Ω source, 10 Ω load): `H = jωM·R_L / ((R_s + jωL_p)(R_L + jωL_s) + ω²M²)`, `|H|` = 0.0617 / 0.3885 / 0.4949, magnitude and phase within 1e-9. Fails today (0)
- [x] 1.2 RC through a switch (100 Ω, `g_on = 1e3`, 1 µF ∥ 1 MΩ): with the switch bit set, `H = Z/(R + 1/g_on + Z)`, `Z = R_L ∥ 1/(jωC)`, `|H|` = 0.998 / 0.8467 / 0.1572 within 1e-9; with the bit clear `|H| < 1e-5` at 100 Hz; with no `mask` the call raises naming `S`. Fails today (0, no error)
- [x] 1.3 Shockley diode, saturable inductor, junction capacitor (the three circuits of `p3_ac.py`): each call raises, the message contains the device name and its kind. Fails today (1, 0.5, 0.9999)
- [x] 1.4 A kernel failure reaches the caller: with `run_mna_sweep_kernel` monkeypatched to raise `RuntimeError("boom")`, `run_mna_sweep(...)` raises that `RuntimeError` even when `dt` and `t_end` are given. Fails today (falls back to impulse-FFT)
- [x] 1.5 C++: for a circuit with R, L, C and one coupled pair, the sweep's `E` equals `C/2` of `assemble_segment_split` entry by entry

## 2. Kernel
- [x] 2.1 `run_mna_sweep`: build `(G, C)` with `assemble_segment_split`, use `E = C/2`; delete `detail::assemble_E_matrix`
- [x] 2.2 Refuse when a branch is `BranchKind::Nonlinear` (name, `StoredKind`), before assembling
- [x] 2.3 Gate: RC, RL and RLC sweeps equal the pre-change values to 1e-12 (`core/tests/benchmarks/test_bench_ac_sweep.cpp` and one new Python check)

## 3. Python wrapper
- [x] 3.1 `mask=` keyword; refusal when `builder.graph.num_switches > 0` and `mask is None`, listing the switch names
- [x] 3.2 Remove the `except Exception`; the impulse-FFT path is selected only by the arguments
- [x] 3.3 Analytic checks for the cells the audit did not probe: VCVS (gain 2 → `|H| = 2` flat), ideal transformer (`n = 0.5` into a resistive load), three-winding `add_multi_winding_transformer` against the impulse-FFT path within 1e-3. A cell that fails is refused by name, not left

## 4. Docs
- [x] 4.1 `CHANGELOG.md` [Unreleased]: the two BREAKING refusals, the transformer fix
- [x] 4.2 `run_mna_sweep` docstring: what the kernel path covers, `mask=`, what it refuses
- [x] 4.3 `python/pulsim/AI_GUIDE.md` and `docs/known-limitations.md`: AC coverage table (kernel sweep: linear elements and switches at a given mask; everything else: time domain)
- [x] 4.4 Stubs via `scripts/generate_stubs.py` if affected

## 5. Verification
- [x] 5.1 `ruff check`
- [x] 5.2 `pytest python/tests`
- [x] 5.3 `ctest -j2`
- [x] 5.4 `openspec validate --all --strict`
- [x] 5.5 Stub check
- [x] 5.6 Re-run `p3_ac.py`: rows 1–2 match the exact column, rows 3–5 raise
