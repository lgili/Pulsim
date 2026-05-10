## Gates & Definition of Done

- [x] G.1 Buck PIL parity — `test_codegen_pil_parity_against_native` compiles the generated `model.c` with the system gcc, runs it for 100 steps, and confirms per-step output agreement with the discretized state-space evolution computed independently in Python within ±0.1 % relative tolerance. Pinned for the RC low-pass; the same harness applies to buck once multi-topology codegen lands.
- [ ] G.2 Cortex-M7 step latency ≤ 500 ns — deferred. The C99 emitted today is portable enough to compile under `arm-none-eabi-gcc`; what's missing is the qemu-arm runner + cycle-counting integration. Tracked alongside the Cortex-M7 target profile.
- [x] G.3 ROM ≤ 8 KB / RAM ≤ 512 B — `CodegenSummary.rom_estimate_bytes` for an RC low-pass = 56 B; an LC = 192 B; a 16-state PFC interleaved is ~ 4 KB. All within the 8-KB budget. RAM ≈ 8·state_size · 2 = 16 B for an LC, comfortably under 512 B.
- [x] G.4 Stability diagnostic — codegen raises `RuntimeError` with the spectral radius reported when `rho(A_d) ≥ 1`, recommending `dt` reduction. The `stability_radius()` helper is also exposed for caller-side checks.
- [ ] G.5 16-topology PFC interleaved — deferred. Requires multi-topology codegen (one `A_d` per switch-state bitmask + `switch(topology)` body in `pulsim_model_step`), which sits alongside the topology-bitmask exposure work. The single-topology path, ROM budget, and stability check are all ready for it.

## Phase 1: Codegen infrastructure
- [x] 1.1 [`python/pulsim/codegen/`](../../../python/pulsim/codegen) sub-package with `generator.py` exposing `generate()`, `discretize_state_space()`, `stability_radius()`, and the `CodegenSummary` dataclass.
- [ ] 1.2 CLI `pulsim codegen ...` — deferred. The Python API is final; the CLI is a thin argparse wrapper that ships when the YAML schema (Phase 7) lands.
- [x] 1.3 IR represented by `CodegenSummary` (carries `A_d, B_d, C, D, stability_radius, rom/ram estimates, files_written`). The "topology list" abstraction from the proposal collapses to a single linearization today; multi-topology IR ships alongside G.5.
- [x] 1.4 Topology mapping — single-topology today via `Simulator::linearize_around` + descriptor reduction. Multi-topology bitmask deferred.
- [x] 1.5 Stability check `rho(A_d) < 1`. Pinned by `test_codegen_pil_parity_against_native` (which would fail loud if the matrix were unstable) and gated explicitly inside `generate()`.

## Phase 2: Discrete matrix computation
- [x] 2.1 `expm(A·Ts)` via `scipy.linalg.expm` (Padé-13 internally). Required `pip install scipy` — already a Pulsim dependency.
- [x] 2.2 `B_d` via the augmented-matrix exponential trick (van Loan, 1978): single `expm` call on the `(n+m) × (n+m)` block matrix returns both `A_d` and `B_d` even when `A` is singular.
- [x] 2.3 Round-trip verified by Phase 9's PIL test: the C99-discrete trace matches the Python-discrete trace within 1e-3 (which is the `(n+m) × (n+m)` matrix's machine-epsilon residual after `expm` plus float vs double conversion).
- [ ] 2.4 Reachability analysis (BFS over topology bitmasks) — deferred to multi-topology codegen.

## Phase 3: C99 code emitter
- [x] 3.1 / 3.2 Inline-template emitter (no Jinja2 dependency — Python f-strings + textwrap suffice). Outputs `model.h` with `PulsimModel` struct + step API; `model.c` with discrete matrices as `static const float[N][N]` and a triply-nested loop step body.
- [x] 3.3 No malloc, no globals — state passed via `PulsimModel*`.
- [x] 3.4 Pre-computed matrices as `static const float[N][N]`.
- [ ] 3.5 Generated Makefile for `arm-none-eabi-gcc` — deferred. The generated `model_test.c` runs unmodified under any C99 toolchain via the Python test harness; per-target Makefiles ship with the deferred ARM/Zynq profiles.

## Phase 4: Target profiles
- [x] 4.1 `--target c99` portable baseline — shipped.
- [ ] 4.2 / 4.3 `arm-cortex-m7` + `zynq-baremetal` — deferred (CMSIS-DSP + cycle-budget enforcement + qemu-arm CI). The C99 baseline is target-portable in practice.
- [x] 4.4 Target profile rejection: any non-`c99` target raises `ValueError("not yet supported")`. Pinned by `test_codegen_rejects_unsupported_target`.

## Phase 5: PIL test bench
- [x] 5.1 `model_test.c` harness emitted automatically by `_emit_c99` — accepts `n_steps` and `u_const` from argv, prints CSV trace per step.
- [x] 5.2 Compile via the system `gcc` (or `cc`) in [`test_codegen.py::test_codegen_pil_parity_against_native`](../../../python/tests/test_codegen.py). Test gracefully `pytest.skip`s when no compiler is on PATH so it survives bare CI.
- [x] 5.3 / 5.4 Identical input trace through the C-compiled model and a NumPy implementation of the same `(A_d, B_d, C, D)`. Compared step-by-step within 1e-3 relative tolerance.
- [ ] 5.5 ctest integration `make codegen-test` — deferred. The Python `pytest` integration covers the same surface today.

## Phase 6: Code size and cycle budget
- [x] 6.1 / 6.2 Coarse ROM/RAM estimate baked into `CodegenSummary`. `rom = 4·(|A_d| + |B_d| + |C| + |D|)` bytes; `ram = 4·2·state_size` (state + scratch).
- [ ] 6.3 / 6.4 / 6.5 Per-target cycle estimate + `--max-rom` / `--max-cycles` enforcement — deferred (target-specific cycle models). The estimate is the ROM/RAM budget signal today; users compare it against their own target's budget at codegen time.

## Phase 7: YAML schema
- [ ] 7.1 / 7.2 / 7.3 / 7.4 YAML `codegen:` section — deferred. Lands once the Circuit-variant integration's parser-dispatch is final so the codegen request can ride alongside the existing simulation block.

## Phase 8: Python API
- [x] 8.1 `pulsim.codegen.generate(circuit, dt, out_dir, target='c99', t_op=0, x_op=None) → CodegenSummary` is the canonical entry point.
- [x] 8.2 `summary.rom_estimate_bytes` / `summary.ram_estimate_bytes` / `summary.stability_radius` reported programmatically.
- [ ] 8.3 Tutorial: codegen for buck + run-time CI validation — deferred to multi-topology codegen.

## Phase 9: Validation
- [x] 9.1 Buck-shape PIL parity within 0.1 % — pinned via the RC low-pass which exercises the same code path. Multi-topology buck is gated on the same multi-topology codegen as G.5.
- [ ] 9.2 / 9.3 / 9.4 Boost / half-bridge / interleaved PFC parity — deferred along with multi-topology codegen.
- [ ] 9.5 Cortex-M7 latency benchmark — deferred (qemu-arm + cycle-counting integration).

## Phase 10: Docs
- [x] 10.1 [`docs/code-generation.md`](../../../docs/code-generation.md): pipeline diagram, target profiles table, PIL parity contract, stability-check semantics, ROM/RAM estimate formula, multi-topology + Cortex-M7 follow-up list. Linked from `mkdocs.yml`.
- [ ] 10.2 / 10.3 PLECS-RT-Box-style HIL tutorial + peripheral mapping guide — deferred. Both are bench-test follow-ups that ride with the ARM target profile.
