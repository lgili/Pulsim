## Why

Hardware-in-the-Loop (HIL) is the highest-margin segment in power-electronics simulation: PLECS RT Box, OPAL-RT eMEGAsim, Typhoon HIL, dSPACE SCALEXIO. Engineers validate firmware against a hard-real-time simulator before connecting actual hardware. To enter this market, Pulsim must generate **deterministic, fixed-step C99 code** from a netlist, executable on embedded targets at sub-microsecond latency.

The technical pattern is well-defined:
1. Reduce the (PWL-mode) circuit to a state-space form A, B, C, D per topology.
2. Pre-compute discrete-time matrices `A_d = expm(A·Ts)`, `B_d = ∫_0^Ts expm(A·τ) B dτ` for each topology at the target step `Ts`.
3. Generate C code that is just: `x = A_d[topology] x + B_d[topology] u; y = C x + D u;` — no allocation, no division, no transcendental.
4. Topology lookup is a switch on the bitmask.

This produces FOC-grade code (∼1 µs step) suitable for ARM Cortex-M7/A9, Zynq, RISC-V soft-cores. With proper drivers it runs on a PLECS RT Box equivalent or a Cortex-M with TI C2000-class peripherals.

This change builds the codegen pipeline; the runtime drivers (PWM peripherals, A/D converters) are out of scope and left to integrators.

## What Changes

### Code Generation Pipeline
- New CLI: `pulsim codegen <netlist.yaml> --target c99 --step 1e-6 --out gen/`.
- Output:
  - `pulsim_model.h` — C99 header with state, input, output structs.
  - `pulsim_model.c` — `model_step(state*, input*, output*)` function, plus topology-resolution helper.
  - `pulsim_topologies.c` — pre-computed `A_d[k]`, `B_d[k]`, `C[k]`, `D[k]` matrices per topology.
  - `pulsim_io.h` — peripheral mapping stubs (declared, user-implemented).
  - `pulsim_makefile` — example build for arm-none-eabi-gcc.
- Generated code targets C99 (no C++), no malloc, no global state, all matrices `static const`.

### Topology Pre-Computation
- For each PWL-mode topology bitmask, compute discrete-time matrices via Padé approximation of `expm`.
- Bound: at most `2^k` topologies for k switching devices; user can declare a "reachability" subset to reduce code size.

### Target Profiles
- `--target c99` — portable, no specific peripherals.
- `--target arm-cortex-m7` — adds CMSIS-DSP fixed-point optional, alignment hints.
- `--target zynq-baremetal` — generates hardware-accelerated matrix-vector via custom IP (stub interface).
- `--target nvidia-jetson` (stretch) — CUDA kernel for parallel topology pre-evaluation.

### Step-Size Validation
- Pre-codegen check: for each topology, compute `eig(A)`. Largest `|eig|` must satisfy `|λ_max · Ts| ≤ 0.5` for stability under Tustin/expm. If not, codegen fails with a recommendation: smaller `Ts` or simplified topology.

### PIL (Processor-In-the-Loop) Test Bench
- After codegen, optional `--pil-bench` produces a Python test harness that compiles the C code with `gcc`/`arm-gcc`, runs identical input traces through both Pulsim native and the generated C, and compares output trace within tolerance.
- This is the regression gate for codegen correctness.

### Code-Size and Cycle Budget
- Per-target estimates: ROM, RAM, peak cycles per `model_step()`.
- Reported in codegen summary; hard fail if exceeds user-set `--max-rom`, `--max-ram`, `--max-cycles`.

### YAML Schema
- New top-level `codegen:` section optional in netlist:
```yaml
codegen:
  target: c99
  step: 1e-6
  reachable_topologies: auto | explicit_list
  outputs: [vout, iload]
  max_rom_kb: 64
  max_ram_kb: 8
```

## Impact

- **New capability**: `code-generation`.
- **Affected specs**: `code-generation` (new), `python-bindings` (codegen API), `netlist-yaml` (codegen section).
- **Affected code**: new `python/pulsim/codegen/` (codegen logic, mostly Python with some C++ helpers for `expm`), new templates under `templates/codegen/`, CLI integration.
- **Performance**: not in real-time runtime (offline), but generated code targets sub-µs step.

## Success Criteria

1. **Buck PIL parity**: generated C code for a 100 kHz buck reproduces native Pulsim within 0.1% on output voltage and inductor current over 100 ms simulated.
2. **Cortex-M7 step latency**: generated `model_step()` for the same buck runs in ≤500 ns on a 240 MHz Cortex-M7 reference target.
3. **Code-size budget**: ROM ≤8 KB, RAM ≤512 B for the buck.
4. **Stability check**: codegen fails with informative diagnostic if `|λ_max · Ts| > 0.5`.
5. **Topology coverage**: 16-topology PFC interleaved successfully generates code with reasonable size budget.
