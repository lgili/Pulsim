## Why
PSIM and PLECS ship dedicated blocks for the standard three-phase control workflow — Clarke / Park transforms, a phase-locked loop, and space-vector modulation. With these primitives, a designer can build a vector-controlled inverter in a handful of YAML lines instead of hand-wiring trigonometry through `math_block` + `gain` chains.

Pulsim's virtual control-block infrastructure (Phase 19 onward) makes this addition cheap: each new block is a single `else if` in `execute_mixed_domain_step` plus a parser registration. Downstream blocks already know how to consume channel values (`duty_from_channel:` etc.), so chaining `pll → park → controller → inverse_park → svm → gates` works without any electrical-domain change.

## What Changes
Add six new virtual control block types (four conceptual capabilities, since forward/inverse are paired):

- `clarke_transform` — Clarke (abc → α β γ). Outputs `<name>.alpha`, `<name>.beta`, `<name>.gamma`.
- `inverse_clarke_transform` — Park inverse (α β γ → abc). Outputs `<name>.a`, `<name>.b`, `<name>.c`.
- `park_transform` — Park (α β → d q 0), takes θ from a channel. Outputs `<name>.d`, `<name>.q`, `<name>.zero`.
- `inverse_park_transform` — (d q → α β), takes θ. Outputs `<name>.alpha`, `<name>.beta`.
- `pll` — locks to a sinusoidal input via a PI loop on the q-axis projection. Outputs `<name>.theta`, `<name>.omega`, `<name>.lock_error`.
- `svm` — Space-Vector Modulation: takes (α, β) reference + DC bus voltage, emits three half-bridge duties `<name>.d_a`, `<name>.d_b`, `<name>.d_c`.

Plus three example benchmarks:

- `three_phase_dq_decoupling` — Open-loop dq-transform demonstration with synthetic 3-phase sine source.
- `pll_grid_sync` — PLL locks to a 60 Hz grid sine source, reports θ and lock error.
- `vector_control_open_loop` — Full chain: grid sine → Clarke → Park → identity controller → inverse Park → SVM → gate duties. Validates the full vector-control wiring composes correctly.

## Impact
- Affected specs: `kernel-v1-core` (new virtual block types) and `benchmark-suite` (3 new benchmarks).
- Affected code: `core/src/v1/yaml_parser.cpp` (alias + arity registration), `core/include/pulsim/v1/runtime_circuit.hpp` (Phase 2 evaluation), `python/bindings.cpp` (likely no change — channel values already exposed), new YAML circuits + baselines.
- Requires C++ rebuild + Python extension re-install (matches the existing build flow).
- Backward-compatible — existing benches don't reference the new types.
