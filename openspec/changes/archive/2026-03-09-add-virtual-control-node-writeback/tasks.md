## 1. Control Node Writeback Contract
- [x] 1.1 Define control-node writeback and resolution order in kernel spec delta.
- [x] 1.2 Document deterministic error behavior for algebraic control loops.

## 2. Kernel Runtime Implementation
- [x] 2.1 Add per-step control-node registry separate from electrical node values.
- [x] 2.2 Resolve control inputs using control-node registry, then electrical nodes.
- [x] 2.3 Write control outputs to channels and control-node registry.
- [x] 2.4 Ensure PWM duty reads post-control channel registry.
- [x] 2.5 Detect and fail deterministic algebraic loops without state.

## 3. Tests and Examples
- [x] 3.1 Add unit test for control-node cascade (GAIN -> GAIN) non-zero output.
- [x] 3.2 Add regression test for cascaded PI chain with PWM duty varying.
- [x] 3.3 Update example 14 boost cascaded control demo to validate convergence.

## 4. Validation
- [x] 4.1 Run targeted kernel and python tests for control cascade behavior.
- [x] 4.2 Run openspec validation.
