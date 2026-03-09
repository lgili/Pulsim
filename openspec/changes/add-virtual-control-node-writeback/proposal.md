## Why
Control blocks currently emit only channels without feeding downstream control nodes, which breaks cascaded control chains (e.g., PI -> SUM -> PI -> PWM). Users must work around this with C_BLOCK, which is unnecessary for typical control topologies.

## What Changes
- Add a control-node writeback registry for virtual control outputs, separate from electrical node values.
- Resolve control block inputs from control-node values first, falling back to electrical node voltages.
- Enforce deterministic evaluation order for control blocks with explicit algebraic loop detection and typed errors.
- Keep existing discrete/hold semantics and PWM duty sourcing from control channels.

## Impact
- Affected specs:
  - kernel-v1-core
- Affected code:
  - core/include/pulsim/v1/runtime_circuit.hpp
  - core/src/v1/runtime_module_adapters.cpp
  - core/tests/test_v1_kernel.cpp
  - python/tests/test_runtime_bindings.py
  - examples/14_boost_pi_cascaded_control_demo.pulsim
