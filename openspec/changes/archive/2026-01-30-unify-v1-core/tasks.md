## 1. Kernel Unification
- [x] 1.1 Create `pulsim/v1/simulation.hpp` + `core/src/v1/simulation.cpp` for a full v1 simulator
- [x] 1.2 Wire DC to `DCConvergenceSolver` with strategy ordering defaults
- [x] 1.3 Implement transient loop with LTE/PI controller and BDF order control
- [x] 1.4 Port switch-event detection and bisection from legacy simulator
- [x] 1.5 Integrate loss accumulation using `pulsim/v1/losses.hpp`
- [x] 1.6 Add streaming/progress callbacks and determinism guarantees

## 2. YAML Netlist (yaml-cpp)
- [x] 2.1 Add `yaml-cpp` via FetchContent in `CMakeLists.txt`
- [x] 2.2 Define YAML schema v1 with mandatory `version`
- [x] 2.2.1 Require `schema` identifier (e.g., pulsim-v1)
- [x] 2.3 Implement YAML parser that builds `pulsim::v1::Circuit`
- [x] 2.4 Support waveforms, models/use, inheritance overrides, and SI suffix parsing
- [x] 2.5 Strict validation + diagnostics for unsupported fields

## 3. Integration & Deprecation
- [x] 3.1 Update examples and docs to YAML
- [x] 3.2 Update Python bindings to use `v1::Simulator` (simplified API)
- [x] 3.3 Remove JSON netlist path and legacy simulator usage

## 4. Validation Gates
- [x] 4.1 Run validation suite (levels 1-4) against new kernel
- [x] 4.2 Add regression tests for event timing and loss accumulation
- [x] 4.3 Document migration guide (JSON -> YAML, old API -> simplified)
