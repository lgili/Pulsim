## Why
The runtime backend solvers and heuristics need to be stressed by more complex and difficult circuits. Users are expected to simulate harsh topologies and the local limit benchmarks currently don't test these edge cases sufficiently. By adding complex stress test benchmarks we can ensure our backend remains robust.

## What Changes
- Add new `ll11_...` to `ll15_...` circuits under `benchmarks/local_limit/circuits` demonstrating difficult convergence situations.
- Append these circuits to `benchmarks/local_limit/benchmarks_local_limit.yaml` with increased `difficulty` ratings.
- Ensure the backend configuration is updated if any of these circuits expose inherent flaws in the solver defaults.

## Impact
- Affected specs: `benchmark-suite`
- Affected code: `benchmarks/local_limit/circuits`, `benchmarks/local_limit/benchmarks_local_limit.yaml`, and potentially `core/src/v1/simulation.cpp` / auto-profiling logic.
