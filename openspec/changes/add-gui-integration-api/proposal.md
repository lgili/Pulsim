## Why

The PulsimGui application requires enhanced APIs in the Pulsim core library to support:
- Real-time simulation progress callbacks with detailed metrics
- Pause/resume simulation capability
- Interactive parameter modification during simulation setup
- Richer metadata for components and simulation results
- Better error reporting for GUI feedback

## What Changes

- **ADDED** Enhanced simulation control with pause/resume capability
- **ADDED** Progress callbacks with detailed metrics (time, iterations, convergence)
- **ADDED** Component metadata API for GUI display (descriptions, valid ranges, units)
- **ADDED** Schematic position hints in circuit representation
- **ADDED** Validation API with detailed error reporting
- **ADDED** Result streaming with configurable decimation
- **MODIFIED** SimulationResult to include more metadata
- **MODIFIED** Python bindings to expose new APIs

## Impact

- Affected specs: New `gui-integration` capability
- Affected code:
  - `core/include/pulsim/simulation.hpp` - pause/resume, callbacks
  - `core/include/pulsim/circuit.hpp` - position hints, metadata
  - `core/src/simulation.cpp` - implementation
  - `python/bindings.cpp` - new bindings
- Backward compatible: Yes (new APIs only)
