## Why
The backend runtime recently passed open-loop stress testing on complex topologies (like LLC, PFC Boost, and Active Clamp Forward). However, a significant portion of real-world power electronics relies on closed-loop regulation. Closed-loop circuits combine stiff high-frequency switching dynamics with much slower control loop dynamics (PI controllers, integrators, and PWM comparators), which often cause solvers to fail or timestep poorly. We need to validate that PulsimCore can simulate mixed-domain (analog + control) closed-loop topologies without failing or taking excessively long.

## What Changes
- Add `ll14_buck_closed_loop.yaml` to `benchmarks/local_limit/circuits`.
- Add `ll15_boost_closed_loop.yaml` to `benchmarks/local_limit/circuits`.
- Add `ll16_flyback_closed_loop.yaml` to `benchmarks/local_limit/circuits`.
- Use the `pi_controller` and `pwm_generator` virtual components to regulate duty cycles based on voltage feedback loops.
- Register these in `benchmarks/local_limit/benchmarks_local_limit.yaml` under `difficulty: 9-closed_loop`.

## Impact
- Affected specs: `benchmark-suite`
- Affected code: `benchmarks/local_limit/circuits` and `benchmarks/local_limit/benchmarks_local_limit.yaml`.
