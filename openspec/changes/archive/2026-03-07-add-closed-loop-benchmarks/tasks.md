## 1. Specification & Benchmarks
- [x] 1.1 Create `ll14_buck_closed_loop.yaml` using a `pi_controller` to regulate output voltage via a `pwm_generator`.
- [x] 1.2 Create `ll15_boost_closed_loop.yaml` using a `pi_controller` and a `pwm_generator` to step up voltage under closed-loop control.
- [x] 1.3 Create `ll16_flyback_closed_loop.yaml` with a coupled inductor (using a T-model equivalent) and closed-loop feedback.
- [x] 1.4 Register the new circuits in `benchmarks/local_limit/benchmarks_local_limit.yaml` under difficulty `9-closed_loop`.

## 2. Validation
- [x] 2.1 Run `benchmarks/local_limit_suite.py` with `--only ll14_buck_closed_loop ll15_boost_closed_loop ll16_flyback_closed_loop --mode both --max-runtime-s 60` and verify the mixed-domain closed-loop circuits complete without crashes/timeouts.
- [x] 2.2 Rerun `make test` to ensure no basic sanity checks are broken by solver/control-loop changes.
