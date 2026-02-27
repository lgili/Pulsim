## 1. Specification & Benchmarks
- [ ] 1.1 Create `ll14_buck_closed_loop.yaml` using a `pi_controller` to regulate output voltage via a `pwm_generator`.
- [ ] 1.2 Create `ll15_boost_closed_loop.yaml` using a `pi_controller` and a `pwm_generator` to step up voltage under closed-loop control.
- [ ] 1.3 Create `ll16_flyback_closed_loop.yaml` with a coupled inductor (using a T-model equivalent) and closed-loop feedback.
- [ ] 1.4 Register the new circuits in `benchmarks/local_limit/benchmarks_local_limit.yaml` under difficulty `9-closed_loop`.

## 2. Validation
- [ ] 2.1 Run `make benchmark-local-limit MAX_RUNTIME=60` and verify the backend can solve the mixed-domain circuits without crashing.
- [ ] 2.2 Rerun `make test` to ensure no basic sanity checks are broken by any solver tweaks.
