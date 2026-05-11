## 1. Audit existing infrastructure
- [ ] 1.1 Document what `python/pulsim/codegen/` currently emits (per-control-block C, full controller, or just signatures).
- [ ] 1.2 Document what `python/pulsim/fmu/` currently exports (FMU for what scope of model — controller, plant, both).
- [ ] 1.3 List the gaps that need closing for an end-to-end co-sim test.

## 2. C-codegen co-sim test
- [ ] 2.1 Implement `python/tests/test_cosim_pi_buck_c_codegen.py`: take `cl_buck_pi.yaml`, extract the PI controller's parameters, emit C source, compile via `subprocess.run(['cc', ...])`, load the .so via `cffi`, and use it in a Python loop that calls `Pulsim.step()` between controller invocations.
- [ ] 2.2 Validate: the resulting V(out) trace matches the all-in-simulator baseline within 1 % at every sample.

## 3. FMU export co-sim test
- [ ] 3.1 Implement `python/tests/test_cosim_pi_buck_fmu.py`: export the PI controller as an FMU via the existing `fmu` module; consume via FMPy in a co-simulation loop.
- [ ] 3.2 Validate: V(out) trace matches within 1 %.

## 4. Python co-sim driver
- [ ] 4.1 `python/examples/runtime_gain_tuning.py`: drive a Pulsim simulation step-by-step, modify the PI gains at runtime based on observed performance, and re-validate the response.

## 5. Documentation
- [ ] 5.1 Document the workflow in `docs/COSIMULATION.md`, including the C-compiler dependency and how to debug a mismatch between the in-sim and generated-code outputs.
- [ ] 5.2 Add a section in `README.md` highlighting that Pulsim supports controller code generation (C + FMU) with validated round-trip parity.
