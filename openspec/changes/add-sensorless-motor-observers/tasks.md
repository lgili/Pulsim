# Tasks — add-sensorless-motor-observers

## 1. Sliding-Mode Observer (PMSM)

- [ ] 1.1 Create `python/pulsim/motors/observers.py` with `SlidingModeObserver` class (stator-frame current + voltage inputs; back-EMF + angle + speed outputs).
- [ ] 1.2 Implement the sliding-surface algorithm with configurable gain `K_sl`, low-pass-filter cutoff `omega_lpf`, and PLL-based angle/speed extraction from the filtered back-EMF.
- [ ] 1.3 Conform to the `BlockChain` block interface (`.step(...)`, named inputs/outputs, `.reset()`).
- [ ] 1.4 pytest covering steady-state lock, speed-reversal hand-off, low-speed degradation flag.

## 2. Flux-MRAS Observer (Induction Motor)

- [ ] 2.1 Add `FluxMRASObserver` to the same observers module. Voltage-model + current-model rotor-flux estimators, error-driven speed adaptation loop.
- [ ] 2.2 Configurable Lyapunov-derived adaptation gains.
- [ ] 2.3 Block-interface conformance + `reset()`.
- [ ] 2.4 pytest covering 10 %–100 % rated speed tracking. Skip until `add-induction-motor-squirrel-cage` lands.

## 3. Showcase examples

- [ ] 3.1 `examples/scripts/run_pmsm_foc_sensorless_smo.py` — PMSM-FOC closed loop using SMO for the Park angle. Startup ramp + load step + speed reversal.
- [ ] 3.2 `examples/scripts/run_im_ifoc_sensorless_mras.py` — IM-IFOC sensorless using MRAS for slip-frequency estimation (depends on IM device).

## 4. Documentation

- [ ] 4.1 `docs/v2/motors-sensorless.md` covering SMO and MRAS algorithms, equations, default-gain derivation, failure-mode discussion.
- [ ] 4.2 Cross-link from existing motors index page.

## 5. Python API surface

- [ ] 5.1 Re-export `SlidingModeObserver` and `FluxMRASObserver` from `python/pulsim/__init__.py` (`__all__`).
- [ ] 5.2 `pulsim.catalog()` discovery entry for both blocks.

## 6. Validation + release

- [ ] 6.1 `openspec validate add-sensorless-motor-observers --strict` clean.
- [ ] 6.2 CI green on all platforms.
- [ ] 6.3 Bundle in the same release as the IM proposal (1.5.0).
- [ ] 6.4 Archive: `openspec archive add-sensorless-motor-observers --yes`.
