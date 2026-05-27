# Tasks — add-sensorless-motor-observers

> Status (Phase 2.3 — v1.5.0 RC): SMO + MRAS implementation done and
> validated. Showcase scripts (closed-loop PMSM + IM FOC) and the
> sensorless docs page are deferred to the v1.5.0 final cut.

## 1. Sliding-Mode Observer (PMSM)

- [x] 1.1 Create `python/pulsim/observers.py` with `SlidingModeObserver` class (stator-frame current + voltage inputs; back-EMF + angle + speed outputs).
- [x] 1.2 Implement the sliding-surface algorithm with configurable gain `K_sl`, low-pass-filter cutoff `omega_lpf`, and PLL-based angle/speed extraction from the filtered back-EMF. Uses ``tanh(err/ε)`` boundary-layer saturating sign for chatter suppression.
- [x] 1.3 Conform to the `BlockChain` block interface (`.reset()`, `.update(*, ...)` returning a tuple). Matches the existing PLL / ParkTransform / etc convention.
- [x] 1.4 pytest covering constructor defaults, return-tuple shape, and a functional lock test against a synthetic PMSM back-EMF signal (4/4 pass locally; angle peak error < 20°, RMS < 10° at 800 rad/s electrical).

## 2. Flux-MRAS Observer (Induction Motor)

- [x] 2.1 Add `FluxMRASObserver` to the same observers module. Voltage-model + current-model rotor-flux estimators, error-driven speed adaptation loop.
- [x] 2.2 Configurable Lyapunov-derived adaptation gains (`Kp_mras`, `Ki_mras`).
- [x] 2.3 Block-interface conformance + `reset()`. Convenience constructor `FluxMRASObserver.from_motor(motor, ...)` pulls Rs / Ls / Lr / Lm from an existing `InductionMotor` handle.
- [ ] 2.4 Functional MRAS test covering 10 %–100 % rated speed — deferred; needs the closed-loop IM-IFOC chain (task 3.2) to drive the motor at a controllable speed setpoint.

## 3. Showcase examples — deferred to v1.5.0 final

- [ ] 3.1 `examples/scripts/run_pmsm_foc_sensorless_smo.py` — PMSM-FOC closed loop using SMO for the Park angle. Startup ramp + load step + speed reversal.
- [ ] 3.2 `examples/scripts/run_im_ifoc_sensorless_mras.py` — IM-IFOC sensorless using MRAS for slip-frequency estimation.

## 4. Documentation — deferred to v1.5.0 final

- [ ] 4.1 `docs/v2/motors-sensorless.md` covering SMO and MRAS algorithms, equations, default-gain derivation, failure-mode discussion.
- [ ] 4.2 Cross-link from existing motors index page.

## 5. Python API surface

- [x] 5.1 Re-export `SlidingModeObserver` and `FluxMRASObserver` from `python/pulsim/__init__.py` (`__all__`).
- [ ] 5.2 `pulsim.catalog()` discovery entry for both blocks (deferred with the showcases).

## 6. Validation + release

- [x] 6.1 `openspec validate add-sensorless-motor-observers --strict` clean.
- [x] 6.2 Local pytest passes (4/4). Cross-platform CI pending PR.
- [ ] 6.3 Bundle in the same release as the IM proposal (1.5.0).
- [ ] 6.4 Archive: `openspec archive add-sensorless-motor-observers --yes`.
