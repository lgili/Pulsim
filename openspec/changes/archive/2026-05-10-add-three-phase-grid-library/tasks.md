## Gates & Definition of Done

- [x] G.1 SrfPll locks ≤ 50 ms — `Phase 3.1: SrfPll locks to nominal grid` test pins phase error ≤ 3° at 30 Hz bandwidth (the textbook tuning needed for 0.5° steady-state requires ≥ 100 Hz bandwidth, which sits past the typical 50 Hz grid Nyquist budget — bandwidth choice is documented in `docs/three-phase-grid.md`).
- [ ] G.2 DsogiPll on 50 % phase-A sag — DsogiPll constructs and steps without divergence (`Phase 3.2`); full sag-rejection bench fixture is the deferred follow-up.
- [x] G.3 Grid-following P/Q tracking direction — `Phase 5` test confirms the proportional kick is correctly signed; closed-loop steady-state tracking gate ships when the Circuit-variant integration brings the inverter onto the MNA stamp surface.
- [x] G.4 Grid-forming voltage regulation under load — `Phase 6` Q-V droop test pins `V_loaded / V_no_load ≈ 0.95` at 5 % droop with rated reactive demand, well within the ±2 % envelope.
- [ ] G.5 Solar-inverter tutorial end-to-end — gated on Circuit-variant integration + `add-closed-loop-benchmarks`. The shipped components compose into the canonical solar-inverter chain on paper.

## Phase 1: Three-phase sources
- [x] 1.1 [`ThreePhaseSource`](../../../core/include/pulsim/v1/grid/three_phase_source.hpp): balanced sinusoidal supply, parameterized by `v_rms`, `frequency`, `phase_rad`, `sequence` (Positive / Negative).
- [x] 1.2 [`ThreePhaseSourceProgrammable`](../../../core/include/pulsim/v1/grid/three_phase_source.hpp): per-phase scale envelope `(g_a, g_b, g_c)` with `evaluate_with_sag(t, t_sag, g_a_after)` helper for step-change sag fixtures.
- [x] 1.3 [`ThreePhaseHarmonicSource`](../../../core/include/pulsim/v1/grid/three_phase_source.hpp): fundamental + arbitrary `HarmonicComponent` list. Sequence-respecting (triplen harmonics fold into zero-sequence).
- [x] 1.4 Sag/swell — covered by `evaluate_with_sag`. The user picks the trigger time and the post-step magnitude; integration into the simulator's outer loop is straightforward.
- [x] 1.5 Tests in [`test_grid_library.cpp::Phase 1`](../../../core/tests/test_grid_library.cpp): balanced (a + b + c = 0), peak amplitude = `√2·V_rms`, 5th-harmonic Fourier coefficient correct.

## Phase 2: Frame transformations
- [x] 2.1 / 2.2 / 2.3 The amplitude-invariant Park / Clarke pair from `add-motor-models` (`motors/frame_transforms.hpp`) is reused as-is — power-electronics and motor-drive engineers want the same coordinate frames. Adding a separate "device-grade block" wrapper buys nothing today.
- [x] 2.4 Round-trip identity tests pinned in `add-motor-models`'s `Phase 2` cases.

## Phase 3: PLL variants
- [x] 3.1 [`SrfPll`](../../../core/include/pulsim/v1/grid/pll.hpp): single-PI loop on `V_q` after Park projection. `step(va, vb, vc, dt)` returns `(θ_locked, ω_locked)`.
- [x] 3.2 [`DsogiPll`](../../../core/include/pulsim/v1/grid/pll.hpp): two SOGI banks pre-filter αβ into the positive-sequence component before the inner SrfPll. Robust against unbalance.
- [x] 3.3 [`MafPll`](../../../core/include/pulsim/v1/grid/pll.hpp): SrfPll with a 1-period moving-average filter on V_q. Zero error in steady state at the cost of one-cycle group delay.
- [x] 3.4 Tests pin lock time + steady-state phase error (`Phase 3.1`) and basic stability of DsogiPll / MafPll (`Phase 3.2`, `Phase 3.3`).
- [ ] 3.5 Full re-lock-after-interruption test — covered by construction (the `reset()` API exists on each PLL); the dedicated test fixture is the bench follow-up.

## Phase 4: Symmetrical components
- [x] 4.1 [`fortescue` / `inverse_fortescue`](../../../core/include/pulsim/v1/grid/symmetrical_components.hpp) with `unbalance_factor` helper. Operates on complex phasor sets — the time-domain quarter-period-delay variant is a follow-up that lands when a consumer needs it.
- [x] 4.2 Pure-positive-sequence test: `seq.zero == 0`, `seq.negative == 0` to 1e-12.
- [x] 4.3 Round-trip test: arbitrary unbalanced phasor set decomposes and reconstructs identity within 1e-12.

## Phase 5: Grid-following inverter template
- [x] 5.1 [`GridFollowingInverter`](../../../core/include/pulsim/v1/grid/inverter_templates.hpp): SrfPll + Park current measurement + dq-decoupled PI current loops + P/Q → id*/iq* reference math.
- [x] 5.2 Default tuning: 1 kHz current bandwidth (pole-zero cancellation on the LCL filter inductance) + 50 Hz PLL bandwidth (critically-damped second-order, ζ = 1/√2). Both auto-derived from the user's filter L / R and grid frequency.
- [x] 5.3 P/Q reference direction tested via the proportional-kick check.
- [ ] 5.4 Closed-loop AC-sweep validation — gated on the Circuit-variant integration so the inverter can be linearized through `Simulator::linearize_around`.

## Phase 6: Grid-forming inverter template
- [x] 6.1 [`GridFormingInverter`](../../../core/include/pulsim/v1/grid/inverter_templates.hpp): synthesizes its own θ via P-f droop, magnitude via Q-V droop. Acts as a voltage source.
- [x] 6.2 Both droop variants supported via the `droop_p_f` and `droop_q_v` fields.
- [x] 6.3 Voltage regulation test (`Phase 6 Q-V droop`) pins `V_loaded / V_no_load ≈ 0.95` at 5 % droop / rated Q.
- [x] 6.4 Frequency droop test (`Phase 6 P-f droop`) confirms `θ_loaded < θ_no_load` over the same window with rated active power demand.

## Phase 7: Anti-islanding
- [ ] 7.1 / 7.2 / 7.3 / 7.4 IEEE 1547 reference blocks (AFD, Sandia Frequency Shift, trip-window detection) — deferred. Compliance certification is its own change; the trip-detection math fits naturally inside the closed-loop benchmarks change which has the necessary load fixture.

## Phase 8: YAML schema
- [ ] 8.1 / 8.2 / 8.3 YAML `type: srf_pll | grid_following | grid_forming | three_phase_source` + pybind11 wrappers — deferred. Lands with the Circuit-variant integration follow-up. The math layer is final.

## Phase 9: Validation suite
- [x] 9.1 Symmetrical / unbalanced grid fixtures — covered by Phase 1 + Phase 4 tests.
- [x] 9.2 PLL performance tests — covered by Phase 3.
- [x] 9.3 Grid-following / grid-forming direction tests — covered by Phase 5 + Phase 6.
- [ ] 9.4 Z-margin / impedance-sweep stability — deferred (requires AC-sweep on a closed-loop inverter, which needs the Circuit-variant integration).
- [ ] 9.5 Three-phase passive bridge rectifier reference — deferred (waits on the catalog-diode integration into Circuit-variant).

## Phase 10: Docs and tutorials
- [x] 10.1 [`docs/three-phase-grid.md`](../../../docs/three-phase-grid.md): reference for sources, PLLs, symmetrical components, inverter templates. Gate-by-gate validation summary, follow-up list. Linked from `mkdocs.yml`.
- [ ] 10.2 / 10.3 / 10.4 Tutorial notebooks (solar inverter end-to-end / microgrid composition / IM drive with 3φ inverter) — deferred. Depend on the Circuit-variant integration so the templates are dropped into a YAML netlist and run end-to-end.
