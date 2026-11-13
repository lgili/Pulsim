## Why

Pulsim's passives are ideal, and it has no line model.

Verified on this branch (review `raw/models.md`, inventory row 1 and
ranked item 13; synthesis M12):

- `Capacitor::Params` is `{C}` and `Inductor::Params` is `{L}`
  (`core/include/pulsim/models/capacitor.hpp:41-43`,
  `models/inductor.hpp`); `add_capacitor` and `add_inductor` take one
  value and an initial condition
  (`builder/circuit_builder.hpp:913-943`). A search of the models, the
  builder and the Python package for ESR or ESL finds nothing, and none
  for a transmission line.
- The only voltage-dependent capacitor is the junction law
  `C0/(1 + v/V0)^m` (`models/nonlinear_capacitor.hpp`), which is not
  the shape of a ceramic capacitor's DC-bias curve (even in `v`,
  falling with `|v|`).
- Winding resistance is a constant; nothing represents its rise with
  frequency.

What it costs, with analytic examples (declared example values, not
parts):

- **Ripple.** A buck with 2 A peak-to-peak inductor ripple at 100 kHz
  into 100 µF: the ideal capacitor gives `ΔI/(8·f·C)` = 25 mV. With an
  ESR of 50 mΩ the resistive ripple alone is `ESR·ΔI` = 100 mV. The
  ideal model under-reports the ripple by a factor of four or more, and
  reports zero capacitor loss.
- **DC bias.** A ceramic capacitor that has lost 80 % of its value at
  its working voltage filters five times worse than its nominal value
  suggests.
- **Cables.** The obvious workaround for a line, a ladder of L and C
  sections, is wrong at fast edges. Measured (`probes/pi_line_step.py`,
  50 Ω, 100 ns line, matched source, open end, ideal step — the exact
  answer is 1.000 V with no overshoot):

  | Sections | Far-end peak | 50 % crossing / delay |
  |---|---|---|
  | 10 | 1.264 V | 1.044 |
  | 20 | 1.268 V | 1.028 |
  | 50 | 1.271 V | 1.015 |
  | 100 | 1.273 V | 1.010 |

  The overshoot stays at 27 % however many sections are used. With a
  20 ns edge (one fifth of the delay) 50 sections give 5.4 % and 20
  sections 10.2 %. A study of motor-terminal overvoltage on a long
  cable reads its answer from exactly this peak.

Other tools (general knowledge, not checked against current
documentation): SPICE has a lossless line element and vendor capacitor
models with ESR, ESL and bias dependence; PLECS and PSIM offer
transmission-line or cable components and series-resistance fields on
their passives.

## What Changes

- **ESR and ESL on the capacitor; series resistance and parallel
  capacitance on the inductor**:
  `add_capacitor(name, from, to, C, c0=None, *, esr=0.0, esl=0.0)`,
  `add_inductor(name, from, to, L, i0=None, *, esr=0.0, c_par=0.0)`
  [Ω, H, F]. Non-zero values expand into a composite of existing
  elements with internal nodes; zero values are bit-identical to today.
  ESR loss appears under the component in `device_loss_summary`.
- **Capacitance against DC bias**: an odd charge table `Q(v)` with even
  `C(v) = dQ/dv` (the symmetry of `FluxTable`), through the charge-law
  variant of `NonlinearCapacitor`; `fit_mlcc_dc_bias(v_bias, c_ratio,
  C_nominal)` builds it from a datasheet "capacitance change against DC
  voltage" curve.
- **AC resistance**: `fit_rac_ladder(f, R_ac, *, L, R_dc=None,
  n_cells=4, tol=0.05)` fits a passive ladder (cells of `R_k ∥ L_k` in
  series with `R_dc` and the remaining inductance) to a winding's
  `R_ac(f)` curve; `add_inductor(..., rac=ladder)` builds it.
- **Transmission line**:
  `add_transmission_line(name, k_pos, k_neg, m_pos, m_neg, *, Z0,
  delay, R=0.0, model="bergeron", sections=None)`.
  - `model="pi"`: `sections` lumped cells of existing elements; works
    in every analysis; `recommend_line_sections(delay, t_rise)` gives
    the count.
  - `model="bergeron"`: a kernel device with the exact travelling-wave
    delay (constant conductance `1/Z0` at each end plus a history
    source), series resistance lumped as R/4, R/2, R/4.
- **YAML**: `esr`, `esl`, `c_par`, `rac` keys on `capacitor` and
  `inductor`; type `transmission_line`; `nonlinear_capacitor` with a
  `table` law.
- No **BREAKING** change.

## Impact

- Affected specs: `device-models`.
- Affected code: `builder/circuit_builder.hpp`, new
  `models/transmission_line.hpp` and `pwl/transmission_line_history.hpp`,
  `pwl/{device_pool,assemble,dc_assemble}.hpp`,
  `solver/{run_transient,trbdf2_transient}.hpp`, `yaml/loader.hpp`,
  `python/bindings.cpp`, new `python/pulsim/passives.py` (fitters),
  `python/pulsim/losses.py`, stubs, docs.
- Dependencies: none required (the roadmap lists none). Related:
  - `add-power-mosfet-datasheet-device`, step 1 — the charge-law
    variant of `NonlinearCapacitor`. The DC-bias table is its
    odd-symmetric case; whichever change lands first brings the
    variant.
  - `add-engine-conformance-matrix` — rows; the Bergeron line is
    refused where a delay cannot be represented (`dsed`, the exact
    propagator, periodic steady-state shooting, AC).
  - `add-fixed-step-resolution-diagnostics` — warns when the step does
    not resolve an ESL resonance or a line delay.
  - `add-unified-device-interface` — not needed: the composites are
    existing linear elements, and the Bergeron line is a constant stamp
    with a history source, not an `f + dq/dt` device.
- ESR/ESL, the R_ac ladder and the π-section line can ship on today's
  structure immediately; the DC-bias table after the charge-law
  variant; the Bergeron line is the only new kernel kind.
