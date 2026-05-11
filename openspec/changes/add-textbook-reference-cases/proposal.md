## Why
Today's regression matrix has Pulsim passing **against itself** (baseline-against-Pulsim). That validates determinism but not absolute correctness. PSIM and PLECS earn credibility because their example libraries reproduce **published** textbook waveforms — engineers can open the PSIM manual, see "Figure 4.5: Buck steady-state from Erickson & Maksimovic § 2.3", and verify the simulator delivers the same numbers.

We need that authority layer.

## What Changes
- Create a new benchmark category `textbook_*` referencing well-known published examples with documented expected values:
  - `textbook_erickson_buck_3_1` — Erickson & Maksimovic § 3.1 (CCM buck steady-state). Compare V_out, ΔI_L, and ΔV_C against the values printed in the textbook.
  - `textbook_erickson_boost_6_2` — § 6.2 boost averaged-model verification: predicted V_out for given duty.
  - `textbook_erickson_buckboost_6_3` — § 6.3 inverting buck-boost.
  - `textbook_mohan_pwm_rectifier_8_4` — Mohan Power Electronics § 8.4 single-phase PWM rectifier.
  - `textbook_kassakian_4_2` — Kassakian, Schlecht, Verghese § 4.2 three-phase diode rectifier RMS / ripple analysis.
- Each YAML carries a `published_values:` block (V_out, ΔI_L, η, etc.) and the validation type is `published`: the runner asserts each measured KPI is within ±2 % of the published number.
- Document each benchmark inline with the textbook citation (chapter, section, figure number, equation number).

## Impact
- Affected specs: `benchmark-suite` (new `published` validation type).
- Affected code: new YAML circuits + new validation handler in `benchmark_runner.py` + a small library of analytical reference functions.
- This is the test category that proves **measurement parity** — without it Pulsim sounds like "another transient simulator".
