# Decisions — add-permeance-magnetic-domain

Implementation decisions taken while applying the change (2026-11-18).

1. **No new device kinds.** A winding is `add_gyrator(g = 1/N)`, a
   permeance and an air gap are `add_capacitor(C = P)`, a magnetic
   resistance is `add_resistor(R = R_m)`, a saturable permeance is a
   `NonlinearCapacitor` with a new charge law. Every engine, the IR, the
   YAML round trip, the conformance table and the AC sweep already handle
   those kinds, so the magnetic domain inherits their support and their
   refusals (a saturable permeance is refused by name on dsed / bdf1
   exactly as a `NonlinearCapacitor`). The design's element table maps
   one to one onto these kinds.

2. **The magnetic-node registry lives in the `DevicePool`**
   (`mark_flux_node`, `flux_nodes`, plus per-branch geometry for the B /
   H outputs), not in the builder: preflight analysis and the DC assembly
   receive only `(graph, pool)`. It is copied with the builder, hashed by
   the IR only when non-empty (every existing hash is unchanged), and
   written by the IR YAML emitter as `magnetic_nodes` /
   `magnetic_geometry` (read back by `from_yaml`). The IR's domain-tagged
   nodes (add-immutable-circuit-ir) can replace it later.

3. **Saturable permeance law = `ChargeLawKind::Tanh`** on the existing
   fields: `cgd_max` = P(0) = μ_r·μ₀A/l, `cgd_min` = μ₀A/l, `V0` = l·H₀.
   Q = C_min·v + (C_max − C_min)·V0·tanh(v/V0) is exactly
   Φ(F) = μ₀A·[F/l + M_s·tanh(F/(l·H₀))]. Reusing fields keeps the IR
   field table, its hashes and the emitter schema unchanged; only the law
   enum gained a value. sech² is evaluated as 4e^{−2|x|}/(1+e^{−2|x|})²
   (no cosh overflow, no 1 − tanh² cancellation).

4. **Monotonicity at 10× the knee is tested as non-decreasing at 1 pA-turn
   and strictly increasing at 1 nA-turn.** At 10× the knee P = μ₀A/l, so a
   1e-12 A step changes Φ by 1.3e-21 Wb against Φ ≈ 2.7e-5 Wb, 5e-17
   relative — below half an ulp. Strict increase there is not
   representable in double precision; task 3.1's wording is met at zero
   and at the knee (strict) and as closely as doubles allow at 10×.

5. **DC flux rows are linear-only.** `charge_rows(graph, pool)` marks the
   magnetic nodes whose incident branches are permeances, magnetic
   resistances and winding magnetic ports only (no other port
   reference), grouped through the resistances and the winding ports.
   A group with such an edge to any other node or to gnd is determined
   by its KCL rows (at DC a resistance shorts the MMF across it; a
   winding fixes it) and gets no flux row. Otherwise the group's lowest row also
   carries Σ±C·(v_from − v_to) = 0: the resistor terms sum to zero over
   the group, so the other rows' KCL makes them vanish there (a group
   of one node has no resistor terms). The conductance floor skips every
   node of a group (gmin-stepping starts at 1e-2 S against permeances
   of ~1e-7 H). Found by the checkpoint review: the first version
   cleared every node a resistance touched, and an internal R_m pair
   then took the floor's answer (F = 0) silently; the final review found
   the same for a winding whose magnetic port touches no gnd (the floor
   split its MMF ±N·i/2).
   A saturable permeance that touches such a node makes the DC operating
   point throw by name: its flux row needs the Newton DC refresh, which
   is being rebuilt by add-unified-device-interface. A saturable
   permeance directly across a winding (no internal node) is open at DC
   like every charge device and works. Φ₀ is 0 (initial conditions on
   permeances are not used by the DC rows). Circuits with no magnetic
   node assemble exactly as before (the row list is empty and
   `stamp_gmin` runs unchanged).

6. **Preflight**: a permeance on a magnetic node counts as a DC path
   (the flux row determines the node), so internal core nodes get no
   1 GΩ tie; a galvanic island holding a magnetic node is refused by
   `require_magnetic_reference` from `run_preflight`, `analyze_preflight`
   and `CircuitBuilder::build()`, whatever `auto_regularize` says.

7. **Deferred to after add-unified-device-interface** (not merged on the
   base of this branch): the hysteretic permeance (4.1), the excess-loss
   element (4.2, 4.3), the loss-element fitter (4.4) and its consistency
   report (4.5). The YAML types `hysteretic_permeance` / `excess_loss`
   are refused by name, pointing at `hysteretic_inductor` and
   `magnetic_resistance`. Rate-dependent loss inside the solve is
   available now through the magnetic resistance (classical eddy loss).

8. **Outputs** are Python accessors over the recorded node potentials:
   `res.flux(name)` = P·F or Φ(F) through the element's law,
   `res.b_field` = Φ/area, `res.h_field` = F/length (geometry elements
   only). A magnetic resistance appears in `device_loss_summary` with
   `kind="magnetic_resistance"`, `category="core_loss"`, its P_avg the
   in-loop loss R_m·(dΦ/dt)². No thermal coupling is added here.

9. **Sign convention of the winding**: `add_winding(name, e_pos, e_neg,
   m_pos, m_neg, N)` is `add_gyrator(e_pos, e_neg, m_pos, m_neg, 1/N)`:
   F(m_pos) − F(m_neg) = N·i with i into e_pos, and the flux rate
   dΦ/dt = v_e/N leaves m_pos into the network. Two windings in series
   on one flux path, each driving flux the same way around it, have a
   positive mutual inductance (the transformer test reverses the
   secondary's magnetic port for that). Windings on two parallel legs
   whose m_pos both face the same yoke drive flux in opposite senses
   around the window: their coupling is negative (k = −0.8 in the EE
   test). Reversing one magnetic port reverses its dot.

10. **Hysteretic permeance = unit winding + one-turn JA inductor (4.1,
   closeout 2026-11-19).** `add_hysteretic_permeance(name, m_a, m_b,
   area, length, Ms, a, alpha, c, k)` builds the gyrator `{name}.w`
   (g = 1, magnetic ports swapped) on the internal node `{name}.e` and the
   existing `add_hysteretic_core_inductor` `{name}` with N = 1, no gap,
   across it. By gyrator duality an inductor behind a unit gyrator is a
   capacitor-type element: the MMF F(m_a) − F(m_b) is the JA inductor's
   current (H = F/l, the JA law's own H for N = 1), the flux through the
   segment is its flux linkage (Φ = A·B) and the flux rate the voltage of
   `{name}.e`. No new kind (decision 1 again): the IR, YAML round trip,
   snapshot (the saturable-inductor block), conformance table and
   refusals are the gyrator's and the JA inductor's (dsed / bdf1 / AC
   refuse the JA kind by name, naming `{name}`). Measured against
   `add_hysteretic_core_inductor` with N = 100 on the 50 Hz drive of
   `test_hysteretic_core_in_loop.py`: currents 1.0e-9 (pwl) / 1.3e-9
   (trbdf2) of the peak; ∮F·dΦ over the steady cycle equals the absorbed
   electrical energy to 1.1e-9. `res.flux` / `b_field` / `h_field` refuse
   it by name (KeyError from `magnetic_element`): Φ depends on the JA state,
   not on F alone; the message gives the MMF and flux-rate recipe. YAML
   type `hysteretic_permeance` (keys `from`, `to`, `area`, `length`,
   `Ms`, `a`, `alpha`, `c`, `k`, optional `M0`) builds the same circuit.

11. **Loss-element fitter (4.4).** `pulsim.fit_core_loss_elements(f, B,
   P_v, *, geometry)` fits the Bertotti split at sinusoidal flux,
   `k_h·f·B^β + k_e·f²·B² + k_x·f^1.5·B^1.5`, by non-negative least squares
   on the relative error inside a bounded 1-D search over β, and maps
   `k_e → R_m = k_e·l/(2π²A)`, `k_x → c = k_x·l/((2π)^1.5·0.5564·√A)`.
   The hysteresis term is reported (it belongs to the JA law, not to these
   elements). The result lists every point it was fitted on; nothing
   reports agreement with them. Round trip 1e-6 on synthetic data; the
   fitted `R_m` dissipates its eddy share `k_e·f²·B²·volume` in a run
   within 1e-4.

12. **Excess-loss element not done (4.2, 4.3, 4.5).** It needs a new
   Newton law, which no existing kind provides: by the same duality as
   decision 10 the design's current-controlled law F(u) = c·u/(u²+ε²)^¼
   becomes a VOLTAGE-controlled conductance i = F(v) on an internal node
   behind a unit winding (v = u = dΦ/dt; dF/du > 0 everywhere, so the
   Jacobian is never singular), i.e. one new two-terminal NewtonStatic
   kind. A new kind reaches the pool (`StoredKind`, entry, `kKindCaps`
   row decided for every engine and the AC sweep), `ir/fields.hpp`, the
   hand stamp (`nonlinear_refresh_*`) and the interface law
   (`device/laws.hpp`), `device_readings` / `loop_feed`,
   `analysis/linearize.hpp`, the IR and YAML emitter, the loader, the
   bindings and stubs, the generated device reference, and every test that
   lists the kinds exactly. Not attempted in the closeout session for its
   size; 4.3 and 4.5 (which uses the element) wait for it.

13. **Excess-loss element shipped (4.2, 4.3, 4.5; 2026-11-19, branch
   feat/core-p5-line-and-excess).** Decision 12's plan, as one new kind
   `StoredKind::ExcessLoss = 29` (`models/magnetic_loss.hpp`): a static
   two-terminal law i = c·v/(v²+ε²)^¼ behind a unit winding
   (`add_excess_loss(name, m_a, m_b, c, eps)` builds `{name}.w` on
   `{name}.e` and the kind `{name}` to ground, as decision 10). The law is
   the design's; `(v²+ε²)^¼` is `sqrt(sqrt(·))` (correctly rounded ops, no
   libm pow). ε is REQUIRED (design). Its rule, derived and checked in
   40-digit mpmath (`probes/excess_loss_truth.py`): under a sine flux rate
   of amplitude Û the regularised loss is below the ideal c·Û^1.5·K,
   K = mean|cos|^1.5 = Γ(5/4)/(√π·Γ(7/4)) = 0.5564178944, by
   δ = (3/4)(ε/Û)² − O((ε/Û)^2.5) — the leading term is
   (ε²/4)·mean|u|^−½ / mean|u|^1.5 and Γ(1/4)Γ(7/4)/(Γ(3/4)Γ(5/4)) = 3;
   measured 6.95e-5 / 7.33e-7 / 7.45e-9 at ε/Û = 1e-2 / 1e-3 / 1e-4
   against 7.5e-5 / 7.5e-7 / 7.5e-9. So ε ≤ Û·√(4δ/3) = 1.15e-3·Û keeps
   the power within 1e-6 of the square-root law. Smooth (analytic on the
   real line) and strictly monotone: F'(u) = c(u²/2+ε²)/(u²+ε²)^(5/4) > 0;
   the C++ test scans ±2000 points at 1e-12 spacing (of max(|centre|, 1))
   around 0, ±ε and ±10 V, checks oddness bit-exactly, the AD slope against
   the closed form (16 ulp) and a central difference (1e-7), ∂/∂p by AD
   against a Richardson difference, and the hand stamp against the
   interface law to 1e-12 (`test_device_laws.cpp`).
   - Every surface: pool (enum, `Entry`, the two counting static_asserts,
     `add_excess_loss`, `excess_loss_params`), `kKindCaps` row
     `NewtonStatic, kNewtonAc` — pwl yes, trbdf2 yes (and esdirk43: a
     static law goes through the stage form unchanged; tested), dsed /
     exact no (not LTI; refused by name by the table, `UnsupportedError`
     naming the branch), bdf1 no (as every Newton kind), AC sweep YES:
     linearised at the DC point, where the flux rate is 0, the small-signal
     model is the regularised core, a magnetic resistance R_m = c/√ε
     (tested against `add_magnetic_resistance(c/√ε)`, 1e-9);
     `test_engine_capabilities.cpp`'s exact AC list has 10 names now. Dual
     path: BOTH — the hand stamp (`make_combined_diode_mosfet_refresh`,
     `stamp_nonlinear_branches<models::ExcessLoss>`) and the interface law
     (`device::laws::ExcessLoss`, a `CurrentLaw` on the same model body)
     are one expression, so they agree to rounding; `static_kind_ported`
     includes it. `ir/fields.hpp` rows `c` [A/V^0.5], `eps` [V] (listed in
     `components()`), `with_value` validates; YAML emitter kind
     `excess_loss` (round trip by hash), the authoring type `excess_loss`
     (`from`, `to`, `c`, `eps`; was refused by name); generated
     `docs/device-reference.md`; bindings `add_excess_loss`, enum value
     (registered LAST: `__members__` order is the table order), stubs.
     Preflight: the kind is a DC path (slope c/√ε at u = 0; without this a
     1 GΩ tie was inserted on `{name}.e`, a spurious loss of 6e-7 of the
     4.3 case). No state: nothing in the snapshot (a static law). Readings:
     `static_device_readings` slot 5, `res.i`, the in-loop recorder
     (`p(X)`, `i(X)`; loop-only like the other Newton readings),
     `device_loss_summary` row `kind='excess_loss'`, `category='core_loss'`,
     `c`, `eps`; `res.power`. NOT fed to a thermal node: the kernel loss
     port is a switching device's junction loss; a core temperature is not
     modelled (stated in docs/losses-and-thermal.md). PSS refuses it with
     the other Newton kinds (`periodic_steady_state._check_circuit`).
   - Numbers (4.3: c = 0.01, ε = 0.01 V, N = 10, 100 V 1 kHz sine;
     U = 10 V): regularised truth 0.175954658818, ideal 0.175954787734.
     pwl dt = T/2000: 0.175954657275 (−8.8e-9 vs regularised, −7.41e-7 vs
     ideal); trbdf2 rtol 1e-10 with the T/2000 cap: −4.0e-9 / −7.37e-7;
     trbdf2 without a cap (1003 steps, the default ceiling): −6.2e-7 /
     −1.35e-6. The circuit is algebraic in u, so a sample is the law to
     Newton tolerance and the error is the trapezoid's (geometric once h
     resolves ε/(ωU)). `device_loss_summary` (piecewise-linear product
     integral, second order): −1.65e-6. f^1.5 law over two decades
     (10 Hz – 1 kHz, fixed flux, ε = 1e-4·U_min): log-log slope
     1.5000000013, each point within 1.0e-8 of c·U^1.5·K. Triangular flux
     (±V square): c·U^1.5 within the edge fraction (3e-4 gate).
   - Round trip (task wording "three-term separation recovered"): synthetic
     sine data → `fit_core_loss_elements` → eddy + excess elements in a run
     at the data's (f, B) → the data's eddy + excess shares within 1e-6
     (the hysteresis share is the fit's k_h·f·B^β: a JA segment is not a
     power law in B, so a JA-in-the-loop round trip at 1e-6 is not a
     well-posed claim). FOUND ON THE WAY: the closeout's
     `_MEAN_ABS_COS_1P5 = 0.5563852309` was wrong (true 0.5564178944,
     5.9e-5); its round-trip test compared the fitter with the same wrong
     constant. Fixed (Γ expression), the test pins the value; the run
     round trip would have failed at 6e-5.
   - 4.5 (consistency report, not gated): `probes/core_loss_consistency.py`
     fits the same synthetic sine data twice (Bertotti elements; Steinmetz
     k·f^a·B^b by log-linear least squares: k = 0.740, a = 1.981,
     b = 1.982) and predicts a triangular flux at 50/100 kHz, 0.1/0.2 T:
     in-loop (eddy + excess simulated, hysteresis from the fit) vs iGSE
     differ by −0.4 %, −0.4 %, −0.4 %, −0.1 % (the synthetic data are
     eddy-dominated, so both reduce to f²B² there). Two models of the same
     data agreeing is not a validation.
