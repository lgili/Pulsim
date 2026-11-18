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
