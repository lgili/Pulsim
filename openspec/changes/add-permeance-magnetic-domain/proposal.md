## Why

Pulsim models a magnetic component as an electrical equivalent: a
saturable inductor with a `λ(i)` law, or a transformer as a T-model
(leakage inductors, one magnetising branch, ideal transformers;
`core/include/pulsim/builder/circuit_builder.hpp:399-466`). That form is
exact for one flux path and, per the review, well executed (`λ`-based
flux row, Hermite `FluxTable` with exact tangents, `GappedCore` swept in
H, Jiles–Atherton inside the Newton loop with an exact tangent). It
cannot express a core with more than one flux path, and it has no
in-loop core loss.

Verified on this branch (review `raw/models.md`, inventory rows for the
transformers and the JA core, ranked item 10; synthesis M8):

- **One magnetising branch.** A three-limb three-phase core, an EE core
  with windings on the outer legs and a gapped centre leg, or any
  integrated magnetic has several coupled flux paths; a T-model has
  one. Example with an analytic answer: for a three-limb core with limb
  permeance 5 µH and an air return path of 50 nH, the zero-sequence
  inductance is `N²/(1/P_L + 3/P_air)`, 0.33 % of the positive-sequence
  inductance; three T-model transformers give 100 %.
- **Hysteresis is rate-independent.** The JA core has static hysteresis
  loss only; the review lists "eddy and excess loss" as omitted.
- **Core loss is post-hoc.** Steinmetz and iGSE are evaluated after the
  run from the recorded current (`core/include/pulsim/analysis/core_loss.hpp`);
  the loss does not load the circuit, damp a resonance or heat a
  thermal node during the run.
- **No gyrator**, so the standard construction of a magnetic network is
  not available (`add-controlled-source-set` adds it).

Other tools (general knowledge, not checked against current
documentation): PLECS has a magnetic domain built on the
permeance–capacitor analogy with saturable and hysteretic permeances
and a magnetic resistance for losses; SPICE users build reluctance
models by hand with controlled sources.

## What Changes

A magnetic-circuit domain in which magnetic nodes are ordinary MNA
nodes, the node potential is the magnetomotive force `F` [A], the
through variable is the flux rate `dΦ/dt` [V], and the product is
power.

- **`add_winding(name, e_pos, e_neg, m_pos, m_neg, turns)`**: the
  gyrator of `add-controlled-source-set` with `g = 1/N`
  (`v = N·dΦ/dt`, `F = N·i`).
- **`add_permeance(name, m_a, m_b, P)`** [H] and
  **`add_air_gap(name, m_a, m_b, area, length)`**
  (`P = μ₀·A/l`): a linear permeance is a capacitor of value `P`.
- **`add_saturable_permeance(name, m_a, m_b, area, length, *, mu_r,
  B_sat | table)`**: `Φ(F) = A·B(F/l)` with the `GappedCore` material
  law in closed form, or a `FluxTable`; stamped in flux form so flux is
  conserved.
- **`add_hysteretic_permeance(name, m_a, m_b, area, length, ja)`**: the
  existing Jiles–Atherton law with its exact tangent, as a permeance.
- **`add_magnetic_resistance(name, m_a, m_b, R_m)`** [S] for
  classical eddy loss (`P = R_m·(dΦ/dt)²`; for a lamination
  `R_m = σ·d²·l/(12·A)`), and
  **`add_excess_loss(name, m_a, m_b, c, eps)`** for the
  `|dB/dt|^1.5` term in a form that is smooth at zero.
- **Outputs**: flux [Wb], flux density [T], field [A/m] and loss [W]
  per magnetic element; core loss appears in `device_loss_summary` and
  can drive a thermal node.
- **DC operating point**: magnetic nodes whose branches are all
  permeances are solved by flux conservation (the reluctance-network
  solution), not left singular.
- **YAML** types `winding`, `permeance`, `air_gap`,
  `saturable_permeance`, `hysteretic_permeance`,
  `magnetic_resistance`, `excess_loss`.
- The T-model saturable transformer and the saturable and hysteretic
  inductors are unchanged. The permeance form of a two-winding
  transformer is required to be equivalent to the T-model below
  saturation.
- No **BREAKING** change.

## Impact

- Affected specs: `magnetic-models`.
- Affected code: `builder/circuit_builder.hpp`,
  `models/{nonlinear_capacitor,flux_table,gapped_core,jiles_atherton}.hpp`
  (reused), new `models/magnetic_loss.hpp`, `pwl/dc_assemble.hpp`
  (flux-conservation rows), builder preflight (magnetic islands),
  `yaml/loader.hpp`, `python/bindings.cpp`,
  `python/pulsim/{magnetic,losses}.py`, stubs, docs.
- Dependencies:
  - `add-controlled-source-set` — required (gyrator). Listed in the
    roadmap.
  - `add-power-mosfet-datasheet-device`, step 1 — the charge-law
    variant of `NonlinearCapacitor`, which the saturable permeance
    reuses with an odd law. If this change lands first it brings that
    variant.
  - `add-unified-device-interface` — needed for the hysteretic
    permeance (state plus `accept`) and the excess-loss element (one
    internal unknown). Linear networks, air gaps, saturable permeances
    and the eddy resistance can ship before it.
  - `add-engine-conformance-matrix` — rows for the new kinds.
  - `add-in-kernel-event-losses` — the thermal node a core loss can
    feed.
  - `add-immutable-circuit-ir` — its domain-tagged nodes replace the
    builder-side registry of magnetic nodes used here.
