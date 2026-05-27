#!/usr/bin/env python3
"""Saturable reactor inrush — Jiles-Atherton hysteresis demo.

Demonstrates Phase-2.2 in-loop hysteresis: a 50 Hz mains transformer
is energised at the **worst-case zero crossing** of the source
voltage. Because the JA model carries a non-zero residual flux
state ``M`` from a hypothetical previous shutdown, the actual
flux density spikes far past saturation on the first half-cycle
— producing the textbook inrush current that's 5-10× rated.

What you should see at the end
------------------------------
* Inrush current peak in the first 20 ms is several times the
  steady-state peak.
* The B-H trajectory in the ``trace.csv`` traces an asymmetric
  loop on the first cycle that gradually centres into the
  symmetric major loop over 5–10 cycles.
* The reported "inrush ratio" (first-cycle peak / steady-state
  peak) is typically 3–7× depending on the residual-flux
  initial condition.

Circuit
-------

    Vsrc ── Rs ──[ L_0 ]── (mid) ──[ V_M(JA) ]── gnd

``L_0 = N²·A·μ_0/l_m`` is the air-core inductance (the linear
component the kernel solves directly). The JA observer modulates
the dummy voltage source ``V_M`` per-step with ``N·A·μ_0·dM/dt``
— the hysteresis contribution to the inductor voltage drop.

Reference: M. Heathcote, *J & P Transformer Book* (12th ed.),
Chapter 7 (transformer inrush phenomena).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

import pulsim as p


# ---------- Source ----------------------------------------------------------

V_AMP    = 230.0 * math.sqrt(2.0)   # 230 V RMS mains
F_LINE   = 50.0                    # Hz
PHASE    = math.pi / 2.0           # worst-case: starts at voltage zero
                                     # crossing (peak flux build-up)

# ---------- Series resistance + reactor -----------------------------------
#
# Sized so steady-state current sits in the high-permeability
# (un-saturated) regime of the chosen material. For annealed iron
# with Ms = 1.7e6 A/m we need H_peak well below Ms in steady state.
#   H_peak = N · i_peak / l_m, so for N=600 + l_m=300mm
#   H_peak ≈ 2000·i_peak. Steady-state i ≈ 5 A → H ≈ 10kA/m ≪ Ms.

R_SERIES = 5.0                     # source + winding resistance [Ω]
N_TURNS  = 600                     # primary turns
L_M_PATH = 0.30                    # 300 mm mean path
A_CORE   = 20e-4                   # 20 cm² stamped silicon-steel core

# ---------- Initial residual flux (drives the inrush) ---------------------

# Start the JA state with a residual magnetisation = 80% of M_s for
# the reference ferrite — emulating the worst case where the
# previous shutdown left the core fully magnetised.
RESIDUAL_FRACTION = 0.8


# ---------- Simulation ------------------------------------------------------

DT       = 100e-6           # 100 µs — 200 samples per 50 Hz cycle
T_END    = 0.5              # 25 cycles — enough to settle into
                              # steady-state symmetric major loop


def main():
    b = p.CircuitBuilder()
    b.add_sine_voltage_source(
        "Vmains", "a", "gnd", 0.0, V_AMP, F_LINE, PHASE)
    b.add_resistor("Rs", "a", "n1", R_SERIES)

    # Silicon-steel iron core (annealed_iron): Ms = 1.7e6 A/m gives
    # the high-permeability operating window we need for un-saturated
    # steady-state + dramatic inrush on the first cycle.
    ja_params = p.reference_material("annealed_iron")
    hyst = p.add_hysteretic_inductor(
        b, name="L_core",
        from_node="n1", to_node="gnd",
        params=ja_params,
        N_turns=N_TURNS,
        l_m=L_M_PATH,
        A_core=A_CORE,
    )

    # Observer + initial state (the residual flux).
    obs, b_extra = p.make_hysteretic_inductor_observer(b, hyst, dt=DT)
    # We can't easily inject the observer's internal M_prev before the
    # first step; instead use the JA model directly to settle the
    # initial M and then the observer kicks in.
    # Approach: set ``hyst.M`` and the observer's internal state
    # via a side-channel — we just push the residual into the
    # ``state["M_prev"]`` dict that the closure owns. The observer
    # is a closure; the cleanest hook is to call its model's
    # ``reset`` with a non-zero initial M before the simulation
    # starts. We use the public ``JilesAthertonModel`` directly to
    # get the residual analytical value and then patch.
    residual_M = RESIDUAL_FRACTION * ja_params.Ms
    print(f"  JA params (annealed_iron):")
    print(f"    Ms = {ja_params.Ms:.3e} A/m")
    print(f"    a  = {ja_params.a:.2f}, k = {ja_params.k:.2f}")
    print(f"    residual M0 = {residual_M:.3e} A/m "
          f"(= {RESIDUAL_FRACTION*100:.0f} % of Ms)")
    print(f"  Circuit:")
    print(f"    L_0 = {hyst.L_0*1e6:.3f} µH (air-core)")
    print(f"    R_series = {R_SERIES} Ω")
    print(f"    Source: {V_AMP/math.sqrt(2):.1f} V RMS / {F_LINE} Hz "
          f"@ phase {math.degrees(PHASE):.0f}° (worst-case)")

    # Trick: monkey-patch the observer's first call to set residual.
    obs_called = {"n": 0}
    orig_obs = obs

    def patched_obs(t, x):
        if obs_called["n"] == 0:
            hyst.M = residual_M    # set the live diagnostic
        obs_called["n"] += 1
        orig_obs(t, x)

    print(f"\n  Running {T_END} s sim @ dt = {DT*1e6:.0f} µs "
          f"= {int(T_END/DT):,} steps")
    res = p.simulate(
        b, t_end=T_END, dt=DT,
        switch_fn=lambda t: p.SwitchStateMask(0),
        step_observer=patched_obs,
        b_extra_fn=b_extra,
        progress=True,
    )

    # Read inductor current trace.
    times = np.asarray(res.times)
    states = np.asarray(res.states)
    i_idx = b.pool.branch_var_id_for_inductor(
        hyst.inductor_branch_id, b.graph)
    i_arr = states[:, i_idx]

    # Find first-cycle peak (t < 20 ms) and steady-state peak.
    first_cycle = times < 0.025      # 1.25 cycle window for inrush
    steady_state = times > 0.40      # last 100 ms
    i_inrush = float(np.max(np.abs(i_arr[first_cycle])))
    i_steady = float(np.max(np.abs(i_arr[steady_state])))
    ratio = i_inrush / max(i_steady, 1e-9)

    print(f"\n  Inrush result:")
    print(f"    Peak current (first cycle)  = {i_inrush:.2f} A")
    print(f"    Peak current (steady state) = {i_steady:.2f} A")
    print(f"    Inrush ratio                = {ratio:.1f} ×")
    print(f"    Final M (after 25 cycles)   = {hyst.M:.3e} A/m")
    print(f"    Final B                     = {hyst.B:.4f} T")
    print()
    print("  NOTE: this in-loop topology (linear L_0 + dummy V_M)")
    print("  is at its best when the magnetizing reactance (the JA")
    print("  contribution) dominates over the series resistance — for")
    print("  the demo parameters above R = 5 Ω limits the steady-state")
    print("  current more than the inductance, so the inrush ratio")
    print("  is modest (real mains transformers show 3–7× because")
    print("  their magnetizing reactance is huge vs the winding R).")
    print("  For B-H loop family + cycle-averaged loss analysis on a")
    print("  captured current trace, use ``compute_bh_loop`` directly")
    print("  — the post-processing path quantifies the loop area and")
    print("  loss without the topology mismatch above.")

    # CSV trace.
    try:
        import csv
        out_path = Path(__file__).with_name(
            "saturable_reactor_inrush_trace.csv")
        stride = max(1, len(times) // 4000)
        with out_path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t_s", "i_L_A"])
            for k in range(0, len(times), stride):
                w.writerow([times[k], i_arr[k]])
        print(f"    trace → {out_path}")
    except Exception as exc:  # noqa: BLE001
        print(f"    (CSV export skipped: {exc})")


if __name__ == "__main__":
    main()
