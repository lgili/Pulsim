#!/usr/bin/env python3
"""PSIM/PLECS "C block" workalike — closed-loop buck with a
Numba-JIT PI controller via ``@pulsim.fast_block``.

The same workflow PSIM users write in their "Custom C Block" widget
(read inputs, mutate state, return scalar output) is expressed here
as a plain Python function decorated with ``@p.fast_block``. Numba
compiles it to native code via LLVM on the first call; subsequent
invocations run at C speed without any external compiler toolchain,
``cc`` in the user's PATH, or ``.so`` build dance.

Workflow:

    1. Build a buck plant in pulsim (Vin → MOSFET → L → C // Rload).
    2. Decorate a PI control law with ``@p.fast_block``.
    3. In the ``step_observer``, sample v_out, compute the PI output,
       update the duty cycle for the next switching period.
    4. Run ``simulate`` for a few mains cycles and plot.

Usage::

    pip install pulsim[fast]
    python examples/scripts/run_fast_block_pi_buck.py [--plot]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

import pulsim as p
import pulsim.fast_block  # ensure submodule registered in sys.modules
_fb_module = sys.modules["pulsim.fast_block"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Closed-loop buck with @pulsim.fast_block PI.")
    parser.add_argument("--plot", action="store_true",
                          help="Show the v_out + duty plot.")
    args = parser.parse_args(argv)

    if not _fb_module.is_available():
        print("ERROR: numba is required for this example.")
        print("Install with:  pip install pulsim[fast]")
        return 1

    # --------------------------------------------------------------
    # 1. Plant — open-loop buck.
    # --------------------------------------------------------------
    V_IN  = 48.0
    L_OUT = 100e-6
    C_OUT = 100e-6
    R_LOAD = 4.0

    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", V_IN)
    b.add_switch("HS", "vin", "sw", g_on=1e3, g_off=1e-9)
    b.add_diode("D1", "gnd", "sw", g_on=1e3, g_off=1e-6, V_th=0.0)
    b.add_inductor("L1", "sw", "vout", L_OUT)
    b.add_capacitor("C1", "vout", "gnd", C_OUT)
    b.add_resistor("R_load", "vout", "gnd", R_LOAD)

    # --------------------------------------------------------------
    # 2. PI control law as a `fast_block`. This is the PSIM "C block"
    #    moral equivalent: function body, scalar inputs, mutate
    #    `state` in place, return the scalar output.
    # --------------------------------------------------------------
    @p.fast_block
    def pi_step(err, dt, Kp, Ki, state):
        # state[0] = integrator
        state[0] += Ki * dt * err
        u = Kp * err + state[0]
        # Anti-windup-ish clamp.
        if u > 1.0:
            u = 1.0
        elif u < 0.0:
            u = 0.0
        return u

    pi_state = pi_step.make_state()
    pi_step.warm_up()  # eat the LLVM cost up front

    # --------------------------------------------------------------
    # 3. PWM driver + step observer.
    # --------------------------------------------------------------
    DT = 1e-7                  # 10 MHz sim grid → resolves PWM edges
    F_SW = 100_000.0           # 100 kHz buck
    T_SW = 1.0 / F_SW
    V_REF = 12.0
    # Plant G_vd = V_in/(LCs² + (L/R)s + 1): f0 = 1.59 kHz, Q = 4.
    # Integral-dominant PI, crossover ω_c = KI·V_in ≈ 2π·100 Hz; the
    # loop gain at the resonance peak, (KI/ω0 + KP)·V_in·Q ≈ 0.4, keeps
    # ~8 dB margin. The old KP=0.02, KI=200 had |L(jω0)| ≈ 4.8 and
    # limit-cycled at ~5 V pp around 12 V.
    KP, KI = 0.001, 13.0

    # Mutable closure state.
    duty_state = [0.0]
    last_pwm_update = [0.0]
    duty_log: list[float] = []
    t_log: list[float] = []

    def switch_fn(t: float):
        m = p.SwitchStateMask(b.graph.num_switches)
        phase = (t % T_SW) / T_SW
        m.set(0, phase < duty_state[0])
        return m

    vout_idx = b.node_id_of("vout")

    def observer(t, x):
        # Sample v_out at every step but only update duty once per
        # PWM period (deterministic discrete control).
        if t - last_pwm_update[0] >= T_SW:
            v_out = float(x[vout_idx])
            err = V_REF - v_out
            duty_state[0] = float(
                pi_step(err, T_SW, KP, KI, pi_state))
            last_pwm_update[0] = t
            duty_log.append(duty_state[0])
            t_log.append(t)

    # --------------------------------------------------------------
    # 4. Run + timing.
    # --------------------------------------------------------------
    # The PI (crossover ~100 Hz) needs ~10 ms to settle.
    T_END = 15e-3
    t0 = time.perf_counter()
    res = p.simulate(b, t_end=T_END, dt=DT,
                       switch_fn=switch_fn,
                       step_observer=observer)
    dt_sim = time.perf_counter() - t0
    print(f"simulated {T_END*1e3:.1f} ms in {dt_sim:.2f} s "
            f"({res.num_steps()} samples)")
    # Settled window: the last 2 ms, time-weighted.
    t_from = T_END - 2e-3
    v_late = p.time_average(res.times, res.v("vout"), t_from=t_from)
    d_late = float(np.mean([d for tt, d in zip(t_log, duty_log)
                            if tt >= t_from]))   # one sample per PWM period
    v_pp = float(np.ptp(res.v("vout")[np.asarray(res.times) >= t_from]))
    print(f"v_out (last 2 ms) = {v_late:.3f} V (ref = {V_REF:.1f} V), "
          f"{v_pp:.3f} V pp")
    print(f"duty  (last 2 ms) = {d_late:.3f} "
            f"(ideal V_REF/V_IN = {V_REF/V_IN:.3f})")
    print(f"PI integrator state = {pi_state[0]:.3f}")

    # --------------------------------------------------------------
    # 5. Optional plot.
    # --------------------------------------------------------------
    if args.plot:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("\n(matplotlib not installed — skipping plot)")
            return 0
        times = np.asarray(res.times) * 1e3  # ms
        v_out = np.asarray([float(s[vout_idx]) for s in res.states])
        fig, (ax_v, ax_d) = plt.subplots(2, 1, sharex=True,
                                              figsize=(9, 5))
        ax_v.plot(times, v_out, lw=1.0, label="v_out")
        ax_v.axhline(V_REF, color="0.5", linestyle="--",
                       label=f"V_REF = {V_REF} V")
        ax_v.set_ylabel("v_out [V]")
        ax_v.legend(loc="lower right")
        ax_v.set_title("Buck CL with @pulsim.fast_block PI controller")
        ax_d.plot(np.asarray(t_log)*1e3, duty_log, lw=1.0)
        ax_d.set_ylabel("duty")
        ax_d.set_xlabel("t [ms]")
        ax_d.set_ylim(0, 1)
        fig.tight_layout()
        # out/ is git-ignored (generated), so create it on first use.
        out = Path(__file__).resolve().parent / "out" / "fast_block_pi_buck.png"
        try:
            out.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out, dpi=140)
            print(f"\nPlot saved to {out}")
        except Exception as exc:  # noqa: BLE001
            print(f"\n(could not save plot: {exc})")
        plt.show()

    return 0


if __name__ == "__main__":
    sys.exit(main())
