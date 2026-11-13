"""Slice 2: the coordinator's list of circuits where today's dsed settle is silently wrong or errors,
on dsed with the settle alone (guard off), the default (settle + guard) and mode_selection='lcp',
against trbdf2 at rtol 1e-10.

    PYTHONPATH=<worktree>/python python3 extra_circuits.py > extra_circuits.out

Per run: the diode law on every STORED sample (an ON diode's current g_on (v - V_f) must not be
negative, an OFF diode's voltage must not exceed max(V_th, V_f)), as the worst reverse current (A)
and the worst forward bias (V) with the diode that carries it, beyond 1e-3 of that diode's peak
current / 1e-3 V; the probed node against trbdf2 (end value and time-weighted mean, np.trapezoid);
the guard / LCP counters; or the error.

The v1 benchmarks (diode_half_wave_rl_load, chopper_with_rc_snubber, cuk_converter, sepic) are
refused by the 2.0 YAML loader; they are rebuilt here with the builder (the half-wave RL case IS
diode_half_wave_rl_load). The double-crossing tank is the circuit of
test_double_crossing_inside_one_step_is_seen; its V_th = 0 failure is not a mode-selection
defect: the turn-on is decided at v = 0, so the armed turn-off predicate (i = g_on v) starts the
next step at exactly 0, which locate_event_in_step_ skips as "just fired" -- the return of i
through zero at 1.89 us inside that 1 us step is never located, and no event = no decision.
"""
import sys
import time
import warnings

import numpy as np

warnings.simplefilter("ignore")
import pulsim as p  # noqa: E402
import pulsim._dsed_dispatch as DD  # noqa: E402

GON, GOFF, T = 1e3, 1e-9, 1e-5


def boost(R, vth):
    def make():
        b = p.CircuitBuilder()
        b.add_voltage_source("Vin", "vin", "gnd", 12.0)
        b.add_inductor("L", "vin", "sw", 100e-6)
        b.add_switch("S", "sw", "gnd", GON, GOFF)
        b.add_diode("D1", "sw", "out", GON, GOFF, vth)
        b.add_capacitor("C", "out", "gnd", 47e-6)
        b.add_resistor("R", "out", "gnd", R)
        fn = p.make_pwm_switch_fn(1 / T, 0.5, b.switch_index_of("S"), b.graph.num_switches, 0.0)
        return b, fn, 400 * T, [("D1", "sw", "out", GON, 0.0, vth)], "out"
    return make


def half_wave_rl(vth):
    def make():
        b = p.CircuitBuilder()
        b.add_sine_voltage_source("Vac", "in", "gnd", 0.0, 10.0, 50.0)
        b.add_diode("D1", "in", "out", 200.0, 1e-6, vth)
        b.add_resistor("R1", "out", "nl", 10.0)
        b.add_inductor("L1", "nl", "gnd", 1e-3)
        return b, None, 60e-3, [("D1", "in", "out", 200.0, 0.0, vth)], "out"
    return make


def double_crossing(vth):
    def make():
        b = p.CircuitBuilder()
        b.add_capacitor("C", "t", "gnd", 100e-9)
        b.add_inductor("L", "t", "gnd", 10e-6, i0=-1.0)
        b.add_diode("D", "t", "k", GON, GOFF, vth)
        b.add_resistor("R", "k", "r", 1e3)
        b.add_voltage_source("Vr", "r", "gnd", 9.5)
        return b, None, 10e-6, [("D", "t", "k", GON, 0.0, vth)], "t"
    return make


def pwm_circuit(build, f, duty, t_end, node, diodes):
    """A v1 benchmark (the 2.0 loader refuses v1 netlists) rebuilt with the builder: the
    vcswitch driven by a 0/10 V PWM against a 5 V threshold is a switch on a duty schedule."""
    def make():
        b = p.CircuitBuilder()
        build(b)
        fn = p.make_pwm_switch_fn(f, duty, b.switch_index_of("S1"), b.graph.num_switches, 0.0)
        return b, fn, t_end, diodes, node
    return make


def chopper(b):
    b.add_voltage_source("Vcc", "vcc", "gnd", 20.0)
    b.add_switch("S1", "vcc", "sw", 200.0, 1e-6)
    b.add_diode("Dfw", "gnd", "sw", 200.0, 1e-6)
    b.add_inductor("L1", "sw", "out", 1e-3, 0.0)
    b.add_resistor("Rload", "out", "gnd", 10.0)
    b.add_resistor("Rsn", "sw", "n_sn", 10.0)
    b.add_capacitor("Csn", "n_sn", "vcc", 100e-9, 0.0)


def cuk(b):
    b.add_voltage_source("Vin", "vin", "gnd", 12.0)
    b.add_inductor("L1", "vin", "sw_node", 47e-6, 0.0)
    b.add_switch("S1", "sw_node", "gnd", 200.0, 1e-6)
    b.add_capacitor("Ccpl", "sw_node", "mid", 22e-6, 24.0)
    b.add_inductor("L2", "mid", "out", 47e-6, 0.0)
    b.add_diode("Drect", "gnd", "mid", 200.0, 1e-6)
    b.add_capacitor("Cout", "out", "gnd", 47e-6, 0.0)
    b.add_resistor("Rload", "out", "gnd", 20.0)


def sepic(b):
    b.add_voltage_source("Vin", "vin", "gnd", 12.0)
    b.add_inductor("L1", "vin", "sw_node", 47e-6, 0.0)
    b.add_switch("S1", "sw_node", "gnd", 200.0, 1e-6)
    b.add_capacitor("Ccpl", "sw_node", "mid", 10e-6, 12.0)
    b.add_inductor("L2", "mid", "gnd", 47e-6, 0.0)
    b.add_diode("Drect", "mid", "out", 200.0, 1e-6)
    b.add_capacitor("Cout", "out", "gnd", 47e-6, 0.0)
    b.add_resistor("Rload", "out", "gnd", 20.0)


CASES = [
    ("boost 20 ohm, V_th = 0", boost(20.0, 0.0)),
    ("boost 20 ohm, V_th = 1e-3", boost(20.0, 1e-3)),
    ("boost 20 ohm, V_th = 1e-2", boost(20.0, 1e-2)),
    ("boost 50 ohm, V_th = 1e-3", boost(50.0, 1e-3)),
    ("boost 500 ohm, V_th = 1e-3", boost(500.0, 1e-3)),
    ("half-wave RL rectifier = benchmark diode_half_wave_rl_load (builder), V_th = 0", half_wave_rl(0.0)),
    ("double-crossing LC tank, V_th = 0", double_crossing(0.0)),
    ("double-crossing LC tank, V_th = 1e-3", double_crossing(1e-3)),
    ("benchmark chopper_with_rc_snubber (builder; duty 0.5 @ 5 kHz)",
     pwm_circuit(chopper, 5e3, 0.5, 1.5e-3, "out", [("Dfw", "gnd", "sw", 200.0, 0.0, 0.0)])),
    ("benchmark cuk_converter (builder; duty 0.5 @ 50 kHz, dead time dropped)",
     pwm_circuit(cuk, 5e4, 0.5, 2e-3, "out", [("Drect", "gnd", "mid", 200.0, 0.0, 0.0)])),
    ("benchmark sepic (builder; duty 0.5 @ 50 kHz, dead time dropped)",
     pwm_circuit(sepic, 5e4, 0.5, 2e-3, "out", [("Drect", "mid", "out", 200.0, 0.0, 0.0)])),
]


def law(b, r, diodes):
    bb = getattr(b, "builder", b)
    S = np.asarray(r.switch_states)
    worst_i, worst_v = (0.0, ""), (0.0, "")
    for name, a, k, gon, vf, vth in diodes:
        on = S[:, bb.switch_index_of(name)].astype(bool)
        vd = np.asarray(r.v(a)) - np.asarray(r.v(k))
        i = gon * (vd - vf)
        peak = max(1e-12, np.abs(np.where(on, i, 0.0)).max())
        ri = -(i[on]).min() if on.any() else 0.0
        fv = (vd[~on] - max(vth, vf)).max() if (~on).any() else 0.0
        if ri > 1e-3 * peak and ri > worst_i[0]:
            worst_i = (ri, name)
        if fv > 1e-3 and fv > worst_v[0]:
            worst_v = (fv, name)
    return worst_i, worst_v


def windows(b, r, name):
    """The diode's switching instants. A stored row carries the mode of the interval that ENDS
    at it (an event row holds the pre-event mode), so the mode changes at t[e], not t[e+1]."""
    bb = getattr(b, "builder", b)
    t = np.asarray(r.times)
    on = np.asarray(r.switch_states)[:, bb.switch_index_of(name)].astype(bool)
    return [("ON " if on[e + 1] else "OFF") + f"@{t[e]:.4e}" for e in np.flatnonzero(np.diff(on.astype(int)))]


def run(make, engine, mode=None, guard=True):
    DD.MODE_GUARD = guard
    try:
        b, fn, t_end, ds, node = make()
        kw = {} if fn is None else {"switch_fn": fn}
        if mode:
            kw["mode_selection"] = mode
        if engine == "trbdf2":
            kw["rtol"] = 1e-10
        t0 = time.perf_counter()
        r = p.simulate(b, t_end=t_end, engine=engine, **kw)
        return b, r, ds, node, time.perf_counter() - t0, None
    except Exception as e:  # noqa: BLE001
        return None, None, None, None, 0.0, f"{type(e).__name__}: {str(e)[:150]}"
    finally:
        DD.MODE_GUARD = True


def tw(t, y):
    return np.trapezoid(y, t) / (t[-1] - t[0])


def main():
    print("pulsim from", p.__file__)
    for label, make in CASES:
        print(f"== {label}")
        b, ref, ds, node, _, err = run(make, "trbdf2")
        if err:
            print(f"   trbdf2: {err}")
        else:
            (ri, rn), (fv, fn_) = law(b, ref, ds)
            print(f"   trbdf2 rtol 1e-10: v({node}) end={np.asarray(ref.v(node))[-1]:.9f} "
                  f"mean={tw(np.asarray(ref.times), np.asarray(ref.v(node))):.9f} "
                  f"law: rev {ri:.3g} A {rn} fwd {fv:.3g} V {fn_}")
        for tag, mode, guard in (("settle", None, False), ("default", None, True), ("lcp", "lcp", True)):
            bb, r, ds2, node2, el, err = run(make, "dsed", mode, guard)
            if err:
                print(f"   {tag:7s}: {err}")
                continue
            (ri, rn), (fv, fn_) = law(bb, r, ds2)
            dg = r.diagnostics or {}
            g, s = dg.get("mode_guard", {}), dg.get("mode_selection", {})
            v = np.asarray(r.v(node2))
            line = (f"   {tag:7s}: events={r.n_events} law: rev {ri:.3g} A {rn} fwd {fv:.3g} V {fn_}"
                    f" | v({node2}) end={v[-1]:.9f} mean={tw(np.asarray(r.times), v):.9f}")
            if ref is not None:
                line += (f" d_end={v[-1] - np.asarray(ref.v(node))[-1]:+.2e} d_mean="
                         f"{tw(np.asarray(r.times), v) - tw(np.asarray(ref.times), np.asarray(ref.v(node))):+.2e}")
            if g:
                line += f" | guard repairs={g['repairs']}/{g['checks']}"
            if s:
                line += f" | lcp piv_max={s['max_pivots_per_event']} pins={s['leak_pins']} ties={s['ties']}"
            print(line + f" [{el:.2f} s]")
            if "double-crossing" in label:
                print(f"            switching: {windows(bb, r, 'D')}")
        if ref is not None and "double-crossing" in label:
            print(f"   trbdf2  switching: {windows(b, ref, 'D')}")
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
