"""Bit-identity probes for update-fixed-step-loop-bookkeeping.

    python bookkeeping_baseline.py save  OUT.npz     # record (before the change)
    python bookkeeping_baseline.py check OUT.npz     # np.array_equal + byte equality vs OUT.npz

Every fixed-engine run of the restart-rule and event-split probe sets,
three times: keyword absent (defaults), `discontinuity_restart=True`,
and the rule with `enable_substep_state_correction=True`; the trbdf2 /
dsed runs once; plus the loop paths this change touches: no `switch_fn`
(default schedule) on pwl and trbdf2, a closed loop (observer writes,
`switch_fn` reads), a dt-retry run, the static path, `store_every`.
Recorded per run: times (by VALUE: their type changes), states, switch
bits, event_iteration_count, commutation events, restart/split counts,
dt-retry times.
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CH = os.path.join(HERE, "..", "..")
sys.path.insert(0, os.path.join(CH, "add-discontinuity-restart-rule", "probes"))
sys.path.insert(0, os.path.join(CH, "add-fixed-step-event-splitting", "probes"))
import pwl_baseline as pb  # noqa: E402
import split_probe as sp  # noqa: E402

import pulsim as p  # noqa: E402

warnings.simplefilter("ignore")
MODES = {"abs": {}, "rule": {"discontinuity_restart": True},
         "split": {"discontinuity_restart": True,
                   "enable_substep_state_correction": True}}


def _rlc(**kw):
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "s", "gnd", 5.0)
    b.add_resistor("R", "s", "a", 1.0)
    b.add_inductor("L", "a", "m", 1e-3)
    b.add_capacitor("C", "m", "gnd", 1e-6)
    return p.simulate(b, t_end=2e-3, **kw)


def _closed_loop(**kw):
    """Hysteretic current control: the observer writes, switch_fn reads."""
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", 24.0)
    b.add_switch("S", "vin", "sw", 1e3, 1e-9)
    b.add_diode("D1", "gnd", "sw", 1e3, 1e-9)
    b.add_inductor("L", "sw", "vout", 100e-6)
    b.add_capacitor("C", "vout", "gnd", 47e-6)
    b.add_resistor("RL", "vout", "gnd", 2.0)
    n, k = b.graph.num_switches, b.switch_index_of("S")
    row = b.pool.branch_var_id_for_inductor(b.branch_index_of("L"), b.graph) \
        if hasattr(b, "branch_index_of") else None
    st = {"on": True}

    def obs(t, x):
        i = x[row] if row is not None else 0.0
        if i > 3.0:
            st["on"] = False
        elif i < 2.0:
            st["on"] = True

    def fn(t):
        m = p.SwitchStateMask(n)
        m.set(k, st["on"])
        return m
    return p.simulate(b, t_end=2e-3, dt=1e-7, engine="pwl", switch_fn=fn,
                      step_observer=obs, **kw)


def _retry(**kw):
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("Vac", "ac", "gnd", 0.0, 400.0, 60.0)
    q = p.IdealDiodeParams()
    q.kappa = 100.0
    b.add_nonlinear_diode("D", "ac", "vout", q, allow_non_passive=True)
    b.add_resistor("R", "vout", "gnd", 50.0)
    b.add_capacitor("C", "vout", "gnd", 100e-6)
    return p.simulate(b, t_end=3.4e-2, dt=1e-3, engine="pwl", **kw)


def _static(**kw):
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "in", "gnd", 10.0)
    b.add_switch("S", "in", "a", 1e3, 1e-9)
    b.add_resistor("R", "a", "gnd", 10.0)
    b.add_diode("D", "a", "k", 1e3, 1e-9)
    b.add_resistor("R2", "k", "gnd", 5.0)
    fn = p.make_pwm_switch_fn(1e4, 0.3, b.switch_index_of("S"), b.graph.num_switches, 0.0)
    return p.simulate(b, t_end=1e-3, dt=1e-6, engine="pwl", switch_fn=fn, **kw)


def runs():
    out = {}
    for tag, flag in MODES.items():
        pb.FLAG.clear()
        pb.FLAG.update(flag)
        for name, res in pb.runs().items():
            if name.startswith(("trbdf2", "dsed")) and flag:
                continue
            out[f"{tag}_{name}"] = res
        out[f"{tag}_rl_off03"] = sp.rl_offgrid(2e-6, 0.3)[0]
        out[f"{tag}_buck4567_T100"] = sp.buck_sched(1.0, 0.4567, 100)
        out[f"{tag}_dcm_grid_T101"] = sp.dcm_grid(101, 60)[0]
        out[f"{tag}_rlc_noswitch_pwl"] = _rlc(dt=1e-6, engine="pwl", **flag)
        out[f"{tag}_rlc_noswitch_store4"] = _rlc(dt=1e-6, engine="pwl", store_every=4, **flag)
        out[f"{tag}_closed_loop"] = _closed_loop(**flag)
        out[f"{tag}_retry"] = _retry(**flag)
        out[f"{tag}_static"] = _static(**flag)
    pb.FLAG.clear()
    out["trbdf2_rlc_noswitch"] = _rlc(engine="trbdf2")
    return out


def arrays(res):
    ev = [(e.t_estimated, e.branch_id, e.new_state) for e in res.commutation_events]
    return {
        "t": np.asarray(res.times, dtype=float),
        "x": np.asarray(res.states),
        "sw": np.asarray(res.switch_states),
        "it": np.asarray(getattr(res, "event_iteration_count", []), dtype=float),
        "ev": np.asarray(ev, dtype=float).reshape(-1, 3),
        "n": np.asarray([getattr(res, "n_restart_steps", 0), getattr(res, "n_split_steps", 0),
                         len(getattr(res, "event_iteration_breaches", []))], dtype=float),
        "retry": np.asarray([d.t for d in getattr(res, "dt_retries", [])], dtype=float),
    }


def save(path):
    data = {}
    for name, res in runs().items():
        for part, arr in arrays(res).items():
            data[f"{name}/{part}"] = arr
    np.savez(path, **data)
    print(f"saved {len(data) // 7} runs to {path}")


def check(path):
    ref = np.load(path)
    bad = 0
    n = 0
    for name, res in runs().items():
        n += 1
        for part, arr in arrays(res).items():
            a = ref[f"{name}/{part}"]
            # By value AND by bytes (signed zeros, task 7.3).
            if not (np.array_equal(a, arr) and a.dtype == arr.dtype
                    and a.tobytes() == np.ascontiguousarray(arr).tobytes()):
                bad += 1
                print(f"DIFFERS {name}/{part}")
    print(f"checked {n} runs: " + ("BIT-IDENTICAL" if bad == 0 else f"{bad} ARRAYS DIFFER"))
    return bad


if __name__ == "__main__":
    if sys.argv[1] == "save":
        save(sys.argv[2])
    else:
        sys.exit(1 if check(sys.argv[2]) else 0)
