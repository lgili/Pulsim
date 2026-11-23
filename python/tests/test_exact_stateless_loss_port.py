"""The exact engine on circuits with NO dynamic state (no L, no C), with a
loss port, a recorder or both (fix/core-exact-stateless-loss-port).

A mode with zero states is a purely algebraic mode: the engine propagates
nothing and every reading is constant between events. The loss port's
closed-form check took ``maxCoeff()`` of the empty state vector, which reads
through a null pointer: the process died (SIGSEGV) where it must either run
or refuse by name. Every run here is a SUBPROCESS with a timeout, so a
regression fails its test instead of taking pytest down.

The truth (no simulator): a 600 V leg, HS (dcp -> a) and LS (a -> gnd),
g_on = 1 kS, g_off = 1 nS, a 10 Ohm load from ``a`` to ground, a 10 kHz
complementary pair with no dead time, the loss port on HS. In each mode the
node voltage is the nodal solve of one node, so the HS current and power are
constants: conduction energy = sum of p * t over the modes; each HS edge is
hard (its current flows forward), so it books the bilinear table at the
open-side voltage and the closed-side current: E (v / 600)(i / 50).
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

import pulsim

V, R, G_ON, G_OFF, F = 600.0, 10.0, 1e3, 1e-9, 10e3
E_ON, E_OFF = 4e-3, 3.5e-3
T_END = 0.975e-3   # HS on [0, 50 us), ...; 10 turn-offs, 9 turn-ons, no edge at t_end
ENGINES = ("pwl", "trbdf2", "dsed", "auto")


def _env() -> dict[str, str]:
    """The child imports the same pulsim as this session (test_ai_guide)."""
    env = dict(os.environ)
    parent = str(Path(pulsim.__file__).parents[1])
    env["PYTHONPATH"] = parent + os.pathsep + env.get("PYTHONPATH", "")
    return env


def _child(case: str, engine: str) -> dict:
    proc = subprocess.run([sys.executable, __file__, case, engine], env=_env(),
                          capture_output=True, timeout=300, text=True)
    assert proc.returncode == 0, (
        f"{case} on {engine}: exit {proc.returncode} (a negative code is a signal: "
        f"the process crashed)\n{proc.stderr[-2000:]}")
    return json.loads(proc.stdout.strip().splitlines()[-1])


def truth() -> dict:
    """HS's booked energies over [0, T_END), closed form."""
    def mode(hs_on: bool):
        g_hs, g_ls = (G_ON, G_OFF) if hs_on else (G_OFF, G_ON)
        va = V * g_hs / (g_hs + g_ls + 1.0 / R)
        i = g_hs * (V - va)
        return va, i, i * (V - va)
    va_on, i_on, p_on = mode(True)
    va_off, _, p_off = mode(False)
    t_on, t_off = 10 * 0.5 / F, T_END - 10 * 0.5 / F
    v_open = V - va_off   # HS blocks this voltage when it is open
    return {"E_cond": p_on * t_on + p_off * t_off, "E_cond_per_s_on": p_on,
            "E_on": 9 * E_ON * (v_open / V) * (i_on / 50.0),
            "E_off": 10 * E_OFF * (v_open / V) * (i_on / 50.0),
            "n_on": 9, "n_off": 10}


# ---------------------------------------------------------------- the child

def _tab(e):
    import numpy as np
    return pulsim.LossTable(v_axis=[0.0, V], i_axis=[0.0, 50.0], tj_axis=[25.0],
                            energy=np.array([[[0.0], [0.0]], [[0.0], [e]]]))


def _build(case: str, engine: str):
    """case: '+'-joined parts. leg (always), load (R), diode (an antiparallel
    diode on LS), cap (a bus capacitor: one state), port (HS loss port),
    rec (a recorder), junction (the port deposits into a node)."""
    parts = set(case.split("+"))
    b = pulsim.CircuitBuilder()
    b.add_voltage_source("Vdc", "dcp", "gnd", V)
    b.add_switch("HS", "dcp", "a", G_ON, G_OFF)
    b.add_switch("LS", "a", "gnd", G_ON, G_OFF)
    if "load" in parts:
        b.add_resistor("Rl", "a", "gnd", R)
    if "diode" in parts:
        b.add_diode("Dl", "gnd", "a", 20.0, G_OFF, V_f=0.7)
    if "cap" in parts:
        b.add_capacitor("Cb", "dcp", "gnd", 1e-6)
    if "port" in parts:
        b.set_device_losses("HS", Tj=25.0, E_on=_tab(E_ON), E_off=_tab(E_OFF))
    if "junction" in parts or "thermal" in parts:   # the port deposits into node tj
        b.add_resistor("Rth", "tj", "gnd", 1.0)
        if "thermal" in parts:   # with a thermal capacitance: one (thermal) state
            b.add_capacitor("Cth", "tj", "gnd", 1e-4)
        b.set_device_losses("HS", junction="tj", E_on=_tab(E_ON), E_off=_tab(E_OFF))
    fn = pulsim.make_dead_time_pwm_pair_fn(F, 0.5, b.switch_index_of("HS"),
                                           b.switch_index_of("LS"),
                                           b.graph.num_switches, 0.0)
    kw = {"dt": 1e-7} if engine == "pwl" else {}
    if "rec" in parts:
        kw["recorders"] = [pulsim.Recorder("r", ["v(a)", "i(HS)"], keep="every",
                                           windows=[(0.0, T_END)])]
    return b, fn, kw


def _main(case: str, engine: str) -> None:
    import warnings
    b, fn, kw = _build(case, engine)
    out: dict = {}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = pulsim.simulate(b, t_end=T_END, switch_fn=fn, engine=engine, **kw)
    except (pulsim.UnsupportedError, ValueError) as e:
        out["refused"] = str(e)
        print(json.dumps(out))
        return
    out["n"] = len(res.times)
    out["h1"] = float(res.times[1] - res.times[0])
    for r in res._device_losses or []:
        if r.device == "HS":
            out.update(E_cond=r.E_cond_electrical, E_on=r.E_on, E_off=r.E_off,
                       n_on=r.n_on, n_off=r.n_off)
    if "rec" in case:
        out["rec"] = res.recorder_feed is not None
    print(json.dumps(out))


# ---------------------------------------------------------------- the tests


@pytest.mark.parametrize("engine", ENGINES)
def test_stateless_leg_loss_port_matches_the_closed_form(engine):
    """The reproducer: every engine runs, and books what the hand truth says."""
    got, want = _child("load+port", engine), truth()
    assert (got["n_on"], got["n_off"]) == (want["n_on"], want["n_off"])
    for k in ("E_on", "E_off"):
        assert got[k] == pytest.approx(want[k], rel=1e-9), k
    if engine == "dsed":   # each constant-power mode integrated in closed form
        assert got["E_cond"] == pytest.approx(want["E_cond"], rel=1e-12)
        return
    # pwl / trbdf2 (and 'auto', which routes to trbdf2 here) start from the
    # zero vector: their t = 0 sample reads p = 0 (sources off), so the first
    # step's quadrature misses at most p_on x that step (measured: pwl 2/3 dt
    # at every t_end, with or without a state; trbdf2 1/(2 sqrt 2) h1).
    # pwl also reads each edge's step with the edge's one-sided sample:
    # +-p_on dt / 2 per edge (a turn-off loses it, a turn-on gains it).
    p_on = want["E_cond_per_s_on"]
    miss = want["E_cond"] - got["E_cond"]
    if engine == "pwl":
        # release: the restart rule starts consistently and reads each edge at
        # its limits: the closed form to rounding (measured -2.6e-16 J of
        # 1.8e-3 J; it missed 4.2e-7 J, within p_on dt (1 + 1/2), without it)
        assert abs(miss) <= 1e-9 * want["E_cond"], miss
        return
    assert 0.0 <= miss <= p_on * got["h1"] * (1 + 1e-9), (miss, got["h1"])


@pytest.mark.parametrize("case", ["port", "diode+port", "port+rec", "load+rec",
                                  "load+port+rec", "cap+port", "load+thermal"])
@pytest.mark.parametrize("engine", ["dsed", "exact", "auto"])
def test_stateless_variants_run_or_refuse_by_name(case, engine):
    """The boundary: no load (leakage only), a diode (still no state), a
    recorder with and without the port, one capacitor (one state)."""
    got = _child(case, engine)
    assert "refused" not in got, got
    if "port" in case or "thermal" in case:
        assert got["n_off"] == 10 and got["n_on"] == 9
        assert math.isfinite(got["E_cond"])


def test_stateless_junction_without_capacitance_is_refused_by_name():
    """A deposit node with no thermal capacitance: refused by name before the run."""
    got = _child("load+junction", "dsed")
    assert "refused" in got and "HS" in got["refused"], got


if __name__ == "__main__":
    _main(sys.argv[1], sys.argv[2])
