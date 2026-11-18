"""G22: a Circuit is immutable across runs. Every simulate() of a circuit
with a stateful C block starts the block from its declared initial
state, so two runs are bit-identical on every engine that runs C blocks,
for a Python step, inline C and an export_c library; only resume_from
continues the block (explicit carry-over). Before: the block's state
carried over (reported 1.383 V, then 1.785 V; _py here 0.358, 0.466,
0.574 V on three runs)."""
from __future__ import annotations

import sys
import warnings

import numpy as np
import pytest

import pulsim as p
from pulsim.c_block import _find_compiler

DT = 1e-6


def _rc():
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "u", "gnd", 1.0)
    b.add_resistor("R", "u", "o", 1e3)
    b.add_capacitor("C", "o", "gnd", 1e-7)
    b.add_resistor("Rc", "c", "gnd", 1.0)
    return b


def _py(b):
    def step(t, dt, inp, out, state):     # an integrator: state matters
        state["acc"] = state.get("acc", 0.25) + inp[0] * 1e3 * dt
        out[0] = state["acc"]
    p.add_c_block(b, inputs=[("v", "o")], outputs=[("v", "c", "gnd")],
                  dt=1e-5, fn=step, state={"acc": 0.25}, name="ctl")


def _inline(b):
    p.add_c_block(b, inputs=[("v", "o")], outputs=[("v", "c", "gnd")],
                  dt=1e-5, name="ctl", code="""
        double* acc = (double*)*state;
        if (acc == 0) { acc = (double*)calloc(1, sizeof(double)); *state = acc; }
        *acc += in[0] * 1e3 * dt;
        out[0] = *acc;""")


def _exported(tmp_path):
    import subprocess
    from pulsim.c_export import export_c
    ch = p.MixedDomainBlockChain(rate=p.Rate(1e-5))
    ch.add("i", p.Integrator(gain=1e3), inputs=dict(x="channel:u", dt="dt"),
           output="y")
    ex = export_c(ch, _rc(), outputs=["y"])
    src = ex.write(tmp_path / "integ.c")
    so = tmp_path / "integ.so"
    r = subprocess.run([_cc(), "-std=c99", "-O2", "-ffp-contract=off",
                        "-shared", "-fPIC", str(src), "-o", str(so), "-lm"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr

    def add(b):
        p.add_c_block(b, inputs=[("v", "o")], outputs=[("v", "c", "gnd")],
                      dt=1e-5, lib=str(so), name="ctl")
    return add


def _cc():
    cc = _find_compiler("c")
    if cc is None:
        if sys.platform == "win32":
            pytest.skip("no C compiler on PATH (Windows runner)")
        pytest.fail("no C compiler found (Linux and macOS runners have one)")
    return cc


ENGINES = [("pwl", dict(dt=DT)), ("trbdf2", {})]


def _two_runs(add, engine, kw):
    b = _rc()
    add(b)
    c = b.build()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r1 = p.simulate(c, 2e-4, engine=engine, **kw)
        r2 = p.simulate(c, 2e-4, engine=engine, **kw)
    return np.asarray(r1.states), np.asarray(r2.states), c


@pytest.mark.parametrize("engine,kw", ENGINES)
@pytest.mark.parametrize("kind", ["python", "inline", "exported"])
def test_simulating_a_circuit_twice_is_bit_identical(kind, engine, kw,
                                                     tmp_path):
    if kind != "python":
        _cc()
    add = {"python": _py, "inline": _inline}.get(kind) or _exported(tmp_path)
    x1, x2, c = _two_runs(add, engine, kw)
    assert x1.shape == x2.shape
    assert x1.tobytes() == x2.tobytes()
    # control: the block integrates (its output moves), so a carried-over
    # state would show
    j = c.to_builder().node_id_of("c")
    assert abs(x1[-1, j] - x1[0, j]) > 1e-3


def test_resume_from_continues_the_block():
    # carry-over is explicit: a resumed run continues the block's state
    b = _rc()
    _py(b)
    c = b.build()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        full = p.simulate(c, 2e-4, dt=DT, engine="pwl")
        # off a block tick (on one, label time is not exact: G22)
        half = p.simulate(c, 1.05e-4, dt=DT, engine="pwl")
        rest = p.simulate(c, 2e-4, dt=DT, engine="pwl",
                          resume_from=half.final_snapshot)
    j = c.to_builder().node_id_of("c")
    xf, xr = np.asarray(full.states), np.asarray(rest.states)
    assert xr[-1, j] == pytest.approx(xf[-1, j], rel=1e-9)



@pytest.mark.xfail(strict=True, reason=(
    "batch-11 review: a SolverSnapshot carries no C-block state; a resume "
    "continues the blocks from the circuit's LAST run (KNOWN_LIMITATIONS, "
    "control-graph track)"))
def test_resume_from_an_older_snapshot_continues_that_runs_block():
    b = _rc()
    _py(b)
    c = b.build()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        full = p.simulate(c, 2e-4, dt=DT, engine="pwl")
        half = p.simulate(c, 1.05e-4, dt=DT, engine="pwl")
        p.simulate(c, 1.5e-4, dt=DT, engine="pwl")        # another run between
        rest = p.simulate(c, 2e-4, dt=DT, engine="pwl",
                          resume_from=half.final_snapshot)
    j = c.to_builder().node_id_of("c")
    xf, xr = np.asarray(full.states), np.asarray(rest.states)
    assert xr[-1, j] == pytest.approx(xf[-1, j], rel=1e-9)

def test_two_circuits_sharing_an_exported_library_start_clean(tmp_path):
    # one static state behind both circuits: B's first run must not start
    # from A's last state (review of the merge: the first run was not reset)
    _cc()
    add = _exported(tmp_path)
    ba, bb = _rc(), _rc()
    add(ba)
    add(bb)
    ca, cb = ba.build(), bb.build()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        xa = np.asarray(p.simulate(ca, 2e-4, dt=DT, engine="pwl").states)
        xb = np.asarray(p.simulate(cb, 2e-4, dt=DT, engine="pwl").states)
    assert xa.tobytes() == xb.tobytes()
