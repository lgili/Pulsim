"""Engine conformance: no silent cell (add-engine-conformance-matrix).

For every device kind and every engine, a transient run either agrees
with the reference engine or raises `pulsim.UnsupportedError` naming the
engine and the device; for every feature keyword, the run either honours
it or raises the same error naming the keyword. Which outcome is expected
is read from the kernel's table, `pulsim.engine_capabilities()`, so a
wrong table entry fails here too.

Reference: `engine='pwl'` at a step for which halving it moves the
observable by less than a quarter of the tolerance. Agreement: at 20
fixed instants, each trace interpolated in TIME (never by index, never an
unweighted mean), the difference is at most 1 % of the reference span.
Presence control: without the device the observable moves by at least
10 % of the span, so a deleted device cannot pass as agreement.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

import pulsim as p
from pulsim import _pulsim as k

ENGINES = ("pwl", "trbdf2", "dsed", "bdf1")
TOL = 0.01          # of the reference span
PRESENCE = 0.10     # of the reference span


class _Closed:
    """A constant all-closed mask with an edge oracle (no edges)."""

    def __init__(self, n):
        self.m = p.SwitchStateMask(n)
        for i in range(n):
            self.m.set(i, True)

    def __call__(self, _t):
        return self.m

    def next_edge_after(self, _t):
        return math.inf


# ---------------------------------------------------------------------
# One circuit per kind: build(with_device) -> builder. `node` is the
# observable, `dev` the branch name the refusal must carry.
# ---------------------------------------------------------------------

def _rc_tail(b, node, R=1e3, C=100e-9):
    b.add_resistor("Rt", node, "o", R)
    b.add_capacitor("Ct", "o", "gnd", C)


def _clamp(add):
    # A sine through 1 kΩ into 10 nF: the device clips the waveform, so
    # the observable has a span (a DC clamp settles in a few µs and
    # leaves nothing to compare).
    def build(on):
        b = p.CircuitBuilder()
        b.add_sine_voltage_source("V", "in", "gnd", 0.0, 10.0, 1e3)
        b.add_resistor("R", "in", "o", 1e3)
        b.add_capacitor("C", "o", "gnd", 10e-9)
        if on:
            add(b)
        return b
    return build


def _gate_load(add):
    def build(on):
        b = p.CircuitBuilder()
        b.add_sine_voltage_source("V", "in", "gnd", 0.0, 10.0, 1e3)
        b.add_voltage_source("Vg", "g", "gnd", 15.0)
        b.add_resistor("R", "in", "o", 1e3)
        b.add_capacitor("C", "o", "gnd", 10e-9)
        if on:
            add(b)
        return b
    return build


def _res(on):
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "in", "gnd", 10.0)
    b.add_resistor("R1", "in", "o", 1e3)
    b.add_capacitor("C", "o", "gnd", 100e-9)
    if on:
        b.add_resistor("Rd", "o", "gnd", 1e3)
    return b


def _vsrc(on):
    b = p.CircuitBuilder()
    if on:
        b.add_voltage_source("Vd", "in", "gnd", 10.0)
    b.add_resistor("R1", "in", "o", 1e3)
    b.add_capacitor("C", "o", "gnd", 100e-9)
    return b


def _switch(on):
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "in", "gnd", 10.0)
    b.add_resistor("Rb", "in", "gnd", 1e3)  # keeps "in" stamped either way
    if on:
        b.add_switch("Sd", "in", "a", 1e3, 1e-9)
    b.add_resistor("R1", "a", "o", 1e3)
    b.add_capacitor("C", "o", "gnd", 100e-9)
    return b


def _cap(on):
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "in", "gnd", 10.0)
    b.add_resistor("R1", "in", "o", 1e3)
    b.add_resistor("R2", "o", "gnd", 1e3)
    if on:
        b.add_capacitor("Cd", "o", "gnd", 100e-9)
    return b


def _ind(on):
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "in", "gnd", 10.0)
    b.add_resistor("R1", "in", "o", 10.0)
    b.add_resistor("R2", "o", "gnd", 1e3)
    if on:
        b.add_inductor("Ld", "o", "gnd", 1e-3)
    return b


def _isrc(on):
    b = p.CircuitBuilder()
    b.add_resistor("R1", "o", "gnd", 1e3)
    b.add_capacitor("C", "o", "gnd", 100e-9)
    if on:
        b.add_current_source("Id", "gnd", "o", 1e-3)
    return b


def _src_rc(add):
    def build(on):
        b = p.CircuitBuilder()
        b.add_resistor("Rb", "in", "gnd", 1e6)
        b.add_resistor("R1", "in", "o", 1e3)
        b.add_capacitor("C", "o", "gnd", 100e-9)
        if on:
            add(b)
        return b
    return build


def _vcvs(on):
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "a", "gnd", 1.0)
    b.add_resistor("Rx", "x", "gnd", 1e6)
    if on:
        b.add_vcvs("Ed", "a", "gnd", "x", "gnd", 5.0)
    _rc_tail(b, "x")
    return b


def _xfmr(on):
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "in", "gnd", 10.0)
    b.add_resistor("Rp", "in", "gnd", 1e3)
    b.add_resistor("Rx", "x", "gnd", 1e6)
    if on:
        b.add_ideal_transformer("Td", "in", "gnd", "x", "gnd", 2.0)
    _rc_tail(b, "x")
    return b


def _coupled(on):
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("V", "in", "gnd", 0.0, 10.0, 1e3)
    b.add_resistor("Rp", "in", "p", 1.0)
    b.add_resistor("Rx", "x", "gnd", 100.0)
    if on:
        b.add_transformer("Kd", "p", "gnd", "x", "gnd", 1e-3, 1e-3, 0.9)
    else:
        b.add_resistor("Rq", "p", "gnd", 100.0)
    return b


def _sat(on):
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "in", "gnd", 10.0)
    b.add_resistor("R1", "in", "o", 10.0)
    b.add_resistor("R2", "o", "gnd", 1e3)
    if on:
        b.add_saturable_inductor("Ld", "o", "gnd", 1e-3, 0.5, 1e-4)
    return b


def _pmsm(on):
    b = p.CircuitBuilder()
    for i, node in enumerate(("ua", "ub", "uc")):
        b.add_sine_voltage_source(f"Vs{i}", node, "gnd", 0.0, 12.0, 30.0,
                                  -2.0 * math.pi / 3.0 * i)
        b.add_resistor(f"Rs{i}", node, node + "2", 1.0)
        b.add_resistor(f"Rl{i}", node + "2", "gnd", 1e3)
    if on:
        b.add_pmsm_mna("Md", "ua2", "ub2", "uc2", "nn", "w", "th",
                       R_s=0.5, L_d=2e-3, L_q=2e-3, psi_pm=0.05,
                       pole_pairs=4, J=1e-3, B=1e-4)
    return b


def _induction(on):
    b = p.CircuitBuilder()
    for i, node in enumerate(("ua", "ub", "uc")):
        b.add_sine_voltage_source(f"Vs{i}", node, "gnd", 0.0, 40.0, 60.0,
                                  -2.0 * math.pi / 3.0 * i)
        b.add_resistor(f"Rs{i}", node, node + "2", 1.0)
        b.add_resistor(f"Rl{i}", node + "2", "gnd", 1e3)
    if on:
        b.add_induction_machine_mna("Md", "ua2", "ub2", "uc2", "nn", "w",
                                    R_s=0.5, L_ls=2e-3, R_r=0.4, L_lr=2e-3,
                                    L_m=60e-3, pole_pairs=2, J=1e-2)
    return b


def _junction_cap(on):
    # tasks.md 1.1: 10 V, 1 kΩ, 100 nF ∥ Cn(1 µF, 1 V, 0.5), 1 MΩ load.
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "in", "gnd", 10.0)
    b.add_resistor("R", "in", "o", 1e3)
    b.add_capacitor("C", "o", "gnd", 100e-9)
    b.add_resistor("Rl", "o", "gnd", 1e6)
    if on:
        b.add_nonlinear_capacitor("Cn", "o", "gnd", 1e-6, 1.0, 0.5)
    return b


def _ndiode_clamp(on):
    # The retired law clamps both half-cycles of a sine, so a slow DC
    # charge (τ = 1 ms) is what gives this kind an observable span.
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "in", "gnd", 10.0)
    b.add_resistor("R", "in", "o", 1e3)
    b.add_capacitor("C", "o", "gnd", 1e-6)
    if on:
        b.add_nonlinear_diode("Dd", "o", "gnd", k.IdealDiodeParams(),
                              allow_non_passive=True)
    return b


def _ctl(add):
    # add-controlled-source-set: a 1 V source (its current, 1 mA through
    # Ra, is the CCCS/CCVS control) drives "x" only through the device.
    def build(on):
        b = p.CircuitBuilder()
        b.add_voltage_source("V", "a", "gnd", 1.0)
        b.add_resistor("Ra", "a", "gnd", 1e3)
        b.add_resistor("Rx", "x", "gnd", 1e3)
        if on:
            add(b)
        _rc_tail(b, "x")
        return b
    return build


def _gyrator(on):
    # A gyrator loaded by 1 uF is an inductor of C/g^2 = 1.11 mH at port 1.
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "in", "gnd", 10.0)
    b.add_resistor("R1", "in", "o", 10.0)
    b.add_resistor("R2", "o", "gnd", 1e3)
    if on:
        b.add_gyrator("Gd", "o", "gnd", "y", "gnd", 0.03)
        b.add_capacitor("Cy", "y", "gnd", 1e-6)
        b.add_resistor("Ry", "y", "gnd", 1e6)
    return b


def _iwave(add):
    def build(on):
        b = p.CircuitBuilder()
        b.add_resistor("R1", "o", "gnd", 1e3)
        b.add_capacitor("C", "o", "gnd", 100e-9)
        if on:
            add(b)
        return b
    return build


# kind -> (build, observable node, device branch, t_end, reference dt,
#          needs a switch_fn)
CELLS = {
    "Resistor": (_res, "o", "Rd", 5e-4, 1e-7, False),
    "VoltageSource": (_vsrc, "o", "Vd", 5e-4, 1e-7, False),
    "Switch": (_switch, "o", "Sd", 5e-4, 1e-7, True),
    "Capacitor": (_cap, "o", "Cd", 5e-4, 1e-7, False),
    "Inductor": (_ind, "o", "Ld", 5e-4, 1e-7, False),
    "Diode": (_clamp(lambda b: b.add_diode("Dd", "o", "gnd", 1e3, 1e-9,
                                           0.701, 0.7)),
              "o", "Dd", 2e-3, 2e-7, False),
    "NonlinearDiode": (_ndiode_clamp, "o", "Dd", 2e-4, 1e-7, False),
    "CurrentSource": (_isrc, "o", "Id", 5e-4, 1e-7, False),
    "PWMVoltageSource": (
        _src_rc(lambda b: b.add_pwm_voltage_source(
            "Vd", "in", "gnd", 10.0, 0.0, 2e3, 0.3)),
        "o", "Vd", 2e-3, 2e-7, False),
    "SineVoltageSource": (
        _src_rc(lambda b: b.add_sine_voltage_source(
            "Vd", "in", "gnd", 0.0, 10.0, 1e3)),
        "o", "Vd", 2e-3, 2e-7, False),
    "PulseVoltageSource": (
        _src_rc(lambda b: b.add_pulse_voltage_source(
            "Vd", "in", "gnd", 0.0, 10.0, 1e-4, 2e-4, 0.0, 2e-5, 2e-5)),
        "o", "Vd", 5e-4, 1e-7, False),
    "MosfetLevel1": (
        _gate_load(lambda b: b.add_mosfet_level1(
            "Md", "o", "gnd", "g", 1e-2, 2.0, with_body_diode=False)),
        "o", "Md", 2e-3, 2e-7, False),
    "IgbtLevel1": (
        _gate_load(lambda b: b.add_igbt_level1("Qd", "o", "gnd", "g")),
        "o", "Qd", 2e-3, 2e-7, False),
    "VCVS": (_vcvs, "o", "Ed", 5e-4, 1e-7, False),
    "SaturableInductor": (_sat, "o", "Ld", 5e-4, 5e-8, False),
    "NonlinearCapacitor": (_junction_cap,
        "o", "Cn", 1e-4, 1e-7, False),
    "ShockleyDiode": (_clamp(lambda b: b.add_shockley_diode(
        "Dd", "o", "gnd")), "o", "Dd", 2e-3, 2e-7, False),
    "LauritzenDiode": (_clamp(lambda b: b.add_lauritzen_diode(
        "Dd", "o", "gnd")), "o", "Dd", 2e-3, 2e-7, False),
    "PmsmMna": (_pmsm, "ua2", "Md", 2e-2, 1e-5, False),
    "InductionMachineMna": (_induction, "ua2", "Md", 2e-2, 1e-5, False),
    "ExternalDevice": (lambda on: _external(on), "o", "Rd", 5e-4, 1e-7, False),
    "IdealTransformer": (_xfmr, "o", "Td", 5e-4, 1e-7, False),
    "VCCS": (_ctl(lambda b: b.add_vccs("Gd", "a", "gnd", "x", "gnd", 5e-3)),
             "o", "Gd", 5e-4, 1e-7, False),
    "CCCS": (_ctl(lambda b: b.add_cccs("Fd", "V", "x", "gnd", 5.0)),
             "o", "Fd", 5e-4, 1e-7, False),
    "CCVS": (_ctl(lambda b: b.add_ccvs("Hd", "V", "x", "gnd", 5e3)),
             "o", "Hd", 5e-4, 1e-7, False),
    "Gyrator": (_gyrator, "o", "Gd", 5e-4, 1e-7, False),
    "SineCurrentSource": (
        _iwave(lambda b: b.add_sine_current_source(
            "Id", "o", "gnd", 0.0, 10e-3, 1e3)),
        "o", "Id", 2e-3, 2e-7, False),
    "PulseCurrentSource": (
        _iwave(lambda b: b.add_pulse_current_source(
            "Id", "o", "gnd", 0.0, 10e-3, 1e-4, 2e-4, 0.0, 2e-5, 2e-5)),
        "o", "Id", 5e-4, 1e-7, False),
    "PwlCurrentSource": (
        _iwave(lambda b: b.add_pwl_current_source(
            "Id", "o", "gnd", [0.0, 1e-4, 3e-4, 5e-4],
            [0.0, 10e-3, -5e-3, 0.0])),
        "o", "Id", 5e-4, 1e-7, False),
}
# Circuit-level cell (not a kind): coupled inductors.
COUPLED = (_coupled, "x", "Kd", 3e-3, 1e-6, False)


def _switch_fn(b, needs):
    if callable(needs):   # a gate schedule built for this circuit
        return needs(b)
    return _Closed(b.graph.num_switches) if needs else None


_XDEV = {}


def _external(on):
    """add-external-device-abi: an external 1 kOhm (the C test model
    `good_resistor`) where `_res` has its resistor; compiled once with the
    system C compiler, or taken from PULSIM_XDEV_TEST_LIBDIR."""
    import os
    import pathlib
    import shutil
    import subprocess
    import sys
    import tempfile
    b = _res(False)
    if not on:
        return b
    if "lib" not in _XDEV:
        d = os.environ.get("PULSIM_XDEV_TEST_LIBDIR")
        if d:
            _XDEV["lib"] = next(str(f) for f in pathlib.Path(d).iterdir()
                                if "pulsim_xdev_contract" in f.name)
        else:
            cc = shutil.which(os.environ.get("CC") or "cc")
            if cc is None or sys.platform == "win32":
                pytest.skip("no C compiler for the external device test model")
            root = pathlib.Path(__file__).resolve().parents[2]
            out = pathlib.Path(tempfile.mkdtemp()) / "contract.so"
            subprocess.run([cc, "-std=c99", "-shared", "-fPIC",
                            f"-I{root / 'core' / 'include'}",
                            str(root / "core/tests/xdev/contract_models.c"),
                            "-o", str(out), "-lm"],
                           check=True, capture_output=True, text=True)
            _XDEV["lib"] = str(out)
    b.add_external_device("Rd", ["o", "gnd"], _XDEV["lib"], "good_resistor",
                          {"G": 1e-3})
    return b


def _run(build, node, t_end, dt, needs_sw, engine, on=True):
    b = build(on)
    sw = _switch_fn(b, needs_sw)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if engine == "bdf1":
            opts = k.SimulationOptions(t_start=0.0, t_end=t_end, dt=dt)
            fn = sw if sw is not None else (
                lambda _t, _n=b.graph.num_switches: p.SwitchStateMask(_n))
            r = k.run_transient_bdf1(b, opts, fn)
            r._builder = b
        elif engine == "pwl":
            r = p.simulate(b, t_end, dt=dt, engine="pwl", switch_fn=sw)
        else:
            r = p.simulate(b, t_end, engine=engine, switch_fn=sw)
    return np.asarray(r.times, float), np.asarray(r.v(node), float)


def _at(t, v, ts):
    return np.interp(ts, t, v)


def _reference(cell):
    build, node, _dev, t_end, dt, sw = cell
    ts = np.linspace(t_end / 20, t_end, 20)
    ref = _at(*_run(build, node, t_end, dt, sw, "pwl"), ts)
    half = _at(*_run(build, node, t_end, dt / 2, sw, "pwl"), ts)
    span = float(ref.max() - ref.min()) or float(np.abs(ref).max())
    assert span > 0
    # The reference is converged: halving dt moves it < TOL/4 of span.
    assert np.max(np.abs(half - ref)) < TOL / 4 * span, "reference dt"
    return ts, ref, span


_REF_CACHE: dict = {}


def _ref(name, cell):
    if name not in _REF_CACHE:
        _REF_CACHE[name] = _reference(cell)
    return _REF_CACHE[name]


def _check_cell(name, cell, engine, supported):
    build, node, dev, t_end, dt, sw = cell
    ts, ref, span = _ref(name, cell)
    if engine == "pwl":
        # Presence control, on the reference engine.
        off = _at(*_run(build, node, t_end, dt, sw, "pwl", on=False), ts)
        assert np.max(np.abs(off - ref)) >= PRESENCE * span, (
            f"{name}: the circuit does not depend on its device")
    try:
        got = _at(*_run(build, node, t_end, dt, sw, engine), ts)
    except p.UnsupportedError as e:
        assert not supported, (
            f"{name} on {engine}: refused but the table says supported: {e}")
        assert e.engine == engine
        # A multi-branch device (the PMSM) names its first branch.
        assert e.device == dev or e.device.startswith(dev + "_"), (
            e.device, dev)
        return
    assert supported, (
        f"{name} on {engine}: ran but the table says unsupported "
        "(a silent cell)")
    err = float(np.max(np.abs(got - ref)))
    assert err <= TOL * span, (
        f"{name} on {engine}: max error {err:.4g} > {TOL * span:.4g} "
        f"(span {span:.4g})")


def test_table_has_one_row_per_kind_and_names_match_the_enum():
    caps = p.engine_capabilities()
    names = [m for m in k.StoredKind.__members__]
    assert list(caps) == names
    for row in caps.values():
        assert set(row) == {"class", "pwl", "trbdf2", "dsed", "bdf1",
                            "ac_sweep"}
        assert row["pwl"], "the reference engine runs every kind"
    assert set(CELLS) == set(caps), "a kind without a conformance cell"


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("kind", list(CELLS))
def test_device_cell(kind, engine):
    supported = p.engine_capabilities()[kind][engine]
    _check_cell(kind, CELLS[kind], engine, supported)


@pytest.mark.parametrize("engine", ENGINES)
def test_coupled_inductor_cell(engine):
    # Circuit-level limit: refused on bdf1 (no mutual stamp: it read 9.38 V
    # off an 18 V span, silently, before this change). dsed extracts the full
    # inductance matrix (add-exact-propagator-strategy, coupled inductors).
    if engine == "bdf1":
        with pytest.raises(p.UnsupportedError) as ei:
            _run(*_args(COUPLED), "bdf1")
        assert (ei.value.engine, ei.value.feature) == (
            "bdf1", "coupled_inductors")
        return
    _check_cell("coupled", COUPLED, engine, True)


def _args(cell):
    build, node, _dev, t_end, dt, sw = cell
    return build, node, t_end, dt, sw


# ---------------------------------------------------------------------
# The two silent device cells this change closes, with their numbers.
# ---------------------------------------------------------------------

def test_junction_capacitor_numbers_and_refusals():
    cell = CELLS["NonlinearCapacitor"]
    build, node, _dev, t_end, dt, _sw = cell
    for eng in ("pwl", "trbdf2"):
        t, v = _run(*_args(cell), eng)
        assert float(np.interp(1e-4, t, v)) == pytest.approx(1.02, abs=0.1)
    t, v = _run(build, node, t_end, dt, False, "pwl", on=False)
    assert float(np.interp(1e-4, t, v)) == pytest.approx(6.30, abs=0.1)
    for eng in ("dsed", "bdf1"):
        with pytest.raises(p.UnsupportedError) as ei:
            _run(*_args(cell), eng)
        e = ei.value
        assert (e.engine, e.device, e.kind) == (eng, "Cn",
                                                "NonlinearCapacitor")
        assert isinstance(e, ValueError)


def _shockley_clamp():
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "in", "gnd", 10.0)
    b.add_resistor("R", "in", "o", 1e3)
    b.add_shockley_diode("D", "o", "gnd")
    b.add_capacitor("C", "o", "gnd", 100e-9)
    return b


def test_refresh_off_is_refused_and_the_raw_binding_stamps_by_default():
    with pytest.raises(p.UnsupportedError) as ei:
        p.simulate(_shockley_clamp(), 1e-4, dt=1e-6, engine="pwl",
                   enable_nonlinear_refresh=False)
    assert (ei.value.engine, ei.value.feature, ei.value.device,
            ei.value.kind) == ("pwl", "enable_nonlinear_refresh", "D",
                               "ShockleyDiode")
    with pytest.raises(p.UnsupportedError) as ei:
        p.simulate(_shockley_clamp(), 1e-4, engine="trbdf2",
                   enable_nonlinear_refresh=False)
    assert ei.value.engine == "trbdf2"
    # The raw binding: default arguments now stamp the diode (it read
    # 10.000 V, the open circuit, before).
    b = _shockley_clamp()
    cache = p.PwlStateSpaceCache(b.graph, b.pool)
    cache.build(1e-6)
    opts = k.SimulationOptions(t_start=0.0, t_end=1e-3, dt=1e-6)
    r = k.run_transient(cache, b.graph, b.pool, opts,
                        lambda _t: p.SwitchStateMask(0))
    r._builder = b
    assert float(r.v("o")[-1]) == pytest.approx(0.594, abs=0.01)


def test_refresh_flag_does_not_move_a_linear_circuit():
    runs = [np.asarray(p.simulate(_rc(), 1e-4, dt=1e-6, engine="pwl",
                                  enable_nonlinear_refresh=f).states)
            for f in (None, True, False)]
    assert np.array_equal(runs[0], runs[1])
    assert np.array_equal(runs[0], runs[2])


# ---------------------------------------------------------------------
# Switch-function adapter: NativeMultiMaskPwm edges on trbdf2.
# ---------------------------------------------------------------------

def _half_bridge(L):
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "in", "gnd", 10.0)
    b.add_switch("S1", "in", "sw", 1e3, 1e-9)
    b.add_switch("S2", "sw", "gnd", 1e3, 1e-9)
    b.add_inductor("L", "sw", "o", L)
    b.add_resistor("R", "o", "gnd", 10.0)
    return b


def test_multimask_narrow_window_lands_every_edge_on_trbdf2():
    T = 1e-3
    b = _half_bridge(1.0)
    n = b.graph.num_switches
    hi = p.SwitchStateMask(n)
    hi.set(0, True)
    lo = p.SwitchStateMask(n)
    lo.set(1, True)
    fn = p.NativeMultiMaskPwm(T, [0.5, 0.501, 1.0], [hi, lo, hi])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = p.simulate(b, 20 * T, engine="trbdf2", switch_fn=fn)
    assert r._trbdf2_stats["n_gate_events"] == 40      # was 0
    t = np.asarray(r.times)
    edges = [(j + f) * T for j in range(20) for f in (0.5, 0.501)]
    worst = max(float(np.min(np.abs(t - e))) for e in edges)
    assert worst <= 1e-12


def test_pwm2_on_trbdf2_unchanged_by_the_adapter():
    T, D = 1e-4, 0.37
    b = _half_bridge(1e-3)
    fn = p.NativePwm2Switch(T, D, b.graph.num_switches)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = p.simulate(b, 10 * T, engine="trbdf2", switch_fn=fn)
    st = r._trbdf2_stats
    # Re-pinned by update-trbdf2-post-event-restart / error-control
    # (integration batch 5): (1104, 2254, 20) -> (1031, 2058, 20); the
    # post-edge restart removes the post-edge step collapse; same 20 edges.
    assert (len(r.times), st["n_solves"], st["n_gate_events"]) == (
        1031, 2058, 20)


# ---------------------------------------------------------------------
# Feature cells: honoured or refused by name, on every engine.
# ---------------------------------------------------------------------

def _rc():
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "in", "gnd", 10.0)
    b.add_resistor("R", "in", "o", 1e3)
    b.add_capacitor("C", "o", "gnd", 100e-9)
    return b


def _sim(engine, b=None, **kw):
    b = b if b is not None else _rc()
    if engine == "pwl":
        kw.setdefault("dt", 1e-6)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return p.simulate(b, 2e-3, engine=engine, **kw)


class _CBlock:
    """Stands in for a registered C block: the refusals fire before it
    is used (the pwl path is covered by test_c_block*.py)."""

    step_observer = None
    b_extra_fn = None


# keyword -> (kwargs, engines that refuse it)
_REFUSED = {
    "store_every": ({"store_every": 4}, ("trbdf2", "dsed")),
    "start_from_dc_op": ({"start_from_dc_op": True}, ("trbdf2", "dsed")),
    "max_dt_halvings": ({"max_dt_halvings": 2}, ("trbdf2", "dsed")),
    "on_cache": ({"on_cache": lambda *a: None}, ("trbdf2", "dsed")),
    "should_continue": ({"should_continue": lambda: True}, ("dsed",)),
    "step_observer": ({"step_observer": lambda t, x: None},
                      ("trbdf2", "dsed")),
    "progress": ({"progress": 50}, ("trbdf2",)),
    "inductor_abs_clamp": ({"inductor_abs_clamp": 10.0},
                           ("trbdf2", "dsed")),
    "inductor_freeze_di_max": ({"inductor_freeze_di_max": 10.0},
                               ("trbdf2", "dsed")),
    "enable_substep_state_correction": (
        {"enable_substep_state_correction": True}, ("trbdf2", "dsed")),
    "discontinuity_restart": (
        {"discontinuity_restart": True}, ("trbdf2", "dsed")),
    "resolution_diagnostics": (
        {"resolution_diagnostics": True}, ("trbdf2", "dsed")),
    # add-complementarity-mode-selection: the shadow runs on pwl, trbdf2 and
    # the dsed event integrator; this RC has no diode: nothing to select
    # (trbdf2 refuses that by name, dsed routes to "auto", which has no hook)
    "mode_selection": ({"mode_selection": "shadow"}, ("trbdf2", "dsed")),
}


@pytest.mark.parametrize("kw", [
    {"discontinuity_restart": True, "enable_substep_state_correction": True},
    {"resolution_diagnostics": True},
])
def test_fixed_step_features_are_honoured_on_pwl(kw):
    """The pwl side of the two cells above: split and diagnostics run and
    leave their trace (the split's counters, the resolution record)."""
    r = _sim("pwl", **kw)
    if "resolution_diagnostics" in kw:
        assert r.resolution is not None
    else:
        assert r.n_split_steps == 0 and "unlocated_gate_steps" in r.counters


@pytest.mark.parametrize("engine", ("trbdf2", "dsed"))
@pytest.mark.parametrize("feature", list(_REFUSED))
def test_fixed_step_feature_refused_by_name(feature, engine):
    kw, refusing = _REFUSED[feature]
    if engine not in refusing:
        pytest.skip("honoured on this engine")
    with pytest.raises(p.UnsupportedError) as ei:
        _sim(engine, **kw)
    assert (ei.value.engine, ei.value.feature) == (engine, feature)


@pytest.mark.parametrize("engine", ("pwl", "trbdf2", "auto", "dsed"))
def test_sensitivities_cell(engine):
    """add-parameter-sensitivities: honoured on the exact path only;
    every other engine refuses the keyword by name (a gradient there
    would be a difference of runs with events on the step grid)."""
    if engine != "dsed":
        with pytest.raises(p.UnsupportedError) as ei:
            _sim(engine, sensitivities=["R.R"])
        assert (ei.value.engine, ei.value.feature) == (
            engine, "sensitivities")
        return
    r = _sim(engine, sensitivities=["R.R"])
    assert r.sensitivity.params == ["R.R"]
    assert r.sensitivity.states.shape[2] == 1


# add-periodic-steady-state-shooting group 5: periodic_steady_state(engine=)
# and SteadySpec(method='pss') x engine. Honoured on the exact stepper
# ('exact' and 'dsed' one engine, bit-identical), 'pwl' and 'trbdf2' (the
# orbit of that engine's transient); anything else refused before a run.
_PSS_T = 1e-5


def _pss_buck(d=None, shockley=False):
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", 24.0)
    b.add_switch("S", "vin", "sw", 1e3, 1e-9)
    if shockley:
        b.add_shockley_diode("D1", "gnd", "sw")
    else:
        b.add_diode("D1", "gnd", "sw", 1e3, 1e-9)
    b.add_inductor("L", "sw", "vout", 100e-6)
    b.add_capacitor("C", "vout", "gnd", 47e-6)
    b.add_resistor("RL", "vout", "gnd", 50.0)
    return b, p.make_pwm_switch_fn(1 / _PSS_T, 0.3, b.switch_index_of("S"),
                                   b.graph.num_switches)


def _pss_kw(engine):
    return {"dt": _PSS_T / 100} if engine in ("pwl", "trbdf2") else {}


@pytest.mark.parametrize("engine", ("pwl", "trbdf2", "dsed", "exact", "bdf1"))
def test_periodic_steady_state_engine_cell(engine):
    b, sf = _pss_buck()
    if engine == "bdf1":
        with pytest.raises(ValueError, match="engine must be"):
            p.periodic_steady_state(b, period=_PSS_T, switch_fn=sf,
                                    engine=engine)
        return
    r = p.periodic_steady_state(b, period=_PSS_T, switch_fn=sf,
                                engine=engine, **_pss_kw(engine))
    v = r.x0[r.state_labels.index("v(vout)")]
    assert v == pytest.approx(8.99624708728943, rel=1e-6)
    assert r._ctx.loop == ("fd-broyden" if engine in ("pwl", "trbdf2")
                           else "kernel")
    # a Newton device is refused by name on every engine, before a run
    bs, sfs = _pss_buck(shockley=True)
    ok, why = p.periodic_steady_state_supported(
        bs, _PSS_T, sfs, engine=engine, **_pss_kw(engine))
    assert not ok and "D1" in why


def test_periodic_steady_state_exact_and_dsed_are_bit_identical():
    out = []
    for eng in ("dsed", "exact"):
        b, sf = _pss_buck()
        r = p.periodic_steady_state(b, period=_PSS_T, switch_fn=sf,
                                    engine=eng)
        out.append((r.x0, r.monodromy, r.n_period_runs))
    assert np.array_equal(out[0][0], out[1][0])
    assert np.array_equal(out[0][1], out[1][1]) and out[0][2] == out[1][2]


@pytest.mark.parametrize("engine", ("auto", "dsed", "exact", "pwl", "trbdf2"))
def test_steady_spec_pss_engine_cell(engine):
    from pulsim.design import Evaluator, SteadySpec

    def build(_d):
        return _pss_buck()
    recs = {}
    for eng in dict.fromkeys(("exact", engine)):
        recs[eng] = Evaluator(build, period=_PSS_T, loads=("RL",),
                              output="vout", engine=eng,
                              options=_pss_kw(eng),
                              steady=SteadySpec("pss")).evaluate({})
    r = recs[engine]
    assert r.status == "ok"
    route = {"auto": "dsed", "exact": "dsed"}.get(engine, engine)
    assert r.counters["route"] == f"pss/{route}"
    assert r.counters["pss_period_runs"] >= 1
    assert r.kpis["v_out"] == pytest.approx(recs["exact"].kpis["v_out"],
                                            rel=1e-6)
    if engine == "dsed":
        assert r.kpis == recs["exact"].kpis


@pytest.mark.parametrize("engine", ("trbdf2", "dsed"))
def test_mode_selection_lcp_refused_by_name_without_the_event_integrator(engine):
    # 'lcp' decides on the dsed event integrator only; this RC routes to "auto"
    with pytest.raises(p.UnsupportedError) as ei:
        _sim(engine, mode_selection="lcp")
    assert (ei.value.engine, ei.value.feature) == (engine, "mode_selection")


@pytest.mark.parametrize("mode", ("shadow", "lcp"))
def test_mode_selection_refused_on_a_diode_free_rk45_dsed_run(mode):
    # rk45 is the event integrator, but with no diode there is nothing to
    # select: refused by name rather than silently ignored
    with pytest.raises(p.UnsupportedError) as ei:
        _sim("dsed", mode_selection=mode, integrator="rk45")
    assert (ei.value.engine, ei.value.feature) == ("dsed", "mode_selection")


def _half_wave():
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("V", "in", "gnd", 0.0, 10.0, 500.0)
    b.add_diode("D", "in", "o", 1e3, 1e-9)
    b.add_resistor("R", "o", "gnd", 100.0)
    b.add_capacitor("C", "o", "gnd", 1e-6)
    return b


@pytest.mark.parametrize("mode", ("shadow", "lcp"))
def test_mode_selection_runs_on_a_diode_circuit_on_trbdf2(mode):
    # task 4.2 on trbdf2: its zero-time cascades are the sites (the cell
    # flipped from refuse to support)
    r = _sim("trbdf2", _half_wave(), mode_selection=mode)
    key = "mode_shadow" if mode == "shadow" else "mode_selection"
    ms = r.diagnostics[key]
    assert sum(ms["sites"].values()) > 0 and ms["errors"] == 0


@pytest.mark.parametrize("mode", ("shadow", "lcp"))
def test_mode_selection_refused_on_a_diode_free_pwl_run(mode):
    with pytest.raises(p.UnsupportedError) as ei:
        _sim("pwl", mode_selection=mode)
    assert (ei.value.engine, ei.value.feature) == ("pwl", "mode_selection")


@pytest.mark.parametrize("mode", ("shadow", "lcp"))
def test_mode_selection_runs_on_pwl(mode):
    # add-complementarity-mode-selection 4.2 / 6.3: the cell flipped from
    # refuse to support on the fixed engine.
    r = _sim("pwl", _half_wave(), mode_selection=mode)
    key = "mode_shadow" if mode == "shadow" else "mode_selection"
    assert r.diagnostics[key]["lcp_count"] > 0


@pytest.mark.parametrize("engine", ("pwl", "trbdf2", "dsed"))
def test_mode_guard_diagnostics_on_pwl_and_dsed(engine):
    # The default-mode guard (add-complementarity-mode-selection task 4,
    # 4b.3 on pwl) judges every settled diode mode and reports there;
    # trbdf2 has no hook yet and carries no such key.
    r = _sim(engine, _half_wave())
    g = (getattr(r, "diagnostics", None) or {}).get("mode_guard")
    if engine == "trbdf2":
        assert g is None
        return
    assert g["checks"] > 0 and g["repairs"] == 0 and g["repair_times"] == []
    assert g["settle_caps"] == 0 and g["lcp"]["capped"] == 0     # DECISIONS #24, #25
    rec = {"class": "passive_reciprocal", "device": "", "kind": ""}
    assert g["passivity"] == rec        # task 5.1 class
    r = _sim(engine, _half_wave(), mode_selection="lcp")
    assert r.diagnostics["mode_selection"]["lcp_count"] > 0
    assert r.diagnostics["mode_selection"]["capped"] == 0
    assert r.diagnostics["mode_selection"]["passivity"] == rec
    assert "mode_guard" not in r.diagnostics


@pytest.mark.parametrize("engine", ("pwl", "trbdf2", "dsed"))
def test_lcp_agrees_with_the_default_settle_on_every_engine(engine):
    # add-complementarity-mode-selection 6.3: one cell per engine (trbdf2
    # calls pwl::select_site at its zero-time cascades).
    r0 = _sim(engine, _half_wave())
    r1 = _sim(engine, _half_wave(), mode_selection="lcp")
    ms = r1.diagnostics["mode_selection"]
    assert ms["errors"] == 0 and ms["capped"] == 0 and ms.get("inconsistent", 0) == 0

    def avg(r):     # time-weighted, exactly clipped to [0, 2 ms]
        t, v = np.asarray(r.times), np.asarray(r.v("o"))
        return float(np.trapezoid(v, t)) / (t[-1] - t[0])
    # pwl, dsed: 0.0 measured; trbdf2 3.1e-9 (its sites decide event instants)
    assert avg(r1) == pytest.approx(avg(r0), rel=1e-6)


@pytest.mark.parametrize("engine", ("dsed", "exact"))
def test_c_blocks_refused(engine):
    # trbdf2 runs C blocks since task 9.3 (test_c_block_engines.py)
    b = _rc()
    b._c_blocks = [_CBlock()]
    with pytest.raises(p.UnsupportedError) as ei:
        _sim(engine, b)
    assert (ei.value.engine, ei.value.feature) == ("dsed", "c_blocks")


@pytest.mark.parametrize("vth", (0.0, 0.7))
def test_diode_drop_at_or_above_threshold_runs_on_dsed(vth):
    # DSED held such a diode ON through reverse bias: the sine clamp
    # read 0.70 V for the whole negative half (10.37 V off). V_f > V_th
    # was refused; V_th = V_f = 0.7 ran silently wrong (min 0.0 V, the
    # reference -9.98 V). add-exact-propagator-strategy D4 decides the
    # turn-off on the post-crossing side, so both now agree with pwl.
    cell = (_clamp(lambda b: b.add_diode("Dd", "o", "gnd", 1e3, 1e-9,
                                         vth, 0.7)),
            "o", "Dd", 2e-3, 2e-7, False)
    _check_cell(f"Diode V_th={vth} V_f=0.7", cell, "dsed", True)


def test_builder_initial_conditions_honoured_on_dsed():
    # add-exact-propagator-strategy D5: c0= maps onto the reduced
    # state (refused by name before).
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "in", "gnd", 10.0)
    b.add_resistor("R", "in", "o", 1e3)
    b.add_capacitor("C", "o", "gnd", 100e-9, c0=2.0)
    r = _sim("dsed", b)
    assert abs(np.asarray(r.v("o"))[0] - 2.0) < 1e-12


def test_observer_on_a_cadence_is_honoured_on_trbdf2():
    calls = []
    _sim("trbdf2", step_observer=lambda t, x: calls.append(t),
         controller_period=1e-4)
    assert len(calls) == 20


def test_observer_on_a_cadence_is_honoured_on_pwl():
    calls = []
    _sim("pwl", step_observer=lambda t, x: calls.append(t),
         controller_period=1e-4)
    assert len(calls) == 20


def _two_bucks():
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "in", "gnd", 24.0)
    for s in ("a", "b"):
        b.add_switch(f"Q{s}", "in", f"sw{s}", 1e2, 1e-9)
        b.add_diode(f"D{s}", "gnd", f"sw{s}", 1e2, 1e-9)
        b.add_inductor(f"L{s}", f"sw{s}", f"out{s}", 1e-3)
        b.add_capacitor(f"C{s}", f"out{s}", "gnd", 47e-6)
        b.add_resistor(f"R{s}", f"out{s}", "gnd", 10.0)
    return b


def _loop(b, out, sw, freq):
    return p.control.bind_pi_to_switch(
        b, pi=p.PIController(Kp=0.02, Ki=20.0, output_min=0.05,
                             output_max=0.95),
        measured=lambda x, i=b.node_id_of(out): x[i], setpoint=12.0,
        switch=sw, freq=freq)


def test_two_closed_loops_on_trbdf2():
    b = _two_bucks()
    la = _loop(b, "outa", "Qa", 20e3)
    lb = _loop(b, "outb", "Qb", 2e3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p.simulate(b, 10e-3, closed_loops=[la, lb], engine="trbdf2")
    assert (len(la.duty_history), len(lb.duty_history)) == (200, 20)


def test_loops_at_two_rates_on_trbdf2():
    b = _two_bucks()
    lb = _loop(b, "outb", "Qb", 2e3)
    mon = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p.simulate(b, 10e-3, closed_loops=[lb], engine="trbdf2",
                   switch_fn=None,
                   step_observer=lambda t, x: mon.append(t),
                   controller_period=5e-5)
    assert (len(lb.duty_history), len(mon)) == (20, 200)


def test_two_closed_loops_on_dsed():
    # add-unified-event-queue group 3: the exact engine's queue ticks them
    b = _two_bucks()
    la = _loop(b, "outa", "Qa", 20e3)
    lb = _loop(b, "outb", "Qb", 2e3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = p.simulate(b, 10e-3, closed_loops=[la, lb], engine="dsed")
    assert (len(la.duty_history), len(lb.duty_history)) == (200, 20)
    assert list(r.control_tick_counts) == [200, 20]


def test_resume_from_refused_on_dsed():
    snap = _sim("pwl").final_snapshot
    with pytest.raises(p.UnsupportedError) as ei:
        _sim("dsed", resume_from=snap)
    assert ei.value.feature == "resume_from"


def test_refusal_is_a_value_error_and_carries_four_fields():
    try:
        _sim("dsed", store_every=4)
    except ValueError as e:      # existing handlers keep catching it
        assert isinstance(e, p.UnsupportedError)
        assert (e.engine, e.feature, e.device, e.kind) == (
            "dsed", "store_every", "", "")
    else:  # pragma: no cover
        pytest.fail("no refusal")


def test_known_limitations_matrix_equals_the_kernel_table():
    import pathlib
    doc = (pathlib.Path(__file__).resolve().parents[2] / "docs"
           / "known-limitations.md").read_text(encoding="utf-8")
    block = doc.split("<!-- engine-capabilities:begin -->")[1].split(
        "<!-- engine-capabilities:end -->")[0]
    lines = [ln for ln in block.strip().splitlines()[2:]]
    cols = ("pwl", "trbdf2", "dsed", "bdf1", "ac_sweep")
    doc_table = {}
    for ln in lines:
        cells = [c.strip() for c in ln.strip("|").split("|")]
        doc_table[cells[0]] = {"class": cells[1], **{
            c: v == "yes" for c, v in zip(cols, cells[2:])}}
    assert doc_table == p.engine_capabilities()


# ---------------------------------------------------------------------
# The kernel AC sweep column for the add-controlled-source-set kinds
# (task 5.1): the controlled ones are stamped, the waveform current
# sources are independent sources and add no AC excitation.
# ---------------------------------------------------------------------

_AC_NEW = ("VCCS", "CCCS", "CCVS", "Gyrator", "SineCurrentSource",
           "PulseCurrentSource", "PwlCurrentSource")


@pytest.mark.parametrize("kind", _AC_NEW)
def test_ac_sweep_cell_of_the_controlled_source_kinds(kind):
    assert p.engine_capabilities()[kind]["ac_sweep"]
    build = CELLS[kind][0]

    def H(on):
        b = build(on)
        if kind in ("VCCS", "CCCS", "CCVS"):
            inp = b.pool.branch_var_id_for_source(b.branch_index_of("V"),
                                                  b.graph)
        else:   # unit current injected at the observable node
            inp = b.node_id_of("o")
        r = k.run_mna_sweep_kernel(
            b.graph, b.pool, p.SwitchStateMask(b.graph.num_switches),
            [1e2, 1e4], inp, b.node_id_of("o"))
        return np.asarray(r.H, dtype=complex)

    on, off = H(True), H(False)
    if kind.endswith("CurrentSource"):
        assert np.array_equal(on, off)
    else:
        assert np.max(np.abs(on - off)) > PRESENCE * np.max(np.abs(on))


# ---------------------------------------------------------------------
# add-power-mosfet-datasheet-device (task 2.9): the charge-law variants
# are the NonlinearCapacitor kind and add_power_mosfet is a composite of
# existing kinds, so they add no table row. Their cells: run where every
# kind they are made of runs (agreeing with pwl), refused by name
# elsewhere, transient and kernel AC sweep alike.
# ---------------------------------------------------------------------

def _law_cap(law):
    def build(on):
        b = p.CircuitBuilder()
        b.add_voltage_source("V", "in", "gnd", 10.0)
        b.add_resistor("R", "in", "o", 1e3)
        b.add_capacitor("C", "o", "gnd", 100e-9)
        b.add_resistor("Rl", "o", "gnd", 1e6)
        if on:
            b.add_nonlinear_capacitor("Cn", "o", "gnd", law())
        return b
    return build


def _pm(on):
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("V", "in", "gnd", 0.0, 10.0, 1e3)
    b.add_voltage_source("Vg", "g", "gnd", 15.0)
    b.add_resistor("R", "in", "o", 1e3)
    b.add_capacitor("C", "o", "gnd", 10e-9)
    if on:
        b.add_power_mosfet(
            "Md", "o", "gnd", "g", K=1e-2, V_T=2.0, C_gs=1e-9, R_g=10.0,
            C_gd=p.ChargeLaw.vdmos_cgd(1e-9, 50e-12, 0.5),
            C_ds=p.ChargeLaw.junction(1e-9, 25.0, 0.5),
            body_diode={"model": "shockley"})
    return b


def _tail(on):
    # An IGBT switched off by its gate: the tail charge (tau_tail, k_tail)
    # keeps conducting after cutoff and slows the collector's rise. The
    # control is the same IGBT with no tail, so presence proves the TAIL
    # stamp runs (not merely the IGBT).
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "in", "gnd", 10.0)
    b.add_pulse_voltage_source("Vg", "g", "gnd", 0.0, 15.0, 0.0, 1e-4,
                               2e-4, 2e-6, 2e-6)
    b.add_resistor("R", "in", "o", 100.0)
    b.add_capacitor("C", "o", "gnd", 100e-9)
    if on:
        b.add_igbt_level1("Qd", "o", "gnd", "g", tau_tail=1e-5, k_tail=0.5)
    else:
        b.add_igbt_level1("Qd", "o", "gnd", "g")
    return b


# name -> (cell, kinds it is made of)
VARIANT_CELLS = {
    "IgbtLevel1[tail]": (
        (_tail, "o", "Qd", 4e-4, 2e-7, False),
        ("IgbtLevel1", "Capacitor", "Resistor", "VoltageSource",
         "PulseVoltageSource")),
    "NonlinearCapacitor[table]": (
        (_law_cap(lambda: p.ChargeLaw.table(p.ChargeTable.from_capacitance(
            [-1.0, 0.0, 5.0, 10.0, 20.0], [1.2e-6, 1e-6, 5e-7, 3e-7, 2e-7]))),
         "o", "Cn", 1e-4, 1e-7, False), ("NonlinearCapacitor",)),
    "NonlinearCapacitor[vdmos_cgd]": (
        (_law_cap(lambda: p.ChargeLaw.vdmos_cgd(1e-6, 1e-7, 0.5)),
         "o", "Cn", 1e-4, 1e-7, False), ("NonlinearCapacitor",)),
    "power_mosfet": (
        (_pm, "o", "Md", 2e-3, 2e-7, False),
        ("MosfetLevel1", "Capacitor", "NonlinearCapacitor", "ShockleyDiode",
         "Resistor")),
}


def _composite_supported(kinds, column):
    caps = p.engine_capabilities()
    return all(caps[kd][column] for kd in kinds)


# Every stateful Newton kind of the kernel's list (pwl::StatefulKinds,
# device/kind.hpp) -> the conformance cell that runs it on every engine.
STATEFUL_CELLS = {
    "nonlinear_capacitor": "NonlinearCapacitor",
    "lauritzen_diode": "LauritzenDiode",
    "pmsm_mna": "PmsmMna",
    "igbt_tail": "IgbtLevel1[tail]",
    "saturable_inductor": "SaturableInductor",
    "external_device": "ExternalDevice",
}


def test_every_stateful_kind_has_a_conformance_cell():
    kinds = k._stateful_kinds()
    assert set(kinds) == set(STATEFUL_CELLS), (
        "a stateful kind without a conformance cell (add it to "
        "STATEFUL_CELLS and give it a cell)")
    for name, cell in STATEFUL_CELLS.items():
        assert cell in CELLS or cell in VARIANT_CELLS, cell
        # the kind-table row whose engine column the cell is checked against
        assert kinds[name] == cell.split("[")[0], (name, kinds[name])


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("name", list(VARIANT_CELLS))
def test_charge_law_and_power_mosfet_cells(name, engine):
    cell, kinds = VARIANT_CELLS[name]
    _check_cell(name, cell, engine, _composite_supported(kinds, engine))


@pytest.mark.parametrize("name", list(VARIANT_CELLS))
def test_charge_law_and_power_mosfet_on_the_ac_sweep(name):
    # add-unified-device-interface 10.2: every kind of these composites is
    # linearised at the DC point (each law's analytic small-signal test is in
    # test_ac_linearize.py); the composite runs and the device is present.
    cell, kinds = VARIANT_CELLS[name]
    assert _composite_supported(kinds, "ac_sweep")

    def H(on):
        b = cell[0](on)
        return np.asarray(k.run_mna_sweep_kernel(
            b.graph, b.pool, p.SwitchStateMask(b.graph.num_switches),
            [1e2, 1e4, 1e6], b.node_id_of("o"), b.node_id_of("o")).H, dtype=complex)

    on = H(True)
    assert np.all(np.isfinite(on))
    if name == "IgbtLevel1[tail]":
        # this cell's DC point has the gate at 0 V: the IGBT is off, so its
        # tail has nothing to linearise (the control differs only by the
        # tail); the conducting tail is tested in test_ac_linearize.py
        return
    assert np.max(np.abs(on - H(False))) > PRESENCE * np.max(np.abs(on))


# ---------------------------------------------------------------------
# add-permeance-magnetic-domain (task 5.4): the magnetic elements are
# composites of existing kinds (winding = Gyrator, permeance / air gap =
# Capacitor, magnetic resistance = Resistor, saturable permeance =
# NonlinearCapacitor with the tanh law), so they add no table row. Their
# cells: run and agree with pwl where every kind they are made of runs,
# refused by name (the device) elsewhere.
# ---------------------------------------------------------------------

def _magnetic(saturable):
    def build(on):
        b = p.CircuitBuilder()
        b.add_voltage_source("V", "in", "gnd", 10.0)
        b.add_resistor("R1", "in", "o", 1.0)
        b.add_resistor("R2", "o", "gnd", 1e3)
        if on:
            b.add_winding("Wd", "o", "gnd", "m1", "gnd", 25)
            b.add_magnetic_resistance("Rm", "m1", "m2", 0.625)  # 1 kOhm across
            if saturable:
                b.add_saturable_permeance("Core", "m2", "m3", 76e-6, 72e-3,
                                          mu_r=2000.0, B_sat=0.35)
            else:
                b.add_permeance("Core", "m2", "m3", 2.653e-6)
            b.add_air_gap("Gap", "m3", "gnd", 76e-6, 0.5e-3)
        return b
    return build


def _magnetic_hysteretic(on):
    # add-permeance-magnetic-domain 4.1: a JA core segment (unit winding +
    # one-turn JA inductor) in series with the gap
    b = _magnetic(False)(False)
    if on:
        ja = p.reference_material("ferrite_n87")
        b.add_winding("Wd", "o", "gnd", "m1", "gnd", 25)
        b.add_magnetic_resistance("Rm", "m1", "m2", 0.625)
        b.add_hysteretic_permeance("Core", "m2", "m3", 76e-6, 72e-3, ja.Ms,
                                   ja.a, ja.alpha, ja.c, ja.k)
        b.add_air_gap("Gap", "m3", "gnd", 76e-6, 0.5e-3)
    return b


MAGNETIC_CELLS = {
    "magnetic[hysteretic_permeance]": (
        (_magnetic_hysteretic, "o", "Core", 5e-4, 1e-7, False),
        ("Gyrator", "Capacitor", "SaturableInductor", "Resistor")),
    "magnetic[linear]": (
        (_magnetic(False), "o", "Core", 5e-4, 1e-7, False),
        ("Gyrator", "Capacitor", "Resistor")),
    "magnetic[saturable_permeance]": (
        (_magnetic(True), "o", "Core", 5e-4, 2e-8, False),
        ("Gyrator", "Capacitor", "NonlinearCapacitor", "Resistor")),
}


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("name", list(MAGNETIC_CELLS))
def test_magnetic_element_cells(name, engine):
    cell, kinds = MAGNETIC_CELLS[name]
    _check_cell(name, cell, engine, _composite_supported(kinds, engine))


MU0_CONF = 4e-7 * np.pi


def test_linear_magnetic_cell_on_the_ac_sweep():
    # Seen from the winding, series magnetic elements are parallel
    # electrical ones: R_m -> N²/R_m, permeance P -> inductance N²·P.
    assert _composite_supported(MAGNETIC_CELLS["magnetic[linear]"][1], "ac_sweep")

    def H(b):
        r = k.run_mna_sweep_kernel(
            b.graph, b.pool, p.SwitchStateMask(b.graph.num_switches),
            [1e2, 1e3, 1e4], b.node_id_of("o"), b.node_id_of("o"))
        return np.asarray(r.H, dtype=complex)

    e = _magnetic(False)(False)
    e.add_resistor("Req", "o", "gnd", 25 ** 2 / 0.625)
    e.add_inductor("Lc", "o", "gnd", 25 ** 2 * 2.653e-6)
    e.add_inductor("Lg", "o", "gnd", 25 ** 2 * MU0_CONF * 76e-6 / 0.5e-3)
    got, want = H(_magnetic(False)(True)), H(e)
    assert np.max(np.abs(got - want)) < 1e-9 * np.max(np.abs(want))


def test_saturable_permeance_refused_by_the_ac_sweep():
    # the NonlinearCapacitor kind is linearised, but not as a permeance on a
    # magnetic node: its DC point needs a nonlinear flux-conservation row
    cell, kinds = MAGNETIC_CELLS["magnetic[saturable_permeance]"]
    b = cell[0](True)
    with pytest.raises(p.UnsupportedError) as ei:
        k.run_mna_sweep_kernel(
            b.graph, b.pool, p.SwitchStateMask(b.graph.num_switches),
            [1e2, 1e4], b.node_id_of("o"), b.node_id_of("o"))
    assert ei.value.engine == "ac_sweep"
    assert ei.value.device == "Core"
    assert ei.value.feature == "nonlinear_permeance"


# ---------------------------------------------------------------------
# Realistic passives (add-realistic-passives 6.1): composites of linear
# R, L, C (and a 0 V return source), so each runs wherever its parts run,
# AC sweep included; the odd DC-bias table is the NonlinearCapacitor kind.
# ---------------------------------------------------------------------
def _passive(add):
    def build(on):
        b = p.CircuitBuilder()
        b.add_sine_voltage_source("V", "in", "gnd", 0.0, 10.0, 2e3)
        b.add_resistor("R", "in", "o", 1e3)
        b.add_resistor("Rl", "o", "gnd", 1e4)
        if on:
            add(b)
        return b
    return build


def _odd_table():
    from pulsim.passives import odd_charge_table
    v = np.linspace(0.0, 40.0, 9)
    return odd_charge_table(v, 1e-6 * 10 * np.arctan(v / 10), 1e-6 / (1 + (v / 10) ** 2))


def _ladder():
    from pulsim.passives import RacLadder
    return RacLadder(R_dc=50.0, L_hf=5e-3, R_k=(500.0,), L_k=(5e-3,))


PASSIVE_CELLS = {
    "capacitor[esr/esl]": (
        (_passive(lambda b: b.add_capacitor("Cp", "o", "gnd", 100e-9, esr=300.0,
                                            esl=1e-3)),
         "o", "Cp", 1e-3, 1e-7, False),
        ("Capacitor", "Inductor", "Resistor")),
    "inductor[esr/c_par/rac]": (
        (_passive(lambda b: b.add_inductor("Lp", "o", "gnd", 10e-3, c_par=1e-9,
                                           rac=_ladder())),
         "o", "Lp", 1e-3, 1e-7, False),
        ("Capacitor", "Inductor", "Resistor")),
    "transmission_line[pi]": (
        (_passive(lambda b: (b.add_transmission_line("T", "o", "gnd", "f", "gnd",
                                                     Z0=1e3, delay=20e-6, sections=4,
                                                     R=10.0),
                             b.add_resistor("Rf", "f", "gnd", 500.0))),
         "o", "T.c0", 1e-3, 1e-7, False),
        ("Capacitor", "Inductor", "Resistor")),
    "nonlinear_capacitor[odd table]": (
        (_passive(lambda b: b.add_nonlinear_capacitor(
            "Cn", "o", "gnd", p.ChargeLaw.table(_odd_table()))),
         "o", "Cn", 1e-3, 1e-7, False),
        ("NonlinearCapacitor",)),
}


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("name", list(PASSIVE_CELLS))
def test_realistic_passive_cells(name, engine):
    cell, kinds = PASSIVE_CELLS[name]
    _check_cell(name, cell, engine, _composite_supported(kinds, engine))


@pytest.mark.parametrize("name", list(PASSIVE_CELLS))
def test_realistic_passive_cells_on_the_ac_sweep(name):
    # The linear composites sweep (their analytic impedances are pinned in
    # test_passives.py); the DC-bias capacitor is refused by name.
    cell, kinds = PASSIVE_CELLS[name]
    b = cell[0](True)
    run = lambda: k.run_mna_sweep_kernel(  # noqa: E731
        b.graph, b.pool, p.SwitchStateMask(b.graph.num_switches),
        [1e2, 1e4], b.node_id_of("o"), b.node_id_of("o"))
    if _composite_supported(kinds, "ac_sweep"):
        assert np.all(np.isfinite(np.asarray(run().H)))
    else:
        with pytest.raises(p.UnsupportedError) as ei:
            run()
        assert (ei.value.engine, ei.value.device) == ("ac_sweep", cell[2])


@pytest.mark.parametrize("engine", ("pwl", "trbdf2", "dsed"))
def test_voltage_sanity_factor_honoured_on_every_engine(engine):
    # A 10 V node against a bound of 0.5 x 10 V: every engine's trace is
    # scanned (dsed returned without the scan, ignoring the keyword).
    kw = {"dt": 1e-6} if engine == "pwl" else {}
    with pytest.warns(UserWarning, match="largest voltage"):
        r = p.simulate(_rc(), 1e-3, engine=engine,
                       voltage_sanity_factor=0.5, **kw)
    assert r._implausible_voltage.peak == pytest.approx(10.0)


# ---------------------------------------------------------------------
# The branch form (add-branch-form-ideal-switch 4.3): a FORM of the
# Switch and Diode kinds with its own engine column, checked like the
# kind cells (no silent cell) and against the docs block.
# ---------------------------------------------------------------------
def _switch_branch(on):
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "in", "gnd", 10.0)
    b.add_resistor("Rb", "in", "gnd", 1e3)
    if on:
        b.add_ideal_switch("Sd", "in", "a", 1e-3, 1e-9)
    b.add_resistor("R1", "a", "o", 1e3)
    b.add_capacitor("C", "o", "gnd", 100e-9)
    return b


FORM_CELLS = {
    "Switch": (_switch_branch, "o", "Sd", 5e-4, 1e-7, True),
    "Diode": (_clamp(lambda b: b.add_diode("Dd", "o", "gnd", form="branch",
                                           R_on=1e-3, g_off=1e-9,
                                           V_th=0.701, V_f=0.7)),
              "o", "Dd", 2e-3, 2e-7, False),
}


def test_branch_form_rows_cover_switch_and_diode():
    caps = k._branch_form_capabilities()
    assert set(caps) == set(FORM_CELLS)
    for row in caps.values():
        assert row["pwl"], "the reference engine runs the branch form"


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("kind", list(FORM_CELLS))
def test_branch_form_cell(kind, engine):
    _check_cell(kind + "[branch]", FORM_CELLS[kind], engine,
                k._branch_form_capabilities()[kind][engine])


# Dependent modes (a switch opening an inductor cut-set, task 5.5): pwl
# re-initialises under the restart rule, trbdf2 always (three SDIRK
# steps), dsed/exact by projection at mode entry (branch-form task 8); pwl
# without the rule refuses by name, bdf1 because the branch form is refused.
_DEP_REFUSAL = {"pwl": "dependent_states",
                "dsed": "branch_form", "bdf1": "branch_form"}


@pytest.mark.parametrize("engine,restart",
                         [(e, False) for e in (*ENGINES, "exact")] + [("pwl", True)])
def test_dependent_states_cell(engine, restart):
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "vin", "gnd", 10.0)
    b.add_ideal_switch("S", "vin", "a", 0.0, 0.0)
    b.add_resistor("R", "a", "m", 1.0)
    b.add_inductor("L", "m", "gnd", 1e-3)

    def fn(t):
        m = p.SwitchStateMask(1)
        m.set(0, t < 2e-4)
        return m
    kw = dict(discontinuity_restart=True) if restart else {}
    if engine in ("pwl", "dsed"):
        kw["dt"] = 5e-6
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if engine == "bdf1":
                r = k.run_transient_bdf1(
                    b, k.SimulationOptions(t_start=0.0, t_end=3e-4, dt=5e-6), fn)
            else:
                r = p.simulate(b, 3e-4, engine=engine, switch_fn=fn, **kw)
    except p.UnsupportedError as e:
        named = "dsed" if engine == "exact" else engine  # the table column
        assert (e.engine, e.feature, e.device) == (named, _DEP_REFUSAL[named], "S")
        assert not (engine == "pwl" and restart), f"pwl+restart refused: {e}"
        return
    assert (engine == "pwl" and restart) or engine in ("trbdf2", "dsed", "exact"), (
        f"{engine} ran a dependent mode silently")
    assert len(r.reinit_records) == 1


@pytest.mark.parametrize("kind", [kd for kd in FORM_CELLS
                                  if not k._branch_form_capabilities()[kd]["dsed"]])
def test_branch_form_exact_alias_refuses_like_dsed(kind):
    out = {}
    for eng in ("dsed", "exact"):
        with pytest.raises(p.UnsupportedError) as ei:
            _run(*_args(FORM_CELLS[kind]), eng)
        e = ei.value
        out[eng] = (e.engine, e.feature, e.device, e.kind)
    assert out["dsed"] == out["exact"] and out["dsed"][1] == "branch_form"


def test_branch_form_matrix_equals_the_docs():
    import pathlib
    doc = (pathlib.Path(__file__).resolve().parents[2] / "docs"
           / "known-limitations.md").read_text(encoding="utf-8")
    block = doc.split("<!-- branch-form-capabilities:begin -->")[1].split(
        "<!-- branch-form-capabilities:end -->")[0]
    cols = ("pwl", "trbdf2", "dsed", "bdf1", "ac_sweep")
    doc_table = {}
    for ln in block.strip().splitlines()[2:]:
        cells = [c.strip() for c in ln.strip("|").split("|")]
        doc_table[cells[0]] = {c: v == "yes" for c, v in zip(cols, cells[1:])}
    assert doc_table == k._branch_form_capabilities()

# ---------------------------------------------------------------------
# add-compiled-control-graph task 4: a control-graph Rate on each engine
# ---------------------------------------------------------------------

def _rated_chain(b, rate):
    ch = p.MixedDomainBlockChain(rate=rate)
    ch.add("g", p.Gain(k=0.5), inputs=dict(x="o"), output="y")
    return ch


@pytest.mark.parametrize("rate,feature", [
    (dict(period=1e-4, compute_delay=2e-5), "compute_delay"),
])
def test_control_rate_delay_time_refused_on_trbdf2(rate, feature):
    b = _rc()
    ch = _rated_chain(b, p.Rate(**rate))
    with pytest.raises(p.UnsupportedError) as ei:
        _sim("trbdf2", b, step_observer=ch.make_step_observer(b))
    assert (ei.value.engine, ei.value.feature) == ("trbdf2", feature)


@pytest.mark.parametrize("rate", [
    dict(period=1e-4, offset=2e-5), dict(period=1e-4, compute_delay=2e-5),
    dict(period=1e-4, compute_delay="one_sample"),
])
def test_control_rate_keywords_honoured_on_pwl(rate):
    b = _rc()
    ch = _rated_chain(b, p.Rate(**rate))
    _sim("pwl", b, step_observer=ch.make_step_observer(b))
    assert ch.tick_counts == [20]     # ceil((2e-3 - offset) / 1e-4)


def test_control_rate_offset_honoured_on_trbdf2():
    b = _rc()
    ch = _rated_chain(b, p.Rate(period=1e-4, offset=2e-5))
    _sim("trbdf2", b, step_observer=ch.make_step_observer(b))
    assert ch.tick_counts == [20]     # ceil((2e-3 - offset) / 1e-4)


@pytest.mark.parametrize("engine", ("dsed", "exact"))
def test_control_rate_offset_honoured_on_dsed(engine):
    b = _rc()
    ch = _rated_chain(b, p.Rate(period=1e-4, offset=2e-5))
    _sim(engine, b, step_observer=ch.make_step_observer(b))
    assert ch.tick_counts == [20]     # ceil((2e-3 - offset) / 1e-4)


def test_control_rate_delay_time_refused_on_dsed():
    b = _rc()
    ch = _rated_chain(b, p.Rate(period=1e-4, compute_delay=2e-5))
    with pytest.raises(p.UnsupportedError) as ei:
        _sim("dsed", b, step_observer=ch.make_step_observer(b))
    assert (ei.value.engine, ei.value.feature) == ("dsed", "compute_delay")


# add-exact-propagator-strategy group 13: engine='exact' is the canonical name of the event-driven
# exact engine, 'dsed' its alias. Every keyword cell of 'dsed' holds under 'exact', and the
# refusal keeps naming 'dsed' (the capability-table column; DECISIONS, group 13 slice).
@pytest.mark.parametrize("feature", [f for f, (_, eng) in _REFUSED.items() if "dsed" in eng])
def test_exact_alias_refuses_what_dsed_refuses(feature):
    kw, _ = _REFUSED[feature]
    with pytest.raises(p.UnsupportedError) as ei:
        _sim("exact", **kw)
    assert (ei.value.engine, ei.value.feature) == ("dsed", feature)


@pytest.mark.parametrize("engine", ("pwl", "trbdf2", "dsed", "exact"))
def test_exact_segments_only_on_the_exact_engine(engine):
    # result.exact['segments'] (group 13 route record) rides on the event integrator's runs only
    r = _sim(engine, _half_wave())
    ex = getattr(r, "exact", None)
    if engine in ("pwl", "trbdf2"):
        assert ex is None
        return
    assert r.engine_used == "dsed"
    segs = ex["segments"]
    assert segs and all(s["exact"] and s["reason"] is None for s in segs)
    assert sum(s["n_segments"] for s in segs) >= 2      # the diode commutates
    assert ex["auto_would_route"] is True               # 13.3 report-only field, same on both names


# Integration batch 6b: 'exact' and 'dsed' are one engine on every device cell and the
# coupled-inductor cell -- the same refusal, or bit-identical times and samples.
@pytest.mark.parametrize("kind", [*CELLS, "coupled"])
def test_exact_and_dsed_are_bit_identical_on_every_cell(kind):
    args = _args(COUPLED if kind == "coupled" else CELLS[kind])
    out = {}
    for eng in ("dsed", "exact"):
        try:
            out[eng] = _run(*args, eng)
        except p.UnsupportedError as e:
            out[eng] = (type(e), e.engine, e.feature, e.device, e.kind)
    a, b = out["dsed"], out["exact"]
    if isinstance(a[0], type):
        assert a == b
        return
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


# Integration batch 9: the exact engine's new cells (allow_implicit_segments, a static circuit,
# a DC start) hold under both names, bit for bit; the keyword is refused by name elsewhere.
def _divider():
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "a", "gnd", 12.0)
    b.add_resistor("R1", "a", "o", 2e3)
    b.add_resistor("R2", "o", "gnd", 1e3)
    return b


def _rc_b_extra():
    b = _rc()
    zero = [0.0] * b.pool.state_size(b.graph)
    return b, (lambda _t: zero)


@pytest.mark.parametrize("cell", ["static", "dc_start", "allow_implicit_segments"])
def test_exact_engine_cells_are_identical_under_both_names(cell):
    out = {}
    for eng in ("dsed", "exact"):
        if cell == "static":
            r = _sim(eng, _divider(), dt=1e-4)
            assert np.abs(np.asarray(r.v("o")) - 4.0).max() < 1e-12      # 12 V * 1k / 3k
        elif cell == "dc_start":
            x = np.asarray(p.compute_dc_op(_rc()), float)
            r = _sim(eng, initial_state=x, dt=1e-4)
        else:
            b, fn = _rc_b_extra()
            with pytest.raises(p.UnsupportedError) as ei:
                _sim(eng, b, b_extra_fn=fn)
            assert (ei.value.engine, ei.value.feature) == ("dsed", "source")
            b, fn = _rc_b_extra()
            r = _sim(eng, b, b_extra_fn=fn, allow_implicit_segments=True)
        out[eng] = (np.asarray(r.times), np.asarray(r.states))
    assert np.array_equal(out["dsed"][0], out["exact"][0])
    assert np.array_equal(out["dsed"][1], out["exact"][1])


@pytest.mark.parametrize("engine", ("pwl", "trbdf2", "auto"))
def test_allow_implicit_segments_is_refused_off_the_exact_engine(engine):
    with pytest.raises(p.UnsupportedError) as ei:
        _sim(engine, allow_implicit_segments=True)
    assert ei.value.feature == "allow_implicit_segments"


# add-in-kernel-event-losses: a device loss port is honoured on the
# fixed-step and the variable-step engines (task 3.1); every other engine
# refuses it by name, with the device, before a step is taken.
@pytest.mark.parametrize("engine", ("pwl", "trbdf2", "dsed", "bdf1", "auto"))
def test_device_losses_cell(engine):
    b = _rc()
    b.add_diode("S", "o", "gnd", 1e3, 1e-9)
    b.add_voltage_source("Vamb", "amb", "gnd", 25.0)
    b.add_resistor("Rth", "tj", "amb", 1.0)
    b.add_capacitor("Cth", "tj", "amb", 1e-3)
    b.set_device_losses("S", junction="tj")
    if engine in ("dsed", "bdf1"):
        with pytest.raises(p.UnsupportedError) as ei:
            if engine == "bdf1":
                k.run_transient_bdf1(
                    b, k.SimulationOptions(t_start=0.0, t_end=1e-4, dt=1e-6),
                    lambda _t: p.SwitchStateMask(b.graph.num_switches))
            else:
                _sim(engine, b)
        assert (ei.value.feature, ei.value.device, ei.value.engine) == (
            "device_losses", "S", engine)
        return
    try:
        r = _sim(engine, b)
    except p.UnsupportedError as e:
        # 'auto' names the engine it resolved to when that one refuses
        assert engine == "auto" and (e.feature, e.device, e.engine) == (
            "device_losses", "S", "dsed")
        return
    assert r.device_energy("S")["E_cond_electrical"] > 0.0
    # 'auto' with a step routes to pwl and honours the port
    r = _sim("auto", b, dt=1e-6)
    assert r.engine_used == "pwl" and r.device_energy("S")["n_on"] >= 0


@pytest.mark.parametrize("engine", ("pwl", "trbdf2", "dsed", "exact"))
def test_reinit_records_on_every_engine(engine):
    # result.reinit_records (the impulse at a mode entry) is on every engine since the fixed
    # engine's D4 records (feat/core-a-ideal-switch): two 1.5 uF capacitors at 8 V and 0 V tied
    # by a 0 V source share 6 uC at t = 0. The exact engine records that start, the same on both
    # names; pwl / trbdf2 record none at t = 0 (measured: their records are at switching events)
    b = p.CircuitBuilder()
    b.add_capacitor("C1", "a", "gnd", 1.5e-6, 8.0)
    b.add_capacitor("C2", "b", "gnd", 1.5e-6, 0.0)
    b.add_voltage_source("V0", "a", "b", 0.0)
    b.add_resistor("R", "a", "gnd", 1e3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = p.simulate(b, 1e-4, engine=engine, dt=1e-6)
    recs = getattr(r, "reinit_records", None)
    if engine in ("pwl", "trbdf2"):
        # the fixed engines' result has the field too (the branch-form ideal
        # switch's records, add-branch-form-ideal-switch D4); they do not
        # reduce dependent states, so nothing is recorded here
        assert not recs
        return
    (rec,) = recs
    assert (rec.kind, rec.sample) == ("capacitor_loop", 0)
    assert rec.lambda_ == pytest.approx(6e-6, rel=1e-12)


# ---------------------------------------------------------------------
# The gated forms of the Diode kind (add-pwl-igbt-and-thyristor 4.5):
# IGBT / GTO (gate held) and thyristor / TRIAC (latching), gate on, each
# clamping the sine like a diode with a drop; their own engine column.
# ---------------------------------------------------------------------
GATED_CELLS = {
    "GateHeld": (_clamp(lambda b: b.add_igbt("Qd", "o", "gnd", 1e-3, 1e9,
                                             V_ce0=0.7, reverse_blocking=True,
                                             V_th=0.701)),
                 "o", "Qd", 2e-3, 2e-7, True),
    "Latching": (_clamp(lambda b: b.add_thyristor("Td", "o", "gnd", 1e-3, 1e9,
                                                  V_f=0.7, V_th=0.701)),
                 "o", "Td", 2e-3, 2e-7, True),
}


def test_gated_rows_cover_both_modes():
    assert set(k._gated_capabilities()) == set(GATED_CELLS)


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("mode", list(GATED_CELLS))
def test_gated_cell(mode, engine):
    _check_cell(mode + "[gated]", GATED_CELLS[mode], engine,
                k._gated_capabilities()[mode][engine])


def test_gated_doc_block_equals_the_kernel_table():
    import pathlib
    doc = (pathlib.Path(__file__).resolve().parents[2] / "docs"
           / "known-limitations.md").read_text(encoding="utf-8")
    block = doc.split("<!-- gated-capabilities:begin -->")[1].split(
        "<!-- gated-capabilities:end -->")[0]
    cols = ("pwl", "trbdf2", "dsed", "bdf1", "ac_sweep")
    doc_table = {}
    for ln in block.strip().splitlines()[2:]:
        cells = [c.strip() for c in ln.strip("|").split("|")]
        doc_table[cells[0]] = {c: v == "yes" for c, v in zip(cols, cells[1:])}
    assert doc_table == k._gated_capabilities()


# ---- the device path (add-unified-device-interface DECISIONS #23) --------
# Generated from the registry: every Newton kind (and Newton variant) cell,
# on every Newton engine, either runs on device_path='interface' and agrees
# with the hand stamps, or refuses by name; which one is the kernel's
# registry (`_device_path_ported`), so a ported kind without a passing cell
# and a refusal the registry does not predict both fail.
_NEWTON_PORTED, _BLOCKS_PORTED = k._device_path_ported()
_BLOCK_OF = {cell: blk for blk, cell in STATEFUL_CELLS.items()}
_PATH_CELLS = {n: c for n, c in CELLS.items() if n in _NEWTON_PORTED}
_PATH_CELLS.update({n: v[0] for n, v in VARIANT_CELLS.items()
                    if n.split("[")[0] in _NEWTON_PORTED})


def _sat_law(add):
    """The saturable inductor under another law, driven by a sine so the
    current reverses (the hysteretic core runs both branches)."""
    def build(on):
        b = p.CircuitBuilder()
        b.add_sine_voltage_source("V", "in", "gnd", 0.0, 10.0, 5e3, 0.0)
        b.add_resistor("R1", "in", "o", 10.0)
        b.add_resistor("R2", "o", "gnd", 1e3)
        if on:
            add(b)
        return b
    return build


# Path-only cells: the table and Jiles-Atherton inductor laws (the device
# path's adapters choose the law per instance; the engine cells above run
# the atan law only).
_PATH_CELLS.update({
    "SaturableInductor[table]": (_sat_law(lambda b: b.add_saturable_inductor_table(
        "Ld", "o", "gnd", [0.0, 0.2, 0.5, 1.0, 2.0],
        [0.0, 2e-4, 4e-4, 5e-4, 6e-4])), "o", "Ld", 4e-4, 5e-8, False),
    "SaturableInductor[ja]": (_sat_law(lambda b: b.add_hysteretic_core_inductor(
        "Ld", "o", "gnd", 100.0, 1e-4, 0.1, 0.0, 4.0e5, 50.0, 5e-5, 0.2,
        30.0)), "o", "Ld", 4e-4, 5e-8, False),
})
_BLOCK_OF.update({"SaturableInductor[table]": "saturable_inductor",
                  "SaturableInductor[ja]": "saturable_inductor"})


def _run_path(cell, engine, path):
    build, node, _dev, t_end, dt, sw = cell
    b = build(True)
    kw = dict(dt=dt) if engine == "pwl" else {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = p.simulate(b, t_end, engine=engine, switch_fn=_switch_fn(b, sw),
                       device_path=path, **kw)
    return (np.asarray(r.times, float), np.asarray(r.v(node), float),
            np.asarray(r.newton.iterations_per_solve))


def test_every_newton_kind_has_a_device_path_cell():
    assert set(_NEWTON_PORTED) <= set(_PATH_CELLS)
    assert set(_BLOCKS_PORTED) == set(STATEFUL_CELLS)


@pytest.mark.parametrize("engine", ["pwl", "trbdf2"])
@pytest.mark.parametrize("name", sorted(_PATH_CELLS))
def test_device_path_cell(name, engine):
    cell = _PATH_CELLS[name]
    base = name.split("[")[0]
    block = _BLOCK_OF.get(name)
    ported = _NEWTON_PORTED[base] and _BLOCKS_PORTED.get(block, True)
    try:
        ta, va, na = _run_path(cell, engine, "legacy")
    except p.UnsupportedError:
        pytest.skip(f"{name} is not on {engine} (kind table)")
    try:
        ti, vi, ni = _run_path(cell, engine, "interface")
    except ValueError as e:
        assert not ported, f"{name} on {engine}: refused but ported: {e}"
        assert "not on the device interface yet" in str(e)
        assert base in str(e) or (block is not None and block in str(e)), e
        return
    assert ported, f"{name} on {engine}: ran on 'interface' but not ported"
    span = float(np.max(va) - np.min(va)) or float(np.max(np.abs(va))) or 1.0
    if engine == "pwl":
        assert np.array_equal(ta, ti)
        err = float(np.max(np.abs(vi - va)))
        # A rounding claim while both paths take the same Newton iterations
        # (1e-12 measured on macOS; bounded at 1e-9 like the equivalence
        # gate, another compiler's libm/FMA differs in the last bits); one
        # iteration apart they differ by up to the Newton tolerance (1e-9
        # absolute by default; DECISIONS #29).
        tol = 1e-9 * span if np.array_equal(na, ni) else 1e-8
        if not np.array_equal(na, ni):   # a stopping difference, not a law one
            assert na.shape == ni.shape and np.count_nonzero(na != ni) <= max(
                3, na.size // 100), f"{name}: Newton counts differ widely"
        assert err <= tol, f"{name}: {err:.3g} vs span {span:.3g}"
    else:
        err = float(np.max(np.abs(np.interp(ta, ti, vi) - va)))
        assert err <= TOL * span, f"{name}: {err:.3g} vs span {span:.3g}"


# ---------------------------------------------------------------------
# The DC column of the machine kinds (add-mna-native-machines 2.7, 3.7):
# "DC held" — the operating point holds the speed / angle at their
# initial values and solves the stator with di/dt = 0. At rest the
# cell's network is linear in closed form: each phase is a Thevenin
# source v_k(0)·1000/1001 behind 1·1000/1001 Ohm, into R_s to a floating
# neutral (sum of currents 0). The AC column is `ac_sweep` above.
# ---------------------------------------------------------------------
@pytest.mark.parametrize("kind", ["PmsmMna", "InductionMachineMna"])
def test_machine_dc_held_cell(kind):
    build, amp = {"PmsmMna": (_pmsm, 12.0),
                  "InductionMachineMna": (_induction, 40.0)}[kind]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        r = p.simulate(build(True), 1e-5, dt=1e-6, engine="pwl",
                       start_from_dc_op=True)
    v = np.array([amp * math.sin(-2.0 * math.pi / 3.0 * i) for i in range(3)])
    vth, rth = v * 1000.0 / 1001.0, 1000.0 / 1001.0
    i_ref = (vth - vth.mean()) / (rth + 0.5)
    got = np.array([float(np.asarray(r.i(f"Md_{x}"))[0]) for x in "abc"])
    assert np.abs(i_ref).max() > 1.0   # control: a real operating point
    assert got == pytest.approx(i_ref, abs=1e-9 * np.abs(i_ref).max())


def _bldc(on):
    b = _pmsm(False)
    if on:
        b.add_bldc_mna("Md", "ua2", "ub2", "uc2", "nn", "w", "th", R_s=0.5,
                       L_s=2e-3, k_e=0.2, pole_pairs=4, J=1e-3, B=1e-4)
    return b


@pytest.mark.parametrize("engine", ENGINES)
def test_pmsm_harmonics_cell(engine):
    # pmsm_mna[harmonics] (3.7): the BLDC's PM-flux series; same kind,
    # same table row as the sinusoidal PMSM
    _check_cell("PmsmMna[harmonics]", (_bldc, "ua2", "Md", 2e-2, 1e-5, False),
                engine, p.engine_capabilities()["PmsmMna"][engine])


def _pmsm_flux_map(on):
    b = _pmsm(False)
    if on:
        # SYNTHETIC saturating, cross-coupled co-energy (test_pmsm_flux_map)
        g = np.arange(-30.0, 30.5, 2.0)
        D, Q = np.meshgrid(g, g, indexing="ij")
        x = np.hypot(D, Q) / 10.0
        f = 1e-3 * np.tanh(x) / np.where(x == 0.0, 1.0, x) + np.where(x == 0.0, 1e-3, 0.0)
        fm = p.FluxMapDQ.from_tables(g, g, 0.05 + f * D, 2e-3 * Q + f * Q,
                                     source="synthetic: conformance")
        b.add_pmsm_flux_map("Md", "ua2", "ub2", "uc2", "nn", "w", "th",
                            R_s=0.5, flux_map=fm, pole_pairs=4, J=1e-3,
                            B=1e-4)
    return b


@pytest.mark.parametrize("engine", ENGINES)
def test_pmsm_flux_map_cell(engine):
    # pmsm_mna[flux_map] (3.7): same kind, same table row
    _check_cell("PmsmMna[flux_map]",
                (_pmsm_flux_map, "ua2", "Md", 2e-2, 1e-5, False),
                engine, p.engine_capabilities()["PmsmMna"][engine])


def _dc_machine(on):
    # a 10 V supply through 1 Ohm, a 100 Ohm bleed: the machine's armature
    # current and back-EMF move the terminal voltage
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "s", "gnd", 10.0)
    b.add_resistor("Rs", "s", "a", 1.0)
    b.add_resistor("Rb", "a", "gnd", 100.0)
    if on:
        b.add_dc_machine("Md", "a", "gnd", "w", R_a=0.5, L_a=1e-3, K=0.1,
                         J=1e-5, B=1e-4)
    return b


_DC_MACHINE_KINDS = ("Resistor", "Inductor", "IdealTransformer", "Capacitor")


@pytest.mark.parametrize("engine", ENGINES)
def test_dc_machine_cell(engine):
    # dc_machine (1.10): a composite of linear kinds, so each engine
    # column is theirs (dsed runs it, D10)
    _check_cell("dc_machine", (_dc_machine, "a", "Md", 2e-2, 1e-5, False),
                engine, _composite_supported(_DC_MACHINE_KINDS, engine))


@pytest.mark.parametrize("name", ["dc_machine", "PmsmMna[harmonics]",
                                  "PmsmMna[flux_map]"])
def test_machine_ac_sweep_cell(name):
    # AC column of the machine rows (1.10, 3.7): the DC machine's linear
    # composite is swept (presence: the machine moves the terminal
    # impedance); the PMSM rows follow the kind's table entry, refused by
    # name when unsupported
    build, node, dev = {"dc_machine": (_dc_machine, "a", "Md"),
                        "PmsmMna[harmonics]": (_bldc, "ua2", "Md"),
                        "PmsmMna[flux_map]": (_pmsm_flux_map, "ua2", "Md")}[name]
    supported = (_composite_supported(_DC_MACHINE_KINDS, "ac_sweep")
                 if name == "dc_machine"
                 else p.engine_capabilities()["PmsmMna"]["ac_sweep"])

    def H(on):
        b = build(on)
        r = k.run_mna_sweep_kernel(
            b.graph, b.pool, p.SwitchStateMask(b.graph.num_switches),
            [1e1, 1e2, 1e3], b.node_id_of(node), b.node_id_of(node))
        return np.asarray(r.H, dtype=complex)

    off = H(False)
    if not supported:
        with pytest.raises(p.UnsupportedError) as ei:
            H(True)
        assert ei.value.engine == "ac_sweep"
        assert ei.value.device == dev or ei.value.device.startswith(dev + "_")
        return
    if name != "dc_machine":
        # the PMSM kind is linearised at an equilibrium only; these driven
        # cells are not one (a current, so a torque the held DC point does
        # not balance): refused by name, the reason stated
        with pytest.raises(p.UnsupportedError) as ei:
            H(True)
        assert ei.value.engine == "ac_sweep" and ei.value.device.startswith(dev + "_")
        assert "torque balance" in str(ei.value)
        return
    on = H(True)
    assert np.max(np.abs(on - off)) > PRESENCE * np.max(np.abs(on))


def test_dc_machine_dc_cell():
    # DC column: the steady state in closed form, ω = (V_a − R_a·B·ω/K)/K
    # with V_a from the 1 Ohm / 100 Ohm divider and the armature current
    b = _dc_machine(True)
    x = p.compute_dc_op(b, gmin=0.0)
    # i_a = B·ω/K; v_a = K·ω + R_a·i_a; (10 − v_a)/1 = v_a/100 + i_a
    K, Ra, B = 0.1, 0.5, 1e-4
    w = 10.0 / ((1 + 1 / 100) * (K + Ra * B / K) + B / K)
    assert x[b.node_id_of("w")] == pytest.approx(w, rel=1e-9)


# Gate EDGES in the gated cells (add-pwl-igbt-and-thyristor 4.5): a PWM
# gate on the IGBT clamp, a phase-controlled pulse on the thyristor
# clamp (latch past the pulse). The schedules declare their edges.
def _gate_schedule(name, make):
    def mk(b):
        try:
            j = b.switch_index_of(name)
        except Exception:  # noqa: BLE001 — the presence control's circuit
            return None
        return make(j, b.graph.num_switches)
    return mk


def _scr_r(on):
    # a phase-controlled half-wave rectifier, 10 Ohm: the turn-off at the
    # voltage zero is continuous, the 60 deg firing edge is the event (a
    # clamp's turn-off sits on a capacitor transient the reference cannot
    # converge at a sample instant)
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("V", "in", "gnd", 0.0, 10.0, 1e3)
    b.add_resistor("R", "o", "gnd", 10.0)
    if on:
        b.add_thyristor("Td", "in", "o", 1e-3, 1e9)
    return b


GATED_EDGE_CELLS = {
    "GateHeld": (GATED_CELLS["GateHeld"][0], "o", "Qd", 2e-3, 2e-7,
                 _gate_schedule("Qd", lambda j, n: p.make_pwm_switch_fn(
                     3e3, 0.37, j, n))),
    "Latching": (_scr_r, "o", "Td", 2e-3, 2e-7,
                 _gate_schedule("Td", lambda j, n: p.make_phase_control_fn(
                     1e3, math.radians(60.0), 1e-4, [j], n))),
}


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("mode", list(GATED_EDGE_CELLS))
def test_gated_cell_with_gate_edges(mode, engine):
    _check_cell(mode + "[gated, edges]", GATED_EDGE_CELLS[mode], engine,
                k._gated_capabilities()[mode][engine])


# AC and DC columns of the gated rows (add-pwl-igbt-and-thyristor 4.5): the
# IGBT and GTO (gate held), the thyristor and the TRIAC (latching). The AC
# sweep stamps a gated device at the supplied mask: g_on (the drop V_f has
# no small-signal part) with its bit on, g_off with it off. Reference: the
# closed-form impedance of the 1 kOhm / 10 nF clamp node with the device as
# that conductance (the TRIAC's two branches in parallel).
_GATED_ADDERS = {
    "igbt": (lambda b: b.add_igbt("Q", "o", "gnd", 1e-3, 1e9, V_ce0=0.7,
                                  reverse_blocking=True, V_th=0.701), 1),
    "gto": (lambda b: b.add_gto("Q", "o", "gnd", 1e-3, 1e9, V_f=0.7,
                                V_th=0.701), 1),
    "thyristor": (lambda b: b.add_thyristor("Q", "o", "gnd", 1e-3, 1e9,
                                            V_f=0.7, V_th=0.701), 1),
    "triac": (lambda b: b.add_triac("Q", "o", "gnd", 1e-3, 1e9, V_f=0.7,
                                    V_th=0.701), 2),
}


@pytest.mark.parametrize("name", list(_GATED_ADDERS))
def test_gated_device_on_the_ac_sweep(name):
    add, n_branches = _GATED_ADDERS[name]
    b = _clamp(add)(True)
    bits = b.switch_indices_of("Q")
    assert len(bits) == n_branches
    f = np.array([1e2, 1e4, 1e6])

    def H(on):
        m = p.SwitchStateMask(b.graph.num_switches)
        for j in (bits if on else ()):
            m.set(j, True)
        return np.asarray(k.run_mna_sweep_kernel(
            b.graph, b.pool, m, list(f), b.node_id_of("o"),
            b.node_id_of("o")).H, dtype=complex)

    for on, g in ((True, 1e3), (False, 1e-9)):
        want = 1.0 / (1e-3 + 2j * np.pi * f * 10e-9 + n_branches * g)
        assert np.max(np.abs(H(on) - want) / np.abs(want)) < 1e-9, (name, on)


@pytest.mark.parametrize("name", list(_GATED_ADDERS))
def test_gated_device_dc_operating_point(name):
    # DC column: 10 A forced through the device with its gate on settles on
    # the drop V_f + i/g_on (the 1 MOhm probe keeps the node defined); the
    # same circuit with the gate off blocks (control first: it conducts).
    add, n_branches = _GATED_ADDERS[name]
    for gate in (True, False):
        b = p.CircuitBuilder()
        b.add_current_source("I", "o", "gnd", 10.0)
        b.add_resistor("Rp", "o", "gnd", 1e6)
        add(b)
        m = p.SwitchStateMask(b.graph.num_switches)
        for j in (b.switch_indices_of("Q") if gate else ()):
            m.set(j, True)
        sw = (lambda m_: (lambda _t: m_))(m)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = p.simulate(b, 1e-5, engine="pwl", dt=1e-6, switch_fn=sw,
                           start_from_dc_op=True)
        v0 = float(np.asarray(r.v("o"))[0])
        if gate:
            # one conducting branch: v = 0.7 + 1e-3·(10 − v/1e6)
            assert v0 == pytest.approx((0.7 + 1e-2) / (1 + 1e-9), abs=1e-9)
        else:
            # blocked: 10 A into 1 MOhm, the off conductance and the DC
            # point's 1e-12 S node floor
            g_off = n_branches * 1e-9
            assert v0 == pytest.approx(10.0 / (1e-6 + g_off + 1e-12), rel=1e-9)


def test_hysteretic_permeance_refused_by_the_ac_sweep():
    # a Jiles-Atherton law has no unique small-signal model (minor loops):
    # the sweep refuses its inductor by name, as for the inductor form
    b = _magnetic_hysteretic(True)
    with pytest.raises(p.UnsupportedError, match="Jiles-Atherton") as ei:
        k.run_mna_sweep_kernel(
            b.graph, b.pool, p.SwitchStateMask(b.graph.num_switches),
            [1e2, 1e4], b.node_id_of("o"), b.node_id_of("o"))
    assert ei.value.device == "Core"
