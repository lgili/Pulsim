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
    return _Closed(b.graph.num_switches) if needs else None


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
    # Circuit-level limit: refused on dsed (also a NotImplementedError,
    # its legacy base) and on bdf1 (no mutual stamp: it read 9.38 V off
    # an 18 V span, silently, before this change).
    if engine == "bdf1":
        with pytest.raises(p.UnsupportedError) as ei:
            _run(*_args(COUPLED), "bdf1")
        assert (ei.value.engine, ei.value.feature) == (
            "bdf1", "coupled_inductors")
        return
    if engine == "dsed":
        with pytest.raises(p.UnsupportedError) as ei:
            _run(*_args(COUPLED), "dsed")
        assert isinstance(ei.value, NotImplementedError)
        assert ei.value.engine == "dsed"
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
    assert (len(r.times), st["n_solves"], st["n_gate_events"]) == (
        1104, 2254, 20)


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
}


@pytest.mark.parametrize("engine", ("trbdf2", "dsed"))
@pytest.mark.parametrize("feature", list(_REFUSED))
def test_fixed_step_feature_refused_by_name(feature, engine):
    kw, refusing = _REFUSED[feature]
    if engine not in refusing:
        pytest.skip("honoured on this engine")
    with pytest.raises(p.UnsupportedError) as ei:
        _sim(engine, **kw)
    assert (ei.value.engine, ei.value.feature) == (engine, feature)


@pytest.mark.parametrize("engine", ("trbdf2", "dsed"))
def test_c_blocks_refused(engine):
    b = _rc()
    b._c_blocks = [_CBlock()]
    with pytest.raises(p.UnsupportedError) as ei:
        _sim(engine, b)
    assert (ei.value.engine, ei.value.feature) == (engine, "c_blocks")


def test_diode_drop_above_threshold_refused_on_dsed():
    # DSED held such a diode ON through reverse bias: the sine clamp
    # read 0.70 V for the whole negative half (10.37 V off), silently.
    b = _clamp(lambda b: b.add_diode("Dd", "o", "gnd", 1e3, 1e-9,
                                     0.0, 0.7))(True)
    with pytest.raises(p.UnsupportedError) as ei:
        _sim("dsed", b)
    e = ei.value
    assert (e.engine, e.feature, e.device, e.kind) == (
        "dsed", "V_f", "Dd", "Diode")


def test_builder_initial_conditions_refused_on_dsed():
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "in", "gnd", 10.0)
    b.add_resistor("R", "in", "o", 1e3)
    b.add_capacitor("C", "o", "gnd", 100e-9, c0=2.0)
    with pytest.raises(p.UnsupportedError) as ei:
        _sim("dsed", b)
    assert ei.value.feature == "initial_conditions"


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


def test_closed_loop_refused_on_dsed():
    b = _two_bucks()
    la = _loop(b, "outa", "Qa", 20e3)
    with pytest.raises(p.UnsupportedError) as ei:
        _sim("dsed", b, closed_loops=[la])
    assert ei.value.feature == "closed_loops"


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


# name -> (cell, kinds it is made of)
VARIANT_CELLS = {
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


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("name", list(VARIANT_CELLS))
def test_charge_law_and_power_mosfet_cells(name, engine):
    cell, kinds = VARIANT_CELLS[name]
    _check_cell(name, cell, engine, _composite_supported(kinds, engine))


@pytest.mark.parametrize("name", list(VARIANT_CELLS))
def test_charge_law_and_power_mosfet_refused_by_the_ac_sweep(name):
    cell, kinds = VARIANT_CELLS[name]
    assert not _composite_supported(kinds, "ac_sweep")
    b = cell[0](True)
    with pytest.raises(p.UnsupportedError) as ei:
        k.run_mna_sweep_kernel(
            b.graph, b.pool, p.SwitchStateMask(b.graph.num_switches),
            [1e2, 1e4], b.node_id_of("o"), b.node_id_of("o"))
    assert ei.value.engine == "ac_sweep"
    assert ei.value.device == cell[2]


@pytest.mark.parametrize("engine", ("pwl", "trbdf2", "dsed"))
def test_voltage_sanity_factor_honoured_on_every_engine(engine):
    # A 10 V node against a bound of 0.5 x 10 V: every engine's trace is
    # scanned (dsed returned without the scan, ignoring the keyword).
    kw = {"dt": 1e-6} if engine == "pwl" else {}
    with pytest.warns(UserWarning, match="largest voltage"):
        r = p.simulate(_rc(), 1e-3, engine=engine,
                       voltage_sanity_factor=0.5, **kw)
    assert r._implausible_voltage.peak == pytest.approx(10.0)
