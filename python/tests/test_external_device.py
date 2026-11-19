"""External devices through the C ABI (add-external-device-abi).

Twins: the saturable inductor and the Shockley diode re-implemented as an
EXTERNAL model (core/tests/xdev/twin_models.cpp, through the C++ helper)
and loaded at run time; they must give the built-in law's numbers. The
test models are compiled here with the system compilers (or taken from
``PULSIM_XDEV_TEST_LIBDIR``, a directory holding the CMake-built
``pulsim_xdev_twins`` / ``pulsim_xdev_contract`` modules); without a
compiler the tests skip. The libraries are illustrations, not part models.
"""
import os
import pathlib
import pickle
import shutil
import subprocess
import sys

import numpy as np
import pytest

import pulsim as p

ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC = ROOT / "core" / "tests" / "xdev"
INC = ROOT / "core" / "include"
EXT = {"win32": ".dll", "darwin": ".so"}.get(sys.platform, ".so")


def _compile(tmp, src, out, cxx):
    tool = shutil.which(os.environ.get("CXX" if cxx else "CC", "") or "") \
        or shutil.which("c++" if cxx else "cc")
    if tool is None or sys.platform == "win32":
        pytest.skip("no C/C++ compiler to build the test model libraries "
                    "(set PULSIM_XDEV_TEST_LIBDIR to CMake-built ones)")
    std = "-std=c++20" if cxx else "-std=c99"
    cmd = [tool, std, "-O2", "-shared", "-fPIC", f"-I{INC}", str(src), "-o",
           str(out)]
    if sys.platform == "darwin":
        cmd.insert(1, "-undefined")
        cmd.insert(2, "dynamic_lookup")
    if not cxx and sys.platform != "darwin":
        cmd.append("-lm")
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return str(out)


@pytest.fixture(scope="module")
def libs(tmp_path_factory):
    d = os.environ.get("PULSIM_XDEV_TEST_LIBDIR")
    if d:
        found = {}
        for f in pathlib.Path(d).iterdir():
            for key in ("twins", "contract"):
                if f"pulsim_xdev_{key}" in f.name:
                    found[key] = str(f)
        return found
    tmp = tmp_path_factory.mktemp("xdev")
    return {
        "twins": _compile(tmp, SRC / "twin_models.cpp", tmp / f"twins{EXT}", True),
        "contract": _compile(tmp, SRC / "contract_models.c",
                             tmp / f"contract{EXT}", False),
    }


SAT = dict(L_0=1e-3, I_sat=1.0, L_residual=1e-4)


def _satind(libs, external):
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("V", "in", "gnd", 0.0, 10.0, 1e3)
    b.add_resistor("R", "in", "a", 0.5)
    if external:
        b.add_external_device("L", ["a", "gnd"], libs["twins"], "satind_twin",
                              {**SAT, "L_0": (1e-3, "H")})
    else:
        b.add_saturable_inductor("L", "a", "gnd", SAT["L_0"], SAT["I_sat"],
                                 SAT["L_residual"])
    return b


SHOCK = dict(I_S=1e-12, n=1.0, V_T=0.025852, G_min=1e-12, i_lim=1e6, BV=0.0)


def _rectifier(libs, external):
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("V", "in", "gnd", 0.0, 20.0, 1e3)
    b.add_resistor("Rs", "in", "a", 1.0)
    if external:
        b.add_external_device("D", ["a", "out"], libs["twins"], "shockley_twin",
                              SHOCK)
    else:
        b.add_shockley_diode("D", "a", "out", SHOCK["I_S"], SHOCK["n"],
                             SHOCK["V_T"], SHOCK["G_min"])
    b.add_capacitor("C", "out", "gnd", 10e-6)
    b.add_resistor("R", "out", "gnd", 100.0)
    return b


def _err(a, b):
    xa, xb = np.asarray(a.states), np.asarray(b.states)
    assert xa.shape == xb.shape
    return float(np.max(np.abs(xa - xb) / np.maximum(np.max(np.abs(xa), axis=0), 1e-9)))


KW = dict(t_end=2e-3, dt=1e-6)


@pytest.mark.parametrize("make", [_satind, _rectifier])
def test_twin_equals_the_builtin_law_on_pwl(libs, make):
    builtin = p.simulate(make(libs, False), device_path="interface", **KW)
    ext = p.simulate(make(libs, True), device_path="interface", **KW)
    assert np.array_equal(np.asarray(builtin.times), np.asarray(ext.times))
    # <= 1e-12 is the local identity gate (macOS, this compiler); another
    # compiler's libm / FMA moves the last bits: 1e-9 elsewhere (CI lesson)
    assert _err(builtin, ext) <= 1e-9


@pytest.mark.parametrize("make", [_satind, _rectifier])
def test_twin_equals_the_builtin_law_on_trbdf2(libs, make):
    kw = dict(t_end=2e-3, dt=1e-6, rtol=1e-6)
    builtin = p.simulate(make(libs, False), engine="trbdf2", device_path="interface", **kw)
    ext = p.simulate(make(libs, True), engine="trbdf2", device_path="interface", **kw)
    t = np.asarray(builtin.times)
    for k in range(np.asarray(builtin.states).shape[1]):
        yi = np.interp(t, np.asarray(ext.times), np.asarray(ext.states)[:, k])
        ya = np.asarray(builtin.states)[:, k]
        scale = max(np.max(np.abs(ya)), 1e-9)
        assert np.max(np.abs(yi - ya)) <= 1e-3 * scale   # 1000x rtol of margin, interpolation


def test_legacy_path_routes_the_external_device_through_its_law(libs):
    a = p.simulate(_satind(libs, True), device_path="legacy", **KW)
    i = p.simulate(_satind(libs, True), device_path="interface", **KW)
    assert _err(a, i) <= 1e-9


def test_dsed_and_dc_operating_point_refuse_by_name(libs):
    with pytest.raises(p.UnsupportedError, match="ExternalDevice"):
        p.simulate(_rectifier(libs, True), engine="dsed", **KW)
    with pytest.raises(ValueError, match="external device 'D'"):
        p.simulate(_rectifier(libs, True), start_from_dc_op=True, **KW)


def test_snapshot_resume_carries_the_external_state_bit_identically(libs):
    def build():
        b = _satind(libs, True)
        b.add_external_device("M", ["in", "gnd"], libs["contract"], "energy_meter",
                              {"G": 1e-3})
        return b
    full = p.simulate(build(), t_end=1e-3, dt=1e-6, device_path="interface")
    first = p.simulate(build(), t_end=5e-4, dt=1e-6, device_path="interface")
    snap = pickle.loads(pickle.dumps(first.final_snapshot))
    assert "external_device" in list(snap.dev_layout_names)
    second = p.simulate(build(), t_end=1e-3, dt=1e-6, device_path="interface",
                        resume_from=snap)
    assert np.array_equal(np.asarray(second.final_snapshot.dev_state),
                          np.asarray(full.final_snapshot.dev_state))
    meter = np.asarray(full.final_snapshot.dev_state)[-1]
    assert meter > 0   # the model's own state (energy) advanced and was carried


def test_non_finite_model_output_in_a_run_is_refused_by_name(libs):
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "a", "gnd", 10.0)   # outside the model's probe box
    b.add_external_device("X", ["a", "gnd"], libs["contract"], "nan_outside")
    with pytest.raises(ValueError, match=r"external device 'X'.*non-finite f\[0\]"):
        p.simulate(b, t_end=1e-5, dt=1e-6)


def test_debug_jacobian_catches_a_lie_the_registration_could_not_see(libs):
    def build(debug):
        b = p.CircuitBuilder()
        b.add_voltage_source("V", "a", "gnd", 4.0)
        b.add_resistor("R", "a", "b", 1.0)
        b.add_external_device("X", ["b", "gnd"], libs["contract"], "liar_outside",
                              debug_jacobian=debug)
        return b
    p.simulate(build(False), t_end=1e-5, dt=1e-6)   # passes registration and runs
    with pytest.raises(ValueError, match="run-time debug check"):
        p.simulate(build(True), t_end=1e-5, dt=1e-6)


def test_registration_refuses_a_lying_model_by_name(libs):
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "a", "gnd", 1.0)
    with pytest.raises(ValueError, match=r"'X1'.*liar_jacobian.*not the derivative"):
        b.add_external_device("X1", ["a", "gnd"], libs["contract"], "liar_jacobian")


def test_parameters_by_name_with_units(libs):
    b = p.CircuitBuilder()
    with pytest.raises(ValueError, match="declares 'H'"):
        b.add_external_device("L", ["a", "gnd"], libs["twins"], "satind_twin",
                              {"L_0": (1.0, "mH")})
    with pytest.raises(ValueError, match="no parameter 'L0'"):
        b.add_external_device("L", ["a", "gnd"], libs["twins"], "satind_twin",
                              {"L0": 1e-3})
    with pytest.raises(ValueError, match="has no model 'nope'"):
        b.add_external_device("L", ["a", "gnd"], libs["twins"], "nope")
