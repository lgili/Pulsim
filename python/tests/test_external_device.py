"""External devices through the C ABI (add-external-device-abi).

Twins: the saturable inductor and the Shockley diode re-implemented as an
EXTERNAL model (core/tests/xdev/twin_models.cpp, through the C++ helper)
and loaded at run time; they must give the built-in law's numbers. The
test models are the CMake-built ``pulsim_xdev_twins`` /
``pulsim_xdev_contract`` modules: every build (the wheel build included)
puts them in ``<build>/xdev_test_models``; ``PULSIM_XDEV_TEST_LIBDIR``
names another directory. Failing both, they are compiled here with the
system compilers; without one the tests skip locally and FAIL on CI (a
skipped portability test proves nothing). The libraries are
illustrations, not part models.
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


def _compile(tmp, src, out, cxx, inc=()):
    tool = shutil.which(os.environ.get("CXX" if cxx else "CC", "") or "") \
        or shutil.which("c++" if cxx else "cc")
    if tool is None or sys.platform == "win32":
        pytest.skip("no C/C++ compiler to build the test model libraries "
                    "(set PULSIM_XDEV_TEST_LIBDIR to CMake-built ones)")
    std = "-std=c++20" if cxx else "-std=c99"
    cmd = [tool, std, "-O2", "-shared", "-fPIC", *[f"-I{i}" for i in inc],
           f"-I{INC}", str(src), "-o", str(out)]
    if sys.platform == "darwin":
        cmd.insert(1, "-undefined")
        cmd.insert(2, "dynamic_lookup")
    if not cxx and sys.platform != "darwin":
        cmd.append("-lm")
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return str(out)


KEYS = ("twins", "contract", "contract_v1")


def _built_libs():
    """The CMake-built modules: PULSIM_XDEV_TEST_LIBDIR, else the newest
    <repo>/build*/[<wheel tag>/]xdev_test_models (pip's build-dir)."""
    d = os.environ.get("PULSIM_XDEV_TEST_LIBDIR")
    dirs = [pathlib.Path(d)] if d else sorted(
        [*ROOT.glob("build/*/xdev_test_models"), *ROOT.glob("build*/xdev_test_models")],
        key=lambda q: q.stat().st_mtime, reverse=True)
    for lib_dir in dirs:
        found = {}
        for f in lib_dir.iterdir():
            if f.suffix not in (".so", ".dll", ".dylib"):
                continue   # .lib / .exp / .pdb beside a Windows .dll
            stem = f.stem.removeprefix("lib")
            for key in KEYS:
                if stem == f"pulsim_xdev_{key}":
                    found[key] = str(f)
        if len(found) == len(KEYS):
            return found
    return None


@pytest.fixture(scope="module")
def libs(tmp_path_factory):
    found = _built_libs()
    if found:
        return found
    tmp = tmp_path_factory.mktemp("xdev")
    try:
        return {
            "twins": _compile(tmp, SRC / "twin_models.cpp", tmp / f"twins{EXT}", True),
            "contract": _compile(tmp, SRC / "contract_models.c",
                                 tmp / f"contract{EXT}", False),
            # the same C models against the header as of c90fc07f
            "contract_v1": _compile(tmp, SRC / "contract_models.c",
                                    tmp / f"contract_v1{EXT}", False,
                                    (SRC / "abi_v1_c90fc07f",)),
        }
    except pytest.skip.Exception:
        if os.environ.get("CI"):
            pytest.fail("CI must build the external-device test models "
                        "(<build>/xdev_test_models): none found, no compiler")
        raise


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


def _satind_hand(libs, external):
    """The same inductor through the adapter's hand-Jacobian path."""
    if not external:
        return _satind(libs, False)
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("V", "in", "gnd", 0.0, 10.0, 1e3)
    b.add_resistor("R", "in", "a", 0.5)
    b.add_external_device("L", ["a", "gnd"], libs["twins"], "satind_hand", SAT)
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


@pytest.mark.parametrize("make", [_satind, _satind_hand, _rectifier])
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


def test_dsed_refuses_by_name(libs):
    with pytest.raises(p.UnsupportedError, match="ExternalDevice"):
        p.simulate(_rectifier(libs, True), engine="dsed", **KW)


# ---- the solve's time reaches the model (ctx->t) --------------------------

def _ramp(lib, R=1e3, G=1e-3):
    """1 V -> R -> a -> ramp conductance g(t) = G (1 + t / 1 ms) -> gnd."""
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "in", "gnd", 1.0)
    b.add_resistor("R", "in", "a", R)
    b.add_external_device("X", ["a", "gnd"], lib, "ramp_conductance", {"G": G})
    return b


def _ramp_exact(t, R=1e3, G=1e-3):
    return 1.0 / (1.0 + R * G * (1.0 + 1e3 * np.asarray(t)))


@pytest.mark.parametrize("which", ["contract", "contract_v1"])
@pytest.mark.parametrize("engine", ["pwl", "trbdf2", "esdirk43"])
def test_a_time_dependent_law_reads_the_solve_time(libs, which, engine):
    """The law g(t) is algebraic, so v_a at every output time is the closed
    form at THAT time: with t stuck at 0 v_a would stay at 0.5 V. The
    library built against the header as of c90fc07f gets the time too
    (esdirk43: each of the five stages at its own time)."""
    b = _ramp(libs[which])
    kw = dict(rtol=1e-6) if engine != "pwl" else {}
    if engine == "esdirk43":
        engine, kw["integrator"] = "trbdf2", "esdirk43"
    res = p.simulate(b, engine=engine, t_end=1e-3, dt=1e-5, **kw)
    t = np.asarray(res.times)
    v = np.asarray(res.states)[:, b.node_id_of("a")]
    assert t[-1] == pytest.approx(1e-3)
    # sample 0 is the zero start (x = 0), not a solve
    assert np.max(np.abs(v[1:] - _ramp_exact(t[1:]))) <= 1e-9   # Newton's tolerance
    assert v[-1] == pytest.approx(1.0 / 3.0, abs=1e-9)


def test_the_static_path_hands_the_model_the_solve_time(libs):
    """run_transient on a dt = 0 cache (no C or L: the static path) reads
    the same law at each grid time; t stuck at 0 kept v_a at 0.5 V."""
    b = _ramp(libs["contract"])
    cache = p.PwlStateSpaceCache(b.graph, b.pool)
    cache.build()
    res = p.run_transient(cache, b.graph, b.pool,
                          p.SimulationOptions(t_start=0.0, t_end=1e-3, dt=1e-5),
                          switch_fn=p.make_combined_switch_fn(b.graph.num_switches, []),
                          enable_nonlinear_refresh=True)
    t = np.asarray(res.times)
    v = np.asarray(res.states)[:, b.node_id_of("a")]
    assert np.max(np.abs(v[1:] - _ramp_exact(t[1:]))) <= 1e-9   # Newton's tolerance


@pytest.mark.parametrize("make", [_satind, _rectifier])
def test_twin_equals_the_builtin_law_on_esdirk43(libs, make):
    # integrator='esdirk43': the external device's q rows take the stage
    # form (base ψ, backward Euler of γh). The twin against the built-in law
    # on the same integrator (same steps), and the built-in against TR-BDF2
    # under a 100 ns ceiling at rtol 1e-10 (the truth here)
    kw = dict(t_end=2e-3, dt=1e-6, rtol=1e-8, atol=1e-11, engine="trbdf2",
              device_path="interface", integrator="esdirk43")
    builtin = p.simulate(make(libs, False), **kw)
    ext = p.simulate(make(libs, True), **kw)
    ref = p.simulate(make(libs, False), **{**kw, "integrator": None, "dt": 1e-7,
                                           "rtol": 1e-10, "atol": 1e-13})
    xb, xe, xr = (np.asarray(r.states) for r in (builtin, ext, ref))
    scale = np.maximum(np.abs(xr).max(axis=0), 1e-9)
    twin = (np.abs(xe[-1] - xb[-1]) / scale).max()
    truth = (np.abs(xb[-1] - xr[-1]) / scale).max()
    # measured: twin 1e-15 / 0 (<= 1e-12 locally; 1e-9 for other libm, CI
    # lesson); truth 1.3e-8 / 5.3e-9, the reference's own error (a 50 ns
    # ceiling quarters it: 3.3e-9 / 1.7e-9)
    assert twin <= 1e-9
    assert truth <= 1e-7


# ---- DC operating point: f(x), df/dx with the storage terms dropped -------

def _dc_rectifier(libs, external):
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "in", "gnd", 5.0)
    b.add_resistor("Rs", "in", "a", 1.0)
    if external:
        b.add_external_device("D", ["a", "out"], libs["twins"], "shockley_twin", SHOCK)
    else:
        b.add_shockley_diode("D", "a", "out", SHOCK["I_S"], SHOCK["n"], SHOCK["V_T"],
                             SHOCK["G_min"])
    b.add_capacitor("C", "out", "gnd", 10e-6)
    b.add_resistor("R", "out", "gnd", 10.0)
    return b


def _dc_satind(libs, external):
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "in", "gnd", 1.0)
    b.add_resistor("R", "in", "a", 0.5)
    if external:
        b.add_external_device("L", ["a", "gnd"], libs["twins"], "satind_twin", SAT)
    else:
        b.add_saturable_inductor("L", "a", "gnd", SAT["L_0"], SAT["I_sat"],
                                 SAT["L_residual"])
    return b


def _dc_x(b):
    return np.asarray(p._pulsim.compute_dc_operating_point(
        b.graph, b.pool, p.SwitchStateMask(b.graph.num_switches)).x)


@pytest.mark.parametrize("make", [_dc_rectifier, _dc_satind])
def test_dc_operating_point_of_the_twin_equals_the_builtin(libs, make):
    xb, xe = _dc_x(make(libs, False)), _dc_x(make(libs, True))
    assert xb.shape == xe.shape
    # <= 1e-12 of scale is the local identity gate; 1e-9 elsewhere (CI lesson)
    assert np.max(np.abs(xb - xe)) <= 1e-9 * np.max(np.abs(xb))
    assert np.max(np.abs(xb)) > 0.5      # a real operating point, not zeros


@pytest.mark.parametrize("make", [_dc_rectifier, _dc_satind])
def test_start_from_dc_op_with_an_external_device_equals_the_builtin(libs, make):
    kw = dict(t_end=2e-5, dt=1e-6, start_from_dc_op=True, device_path="interface")
    rb, re_ = p.simulate(make(libs, False), **kw), p.simulate(make(libs, True), **kw)
    xb, xe = np.asarray(rb.states), np.asarray(re_.states)
    assert np.max(np.abs(xb[0] - xe[0])) <= 1e-9 * np.max(np.abs(xb[0]))
    # the truth: a DC source, so the run stays AT the DC point. (The built-in
    # saturable inductor does not: its flux history is not seeded from the
    # DC point on pwl and its current collapses at step 1 — recorded in
    # DECISIONS #19, not this track's to change.)
    assert np.max(np.abs(xe - xe[0])) <= 1e-9 * np.max(np.abs(xe))


def test_dc_operating_point_hands_the_model_t_eval(libs):
    b = _ramp(libs["contract"])
    x = np.asarray(p._pulsim.compute_dc_operating_point(
        b.graph, b.pool, p.SwitchStateMask(0), t_eval=5e-4).x)
    # the DC solve's own conductance floor (1e-9 S at the node) moves it 1.6e-10
    assert x[b.node_id_of("a")] == pytest.approx(float(_ramp_exact(5e-4)), abs=1e-9)
    assert abs(x[b.node_id_of("a")] - 0.5) > 0.09    # t_eval reached the law


# ---- AC sweep: linearised at the DC point (df/dx + jw dq/dx) --------------

FREQS = np.array([10.0, 1e3, 1e5])


def test_ac_sweep_of_the_external_shockley_is_its_small_signal_resistance(libs):
    """Biased near 1 A: H = r_d / (R_s + r_d), r_d = n V_T / (I + I_S) (G_min
    in parallel), from the DC point the sweep solved itself."""
    b = _dc_rectifier(libs, True)
    res = p.run_mna_sweep(b, output_idx=b.node_id_of("out"), freqs=FREQS,
                          v_in_branch_id=b.branch_id_of("V"))
    x = _dc_x(_dc_rectifier(libs, True))
    va, vo = x[b.node_id_of("a")], x[b.node_id_of("out")]
    i_dc = vo / 10.0 + 0.0                     # the load current = the diode current
    assert 0.3 < i_dc < 1.0
    vd = va - vo
    g = SHOCK["I_S"] / (SHOCK["n"] * SHOCK["V_T"]) * np.exp(vd / (SHOCK["n"] * SHOCK["V_T"])) \
        + SHOCK["G_min"]
    r_d = 1.0 / g
    assert r_d == pytest.approx(SHOCK["n"] * SHOCK["V_T"] / (i_dc - SHOCK["G_min"] * vd
                                                         + SHOCK["I_S"]), rel=1e-9)
    w = 2 * np.pi * FREQS
    zl = 1.0 / (1.0 / 10.0 + 1j * w * 10e-6)   # R || C at the output
    exact = zl / (1.0 + r_d + zl)
    np.testing.assert_allclose(np.asarray(res.H), exact, rtol=1e-9)


def test_ac_sweep_of_the_external_saturable_inductor_is_its_incremental_inductance(libs):
    """At I = 2 A: L(I) = dλ/di = L_res + (L_0 - L_res) / (1 + (I/I_sat)^2) —
    the same sweep as a built-in linear inductor of that L."""
    b = _dc_satind(libs, True)
    res = p.run_mna_sweep(b, output_idx=b.node_id_of("a"), freqs=FREQS,
                          v_in_branch_id=b.branch_id_of("V"))
    i_dc = 1.0 / 0.5
    L = SAT["L_residual"] + (SAT["L_0"] - SAT["L_residual"]) / (1 + (i_dc / SAT["I_sat"]) ** 2)
    lin = p.CircuitBuilder()
    lin.add_voltage_source("V", "in", "gnd", 1.0)
    lin.add_resistor("R", "in", "a", 0.5)
    lin.add_inductor("L", "a", "gnd", L)
    ref = p.run_mna_sweep(lin, output_idx=lin.node_id_of("a"), freqs=FREQS,
                          v_in_branch_id=lin.branch_id_of("V"))
    np.testing.assert_allclose(np.asarray(res.H), np.asarray(ref.H), rtol=1e-9)
    w = 2 * np.pi * FREQS
    np.testing.assert_allclose(np.asarray(res.H), 1j * w * L / (0.5 + 1j * w * L), rtol=1e-9)


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


def test_debug_jacobian_accepts_an_honest_kink_with_one_sided_slopes(libs):
    """A model that does NOT claim SMOOTH declares the backward slope at its
    kink (v = 0); the central quotient averages both sides (0.75 G) and
    matches neither. Registration (the probe centre is v = 0) and the
    run-time debug check accept either one-sided derivative."""
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("V", "in", "gnd", 0.0, 1.0, 1e3)
    b.add_resistor("R", "in", "a", 100.0)
    b.add_external_device("X", ["a", "gnd"], libs["contract"], "honest_kink",
                          debug_jacobian=True)
    res = p.simulate(b, t_end=1e-3, dt=1e-5)
    va = np.asarray(res.states)[:, b.node_id_of("a")]
    assert np.min(va) < 0 < np.max(va)    # the run crossed the kink both ways


def test_a_rebuild_with_the_same_size_and_mtime_is_refused(libs, tmp_path):
    if sys.platform == "win32":
        pytest.skip("Windows locks a loaded DLL: it cannot change on disk")
    lib = tmp_path / f"same{EXT}"
    shutil.copyfile(libs["contract"], lib)
    keep = p.CircuitBuilder()
    keep.add_external_device("X", ["a", "gnd"], str(lib), "good_resistor")
    st = os.stat(lib)
    data = bytearray(lib.read_bytes())
    data[-1] ^= 0xFF                       # one byte: same size
    # replaced the way a linker does it (a new file renamed over the path).
    # Writing into the loaded file itself truncates it, and Linux then drops
    # the image's relocated pages: the process dies at the next call into it.
    new = tmp_path / f"same.new{EXT}"
    new.write_bytes(bytes(data))
    os.utime(new, ns=(st.st_atime_ns, st.st_mtime_ns))   # same mtime
    os.replace(new, lib)
    assert os.stat(lib).st_size == st.st_size and os.stat(lib).st_mtime_ns == st.st_mtime_ns
    with pytest.raises(ValueError, match="changed on disk"):
        p.CircuitBuilder().add_external_device("Y", ["a", "gnd"], str(lib), "good_resistor")
    del keep


@pytest.mark.skipif((os.cpu_count() or 1) < 2, reason="a race needs two cores to fire")
def test_registration_refuses_a_model_that_is_not_reentrant(libs):
    """liar_racy is pure when called serially (it passes the repeat and the
    interleaved checks) but keeps its scratch in a static: concurrent calls
    corrupt each other. The threaded registration probe sees it."""
    with pytest.raises(ValueError, match="not reentrant"):
        p.CircuitBuilder().add_external_device("X", ["a", "gnd"], libs["contract"],
                                               "liar_racy")


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


@pytest.mark.skipif(sys.platform == "win32",
                    reason="Windows locks a loaded DLL: it cannot change on disk")
def test_a_library_rebuilt_while_loaded_is_refused_not_silently_stale(libs, tmp_path):
    import gc
    lib = tmp_path / f"copy{EXT}"
    shutil.copyfile(libs["contract"], lib)
    keep = p.CircuitBuilder()
    keep.add_external_device("X", ["a", "gnd"], str(lib), "good_resistor")
    with open(lib, "ab") as fh:   # a "rebuild": same path, new size
        fh.write(b"\0" * 64)
    b = p.CircuitBuilder()
    with pytest.raises(ValueError, match="changed on disk since it was loaded"):
        b.add_external_device("Y", ["a", "gnd"], str(lib), "good_resistor")
    del keep, b
    gc.collect()   # the last user gone: Pulsim closes the library
    # the new build loads only if the system really unloaded the old image;
    # otherwise it is refused again, never silently stale
    try:
        p.CircuitBuilder().add_external_device("Z", ["a", "gnd"], str(lib), "good_resistor")
    except ValueError as exc:
        assert "still mapped" in str(exc)


def test_get_include_holds_the_stable_header():
    inc = pathlib.Path(p.get_include())
    assert (inc / "pulsim" / "xdev" / "pulsim_xdev.h").is_file()


def test_threaded_sweep_with_an_external_device_equals_the_serial_run(libs):
    """The host is reentrant: per-instance buffers and state, no globals
    but the mutex-guarded library registry. Eight points on eight threads
    (each registers its own instance concurrently) equal the serial run to
    the bit (same code path, same machine)."""
    def factory(R):
        b = _rectifier(libs, True)
        b.add_resistor("Rx", "out", "gnd", R)
        return b

    def kpi(res, _):
        x = np.asarray(res.states)
        return {"last": float(x[-1].sum()), "peak": float(np.abs(x).max())}

    grid = dict(params={"R": [50.0 + 10 * k for k in range(8)]}, kpi_fn=kpi,
                verbose=False, t_end=5e-4, dt=1e-6)
    serial = p.sweep(factory, **grid)
    threaded = p.sweep(factory, parallel=True, backend="thread", n_workers=8, **grid)
    for k in ("last", "peak"):
        assert np.array_equal(serial.kpis[k], threaded.kpis[k])


@pytest.mark.parametrize("engine", ["pwl", "trbdf2"])
def test_recorders_with_an_external_device_in_the_circuit(libs, engine):
    """In the loop the external device has no current law: the other
    readings are fed in the loop, and its own i() falls back to the stored
    arrays as every no-law pool kind does, where res.i refuses it by name."""
    kw = dict(dt=1e-6) if engine == "pwl" else dict(rtol=1e-6)
    b = _rectifier(libs, True)
    res = p.simulate(b, 2e-3, engine=engine, device_path="interface", **kw,
                     recorders=[p.Recorder("r", ["v(out)"])])
    assert res.recorder_feed["mode"] == "in-loop", res.recorder_feed
    np.testing.assert_allclose(res.recorded["r"]["v(out)"],
                               np.asarray(res.states)[:, b.node_id_of("out")],
                               rtol=1e-12, atol=0.0)
    with pytest.raises(NotImplementedError, match="branch 'D'"):
        p.simulate(_rectifier(libs, True), 2e-3, engine=engine,
                   device_path="interface", **kw,
                   recorders=[p.Recorder("r", ["i(D)"])])
