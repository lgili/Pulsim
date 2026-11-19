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
@pytest.mark.parametrize("engine", ["pwl", "trbdf2"])
def test_a_time_dependent_law_reads_the_solve_time(libs, which, engine):
    """The law g(t) is algebraic, so v_a at every output time is the closed
    form at THAT time: with t stuck at 0 v_a would stay at 0.5 V. The
    library built against the header as of c90fc07f gets the time too."""
    b = _ramp(libs[which])
    kw = dict(rtol=1e-6) if engine == "trbdf2" else {}
    res = p.simulate(b, engine=engine, t_end=1e-3, dt=1e-5, **kw)
    t = np.asarray(res.times)
    v = np.asarray(res.states)[:, b.node_id_of("a")]
    assert t[-1] == pytest.approx(1e-3)
    # sample 0 is the zero start (x = 0), not a solve
    assert np.max(np.abs(v[1:] - _ramp_exact(t[1:]))) <= 1e-9   # Newton's tolerance
    assert v[-1] == pytest.approx(1.0 / 3.0, abs=1e-9)


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
    lib.write_bytes(bytes(data))
    os.utime(lib, ns=(st.st_atime_ns, st.st_mtime_ns))   # same mtime
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


# ---- 3.1 internal unknown: the Lauritzen diode with its charge as a row ---

LAUR = dict(I_S=1e-12, n=1.0, V_T=0.025852, tau=1e-7, T_M=1e-8, G_min=1e-12)


def _recovery(libs, external):
    """The double-pulse test of test_lauritzen_diode: 20 A freewheeling in
    D1 until S1 turns on at 1 us and commutates it (reverse recovery)."""
    b = p.CircuitBuilder()
    b.add_voltage_source("Vdc", "dc", "gnd", 400.0)
    b.add_inductor("Lload", "dc", "sw", 200e-6, i0=20.0)
    b.add_inductor("Lstray", "d2", "dc", 50e-9, i0=20.0)
    if external:
        b.add_external_device("D1", ["sw", "d2"], libs["twins"], "lauritzen_twin", LAUR)
    else:
        b.add_lauritzen_diode("D1", "sw", "d2", **LAUR)
    b.add_switch("S1", "sw", "gnd", 1e3, 1e-9)
    b.add_resistor("Rsnub", "sw", "gnd", 1e5)
    n, k = b.graph.num_switches, b.switch_index_of("S1")

    def sf(t):
        m = p.SwitchStateMask(n)
        if t >= 1e-6:
            m.set(k, True)
        return m
    return b, sf


def _recovery_err(a, b):
    """Worst relative gap of the node voltages and the two inductor
    currents (the external run has one more state column, q_M)."""
    worst = 0.0
    for get, keys in ((a.v, ("dc", "sw", "d2")), (a.i, ("Lload", "Lstray"))):
        for key in keys:
            ya = np.asarray(get(key))
            yb = np.asarray((b.v if get == a.v else b.i)(key))
            worst = max(worst, float(np.max(np.abs(ya - yb)) / max(np.max(np.abs(ya)), 1e-9)))
    return worst


@pytest.mark.parametrize("path", ["legacy", "interface"])
def test_lauritzen_twin_with_an_internal_charge_equals_the_builtin_on_pwl(libs, path):
    kw = dict(t_end=1.5e-6, dt=2e-10, engine="pwl", device_path=path)
    bb, sf = _recovery(libs, False)
    builtin = p.simulate(bb, switch_fn=sf, **kw)
    be, sf = _recovery(libs, True)
    ext = p.simulate(be, switch_fn=sf, **kw)
    assert np.array_equal(np.asarray(builtin.times), np.asarray(ext.times))
    assert np.asarray(ext.i("Lstray")).min() < -1.0   # it recovers
    # local gate 1e-10 (spec); 1e-9 for another compiler's libm (CI lesson)
    assert _recovery_err(builtin, ext) <= 1e-9


def test_lauritzen_twin_on_trbdf2(libs):
    kw = dict(t_end=1.5e-6, dt=2e-10, engine="trbdf2", rtol=1e-6, device_path="interface")
    bb, sf = _recovery(libs, False)
    builtin = p.simulate(bb, switch_fn=sf, **kw)
    be, sf = _recovery(libs, True)
    ext = p.simulate(be, switch_fn=sf, **kw)
    t = np.asarray(builtin.times)
    ia = np.asarray(builtin.i("Lstray"))
    ib = np.interp(t, np.asarray(ext.times), np.asarray(ext.i("Lstray")))
    assert ib.min() < -1.0
    assert np.max(np.abs(ia - ib)) <= 1e-3 * np.max(np.abs(ia))   # interpolation margin


# ---- 2.2 charge only: the junction capacitor --------------------------------

JCAP = dict(C0=2e-9, V0=25.0, m=0.5)


def _charge(libs, external, record=False):
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "in", "gnd", 400.0)
    b.add_resistor("R", "in", "a", 10.0)
    if external:
        b.add_external_device("C", ["a", "gnd"], libs["twins"], "junction_cap_twin", JCAP,
                              record=record)
        # the built-in stamps G_min = 1e-12 S on each of its nodes (assemble.hpp:
        # a Coss row is structurally empty otherwise): 4e-10 A at 400 V
        b.add_resistor("Gmin", "a", "gnd", 1e12)
    else:
        b.add_nonlinear_capacitor("C", "a", "gnd", JCAP["C0"], JCAP["V0"], JCAP["m"])
    return b


@pytest.mark.parametrize("path", ["legacy", "interface"])
def test_junction_capacitor_twin_equals_add_nonlinear_capacitor_on_pwl(libs, path):
    kw = dict(t_end=2e-6, dt=1e-9, engine="pwl", device_path=path)
    builtin = p.simulate(_charge(libs, False), **kw)
    ext = p.simulate(_charge(libs, True), **kw)
    assert np.array_equal(np.asarray(builtin.times), np.asarray(ext.times))
    assert _err(builtin, ext) <= 1e-12   # the spec gate; measured 2.8e-15
    # the charge that entered the device: the integral of its own pin current
    # (the generic reading, q-row companion current). Not the resistor's
    # current: its t = 0 sample (40 A, v = 0) is a value no step integrated.
    rec = p.simulate(_charge(libs, True, record=True), **kw)
    t = np.asarray(rec.times)
    q = float(np.trapezoid(np.asarray(rec.device_output("C", "i(pos)")), t))
    assert np.asarray(ext.v("a"))[-1] == pytest.approx(400.0, abs=1e-6)
    assert q == pytest.approx(312.31e-9, rel=2e-5)


# ---- 3.2 kernel-owned state: rejected steps, resume, steady state ---------

def _metered(libs, record=False):
    b = _satind(libs, True)
    b.add_external_device("M", ["in", "gnd"], libs["contract"], "metered_resistor",
                          {"G": 1e-3}, record=record)
    return b


def test_rejected_trbdf2_steps_leave_no_trace_in_the_model_state(libs):
    """accept runs only for accepted steps: the meter's energy (rectangle
    rule over the accepted steps, x at each step's end) is reproduced from
    the stored grid alone, on a run that rejected steps."""
    b = _rectifier(libs, True)   # the diode's turn-on rejects steps; the smooth satind does not
    b.add_external_device("M", ["in", "gnd"], libs["contract"], "metered_resistor", {"G": 1e-3})
    res = p.simulate(b, engine="trbdf2", t_end=2e-3, dt=1e-5, rtol=1e-6,
                     device_path="interface")
    assert res.stats.n_reject > 0
    t = np.asarray(res.times)
    v = np.asarray(res.v("in"))
    e = float(np.sum(np.diff(t) * 1e-3 * v[1:] ** 2))
    meter = float(np.asarray(res.final_snapshot.dev_state)[-1])
    assert meter == pytest.approx(e, rel=1e-12)


def test_trbdf2_snapshot_resume_carries_the_model_state(libs):
    kw = dict(engine="trbdf2", dt=1e-6, rtol=1e-6, device_path="interface")
    full = p.simulate(_metered(libs), t_end=1e-3, **kw)
    first = p.simulate(_metered(libs), t_end=5e-4, **kw)
    snap = pickle.loads(pickle.dumps(first.final_snapshot))
    second = p.simulate(_metered(libs), t_end=1e-3, resume_from=snap, **kw)
    a = np.asarray(second.final_snapshot.dev_state)
    b = np.asarray(full.final_snapshot.dev_state)
    # the resumed run restarts its step controller at the snapshot: the
    # same physics on another grid, not the same bits (measured 1.2e-4)
    np.testing.assert_allclose(a, b, rtol=1e-3, atol=1e-12)
    # exact on its own grid: the meter resumed from the snapshot's energy and
    # added the rectangle rule over the second run's accepted steps only
    e0 = float(np.asarray(first.final_snapshot.dev_state)[-1])
    t = np.asarray(second.times)
    v = np.asarray(second.v("in"))
    assert e0 > 0
    assert a[-1] == pytest.approx(e0 + float(np.sum(np.diff(t) * 1e-3 * v[1:] ** 2)), rel=1e-12)


@pytest.mark.parametrize("kw", [dict(engine="pwl", dt=1e-6),
                                dict(engine="trbdf2", rtol=1e-7, atol=1e-10)])
def test_device_output_covers_a_pwm_run_with_gate_edges_and_a_diode(libs, kw):
    """trbdf2 commits a step, then a gate-edge restart moves t in the same
    iteration: every committed sample needs its trace row (review #1)."""
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", 24.0)
    b.add_mosfet("Q1", "vin", "sw")
    b.add_diode("D1", "gnd", "sw", 1e3, 1e-9)
    b.add_inductor("L", "sw", "vout", 100e-6)
    b.add_capacitor("C", "vout", "gnd", 47e-6)
    b.add_resistor("RL", "vout", "gnd", 10.0)
    b.add_external_device("M", ["vout", "gnd"], libs["contract"], "metered_resistor",
                          {"G": 1e-3}, record=True)
    fn = p.make_pwm_switch_fn(100e3, 0.5, b.switch_index_of("Q1"), b.graph.num_switches)
    res = p.simulate(b, switch_fn=fn, t_end=2e-4, **kw)
    v = np.asarray(res.v("vout"))
    np.testing.assert_allclose(res.device_output("M", "i(a)"), 1e-3 * v, rtol=0, atol=1e-15)


def test_steady_state_refuses_the_model_state_by_name(libs):
    """The period map copies named state blocks only; the external block is
    not one yet, so steady_state refuses instead of restarting the model's
    state every period (the affine map would see no state at all)."""
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("V", "in", "gnd", 0.0, 10.0, 1e3)
    b.add_resistor("R", "in", "a", 0.5)
    b.add_inductor("L", "a", "gnd", 1e-3)
    b.add_external_device("M", ["in", "gnd"], libs["contract"], "metered_resistor", {"G": 1e-3})
    with pytest.raises(NotImplementedError, match="external_device"):
        p.steady_state(b, period=1e-3, dt=1e-6, device_path="interface")
    with pytest.raises(p.UnsupportedError, match="stateful"):
        p.steady_state(b, period=1e-3)      # no dt: periodic_steady_state


# ---- 3.4 readings: pin currents, power, named outputs ----------------------

@pytest.mark.parametrize("engine", ["pwl", "trbdf2"])
def test_device_output_reads_the_generic_readings_and_the_named_outputs(libs, engine):
    kw = dict(rtol=1e-6) if engine == "trbdf2" else {}
    res = p.simulate(_metered(libs, record=True), engine=engine, t_end=1e-3, dt=1e-6,
                     device_path="interface", **kw)
    t = np.asarray(res.times)
    v = np.asarray(res.v("in"))
    np.testing.assert_allclose(res.device_output("M", "i(a)"), 1e-3 * v, rtol=0, atol=1e-15)
    np.testing.assert_allclose(res.device_output("M", "i(b)"), -1e-3 * v, rtol=0, atol=1e-15)
    np.testing.assert_allclose(res.device_output("M", "power"), 1e-3 * v * v, rtol=0, atol=1e-13)
    np.testing.assert_allclose(res.device_output("M", "p_law"), 1e-3 * v * v, rtol=0, atol=1e-13)
    e = np.concatenate([[0.0], np.cumsum(np.diff(t) * 1e-3 * v[1:] ** 2)])
    np.testing.assert_allclose(res.device_output("M", "energy"), e, rtol=1e-12, atol=1e-18)
    with pytest.raises(KeyError, match="has"):
        res.device_output("M", "nope")
    with pytest.raises(ValueError, match="not recorded"):
        res.device_output("L", "power")


def test_device_output_pin_current_includes_the_charge_rate(libs):
    """A q-only device: its pin current is the committed dq/dt, which the
    series resistor carries (KCL), to Newton's tolerance."""
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "in", "gnd", 400.0)
    b.add_resistor("R", "in", "a", 10.0)
    b.add_external_device("C", ["a", "gnd"], libs["twins"], "junction_cap_twin", JCAP,
                          record=True)
    res = p.simulate(b, engine="pwl", t_end=2e-7, dt=1e-9)
    i_r = (400.0 - np.asarray(res.v("a"))) / 10.0
    i_c = res.device_output("C", "i(pos)")
    np.testing.assert_allclose(i_c[1:], i_r[1:], rtol=0, atol=1e-6 * np.max(np.abs(i_r)))


def test_device_output_without_record_is_refused(libs):
    res = p.simulate(_metered(libs), t_end=1e-5, dt=1e-6)
    with pytest.raises(ValueError, match="record=True"):
        res.device_output("M", "power")


@pytest.mark.parametrize("model, frag", [("liar_output_name", "generic reading"),
                                         ("liar_events", "event functions")])
def test_outputs_block_refusals(libs, model, frag):
    b = p.CircuitBuilder()
    with pytest.raises(ValueError, match=frag):
        b.add_external_device("X", ["a", "gnd"], libs["contract"], model)


def test_a_library_built_before_the_outputs_block_has_no_outputs(libs):
    """contract_v1 (the header as of c90fc07f) has no appended block: its
    energy meter loads, records the generic readings and nothing more."""
    b = _satind(libs, True)
    b.add_external_device("M", ["in", "gnd"], libs["contract_v1"], "energy_meter",
                          {"G": 1e-3}, record=True)
    res = p.simulate(b, t_end=1e-4, dt=1e-6)
    assert list(res._xdev_trace.columns("M")) == ["i(a)", "i(b)", "power"]
    with pytest.raises(ValueError, match="no model 'metered_resistor'"):
        p.CircuitBuilder().add_external_device("M", ["a", "gnd"], libs["contract_v1"],
                                               "metered_resistor")


# ---- 1.6 components(): the library and its content hash -------------------

def test_components_entry_names_the_library_and_its_hash(libs):
    b = _rectifier(libs, True)
    d = next(c for c in b.components() if c["name"] == "D")
    assert d["kind"] == "external_device"
    assert d["model"] == "shockley_twin"
    assert os.path.samefile(d["library"], libs["twins"])
    h = d["library_hash"]
    assert h.startswith("fnv1a64:") and len(h) == len("fnv1a64:") + 16
    assert d["params"]["I_S"] == SHOCK["I_S"] and d["params"]["n"] == SHOCK["n"]
    assert [p_ for p_ in d["pins"]] == [b.node_id_of("a"), b.node_id_of("out")]


# ---- 1.7 YAML: native code only from libraries the caller lists -----------

def _netlist(lib, extra=""):
    return f"""
circuit:
  devices:
    - {{type: voltage_source, name: V, from: in, to: gnd, V: 5.0}}
    - {{type: resistor, name: Rs, from: in, to: a, R: 1.0}}
    - type: external_device
      name: D
      nodes: [a, out]
      lib: {lib}
      model: shockley_twin
      params: {{I_S: 1.0e-12, n: [1.0, ""], V_T: [0.025852, V]}}{extra}
    - {{type: resistor, name: R, from: out, to: gnd, R: 10.0}}
simulation: {{t_end: 1.0e-5, dt: 1.0e-6}}
"""


def test_yaml_external_device_is_refused_without_an_allow_list(libs):
    with pytest.raises(RuntimeError, match=r"D.*allow_native_code=\["):
        p.load_yaml_string(_netlist(libs["twins"]))


def test_yaml_allow_native_code_true_is_refused(libs):
    with pytest.raises(ValueError, match="pass the library paths"):
        p.load_yaml_string(_netlist(libs["twins"]), allow_native_code=True)
    for one in (libs["twins"], pathlib.Path(libs["twins"])):   # a list, not one path
        with pytest.raises(TypeError, match="not one path"):
            p.load_yaml_string(_netlist(libs["twins"]), allow_native_code=one)


def test_yaml_external_device_loads_a_listed_library_or_directory(libs, tmp_path):
    lib = pathlib.Path(libs["twins"])
    for allow in ([str(lib)], [lib.parent]):
        lc = p.load_yaml_string(_netlist(lib), allow_native_code=allow)
        d = next(c for c in lc.builder.components() if c["name"] == "D")
        assert d["kind"] == "external_device" and d["params"]["I_S"] == 1e-12
    # a relative lib: is taken from the file's folder; symlinks are followed
    # before the comparison
    if sys.platform != "win32":
        link = tmp_path / f"alias{lib.suffix}"
        link.symlink_to(lib)
        f = tmp_path / "net.yaml"
        f.write_text(_netlist(link.name), encoding="utf-8")
        lc = p.load_yaml_file(str(f), allow_native_code=[lib])
        b = p.CircuitBuilder()
        b.add_voltage_source("V", "in", "gnd", 5.0)
        b.add_resistor("Rs", "in", "a", 1.0)
        b.add_external_device("D", ["a", "out"], str(lib), "shockley_twin",
                              {"I_S": 1e-12})
        b.add_resistor("R", "out", "gnd", 10.0)
        ra = p.simulate(lc.builder, t_end=1e-5, dt=1e-6)
        rb = p.simulate(b, t_end=1e-5, dt=1e-6)
        assert np.array_equal(np.asarray(ra.states), np.asarray(rb.states))
        # the link's directory is not the library's: listing it allows nothing
        with pytest.raises(RuntimeError, match="not in allow_native_code"):
            p.load_yaml_file(str(f), allow_native_code=[tmp_path])


def test_yaml_external_device_in_a_subsystem_takes_the_block_nodes(libs):
    lib = libs["twins"]
    doc = f"""
circuit:
  devices:
    - {{type: voltage_source, name: V, from: in, to: gnd, V: 5.0}}
    - {{type: resistor, name: R, from: out, to: gnd, R: 10.0}}
  subsystems:
    - name: blk
      ports: {{p: in, n: out}}
      devices:
        - {{type: resistor, name: Rs, from: p, to: a, R: 1.0}}
        - {{type: external_device, name: D, nodes: [a, n], lib: {lib},
           model: shockley_twin, params: {{I_S: 1.0e-12}}}}
"""
    c = p.load_yaml_string(doc, allow_native_code=[lib]).builder.build()
    assert c.nodes == ["in", "out", "blk/a"]   # the port n is out, a is scoped


def test_yaml_external_device_keys_are_strict(libs):
    lib = libs["twins"]
    with pytest.raises(RuntimeError, match="unknown key 'code'"):
        p.load_yaml_string(_netlist(lib, "\n      code: 'return 0;'"),
                           allow_native_code=[lib])
    bad = _netlist(lib).replace("I_S: 1.0e-12", "I_SS: 1.0e-12")
    with pytest.raises(ValueError, match="no parameter 'I_SS'"):
        p.load_yaml_string(bad, allow_native_code=[lib])
    bad = _netlist(lib).replace("V_T: [0.025852, V]", "V_T: [25.852, mV]")
    with pytest.raises(ValueError, match="declares 'V'"):
        p.load_yaml_string(bad, allow_native_code=[lib])
