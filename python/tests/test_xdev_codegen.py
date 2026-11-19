"""Equations to C (`pulsim.xdev`, add-external-device-abi section 4).

SymPy laws compiled to C99 and loaded through the external-device ABI: the
generated C against the symbolic law (mpmath through ``lambdify``), the
SymPy Shockley diode against the built-in, the saturable inductor (an
internal unknown and a charge) against its hand-written twin, the
smoothness lint, the monotonicity scan, the trust boundary on names.

Needs SymPy and a C compiler: without them the tests skip locally and FAIL
on CI (``CI`` set; ``pip install .[dev]`` brings SymPy). Windows skips: the
native cache drives a gcc/clang-style command line only.
"""
import hashlib
import os
import re
import subprocess
import sys

import numpy as np
import pytest

import pulsim as p
from pulsim import c_block, xdev

from .test_external_device import KW, SAT, SHOCK, _err, _rectifier, _satind, libs  # noqa: F401

try:
    import sympy as sp
except ImportError:
    sp = None


def _need():
    why = None
    if sys.platform == "win32":
        pytest.skip("xdev codegen: the native cache has no MSVC command line")
    if sp is None:
        why = "SymPy is not installed (pip install pulsim[codegen])"
    elif c_block._find_compiler("c") is None:
        why = "no C compiler"
    if why:
        if os.environ.get("CI"):
            pytest.fail(f"CI must run the xdev codegen tests: {why}")
        pytest.skip(why)


@pytest.fixture(autouse=True)
def _codegen_ready(request):
    if not request.node.name.startswith("test_missing_"):
        _need()


# identity gate on this machine; another compiler / libm moves the last bits
# (CI lesson: two code paths agree to assert_allclose, not to the bit)
LOCAL = sys.platform == "darwin" and not os.environ.get("CI")
TWIN_TOL = 1e-12 if LOCAL else 1e-9


# ---- the test laws ---------------------------------------------------------

def shockley_law():
    """models/shockley_diode.hpp: exponential continued by its tangent above
    v_lim = n V_T ln(i_lim / I_S + 1), optional breakdown, G_min."""
    a, k = xdev.pins("anode", "cathode")
    I_S, n, V_T, G_min, i_lim, BV = xdev.params(
        I_S=(1e-12, "A", 1e-30, 1.0), n=(1.0, "", 0.1, 10.0), V_T=(0.025852, "V", 1e-3, 1.0),
        G_min=(1e-12, "S", 0.0, 1.0), i_lim=(1e6, "A", 1e-6, 1e12), BV=(0.0, "V", 0.0, 1e5))
    vte = n * V_T
    vl = vte * sp.log(i_lim / I_S + 1)
    el = sp.exp(vl / vte)

    def branch(u):
        return sp.Piecewise((I_S * (el - 1) + I_S * el / vte * (u - vl), u > vl),
                            (I_S * (sp.exp(u / vte) - 1), True))

    u = a - k
    i = branch(u) - sp.Piecewise((branch(-BV - u), BV > 0), (0, True)) + u * G_min
    return dict(name="shockley_gen", pins=[a, k], f=[i, -i])


def satind_law():
    """The saturable inductor: internal row = the branch current i, its flux
    as a charge: f = {i, -i, v_from - v_to}, q = {0, 0, -lambda(i)}."""
    fr, to = xdev.pins("from", "to")
    i = xdev.internal("i")
    L0, Is, Lr = xdev.params(L_0=(1e-3, "H", 1e-12, 1e3), I_sat=(1.0, "A", 1e-9, 1e6),
                             L_residual=(1e-4, "H", 0.0, 1e3))
    lam = Lr * i + (L0 - Lr) * Is * sp.atan(i / Is)
    return dict(name="satind_gen", pins=[fr, to], internal=[i], f=[i, -i, fr - to],
                q=[0, 0, -lam], probe={fr: (-10, 10), to: (-10, 10), i: (-20, 20)})


def mixed_law():
    """exp(u) - 1 and log(1 + z) shapes, an internal unknown, charges."""
    a, b = xdev.pins("a", "b")
    z = xdev.internal("z")
    G, K, V0, C0, C1, A = xdev.params(G=(1e-3, "S", 0, 1), K=(1e-6, "A", 0, 1),
                                      V0=(0.5, "V", 1e-3, 10), C0=(1e-9, "F", 0, 1),
                                      C1=(1e-12, "F/V^2", 0, 1), A=(2.0, "", 0.1, 10))
    u = a - b
    i = G * u + K * (sp.exp(u / V0) - 1) + 1e-3 * z
    qp = C0 * u + C1 * u ** 3 / 3
    return dict(name="mixed_gen", pins=[a, b], internal=[z],
                f=[i, -i, z - A * sp.log(1 + u ** 2)], q=[qp, -qp, 1e-9 * (z + z ** 3 / 3)],
                probe={z: (-2, 2)})


def meter_law():
    """A state (energy, rectangle rule over the accepted steps) and named
    outputs: contract_models.c's metered_resistor."""
    a, b = xdev.pins("a", "b")
    G = xdev.params(G=(1e-3, "S", 1e-9, 1e3))
    E = xdev.state("E", init=0)
    v = a - b
    return dict(name="meter_gen", pins=[a, b], f=[G * v, -G * v],
                update={E: E + xdev.h * G * v ** 2},
                outputs={"energy": (E, "J"), "p_law": (G * v ** 2, "W")})


LAWS = {"shockley": shockley_law, "satind": satind_law, "mixed": mixed_law,
        "meter": meter_law}


def _compile(which, **kw):
    law = LAWS[which]()
    law.update(kw)
    return xdev.compile(**law), law


# ---- 4.1 / 4.3: generated C against the symbolic law -----------------------

@pytest.mark.parametrize("which", list(LAWS))
def test_generated_c_matches_the_symbolic_law_at_1000_points(which):
    """f, q and both Jacobians from the library's own eval against mpmath
    (lambdify, 30 digits) of the DECLARED expressions and of their SymPy
    derivatives, at 1000 random points (the Shockley box widened past its
    tangent continuation). Bound: 1e-13 relative, floor 1e-13 of the
    entry's scale over the points (a printer / CSE / pattern bug is an O(1)
    error; the floor only forgives entries near their own zero, where
    double rounding of the inputs dominates)."""
    import mpmath
    dev, law = _compile(which)
    unk = law["pins"] + law.get("internal", [])
    R = len(unk)
    q = law.get("q", [0] * R)
    prm = dev.param_syms
    jf = [[sp.diff(e, x) for x in unk] for e in law["f"]]
    jq = [[sp.diff(sp.sympify(e), x) for x in unk] for e in q]
    ref = sp.lambdify(unk + prm, [law["f"], q, jf, jq], modules="mpmath")
    rng = np.random.default_rng(20261119)
    box = [law.get("probe", {}).get(x, (-1.0, 1.0)) for x in unk]
    if which == "shockley":
        box[0] = (-2.0, 2.0)            # v_lim = 1.07 V: both pieces
    pv = [f[1] for f in dev.fields]
    worst = 0.0
    got, want = [], []
    mpmath.mp.dps = 30
    for _ in range(1000):
        x = np.array([rng.uniform(lo, hi) for lo, hi in box])
        f, qv, Jf, Jq = dev.eval(x)
        r = ref(*[mpmath.mpf(float(v)) for v in x], *[mpmath.mpf(v) for v in pv])
        got.append(np.concatenate([f, qv, Jf.ravel(), Jq.ravel()]))
        want.append(np.array([float(v) for v in (*r[0], *r[1], *sum(r[2], []), *sum(r[3], []))]))
    got, want = np.array(got), np.array(want)
    scale = np.max(np.abs(want), axis=0)
    with np.errstate(invalid="ignore"):     # 0/0 for entries that are identically 0
        err = np.abs(got - want) / np.maximum(np.abs(want), scale)
    worst = float(np.max(np.nan_to_num(err, nan=0.0)))
    assert np.all(np.isfinite(got))
    assert np.all(np.abs(got - want) <= 1e-13 * np.maximum(np.abs(want), scale)), worst
    if which == "mixed":
        assert "expm1(" in dev.source and "log1p(" in dev.source


def test_the_pattern_is_the_structurally_nonzero_derivatives():
    dev, _ = _compile("satind")
    assert dev.f_pattern == [(0, 2), (1, 2), (2, 0), (2, 1)]
    assert dev.q_pattern == [(2, 2)]
    dev, _ = _compile("mixed")
    assert dev.f_pattern == [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2),
                             (2, 0), (2, 1), (2, 2)]
    assert dev.q_pattern == [(0, 0), (0, 1), (1, 0), (1, 1), (2, 2)]


def test_same_law_gives_byte_identical_source_in_another_process(tmp_path):
    """The cache key is the source: it must not depend on SymPy's Dummy
    numbering (the other process makes 1000 Dummies first)."""
    dev, _ = _compile("shockley")
    again, _ = _compile("shockley")
    assert again.source == dev.source and again.lib == dev.lib
    script = tmp_path / "again.py"
    script.write_text(
        "import hashlib, sympy as sp, sys\n"
        "[sp.Dummy() for _ in range(1000)]\n"
        "from python.tests.test_xdev_codegen import _compile\n"
        "print(hashlib.sha256(_compile('shockley')[0].source.encode()).hexdigest())\n",
        encoding="utf-8")
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(
        [os.path.dirname(os.path.dirname(p.__file__)), root, os.environ.get("PYTHONPATH", "")]))
    r = subprocess.run([sys.executable, str(script)], cwd=root, env=env,
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    assert r.stdout.strip() == hashlib.sha256(dev.source.encode()).hexdigest()


def test_generated_c_is_pedantic_c99_without_warnings(tmp_path):
    """Every test law's C through `-std=c99 -pedantic -Wall -Wextra -Werror`
    (C90-style declarations too) on each compiler present (cc; gcc-15 / gcc /
    clang when installed; CI's gcc on Linux)."""
    import shutil
    ccs = {}
    for name in (c_block._find_compiler("c"), "gcc-15", "gcc", "clang"):
        path = shutil.which(name) if name else None
        if path:
            ccs.setdefault(os.path.realpath(path), path)
    devs = {w: _compile(w)[0] for w in LAWS}
    devs["kinked"] = _kinked(lambda u: sp.Max(u, 0) + sp.Abs(u - 0.5), allow_nonsmooth=True)
    for which, dev in devs.items():
        src = tmp_path / f"{which}.c"
        src.write_text(dev.source, encoding="utf-8")
        for cc in ccs.values():
            r = subprocess.run([cc, "-std=c99", "-pedantic", "-Wall", "-Wextra", "-Werror",
                                "-Wdeclaration-after-statement", "-fsyntax-only",
                                f"-I{p.get_include()}", str(src)],
                               capture_output=True, text=True)
            assert r.returncode == 0, f"{which} ({cc}):\n{r.stderr}"
        # pure: every static is const data or a function; no M_PI / near / far
        assert not re.search(r"static (?!const |int32_t PULSIM_XDEV_CALL )", dev.source), which
        assert not re.search(r"\b(M_PI|M_E|near|far)\b", dev.source), which


# ---- 4.3: the twins in a circuit -------------------------------------------

def test_sympy_shockley_equals_the_builtin_on_pwl():
    """The C1 Piecewise (tangent continuation) is proved by the lint: the
    model keeps SMOOTH and passes the registration checks."""
    dev, _ = _compile("shockley")
    assert dev.smooth
    builtin = p.simulate(_rectifier(None, False), device_path="interface", **KW)
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("V", "in", "gnd", 0.0, 20.0, 1e3)
    b.add_resistor("Rs", "in", "a", 1.0)
    dev.add_to(b, "D", "a", "out", **SHOCK)
    b.add_capacitor("C", "out", "gnd", 10e-6)
    b.add_resistor("R", "out", "gnd", 100.0)
    ext = p.simulate(b, device_path="interface", **KW)
    assert np.array_equal(np.asarray(builtin.times), np.asarray(ext.times))
    assert _err(builtin, ext) <= TWIN_TOL


def test_sympy_saturable_inductor_equals_its_hand_written_twin_on_pwl(libs):  # noqa: F811
    dev, _ = _compile("satind")
    assert dev.smooth
    twin = p.simulate(_satind(libs, True), device_path="interface", **KW)
    builtin = p.simulate(_satind(libs, False), device_path="interface", **KW)
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("V", "in", "gnd", 0.0, 10.0, 1e3)
    b.add_resistor("R", "in", "a", 0.5)
    dev.add_to(b, "L", "a", "gnd", **SAT)
    gen = p.simulate(b, device_path="interface", **KW)
    assert _err(twin, gen) <= TWIN_TOL
    assert _err(builtin, gen) <= TWIN_TOL


def test_state_update_and_outputs_equal_the_hand_written_meter(libs):  # noqa: F811
    """A state (energy, rectangle rule over the accepted steps) and named
    outputs, against contract_models.c's metered_resistor."""
    dev, _ = _compile("meter")

    def run(gen):
        b = _satind(libs, True)
        if gen:
            dev.add_to(b, "M", "in", "gnd", record=True, G=1e-3)
        else:
            b.add_external_device("M", ["in", "gnd"], libs["contract"], "metered_resistor",
                                  {"G": 1e-3}, record=True)
        return p.simulate(b, t_end=1e-3, dt=1e-6, device_path="interface")

    hand, gen = run(False), run(True)
    for key in ("energy", "p_law"):
        np.testing.assert_allclose(gen.device_output("M", key), hand.device_output("M", key),
                                   rtol=1e-12, atol=1e-18)
    assert gen.device_output("M", "energy")[-1] > 0


# ---- 4.2: the smoothness lint ----------------------------------------------

def _kinked(expr_of_u, **kw):
    a, k = xdev.pins("a", "k")
    G = xdev.params(G=(1e-3, "S", 1e-9, 1.0))
    i = G * expr_of_u(a - k) + 1e-9 * (a - k)
    return xdev.compile("kinked", [a, k], [i, -i], **kw)


@pytest.mark.parametrize("law, frag", [
    (lambda u: sp.Abs(u), "Abs("),
    (lambda u: sp.sign(u) * u ** 2, "sign("),
    (lambda u: sp.Max(u, 0), "Max("),
    (lambda u: sp.Min(u, 0), "Min("),
    (lambda u: sp.Heaviside(u) * u, "Heaviside("),
    (lambda u: sp.floor(u), "floor("),
    (lambda u: sp.Piecewise((u, u > 0), (u / 2, True)), "Piecewise("),
])
def test_lint_refuses_a_kink_with_its_subexpression(law, frag):
    with pytest.raises(ValueError, match="not C1") as e:
        _kinked(law)
    assert frag in str(e.value) and "allow_nonsmooth" in str(e.value)


def test_allow_nonsmooth_drops_the_claim_and_registers_with_one_sided_slopes():
    # two kinks at different points (at ONE point, two kinks may take opposite
    # sides at the tie: SymPy orders Max's arguments canonically)
    dev = _kinked(lambda u: sp.Max(u, 0) + sp.Abs(u - 0.5) / 4, allow_nonsmooth=True)
    assert not dev.smooth and "Max" not in dev.source and "fabs" not in dev.source
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("V", "in", "gnd", 0.0, 5.0, 1e3)
    b.add_resistor("R", "in", "a", 100.0)
    dev.add_to(b, "X", "a", "gnd")          # the registration checks pass
    res = p.simulate(b, t_end=1e-3, dt=1e-6, device_path="interface")
    v = np.asarray(res.v("a"))
    i_x = 1e-3 * (np.maximum(v, 0) + np.abs(v - 0.5) / 4) + 1e-9 * v
    vin = 5.0 * np.sin(2 * np.pi * 1e3 * np.asarray(res.times))
    np.testing.assert_allclose(((vin - v) / 100.0)[1:], i_x[1:], rtol=0, atol=1e-9)  # t=0: initial guess
    for at, sides in ((0.0, (1e-3 * 0.75, -1e-3 * 0.25)), (0.5, (1e-3 * 1.25, 1e-3 * 0.75))):
        J = dev.eval([at, 0.0])[2][0, 0] - 1e-9     # AT a kink: one side's slope
        assert min(abs(J - s_) for s_ in sides) <= 1e-15, (at, J)


def test_lint_accepts_a_proved_c1_piecewise_and_parameter_only_kinks():
    # a parabola joined to its tangent at u = 1: C1, not C2
    dev = _kinked(lambda u: sp.Piecewise((2 * u - 1, u > 1), (u ** 2, True)))
    assert dev.smooth
    a, k = xdev.pins("a", "k")
    G = xdev.params(G=(1e-3, "S", -1.0, 1.0))
    i = sp.Abs(G) * (a - k)                  # |G| is constant in the unknowns
    assert xdev.compile("abs_param", [a, k], [i, -i]).smooth
    # a Piecewise whose value matches but whose slope does not is a kink
    with pytest.raises(ValueError, match="not C1"):
        _kinked(lambda u: sp.Piecewise((2 * u - 1, u > 1), (u, True)))


# ---- 4.4: the monotonicity scan --------------------------------------------

def test_scan_monotone_finds_the_langevin_cancellation_near_zero():
    """coth(u) - 1/u without its series (the project's 1 pA lesson): the
    registration box sees a monotone law; the scan finds it decreasing
    where the two 1/u-sized terms cancel, and the NaN at u = 0."""
    a, k = xdev.pins("a", "k")
    G = xdev.params(G=(1.0, "S", 1e-9, 1e3))
    u = a - k
    i = G * (sp.coth(u) - 1 / u)
    dev = xdev.compile("langevin_raw", [a, k], [i, -i])
    scan = xdev.scan_monotone(dev, "a", -1.0, 1.0)
    assert not scan.monotone
    assert scan.decreasing, "the cancellation must show as a decrease"
    assert all(max(abs(lo), abs(hi)) < 1e-3 for lo, hi in scan.decreasing), scan.decreasing
    assert 0.0 in scan.nonfinite
    # the same box away from the cancellation is monotone
    assert xdev.scan_monotone(dev, "a", 0.01, 1.0).monotone
    # and a smooth honest law is monotone everywhere, near zero included
    t = G * sp.tanh(u)
    assert xdev.scan_monotone(xdev.compile("tanh_law", [a, k], [t, -t]), "a", -1, 1).monotone


def test_scan_monotone_reads_a_library_model_without_sympy(libs):  # noqa: F811
    dev = xdev.load(libs["contract"], "liar_nonmonotone")
    assert not xdev.scan_monotone(dev, "a", -1.0, 1.0, params={"G": 1.0}).monotone
    assert xdev.scan_monotone(xdev.load(libs["contract"], "good_resistor"), 0, -1, 1).monotone


# ---- 1.6: the free function --------------------------------------------------

GOOD_C = r"""
#define PULSIM_XDEV_DECLARE_ENTRY
#include "pulsim/xdev/pulsim_xdev.h"
static const char *const PINS[2] = {"a", "b"};
static const pulsim_xdev_field FG[1] = {{"G", "S", 1e-3, 1e-9, 1e3}};
static const uint32_t FR[4] = {0, 0, 1, 1}, FC[4] = {0, 1, 0, 1};
static const double LO[2] = {-1, -1}, HI[2] = {1, 1};
static int32_t PULSIM_XDEV_CALL ev(const pulsim_xdev_ctx *c, const double *x, double *f,
                                   double *q, double *df, double *dq) {
    const double g = c->p[0], i = g * (x[0] - x[1]);
    (void)dq;
    f[0] = i; f[1] = -i; q[0] = 0; q[1] = 0;
    df[0] = g; df[1] = -g; df[2] = -g; df[3] = g;
    return 0;
}
static const pulsim_xdev_descriptor D = {PULSIM_XDEV_ABI_VERSION,
    (uint32_t)sizeof(pulsim_xdev_descriptor), "inline_resistor", 2, 0, 1, 0, 4, 0, 0,
    PULSIM_XDEV_CLAIM_SMOOTH, PINS, FG, FR, FC, NULL, NULL, LO, HI, NULL, ev, NULL, NULL,
    NULL, NULL, NULL, {NULL, NULL, NULL, NULL}, 0, 0, NULL, NULL};
PULSIM_XDEV_EXTERN_C PULSIM_XDEV_EXPORT const pulsim_xdev_descriptor *PULSIM_XDEV_CALL
pulsim_xdev_get(uint32_t index) { return index == 0 ? &D : NULL; }
"""


def test_add_external_device_from_code_and_from_lib(libs):  # noqa: F811
    def run(**kw):
        b = p.CircuitBuilder()
        b.add_voltage_source("V", "in", "gnd", 2.0)
        b.add_resistor("R", "in", "a", 1000.0)
        assert xdev.add_external_device(b, "X", ["a", "gnd"], params={"G": 1e-3}, **kw) is b
        return p.simulate(b, t_end=1e-5, dt=1e-6, device_path="interface")

    np.testing.assert_allclose(run(code=GOOD_C).v("a")[-1], 1.0, rtol=1e-12)
    assert p.add_external_device is xdev.add_external_device
    lib = str(xdev._compile_c(GOOD_C, "c", who="t"))
    np.testing.assert_allclose(run(lib=lib).v("a")[-1], 1.0, rtol=1e-12)
    np.testing.assert_allclose(run(lib=libs["contract"], model="good_resistor").v("a")[-1],
                               1.0, rtol=1e-12)
    with pytest.raises(ValueError, match="models"):
        run(lib=libs["contract"])            # several models: name one
    with pytest.raises(ValueError, match="exactly one"):
        xdev.add_external_device(p.CircuitBuilder(), "X", ["a", "gnd"])


# ---- trust boundary ----------------------------------------------------------

@pytest.mark.parametrize("bad", ['a"; system("x', "int", "near", "_x", "a b", "", "é",
                                 "x" * 64])
def test_names_reaching_the_c_are_strict_identifiers(bad):
    with pytest.raises(ValueError, match="refused"):
        xdev.pins(bad)
    with pytest.raises(ValueError, match="refused"):
        xdev.params(**{bad: 1.0})
    a, k = xdev.pins("a", "k")
    with pytest.raises(ValueError, match="refused"):
        xdev.compile(bad, [a, k], [a - k, k - a])


def test_units_and_foreign_symbols_are_refused():
    with pytest.raises(ValueError, match="unit"):
        xdev.params(G=(1.0, 'S"'))
    with pytest.raises(ValueError, match="unit"):
        xdev.params(G=(1.0, "S??="))
    a, k = xdev.pins("a", "k")
    z = sp.Symbol("z")
    with pytest.raises(ValueError, match="not declared"):
        xdev.compile("foreign", [a, k], [z * (a - k), -z * (a - k)])
    with pytest.raises(ValueError, match="not declared"):   # a pin not listed
        xdev.compile("unlisted", [a], [a - k])
    with pytest.raises(ValueError, match="update"):
        xdev.compile("h_in_law", [a, k], [xdev.h * a, -xdev.h * a])


def test_missing_sympy_names_the_extra(monkeypatch):
    monkeypatch.setattr(xdev, "_NS", {})
    monkeypatch.setitem(sys.modules, "sympy", None)
    with pytest.raises(ImportError, match=r"pip install pulsim\[codegen\]"):
        xdev.pins("a")


def test_missing_compiler_is_a_clear_error(monkeypatch):
    if sp is None:
        pytest.skip("SymPy is not installed")
    monkeypatch.setattr(c_block, "_find_compiler", lambda lang: None)
    a, k = xdev.pins("a", "k")
    with pytest.raises(RuntimeError, match="no C compiler"):
        xdev.compile("nocc", [a, k], [a - k + 0.5 * (a - k) ** 3, k - a - 0.5 * (a - k) ** 3])


def test_pyproject_declares_the_codegen_extra():
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    text = open(os.path.join(root, "pyproject.toml"), encoding="utf-8").read()
    assert "sympy" in text.split("codegen = [")[1].split("]")[0]
    assert "sympy" in text.split("dev = [")[1].split("]")[0]   # CI installs .[dev]
