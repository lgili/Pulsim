"""Flux-map PMSM (add-mna-native-machines group 4).

`FluxMapDQ.from_tables` fits ONE co-energy W'(i_d, i_q) (a C2 bicubic
spline) to both flux tables, so lambda = grad W' is reciprocal by
construction and the incremental inductances (the Hessian) are
continuous: what the Newton stamp needs. Truths used here:

* closed form: a map sampled from L_d, L_q, psi_pm reproduces the
  constant-L machine (4.3);
* reciprocity and the analytic Hessian against central differences of
  the kernel's own lambda (4.4), and refusals by name;
* an independent SciPy integration of the ANALYTIC saturating,
  cross-coupled law (`_machine_reference/flux_map_reference.py`, pinned
  below) for a locked-rotor step (4.5) on pwl and trbdf2;
* energy balance of a saturated free-acceleration start.

Every map here is SYNTHETIC (declared test parameters, no real motor).
"""

import math
import warnings

import numpy as np
import pytest

import pulsim as p

RS, PP = 0.5, 4
# analytic co-energy of the reference (flux_map_reference.py)
PSI, L_Q2, L_S, I_S = 0.05, 2e-3, 1e-3, 10.0

# (t, i_d, i_q) from flux_map_reference.py: Radau and DOP853 at rtol 1e-12
REF = (
    (0.0005, 2.6019533947331928, 1.228499265582921),
    (0.001, 5.04392493486622, 2.4734454821586094),
    (0.002, 9.322734097233159, 4.864251144306278),
    (0.004, 12.452866390246943, 8.64217614319871),
    (0.008, 12.331868843493591, 12.821421206825661),
    (0.016, 12.057665354479338, 15.449350997594008),
)
T_RISE = 50e-6   # the reference's voltage ramp


def _sat_flux(i_d, i_q):
    """lambda of W' = psi·i_d + ½·L_q2·i_q² + L_s·I_s²·log cosh(|i|/I_s)."""
    r = np.hypot(i_d, i_q)
    x = r / I_S
    f = np.where(x < 1e-3, L_S * (1 - x * x / 3),
                 L_S * np.tanh(x) / np.where(x < 1e-3, 1.0, x))
    return PSI + f * i_d, L_Q2 * i_q + f * i_q


def _grid(lo, hi, step):
    return np.arange(lo, hi + step / 2, step)


def _sat_map(lo=-6.0, hi=26.0, step=1.0, scale_q=1.0):
    g = _grid(lo, hi, step)
    D, Q = np.meshgrid(g, g, indexing="ij")
    ld, lq = _sat_flux(D, Q)
    return p.FluxMapDQ.from_tables(g, g, ld, scale_q * lq,
                                   source="synthetic: log-cosh test map")


def _linear_map(L_d=1e-3, L_q=3e-3, psi=0.05):
    g = _grid(-30.0, 30.0, 10.0)
    D, Q = np.meshgrid(g, g, indexing="ij")
    return p.FluxMapDQ.from_tables(g, g, L_d * D + psi, L_q * Q,
                                   source="synthetic: linear")


def _sim(b, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return p.simulate(b, **kw)


# ---------------------------------------------------------------
# 4.3: the linear map is the constant-L machine
# ---------------------------------------------------------------

def _sine_fed(add, *, v_peak=12.0, f_e=30.0):
    b = p.CircuitBuilder()
    for k, node in enumerate(("ua", "ub", "uc")):
        b.add_sine_voltage_source(
            f"Vs_{'abc'[k]}", node, "gnd", v_dc=0.0, v_amplitude=v_peak,
            frequency=f_e, phase=-2.0 * math.pi / 3.0 * k)
    add(b)
    return b


@pytest.mark.parametrize("engine", ["pwl", "trbdf2"])
def test_linear_map_reproduces_the_constant_inductance_machine(engine):
    fm = _linear_map()
    mech = dict(pole_pairs=PP, J=1e-3, B=1e-4)
    ref = _sim(_sine_fed(lambda b: b.add_pmsm_mna(
        "M1", "ua", "ub", "uc", "nn", "w", "th", R_s=RS, L_d=1e-3,
        L_q=3e-3, psi_pm=0.05, **mech)), t_end=0.05, dt=2e-5, engine=engine)
    got = _sim(_sine_fed(lambda b: b.add_pmsm_flux_map(
        "M1", "ua", "ub", "uc", "nn", "w", "th", R_s=RS, flux_map=fm,
        **mech)), t_end=0.05, dt=2e-5, engine=engine)
    assert fm.fit_residual < 1e-12
    assert np.asarray(got.times) == pytest.approx(np.asarray(ref.times), rel=1e-9)
    w_ref = np.asarray(ref.v("w"))
    assert np.abs(w_ref).max() > 10.0   # control: the rotor really turns
    sigs = [lambda r: r.v("w"), lambda r: r.v("th"),
            lambda r: r.i("M1_a"), lambda r: r.i("M1_b")]
    if engine == "pwl":   # the machine traces are recorded on pwl
        sigs.append(lambda r: r.signal("M1.T_em"))
    for sig in sigs:
        a, b_ = np.asarray(sig(got)), np.asarray(sig(ref))
        assert np.abs(a - b_).max() <= 1e-9 * np.abs(b_).max()


# ---------------------------------------------------------------
# 4.2 / 4.4: the kernel law — reciprocity, Hessian, monotone to 1 pA
# ---------------------------------------------------------------

def test_reciprocity_and_hessian_at_random_points():
    fm = _sat_map()
    rng = np.random.default_rng(7)
    d = 1e-4
    worst_rec = worst_h = 0.0
    for i_d, i_q in rng.uniform(-6.0, 26.0, size=(100, 2)):
        _, _, _, hdd, hdq, hqq = fm.evaluate(i_d, i_q)
        dld_dq = (fm.evaluate(i_d, i_q + d)[1] - fm.evaluate(i_d, i_q - d)[1]) / (2 * d)
        dlq_dd = (fm.evaluate(i_d + d, i_q)[2] - fm.evaluate(i_d - d, i_q)[2]) / (2 * d)
        dld_dd = (fm.evaluate(i_d + d, i_q)[1] - fm.evaluate(i_d - d, i_q)[1]) / (2 * d)
        dlq_dq = (fm.evaluate(i_d, i_q + d)[2] - fm.evaluate(i_d, i_q - d)[2]) / (2 * d)
        worst_rec = max(worst_rec, abs(dld_dq - dlq_dd))
        worst_h = max(worst_h, abs(dld_dd - hdd), abs(dld_dq - hdq),
                      abs(dlq_dq - hqq))
    assert worst_rec <= 1e-9 * L_S
    assert worst_h <= 1e-7 * L_S
    # the map really is cross-coupled and saturating (control)
    assert abs(fm.evaluate(12.0, 16.0)[4]) > 0.05 * L_S
    assert fm.evaluate(20.0, 0.0)[3] < 0.2 * fm.evaluate(0.0, 0.0)[3]


@pytest.mark.parametrize("axis", [0, 1])
def test_flux_is_monotone_at_picoampere_spacing(axis):
    fm = _sat_map()
    for x0 in (-5.0, 0.0, 3.3, 12.0, 25.9, 28.5):   # 28.5 A: in the band past the grid
        pt = [x0, 7.0] if axis == 0 else [7.0, x0]
        lam = []
        for k in range(21):
            q = list(pt)
            q[axis] += k * 1e-12
            lam.append(fm.evaluate(*q)[1 + axis])
        assert np.all(np.diff(lam) >= 0.0), (x0, np.diff(lam))
        h = fm.evaluate(*pt)[3 if axis == 0 else 5]
        step = [1e-12, 0.0] if axis == 0 else [0.0, 1e-12]
        inc = fm.increment(pt[0], pt[1], *step)[axis]
        # the increment form the flux row uses: H·δ to round-off in δ
        assert inc == pytest.approx(h * 1e-12, rel=1e-9)
    # across a knot line, in the increment form the flux row uses: 1 pA
    # steps from 10 pA below a knot stay positive and equal H·δ
    for knot in (0.0, 12.0, 26.0):
        for k in range(21):
            x = knot - 10e-12 + k * 1e-12
            pt = [x, 7.0] if axis == 0 else [7.0, x]
            step = [1e-12, 0.0] if axis == 0 else [0.0, 1e-12]
            inc = fm.increment(pt[0], pt[1], *step)[axis]
            h = fm.evaluate(*pt)[3 if axis == 0 else 5]
            assert inc > 0.0 and inc == pytest.approx(h * 1e-12, rel=1e-9)


def test_increment_equals_the_gradient_difference_across_knots():
    fm = _sat_map()
    rng = np.random.default_rng(3)
    for _ in range(50):
        x0 = rng.uniform(-8.0, 28.0, 2)
        dx = rng.uniform(-3.0, 3.0, 2)
        g1, g0 = fm.evaluate(*(x0 + dx)), fm.evaluate(*x0)
        inc = fm.increment(x0[0], x0[1], dx[0], dx[1])
        # exact in exact arithmetic; the direct difference carries the
        # cells' join round-off (W' ~ 1 J differenced into lambda ~ 0.06 Wb)
        assert inc[0] == pytest.approx(g1[1] - g0[1], abs=1e-13)
        assert inc[1] == pytest.approx(g1[2] - g0[2], abs=1e-13)


def test_a_non_reciprocal_table_is_refused_naming_the_defect():
    # lambda_q scaled by 1.1: the cross terms disagree by 10 % (a field
    # that is not a gradient). The tables' reciprocity defect is 9.1 % of
    # the cross inductance (the reciprocal table: 0.19 %, central-difference
    # truncation), refused at the default tol of 2 %. The fit residual of
    # the same table is only 0.16 % of the flux range, which the PM and
    # self terms dominate: it alone would not catch it (D22).
    g = _grid(-6.0, 26.0, 1.0)
    D, Q = np.meshgrid(g, g, indexing="ij")
    ld, lq = _sat_flux(D, Q)
    with pytest.raises(ValueError, match=r"log-cosh 1.1.*reciprocity defect 0\.09"):
        p.FluxMapDQ.from_tables(g, g, ld, 1.1 * lq,
                                source="synthetic: log-cosh 1.1")
    assert _sat_map().fit_residual < 1e-6   # the reciprocal table: round-off


def test_a_poor_fit_is_refused_naming_the_residual():
    g = _grid(-6.0, 26.0, 2.0)
    D, Q = np.meshgrid(g, g, indexing="ij")
    ld, lq = _sat_flux(D, Q)
    with pytest.raises(ValueError, match=r"stiff.*fit residual"):
        p.FluxMapDQ.from_tables(g, g, ld, lq, smooth=1e3,
                                source="synthetic: stiff")


def test_a_non_positive_definite_table_is_refused():
    g = _grid(-10.0, 10.0, 2.0)
    D, Q = np.meshgrid(g, g, indexing="ij")
    with pytest.raises(ValueError, match="not positive definite"):
        p.FluxMapDQ.from_tables(g, g, -1e-3 * D, 1e-3 * Q, source="synthetic: bad")


def test_a_too_coarse_grid_is_refused():
    g = np.array([-1.0, 1.0])
    D, Q = np.meshgrid(g, g, indexing="ij")
    with pytest.raises(ValueError, match="does not determine"):
        p.FluxMapDQ.from_tables(g, g, 1e-3 * D, 1e-3 * Q, source="synthetic: coarse")


def test_builder_refuses_inconsistent_origin_values_and_harmonics():
    fm = _sat_map()
    b = p.CircuitBuilder()
    with pytest.raises(ValueError, match=r"M1.*values at i = 0"):
        b.add_pmsm_mna("M1", "a", "b", "c", "n", "w", "th", R_s=RS,
                       L_d=1e-3, L_q=3e-3, psi_pm=0.05, pole_pairs=PP, J=1.0,
                       flux_map=fm)
    with pytest.raises(ValueError, match=r"M2.*harmonics"):
        b.add_pmsm_mna("M2", "a", "b", "c", "n", "w", "th", R_s=RS,
                       L_d=1e-3, L_q=3e-3, psi_pm=0.05, pole_pairs=PP, J=1.0,
                       harmonics=[(1, 0.05, 0.0)], flux_map=fm)


# ---------------------------------------------------------------
# 4.5: saturated locked-rotor step against SciPy
# ---------------------------------------------------------------

def _locked(fm):
    return _locked_scaled(fm, 1.0)


def _locked_scaled(fm, k):
    v_d, v_q = 6.0 * k, 8.0 * k   # theta_e = 0: abc from the inverse Clarke
    s3 = math.sqrt(3.0) / 2.0
    va = (v_d, -v_d / 2 + s3 * v_q, -v_d / 2 - s3 * v_q)
    b = p.CircuitBuilder()
    for k, node in enumerate(("ua", "ub", "uc")):   # ramp 0 -> V, as the reference
        b.add_pulse_voltage_source(f"V{'abc'[k]}", node, "gnd", 0.0, va[k],
                                   0.0, 1.0, rise_time=T_RISE)
    b.add_pmsm_flux_map("M1", "ua", "ub", "uc", "nn", "w", "th", R_s=RS,
                        flux_map=fm, pole_pairs=PP, J=1.0)
    b.add_voltage_source("Vw", "w", "gnd", 0.0)   # rotor locked
    return b


def _dq_at(res, t):
    ts = np.asarray(res.times)
    ia, ib, ic = (np.asarray(res.i(f"M1_{x}")) for x in "abc")
    i_d, i_q = ia, (ib - ic) / math.sqrt(3.0)
    return np.interp(t, ts, i_d), np.interp(t, ts, i_q)


@pytest.mark.parametrize("engine,dt,tol", [("pwl", 1e-6, 1e-4),
                                           ("trbdf2", 1e-6, 1e-4)])
def test_saturated_locked_rotor_step_matches_scipy(engine, dt, tol):
    res = _sim(_locked(_sat_map()), t_end=REF[-1][0], dt=dt, engine=engine)
    worst = 0.0
    for t, d_ref, q_ref in REF:
        i_d, i_q = _dq_at(res, t)
        worst = max(worst, abs(i_d - d_ref), abs(i_q - q_ref))
    assert worst <= tol * 20.0, worst   # 20 A = |i| at steady state
    # control: the unsaturated machine (the map's inductances at i = 0)
    # misses the reference by far more than the tolerance
    lin = _sim(_locked(_linear_map(L_d=L_S, L_q=L_Q2 + L_S, psi=PSI)),
               t_end=REF[-1][0], dt=dt, engine=engine)
    miss = max(max(abs(a - d), abs(b_ - q)) for (t, d, q) in REF
               for a, b_ in [_dq_at(lin, t)])
    assert miss > 100 * tol * 20.0, miss


# ---------------------------------------------------------------
# Energy balance of a saturated start
# ---------------------------------------------------------------

def test_energy_balance_of_a_saturated_free_acceleration():
    fm = _sat_map(lo=-30.0, hi=30.0, step=1.5)
    J, B = 2e-4, 1e-4
    b = _sine_fed(lambda b: b.add_pmsm_flux_map(
        "M1", "ua", "ub", "uc", "nn", "w", "th", R_s=RS, flux_map=fm,
        pole_pairs=PP, J=J, B=B), v_peak=12.0, f_e=20.0)
    res = _sim(b, t_end=0.04, dt=2e-6, engine="pwl")
    t = np.asarray(res.times)
    vn = np.asarray(res.v("nn"))
    cur = [np.asarray(res.i(f"M1_{x}")) for x in "abc"]
    v = [np.asarray(res.v(n)) - vn for n in ("ua", "ub", "uc")]
    w = np.asarray(res.v("w"))
    i_d = np.asarray(res.signal("M1.i_d"))
    i_q = np.asarray(res.signal("M1.i_q"))
    p_in = sum(vk * ik for vk, ik in zip(v, cur))
    p_cu = RS * sum(ik * ik for ik in cur)
    ev = np.array([fm.evaluate(a, c) for a, c in zip(i_d, i_q)])
    w_field = 1.5 * (ev[:, 1] * i_d + ev[:, 2] * i_q - ev[:, 0])
    e_in = np.trapezoid(p_in, t)
    e_out = (np.trapezoid(p_cu, t) + (w_field[-1] - w_field[0])
             + 0.5 * J * (w[-1] ** 2 - w[0] ** 2) + B * np.trapezoid(w * w, t))
    assert np.hypot(i_d, i_q).max() > 2 * I_S   # control: deep saturation
    assert abs(w).max() > 10.0                  # control: it accelerates
    # measured 1.4e-9 (5.5e-9 at 4 us: second order)
    assert abs(e_in - e_out) <= 2e-8 * abs(e_in), (e_in, e_out)


# ---------------------------------------------------------------
# DC operating point with held mechanics
# ---------------------------------------------------------------

def test_dc_point_at_rest_is_v_over_r_and_spinning_is_refused():
    v = np.array([10.0, -4.0, -6.0])

    def build(omega0):
        b = p.CircuitBuilder()
        for k, node in enumerate(("ua", "ub", "uc")):
            b.add_voltage_source(f"V{'abc'[k]}", node, "gnd", float(v[k]))
        b.add_pmsm_flux_map("M1", "ua", "ub", "uc", "nn", "w", "th", R_s=RS,
                            flux_map=_sat_map(lo=-30.0, hi=30.0, step=1.5),
                            pole_pairs=PP, J=1.0, omega0=omega0, theta0=0.3)
        return b

    res = _sim(build(0.0), t_end=2e-5, dt=1e-6, engine="pwl",
               start_from_dc_op=True)
    i0 = np.array([float(np.asarray(res.i(f"M1_{x}"))[0]) for x in "abc"])
    assert i0 == pytest.approx((v - v.mean()) / RS, abs=1e-9)
    with pytest.raises(Exception, match=r"M1_a.*omega0 = 0"):
        _sim(build(30.0), t_end=2e-5, dt=1e-6, engine="pwl",
             start_from_dc_op=True)


def test_an_accepted_state_off_the_map_is_refused_by_name():
    # steady i = (24, 32) A: past the i_q grid (26 A) and its band (3.2 A)
    b = _locked(_sat_map())
    big = _locked_scaled(_sat_map(), 2.0)
    with pytest.raises(Exception, match=r"M1_a.*left flux map 'synthetic: log-cosh"):
        _sim(big, t_end=REF[-1][0], dt=2e-6, engine="pwl")
    _sim(b, t_end=2e-4, dt=2e-6, engine="pwl")   # control: on the map it runs


def test_dsed_refuses_the_flux_map_machine():
    with pytest.raises(Exception, match="PmsmMna"):
        _sim(_locked(_sat_map()), t_end=1e-4, dt=1e-6, engine="dsed")
