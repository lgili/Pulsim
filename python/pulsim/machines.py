"""MNA-native electrical machines built from existing kernel elements.

``CircuitBuilder.add_dc_machine`` (openspec ``add-mna-native-machines``,
step 1) puts a DC machine INSIDE the circuit matrix:

    a_pos ─ R_a ─ L_a ─ e ─[ secondary ]─ a_neg       v_e − v_a_neg = K·ω
                           ideal transformer, n = K
    omega ─[ primary ]─ gnd                           torque K·i into omega
    omega ─ C = J ─ gnd,  omega ─ R = 1/B ─ gnd,  I = T_load out of omega

The speed ω [rad/s] is the voltage of ``omega_node`` and torque [N·m]
is current, so ``J·dω/dt = K·i − B·ω − T_load`` is the node's KCL. No
quantity is taken from the previous step: the machine is second order
on ``pwl``, runs unchanged on ``trbdf2`` and appears in the DC
operating point (inductor short, inertia open, so
``i = T_load/K`` and ``ω = (V − R_a·i)/K`` when ``B = 0``). Power is
conserved by construction: ``K·ω·i`` leaves the armature and enters
the shaft.

It replaces the forward-Euler observer of ``pulsim.add_dc_motor``
(first order: 13.6 % speed error at dt = 100 µs on the
``add-mna-native-machines`` probe, against 0.48 % here).

Sub-elements (only those with a non-zero value are created):
``{name}.ra`` (resistor), ``{name}.la`` (inductor, its current is the
armature current), ``{name}.em`` (ideal transformer), ``{name}.j``
(capacitor), ``{name}.b`` (resistor ``1/B``), ``{name}.tl`` (current
source).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ._pulsim import CircuitBuilder

__all__ = ["flux_map_from_tables", "install", "pm_harmonics_from_waveform",
           "trapezoid_harmonics"]


def _check(name: str, label: str, value: float, *, positive: bool) -> float:
    v = float(value)
    ok = math.isfinite(v) and (v > 0.0 if positive else v >= 0.0)
    if not ok:
        rule = "> 0" if positive else ">= 0"
        raise ValueError(
            f"add_dc_machine({name!r}): {label} must be finite and "
            f"{rule} (got {value!r}).")
    return v


def add_dc_machine(self, name: str, a_pos: str, a_neg: str,
                   omega_node: str, *, R_a: float, L_a: float,
                   K: Optional[float] = None, J: float, B: float = 0.0,
                   T_load: float = 0.0, theta_node: Optional[str] = None,
                   i0: Optional[float] = None, omega0: float = 0.0,
                   Ke: Optional[float] = None, Kt: Optional[float] = None
                   ) -> "CircuitBuilder":
    """Add a permanent-magnet DC machine solved inside the matrix.

    Parameters [SI]: ``R_a`` [Ω, ≥ 0], ``L_a`` [H, > 0], ``K``
    [V·s/rad = N·m/A, > 0], ``J`` [kg·m², > 0], ``B`` [N·m·s/rad,
    ≥ 0], ``T_load`` [N·m, constant, opposing positive speed]. ``i0``
    is the initial armature current [A] and ``omega0`` the initial
    speed [rad/s]. ``omega_node`` is created if new; read the speed
    with ``result.v(omega_node)`` and the armature current with
    ``result.i(f"{name}.la")``. A speed-dependent load is any element
    you connect to ``omega_node`` (torque = current out of it).

    ``Ke`` / ``Kt`` are the legacy ``add_dc_motor`` spellings: in SI
    units they are the same constant, so a call with ``Kt != Ke`` is
    refused (the coupling would not conserve power).

    ``theta_node`` (optional) gets the shaft angle [rad]: a unit VCCS
    ``{name}.th`` from the speed node into a 1 F capacitor ``{name}.ct``,
    so ``v(theta_node) = ∫ω dt`` (starting at 0).
    """
    given = {k: v for k, v in (("K", K), ("Ke", Ke), ("Kt", Kt))
             if v is not None}
    if not given:
        raise ValueError(f"add_dc_machine({name!r}): K is required "
                         "(V·s/rad, equal to the torque constant N·m/A).")
    values = {float(v) for v in given.values()}
    if len(values) > 1:
        raise ValueError(
            f"add_dc_machine({name!r}): got different constants "
            f"{given}. In SI units the back-EMF constant Ke [V·s/rad] "
            "and the torque constant Kt [N·m/A] are the same number "
            "(power K·ω·i leaves the armature and enters the shaft); "
            "a machine with Kt != Ke would create or destroy energy. "
            "Pass one K; convert datasheet units (e.g. V/krpm: "
            "K = Ke_V_per_krpm · 60 / (2π·1000)).")
    k = _check(name, "K", values.pop(), positive=True)
    r_a = _check(name, "R_a", R_a, positive=False)
    l_a = _check(name, "L_a", L_a, positive=True)
    j = _check(name, "J", J, positive=True)
    b = _check(name, "B", B, positive=False)
    t_load = float(T_load)
    if not math.isfinite(t_load):
        raise ValueError(f"add_dc_machine({name!r}): T_load must be "
                         f"finite (got {T_load!r}).")

    l_from = f"{name}.ra_la" if r_a > 0.0 else a_pos
    emf = f"{name}.emf"
    if r_a > 0.0:
        self.add_resistor(f"{name}.ra", a_pos, l_from, r_a)
    self.add_inductor(f"{name}.la", l_from, emf, l_a, i0)
    # Secondary = armature side (v_s = K·v_p), primary = speed node:
    # the reflected primary current K·i is the torque into omega.
    self.add_ideal_transformer(f"{name}.em", omega_node, "gnd",
                               emf, a_neg, k)
    self.add_capacitor(f"{name}.j", omega_node, "gnd", j, float(omega0))
    if b > 0.0:
        self.add_resistor(f"{name}.b", omega_node, "gnd", 1.0 / b)
    if t_load != 0.0:
        # A current source pushes I out of n_pos: n_pos = gnd draws
        # T_load out of the speed node.
        self.add_current_source(f"{name}.tl", "gnd", omega_node, t_load)
    if theta_node is not None:
        # θ = ∫ω dt: a unit VCCS from the speed into a 1 F capacitor.
        self.add_vccs(f"{name}.th", omega_node, "gnd", theta_node, "gnd",
                      1.0)
        self.add_capacitor(f"{name}.ct", theta_node, "gnd", 1.0, 0.0)
    return self


def trapezoid_harmonics(k_e: float, pole_pairs: int, *,
                        flat_top_deg: float = 120.0,
                        n_harmonics: int = 25) -> list:
    """PM-flux harmonic series ``[(h, psi_h, phi_h), ...]`` of a
    trapezoidal back-EMF (``add-mna-native-machines`` design, BLDC).

    ``k_e`` [V·s/rad] is the phase back-EMF on the flat top per unit
    MECHANICAL speed; the flat top is ``flat_top_deg`` electrical degrees
    wide and the ramps are straight. With ramp half-width
    ``gamma = (180° − flat_top)/2``, odd ``h`` up to ``n_harmonics``::

        psi_h = (k_e/pp)·4·sin(h·gamma)/(pi·gamma·h³),   phi_h = 0

    so ``lambda_pm,a(theta_e) = sum psi_h·cos(h·theta_e)`` and the
    back-EMF ``-pp·omega·dlambda/dtheta_e`` is the truncated trapezoid
    (RMS deviation 0.40 % of the flat top at 25 harmonics, 120°).
    """
    k_e, pp = float(k_e), int(pole_pairs)
    if not (math.isfinite(k_e) and k_e > 0.0) or pp < 1:
        raise ValueError("trapezoid_harmonics: k_e must be > 0 and "
                         "pole_pairs a positive integer")
    if not 0.0 <= flat_top_deg < 180.0:
        raise ValueError("trapezoid_harmonics: flat_top_deg must be in "
                         "[0, 180)")
    gamma = math.radians(180.0 - flat_top_deg) / 2.0
    out = []
    for h in range(1, int(n_harmonics) + 1, 2):
        s = (math.sin(h * gamma) / (h * gamma)) if gamma > 0 else 1.0
        out.append((float(h), (k_e / pp) * 4.0 * s / (math.pi * h * h),
                    0.0))
    return out


def pm_harmonics_from_waveform(theta_e, emf, omega_e: float, *,
                               n_harmonics: int = 25) -> list:
    """PM-flux series ``[(h, psi_h, phi_h), ...]`` from one electrical
    period of a measured phase-a back-EMF.

    ``theta_e`` are ``N`` uniformly spaced electrical angles covering
    one period (``[0, 2π)``, endpoint excluded), ``emf`` the phase-a
    back-EMF [V] at those angles, recorded at the constant ELECTRICAL
    speed ``omega_e`` [rad/s] with the motor convention ``e = dλ/dt``.
    Returns the harmonics ``1..n_harmonics`` (the DC term is dropped:
    a constant flux has no back-EMF and no torque).
    """
    import numpy as np

    th = np.asarray(theta_e, dtype=float)
    e = np.asarray(emf, dtype=float)
    n = th.size
    if e.size != n or n < 2 * n_harmonics + 2:
        raise ValueError("pm_harmonics_from_waveform: theta_e and emf "
                         "must have the same length, >= 2·n_harmonics + 2")
    if not (math.isfinite(omega_e) and omega_e != 0.0):
        raise ValueError("pm_harmonics_from_waveform: omega_e must be "
                         "finite and non-zero")
    c = np.fft.rfft(e / omega_e) / n * 2.0           # dλ/dθ_e coefficients
    out = []
    for h in range(1, int(n_harmonics) + 1):
        ch = c[h] * np.exp(-1j * h * th[0])          # phase at theta = 0
        amp = abs(ch)
        if amp == 0.0:
            continue
        # dλ/dθ = amp·cos(hθ + β) = −h·psi·sin(hθ + φ), φ = β − π/2.
        out.append((float(h), float(amp / h),
                    float(np.angle(ch) - math.pi / 2.0)))
    return out


def add_bldc_mna(self, name: str, phase_a: str, phase_b: str, phase_c: str,
                 neutral: str, omega_node: str, theta_node: str, *,
                 R_s: float, L_s: float, k_e: float, pole_pairs: int,
                 J: float, B: float = 0.0, T_load: float = 0.0,
                 flat_top_deg: float = 120.0, n_harmonics: int = 25,
                 omega0: float = 0.0, theta0: float = 0.0
                 ) -> "CircuitBuilder":
    """Add an MNA-native BLDC: ``add_pmsm_mna`` (round rotor,
    ``L_d = L_q = L_s``) with the PM flux of a trapezoidal back-EMF
    (:func:`trapezoid_harmonics`). Back-EMF and torque come from the
    same series and its exact derivatives, so the torque is consistent
    with the back-EMF (co-energy). ``k_e`` [V·s/rad]: phase back-EMF on
    the flat top per unit mechanical speed."""
    harm = trapezoid_harmonics(k_e, pole_pairs, flat_top_deg=flat_top_deg,
                               n_harmonics=n_harmonics)
    return self.add_pmsm_mna(
        name, phase_a, phase_b, phase_c, neutral, omega_node, theta_node,
        R_s=R_s, L_d=L_s, L_q=L_s, psi_pm=harm[0][1],
        pole_pairs=pole_pairs, J=J, B=B, T_load=T_load, omega0=omega0,
        theta0=theta0, harmonics=harm)


# ---------------------------------------------------------------------------
# Flux maps (add-mna-native-machines group 4)
# ---------------------------------------------------------------------------

def _bspline_derivs(t, span: int, x: float, nbasis: int):
    """D^r B_{i,3}(x), r = 0..3, for every basis i, taking x in the knot
    span ``[t[span], t[span + 1])`` (its right limit at ``t[span]``) --
    Cox-de Boor with the derivative recursion, exact up to round-off."""
    import functools

    @functools.lru_cache(maxsize=None)
    def b(i: int, p: int, r: int) -> float:
        if p == 0:
            return 1.0 if (r == 0 and i == span) else 0.0
        out = 0.0
        d1 = t[i + p] - t[i]
        d2 = t[i + p + 1] - t[i + 1]
        if r == 0:
            if d1 > 0:
                out += (x - t[i]) / d1 * b(i, p - 1, 0)
            if d2 > 0:
                out += (t[i + p + 1] - x) / d2 * b(i + 1, p - 1, 0)
        else:
            if d1 > 0:
                out += p * b(i, p - 1, r - 1) / d1
            if d2 > 0:
                out -= p * b(i + 1, p - 1, r - 1) / d2
        return out

    return [[b(i, 3, r) for r in range(4)] for i in range(nbasis)]


def _d3_jumps(x, nb):
    """Jump of D^3 B_i across the first and last interior knot (one row
    per knot, not-a-knot end conditions)."""
    import numpy as np

    t = np.concatenate([[x[0]] * 3, x, [x[-1]] * 3])
    out = []
    for j in sorted({1, x.size - 2}):
        if 1 <= j <= x.size - 2:
            right = _bspline_derivs(t, j + 3, x[j], nb)
            left = _bspline_derivs(t, j + 2, x[j], nb)
            out.append(np.array([r[3] - l_[3] for r, l_ in zip(right, left)]))
    return out


def flux_map_from_tables(i_d, i_q, lambda_d, lambda_q, *, source: str,
                         tol: float = 0.02, smooth: float = 0.0):
    """Fit a :class:`FluxMapDQ` to flux tables ``lambda_d[a, b]``,
    ``lambda_q[a, b]`` at ``(i_d[a], i_q[b])`` [A, Wb].

    The map is the gradient of ONE co-energy ``W'(i_d, i_q)``, a C2 tensor
    cubic B-spline with knots at the table grid, fitted to both tables at
    once by least squares, so ``dlambda_d/di_q == dlambda_q/di_d`` holds by
    construction. A measured table is never exactly reciprocal. Refused by
    name (``source``):

    * a reciprocity defect above ``tol``: max |dlambda_d/di_q -
      dlambda_q/di_d| of the tables (central differences) over the cross
      inductance (floor 5 % of the self inductance) -- the machine would
      create or destroy energy over a current cycle;
    * a fit residual above ``tol`` (max |fit - table| over the grid, as a
      fraction of the larger flux range; stored as ``fit_residual``);
    * a grid too coarse to determine the co-energy (rank-deficient fit;
      pass ``smooth > 0`` to regularise it);
    * an incremental inductance matrix that is not positive definite on
      the grid or half a cell beyond it (checked by the kernel).

    ``smooth`` adds a second-difference penalty on the spline
    coefficients (weight ``smooth``, default 0 = pure least squares).
    ``source`` records the provenance (FEA file, measurement, or
    ``"synthetic: ..."``); no map ships with Pulsim.
    """
    import numpy as np

    from ._pulsim import FluxMapDQ  # type: ignore[import-not-found]

    xd = np.asarray(i_d, dtype=float)
    xq = np.asarray(i_q, dtype=float)
    ld = np.asarray(lambda_d, dtype=float)
    lq = np.asarray(lambda_q, dtype=float)
    who = f"flux map '{source}'"
    if xd.ndim != 1 or xq.ndim != 1 or xd.size < 2 or xq.size < 2:
        raise ValueError(f"{who}: i_d and i_q must be 1-D with >= 2 points")
    if ld.shape != (xd.size, xq.size) or lq.shape != ld.shape:
        raise ValueError(f"{who}: lambda_d and lambda_q must have shape "
                         f"(len(i_d), len(i_q)) = {(xd.size, xq.size)}")
    if not (np.all(np.isfinite(ld)) and np.all(np.isfinite(lq))):
        raise ValueError(f"{who}: the flux tables must be finite")
    if np.any(np.diff(xd) <= 0) or np.any(np.diff(xq) <= 0):
        raise ValueError(f"{who}: i_d and i_q must be strictly increasing")

    # Reciprocity of the TABLES themselves: dlambda_d/di_q - dlambda_q/di_d
    # by central differences on the interior, against the cross terms (floor
    # 5 % of the self terms, so an uncoupled table is not judged on noise).
    # The fit residual alone is relative to the whole flux range, which the
    # PM and self terms dominate: a 10 % cross mismatch showed as 0.16 %.
    if xd.size >= 3 and xq.size >= 3:
        cd = np.gradient(ld, xq, axis=1)[1:-1, 1:-1]
        cq = np.gradient(lq, xd, axis=0)[1:-1, 1:-1]
        self_ = max(np.abs(np.gradient(ld, xd, axis=0)).max(),
                    np.abs(np.gradient(lq, xq, axis=1)).max())
        cross = max(np.abs(cd).max(), np.abs(cq).max(), 0.05 * self_, 1e-300)
        defect = float(np.abs(cd - cq).max() / cross)
        if defect > tol:
            raise ValueError(
                f"{who}: reciprocity defect {defect:.4g} (max |dlambda_d/di_q "
                f"- dlambda_q/di_d| over the cross inductance) exceeds tol = "
                f"{tol:g}: the tables are not the gradient of one co-energy "
                f"and would create or destroy energy over a current cycle")

    def axis(x):
        t = np.concatenate([[x[0]] * 3, x, [x[-1]] * 3])
        nb = x.size + 2
        # D^r B at each grid point (the last point from the last span)
        at_pts = [_bspline_derivs(t, min(a, x.size - 2) + 3, x[a], nb)
                  for a in range(x.size)]
        # power-basis rows per cell: D^r B(x_a+) / r!
        cells = [[[d[r] / math.factorial(r) for r in range(4)]
                  for d in _bspline_derivs(t, a + 3, x[a], nb)]
                 for a in range(x.size - 1)]
        return np.array(at_pts), np.array(cells), nb

    Pd, Cd, nbd = axis(xd)   # Pd[a, i, r], Cd[a, i, m]
    Pq, Cq, nbq = axis(xq)
    n = nbd * nbq
    rows = []
    rhs = []
    for a in range(xd.size):
        for b in range(xq.size):
            rows.append(np.outer(Pd[a, :, 1], Pq[b, :, 0]).ravel())
            rhs.append(ld[a, b])
            rows.append(np.outer(Pd[a, :, 0], Pq[b, :, 1]).ravel())
            rhs.append(lq[a, b])
    n_data = len(rows)
    # W' is defined up to a constant: pin W'(i_d[0], i_q[0]) = 0.
    rows.append(np.outer(Pd[0, :, 0], Pq[0, :, 0]).ravel())
    rhs.append(0.0)
    if smooth > 0:
        w = math.sqrt(smooth)
        for di, dj in ((1, 0), (0, 1)):   # second differences along each axis
            for i in range(nbd - 2 * di):
                for j in range(nbq - 2 * dj):
                    r = np.zeros(n)
                    for k, wk in enumerate((w, -2 * w, w)):
                        r[(i + k * di) * nbq + (j + k * dj)] = wk
                    rows.append(r)
                    rhs.append(0.0)
    A = np.array(rows)
    y = np.array(rhs)
    # Least squares on the data; gradient data at the knots leave a few
    # spline modes free, which the not-a-knot conditions (W' C3 across the
    # first and last interior knot lines, exact for a cubic) fix without
    # touching the fit.
    U, S, Vt = np.linalg.svd(A, full_matrices=True)
    rank = int(np.sum(S > S[0] * 1e-11))
    c = Vt[:rank].T @ ((U[:, :rank].T @ y) / S[:rank])
    N = Vt[rank:].T
    if N.shape[1]:
        K = []
        for P_jump, other, along_d in ((_d3_jumps(xd, nbd), nbq, True),
                                       (_d3_jumps(xq, nbq), nbd, False)):
            for jump in P_jump:
                for o in range(other):
                    r = np.zeros((nbd, nbq))
                    if along_d:
                        r[:, o] = jump
                    else:
                        r[o, :] = jump
                    K.append(r.ravel())
        KN = np.array(K) @ N if K else np.zeros((0, N.shape[1]))
        if KN.shape[0] < N.shape[1] or np.linalg.matrix_rank(KN) < N.shape[1]:
            raise ValueError(
                f"{who}: the {xd.size} x {xq.size} grid does not determine the "
                f"co-energy spline ({n} coefficients, rank {rank}); use a "
                f"finer table or smooth > 0")
        c = c + N @ np.linalg.lstsq(KN, -(np.array(K) @ c), rcond=None)[0]
    fit = A[:n_data] @ c
    scale = max(np.ptp(ld), np.ptp(lq), 1e-300)
    resid = float(np.max(np.abs(fit - y[:n_data])) / scale)
    if resid > tol:
        raise ValueError(
            f"{who}: fit residual {resid:.4g} of the flux range exceeds tol = "
            f"{tol:g}: the tables are not the gradient of one co-energy "
            f"(dlambda_d/di_q != dlambda_q/di_d, a non-reciprocal map that "
            f"would create or destroy energy over a current cycle)")
    C = c.reshape(nbd, nbq)
    coef = np.einsum("ij,aim,bjn->abmn", C, Cd, Cq)
    return FluxMapDQ(list(xd), list(xq), list(coef.ravel()), source, resid)


def add_pmsm_flux_map(self, name: str, phase_a: str, phase_b: str,
                      phase_c: str, neutral: str, omega_node: str,
                      theta_node: str, *, R_s: float, flux_map,
                      pole_pairs: int, J: float, B: float = 0.0,
                      T_load: float = 0.0, L_0: float = 0.0,
                      omega0: float = 0.0, theta0: float = 0.0
                      ) -> "CircuitBuilder":
    """Add an MNA-native PMSM whose dq law is the saturating,
    cross-coupled ``flux_map`` (:func:`flux_map_from_tables`):
    ``add_pmsm_mna`` with ``L_d``, ``L_q``, ``psi_pm`` set to the map's
    values at ``i = 0``. Torque ``(3/2)·pp·(lambda_d·i_q - lambda_q·i_d)``
    from the same co-energy. The held-mechanics DC point is available at
    ``omega0 = 0`` only."""
    _, psi, _, l_dd, _, l_qq = flux_map.evaluate(0.0, 0.0)
    return self.add_pmsm_mna(
        name, phase_a, phase_b, phase_c, neutral, omega_node, theta_node,
        R_s=R_s, L_d=l_dd, L_q=l_qq, psi_pm=psi, pole_pairs=pole_pairs,
        J=J, B=B, T_load=T_load, L_0=L_0, omega0=omega0, theta0=theta0,
        flux_map=flux_map)


def install(builder_cls) -> None:
    """Patch ``add_dc_machine`` / ``add_bldc_mna`` onto ``builder_cls``
    (idempotent)."""
    for fn in (add_dc_machine, add_bldc_mna, add_pmsm_flux_map):
        if not hasattr(builder_cls, fn.__name__):
            fn.__qualname__ = f"CircuitBuilder.{fn.__name__}"
            setattr(builder_cls, fn.__name__, fn)
