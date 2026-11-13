"""Realistic passives built from existing kernel elements.

openspec ``add-realistic-passives``. Everything here is a composite of
linear R, L, C (no new kernel kind), so it runs on every engine and in
the DC and AC analyses exactly like the elements it is made of.

* ``add_capacitor(..., esr=, esl=)``:
  ``from ─ {name}.esr ─ {name}.esl ─ {name} ─ to``
* ``add_inductor(..., esr=, c_par=, rac=)``:
  ``from ─ {name}.esr ─ {name} ─ [rac cells] ─ to``, ``{name}.cpar``
  from ``from`` to ``to``.
* ``fit_rac_ladder`` fits ``R_dc + Σ R_k·(ωτ_k)²/(1+(ωτ_k)²)`` to a
  winding's ``R_ac(f)`` curve (a fit reproduces its own data by
  construction: that is not a validation).
* ``add_transmission_line(..., model="pi")``: lumped π sections.

``{name}`` stays the ideal element, so ``c0``, ``i0`` and
``result.i(name)`` keep their meaning; a parameter left at zero adds
nothing (the call is the native one, bit for bit). The ESR loss is the
resistor row ``{name}.esr`` of ``device_loss_summary``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Sequence

import numpy as np

if TYPE_CHECKING:
    from ._pulsim import CircuitBuilder

__all__ = [
    "RacLadder",
    "fit_rac_ladder",
    "recommend_line_sections",
    "install",
]


def _nonneg(who: str, label: str, value) -> float:
    v = float(value)
    if not (math.isfinite(v) and v >= 0.0):
        raise ValueError(f"{who}: {label} must be finite and >= 0 "
                         f"(got {value!r}).")
    return v


# ---------------------------------------------------------------------------
# AC-resistance ladder
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RacLadder:
    """``R_dc ─ L_hf ─ [R_1 ∥ L_1] ─ … ─ [R_n ∥ L_n]``.

    ``R_k``/``L_k`` are the cells, ``band`` the (f_min, f_max) of the
    data the ladder was fitted to (it extrapolates outside it as its
    last cell dictates), ``residual`` the largest relative deviation
    from that data, ``source`` a free-text provenance note.
    """
    R_dc: float
    L_hf: float
    R_k: tuple
    L_k: tuple
    band: tuple = (0.0, 0.0)
    residual: float = 0.0
    source: str = ""

    def r_ac(self, f) -> np.ndarray:
        """Series resistance [Ω] of the ladder at frequency ``f`` [Hz]."""
        return np.real(self.impedance(f))

    def impedance(self, f) -> np.ndarray:
        """Complex impedance [Ω] of the whole ladder at ``f`` [Hz]."""
        jw = 2j * np.pi * np.asarray(f, dtype=float)
        z = self.R_dc + jw * self.L_hf
        for r, lk in zip(self.R_k, self.L_k):
            z = z + (r * jw * lk) / (r + jw * lk)
        return z


def fit_rac_ladder(f: Sequence[float], R_ac: Sequence[float], *, L: float,
                   R_dc: Optional[float] = None, n_cells: int = 4,
                   tol: float = 0.05, name: str = "inductor",
                   source: str = "") -> RacLadder:
    """Fit a passive ``R ∥ L`` ladder to a winding's ``R_ac(f)`` curve.

    Corner frequencies ``1/(2π·τ_k)`` are log-spaced across the data
    band; the cell resistances come from a non-negative least-squares
    fit (relative error), so every element is ≥ 0 and the ladder is
    passive. ``L_k = R_k·τ_k`` and ``L_hf = L − Σ L_k``, where ``L`` is
    the low-frequency inductance [H]. ``R_dc`` [Ω] is fitted when not
    given.

    Refused, naming ``name``: ``L_hf ≤ 0`` (the resistance rise needs
    more cell inductance than the component has) and a largest relative
    residual above ``tol``.
    """
    from scipy.optimize import nnls

    who = f"fit_rac_ladder({name!r})"
    f = np.asarray(f, dtype=float)
    r = np.asarray(R_ac, dtype=float)
    if f.ndim != 1 or f.shape != r.shape or f.size < 2:
        raise ValueError(f"{who}: f and R_ac must be 1-D arrays of the "
                         "same length (at least 2 points).")
    if not (np.all(np.isfinite(f)) and np.all(f > 0)
            and np.all(np.isfinite(r)) and np.all(r > 0)):
        raise ValueError(f"{who}: f and R_ac must be finite and > 0.")
    if int(n_cells) < 1:
        raise ValueError(f"{who}: n_cells must be >= 1.")
    L = float(L)
    if not (math.isfinite(L) and L > 0):
        raise ValueError(f"{who}: L must be finite and > 0 (got {L!r}).")

    f_lo, f_hi = float(f.min()), float(f.max())
    corners = np.logspace(math.log10(f_lo), math.log10(f_hi), int(n_cells))
    tau = 1.0 / (2 * np.pi * corners)
    x = np.outer(2 * np.pi * f, tau) ** 2
    cols = x / (1.0 + x)
    target = r.copy()
    if R_dc is None:
        cols = np.column_stack([np.ones_like(f), cols])
    else:
        target = target - _nonneg(who, "R_dc", R_dc)
    # Relative least squares: every point weighted by 1/R_ac.
    sol, _ = nnls(cols / r[:, None], target / r)
    if R_dc is None:
        r_dc, r_k = float(sol[0]), sol[1:]
    else:
        r_dc, r_k = float(R_dc), sol
    l_k = r_k * tau
    l_hf = L - float(np.sum(l_k))
    if not l_hf > 0.0:
        raise ValueError(
            f"{who}: the fitted cells need {np.sum(l_k):.6g} H of "
            f"inductance but L is {L:.6g} H (deficit {-l_hf:.6g} H). "
            "The requested resistance rise cannot be realised by a "
            "passive ladder inside this inductance.")
    keep = r_k > 0.0
    ladder = RacLadder(R_dc=r_dc, L_hf=l_hf,
                       R_k=tuple(float(v) for v in r_k[keep]),
                       L_k=tuple(float(v) for v in l_k[keep]),
                       band=(f_lo, f_hi), source=source)
    res = float(np.max(np.abs(ladder.r_ac(f) - r) / r))
    if res > tol:
        raise ValueError(
            f"{who}: largest relative residual {res:.3g} exceeds tol "
            f"{tol:g} with {n_cells} cells; add cells or relax tol.")
    return RacLadder(R_dc=ladder.R_dc, L_hf=ladder.L_hf, R_k=ladder.R_k,
                     L_k=ladder.L_k, band=ladder.band, residual=res,
                     source=source)


# ---------------------------------------------------------------------------
# Transmission line (π sections)
# ---------------------------------------------------------------------------

def recommend_line_sections(delay: float, t_rise: float) -> int:
    """π sections for a line of one-way ``delay`` driven by edges of
    ``t_rise``: ``ceil(11·delay/t_rise)``. Empirical, from two measured
    points (5.4 % overshoot with 50 sections at ``t_rise = delay/5``,
    10.2 % with 20); an ideal step overshoots 27 % at ANY count."""
    d, tr = float(delay), float(t_rise)
    if not (math.isfinite(d) and d > 0 and math.isfinite(tr) and tr > 0):
        raise ValueError("recommend_line_sections: delay and t_rise must "
                         "be finite and > 0.")
    return int(math.ceil(11.0 * d / tr))


def add_transmission_line(self, name: str, k_pos: str, k_neg: str,
                          m_pos: str, m_neg: str, *,
                          Z0: Optional[float] = None,
                          delay: Optional[float] = None, R: float = 0.0,
                          model: str = "pi", sections: Optional[int] = None,
                          L_per_m: Optional[float] = None,
                          C_per_m: Optional[float] = None,
                          length: Optional[float] = None) -> "CircuitBuilder":
    """Two-conductor lossless (or series-lossy) line, differential mode.

    Give ``Z0`` [Ω] and one-way ``delay`` [s], or ``L_per_m`` [H/m],
    ``C_per_m`` [F/m] and ``length`` [m]. ``R`` [Ω] is the total series
    resistance, distributed over the sections.

    ``model="pi"``: ``sections`` cells of a series ``Z0·delay/N``
    inductance with ``delay/(Z0·N)`` shunt capacitance split at its
    ends. Usable in every analysis; cut-off ``N/(π·delay)``. LIMITATION
    (measured, openspec add-realistic-passives): an ideal step
    overshoots the exact answer by 27 % at the far end of an open,
    source-matched line whatever ``N`` is (1.271 V instead of 1.000 V
    with 50 sections); a finite edge needs
    :func:`recommend_line_sections` sections to stay near 5 %.

    ``model="bergeron"`` (exact delay) needs a kernel device that does
    not exist yet and is refused by name.

    Sub-elements: ``{name}.l{k}``, ``{name}.c{k}``, ``{name}.r{k}``;
    internal nodes ``{name}.n{k}``; ``{name}.ret`` (0 V source) joins
    ``m_neg`` to ``k_neg`` when they differ (the return conductor's
    impedance is lumped into the forward one).
    """
    who = f"add_transmission_line({name!r})"
    if model == "bergeron":
        from ._refusal import unsupported
        raise unsupported(
            "", "model",
            f"{who}: model='bergeron' needs a travelling-wave kernel "
            "device (openspec add-realistic-passives, section 5) that "
            "is not implemented yet. Use model='pi' with "
            "sections=recommend_line_sections(delay, t_rise).",
            device=name)
    if model != "pi":
        raise ValueError(f"{who}: model must be 'pi' or 'bergeron' "
                         f"(got {model!r}).")
    per_m = (L_per_m, C_per_m, length)
    if any(v is not None for v in per_m):
        if Z0 is not None or delay is not None or None in per_m:
            raise ValueError(f"{who}: give either (Z0, delay) or all of "
                             "(L_per_m, C_per_m, length).")
        lp, cp, ln = (float(v) for v in per_m)
        if not all(math.isfinite(v) and v > 0 for v in (lp, cp, ln)):
            raise ValueError(f"{who}: L_per_m, C_per_m and length must "
                             "be finite and > 0.")
        Z0, delay = math.sqrt(lp / cp), ln * math.sqrt(lp * cp)
    if Z0 is None or delay is None:
        raise ValueError(f"{who}: Z0 and delay are required.")
    z0, tau = float(Z0), float(delay)
    if not (math.isfinite(z0) and z0 > 0):
        raise ValueError(f"{who}: Z0 must be finite and > 0 (got {Z0!r}).")
    if not (math.isfinite(tau) and tau > 0):
        raise ValueError(f"{who}: delay must be finite and > 0 "
                         f"(got {delay!r}).")
    r_tot = _nonneg(who, "R", R)
    if sections is None or int(sections) != sections or int(sections) < 1:
        raise ValueError(f"{who}: sections must be an integer >= 1 "
                         f"(got {sections!r}); see recommend_line_sections.")
    n = int(sections)
    l_sec, c_sec = z0 * tau / n, tau / (z0 * n)
    nodes = [k_pos] + [f"{name}.n{k}" for k in range(1, n)] + [m_pos]
    for k in range(n + 1):
        c = c_sec if 0 < k < n else 0.5 * c_sec
        self.add_capacitor(f"{name}.c{k}", nodes[k], k_neg, c)
    for k in range(n):
        a, b = nodes[k], nodes[k + 1]
        if r_tot > 0.0:
            mid = f"{name}.m{k}"
            self.add_resistor(f"{name}.r{k}", a, mid, r_tot / n)
            a = mid
        self.add_inductor(f"{name}.l{k}", a, b, l_sec)
    if m_neg != k_neg:
        self.add_voltage_source(f"{name}.ret", m_neg, k_neg, 0.0)
    return self


# ---------------------------------------------------------------------------
# Builder wrappers
# ---------------------------------------------------------------------------

def _wrap_capacitor(native, native_l):
    def add_capacitor(self, name: str, n_pos: str, n_neg: str,
                      C_farads: float, c0: Optional[float] = None, *,
                      esr: float = 0.0, esl: float = 0.0) -> "CircuitBuilder":
        who = f"add_capacitor({name!r})"
        r = _nonneg(who, "esr", esr)
        lser = _nonneg(who, "esl", esl)
        if r == 0.0 and lser == 0.0:
            return native(self, name, n_pos, n_neg, C_farads, c0)
        a = n_pos
        if r > 0.0:
            nxt = f"{name}.esr_n"
            self.add_resistor(f"{name}.esr", a, nxt, r)
            a = nxt
        if lser > 0.0:
            nxt = f"{name}.esl_n"
            native_l(self, f"{name}.esl", a, nxt, lser, None)
            a = nxt
        return native(self, name, a, n_neg, C_farads, c0)

    add_capacitor.__qualname__ = "CircuitBuilder.add_capacitor"
    add_capacitor.__doc__ = (native.__doc__ or "") + (
        "\nOptional keyword parasitics (openspec add-realistic-passives): "
        "esr [Ω] and esl [H] in series, expanded into {name}.esr and "
        "{name}.esl with internal nodes; zero adds nothing. c0 is the "
        "voltage of the ideal capacitor {name}.")
    add_capacitor._pulsim_native = native  # type: ignore[attr-defined]
    return add_capacitor


def _wrap_inductor(native, native_c):
    def add_inductor(self, name: str, n_pos: str, n_neg: str,
                     L_henries: float, i0: Optional[float] = None, *,
                     esr: float = 0.0, c_par: float = 0.0,
                     rac=None) -> "CircuitBuilder":  # rac: a RacLadder
        who = f"add_inductor({name!r})"
        r = _nonneg(who, "esr", esr)
        cp = _nonneg(who, "c_par", c_par)
        if r == 0.0 and cp == 0.0 and rac is None:
            return native(self, name, n_pos, n_neg, L_henries, i0)
        l_main = L_henries
        a = n_pos
        if rac is not None:
            if r > 0.0:
                raise ValueError(f"{who}: give esr or rac, not both "
                                 "(the ladder's R_dc is the series "
                                 "resistance).")
            l_tot = rac.L_hf + sum(rac.L_k)
            if not math.isclose(l_tot, float(L_henries), rel_tol=1e-9):
                raise ValueError(
                    f"{who}: the ladder's low-frequency inductance "
                    f"{l_tot:.9g} H differs from L = {L_henries!r}; fit "
                    "it with the same L.")
            r = rac.R_dc
            l_main = rac.L_hf
        if r > 0.0:
            nxt = f"{name}.esr_n"
            self.add_resistor(f"{name}.esr", a, nxt, r)
            a = nxt
        if rac is None:
            native(self, name, a, n_neg, l_main, i0)
        else:
            cells = len(rac.R_k)
            nxt = f"{name}.rac_n0" if cells else n_neg
            native(self, name, a, nxt, l_main, i0)
            for k, (rk, lk) in enumerate(zip(rac.R_k, rac.L_k)):
                a = nxt
                nxt = f"{name}.rac_n{k + 1}" if k + 1 < cells else n_neg
                self.add_resistor(f"{name}.rac_r{k}", a, nxt, rk)
                # At DC the cell current flows in L_k (R_k is bypassed).
                native(self, f"{name}.rac_l{k}", a, nxt, lk, i0)
        if cp > 0.0:
            native_c(self, f"{name}.cpar", n_pos, n_neg, cp, None)
        return self

    add_inductor.__qualname__ = "CircuitBuilder.add_inductor"
    add_inductor.__doc__ = (native.__doc__ or "") + (
        "\nOptional keyword parasitics (openspec add-realistic-passives): "
        "esr [Ω] in series ({name}.esr), c_par [F] across the terminals "
        "({name}.cpar), rac=fit_rac_ladder(...) for a frequency-"
        "dependent winding resistance; zero adds nothing. i0 is the "
        "current of the ideal inductor {name}.")
    add_inductor._pulsim_native = native  # type: ignore[attr-defined]
    return add_inductor


def install(builder_cls) -> None:
    """Patch the parasitic keywords and ``add_transmission_line`` onto
    ``builder_cls`` (idempotent)."""
    cap = builder_cls.add_capacitor
    ind = builder_cls.add_inductor
    if not hasattr(cap, "_pulsim_native"):
        builder_cls.add_capacitor = _wrap_capacitor(cap, ind)
        builder_cls.add_inductor = _wrap_inductor(ind, cap)
    if not hasattr(builder_cls, "add_transmission_line"):
        add_transmission_line.__qualname__ = (
            "CircuitBuilder.add_transmission_line")
        builder_cls.add_transmission_line = add_transmission_line
