"""Forward parameter sensitivities on the exact path.

``add-parameter-sensitivities``, groups 2, 3 and 7 on today's ``dsed``
exact stepper. Between events a PWL circuit with DC sources is LTI,
``x' = A x + b``; with ``xb = (x, 1)`` and ``Ab = [[A, b], [0, 0]]`` one
step is ``xb(t + h) = exp(Ab h) xb(t)``, and its derivative with respect
to a parameter ``p`` is

    S(t + h) = exp(Ab h) S(t) + (d exp(Ab h)/dp) xb(t)

with the Frechet derivative of the exponential by Daleckii-Krein in the
eigenbasis of ``Ab``:

    d exp(Ab h) = V (F o (V^-1 dAb V)) V^-1,
    F_ij = (e^{mu_i h} - e^{mu_j h}) / (mu_i - mu_j) = h e^{a} phi1(b - a)

(``a`` the scaled eigenvalue with the larger real part, so ``phi1``
never overflows and close eigenvalues lose nothing). When the basis is
near-defective (critical damping) the same derivative comes from the
block exponential ``exp([[Ab, 0], [dAb, Ab]] h)`` (Van Loan 1978) by
Pade [13/13]; when neither is accurate the sensitivity is refused.

``dAb/dp`` is exact: every entry of ``A``, ``b`` and of the node-voltage
recovery is a linear-fractional (Moebius) function of an R, L, C or
source value (a rank-one stamp in a Schur complement, or a row scaled by
``1/C``, ``1/L``), and for a Moebius function the harmonic combination of
two one-sided differences, ``2 a b / (delta (b - a))``, IS the
derivative. It is formed at two step sizes and the two must agree, or
the parameter is refused.

Events (D3): at a state event ``g(x, p) = 0`` (a PWL diode)

    dtau/dp = -(grad g . S- + g_p) / (grad g . f-),  S+ = S- + (f- - f+) dtau/dp

and at a gate edge whose instant depends on the duty, ``dtau/dD = T``.
``grad g . f- = 0`` is a grazing event: the gradient does not exist
there and the run refuses it. A predicate that comes close to zero
inside a segment without crossing (a near miss) is flagged: the
gradient is exact for the event pattern taken, and a parameter change
of the size of the margin creates an event and changes it.

Every step of the engine's trajectory is re-derived from these
propagators; a step the formulas do not reproduce (a numeric step of a
mask without an exact propagator) refuses the run rather than
returning the derivative of a different map.
"""

from __future__ import annotations

import math
import re
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

import numpy as np

__all__ = [
    "SensitivityResult",
    "KpiValue",
    "value_and_grad",
    "simulate_sensitivities",
]

_U = np.finfo(float).eps
_TIER_BOUND = 1e-10           # u*kappa and u*||A h||_1, as exact_lti.hpp
_MAX_PARAMS = 64
_GRAZE = 1e-9                 # |grad g . f-| <= this * |grad g| |f-|
_NEAR_MISS = 1e-6             # interior min |g| <= this * max|v_D|
_MOEBIUS_RTOL = 1e-7
_STEP_RTOL = 1e-10            # re-derived step against the engine's sample
_BRANCH_RE = re.compile(r"\(branch #(\d+)\)")
_GATE, _DIODE_ON, _DIODE_OFF = 0, 1, 2

# kind -> (accepted field names, builder update method, params key)
_FIELDS = {
    "resistor": (("R", "R_ohms"), "update_resistor_R", "R_ohms"),
    "inductor": (("L", "L_henries"), "update_inductor_L", "L_henries"),
    "capacitor": (("C", "C_farads"), "update_capacitor_C", "C_farads"),
    "voltage_source": (("V",), "update_voltage_source_V", "V"),
}


def _unsup(feature, msg, *, device="", kind="", engine="dsed"):
    from ._refusal import unsupported
    return unsupported(engine, feature, msg, device=device, kind=kind)


# ----------------------------------------------------------------------
# phi functions (complex, vectorised)
# ----------------------------------------------------------------------

def _expm1c(s):
    a, b = s.real, s.imag
    sh = np.sin(b / 2)
    with np.errstate(over="ignore", invalid="ignore"):
        return (np.expm1(a) * np.cos(b) - 2 * sh * sh) \
            + 1j * np.exp(a) * np.sin(b)


def _phi1(s):
    s = np.asarray(s, dtype=complex)
    out = np.ones_like(s)
    nz = s != 0
    out[nz] = _expm1c(s[nz]) / s[nz]
    return out


def _psi(k, m):
    """``psi_k(m) = int_0^1 th^k e^{th m} dth`` (the k-th derivative of
    phi1), k = 0..kmax, shape (kmax + 1,) + m.shape."""
    m = np.asarray(m, dtype=complex)
    out = np.empty((k + 1,) + m.shape, dtype=complex)
    small = np.abs(m) <= 1.0
    # series for |m| <= 1: sum_j m^j / (j! (j + k + 1))
    term = np.ones_like(m)
    acc = np.zeros((k + 1,) + m.shape, dtype=complex)
    for j in range(30):
        for kk in range(k + 1):
            acc[kk] += term / (j + kk + 1)
        term = term * m / (j + 1)
    # forward recurrence otherwise: psi_k = (e^m - k psi_{k-1}) / m
    ms = np.where(small, 1.0, m)
    with np.errstate(over="ignore", invalid="ignore"):
        em = np.exp(ms)
        rec = _phi1(ms)
        out[0] = np.where(small, acc[0], rec)
        for kk in range(1, k + 1):
            rec = (em - kk * rec) / ms
            out[kk] = np.where(small, acc[kk], rec)
    return out


def _dd_exp(a, b):
    """``(e^a - e^b)/(a - b)`` (``e^a`` when equal), no overflow on the
    decaying side and no cancellation for close arguments."""
    a, b = np.broadcast_arrays(np.asarray(a, complex), np.asarray(b, complex))
    swap = a.real < b.real
    hi = np.where(swap, b, a)
    lo = np.where(swap, a, b)
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        return np.exp(hi) * _phi1(lo - hi)


def _dd_phi1(a, b):
    """``(phi1(a) - phi1(b))/(a - b)`` (``phi1'(a)`` when equal): direct
    when the points are apart, Taylor about the midpoint otherwise."""
    a, b = np.broadcast_arrays(np.asarray(a, complex), np.asarray(b, complex))
    d = a - b
    m = 0.5 * (a + b)
    near = np.abs(d) <= 1e-3 * np.maximum(1.0, np.abs(m.real))
    with np.errstate(divide="ignore", invalid="ignore"):
        direct = (_phi1(a) - _phi1(b)) / np.where(near, 1.0, d)
    ps = _psi(5, m)
    taylor = ps[1] + ps[3] * d ** 2 / 24 + ps[5] * d ** 4 / 1920
    return np.where(near, taylor, direct)


# ----------------------------------------------------------------------
# Pade [13/13] exponential (Higham 2005), numpy only
# ----------------------------------------------------------------------

_B13 = (64764752532480000.0, 32382376266240000.0, 7771770303897600.0,
        1187353796428800.0, 129060195264000.0, 10559470521600.0,
        670442572800.0, 33522128640.0, 1323241920.0, 40840800.0,
        960960.0, 16380.0, 182.0, 1.0)


def _expm(M):
    nrm = float(np.abs(M).sum(axis=0).max()) if M.size else 0.0
    s = max(0, int(math.ceil(math.log2(nrm / 5.371920351148152)))) \
        if nrm > 5.371920351148152 else 0
    A = M / 2.0 ** s
    eye = np.eye(M.shape[0])
    A2 = A @ A
    A4 = A2 @ A2
    A6 = A4 @ A2
    b = _B13
    U = A @ (A6 @ (b[13] * A6 + b[11] * A4 + b[9] * A2)
             + b[7] * A6 + b[5] * A4 + b[3] * A2 + b[1] * eye)
    V = A6 @ (b[12] * A6 + b[10] * A4 + b[8] * A2) \
        + b[6] * A6 + b[4] * A4 + b[2] * A2 + b[0] * eye
    E = np.linalg.solve(V - U, V + U)
    for _ in range(s):
        E = E @ E
    return E


# ----------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------

@dataclass
class _Param:
    name: str
    comp: str
    kind: str                 # resistor / ... / "duty"
    value: float
    setter: Optional[Callable] = None
    bit: int = -1
    period: float = 0.0


def _resolve(builder, names, switch_fn) -> list:
    if isinstance(names, str):
        names = [names]
    names = list(names)
    if not names:
        raise ValueError("sensitivities: give at least one parameter name "
                         "('<component>.<field>').")
    if len(names) > _MAX_PARAMS:
        raise _unsup(
            "sensitivities", f"sensitivities: {len(names)} parameters; "
            f"the forward method is limited to {_MAX_PARAMS} per run "
            "(one propagation each). Split the request.")
    if len(set(names)) != len(names):
        raise ValueError(f"sensitivities: duplicate names in {names}.")
    comps = {c["name"]: c for c in builder.components()}
    out = []
    for nm in names:
        if not isinstance(nm, str) or "." not in nm:
            raise ValueError(
                f"sensitivities: {nm!r} is not '<component>.<field>'.")
        cname, fld = nm.rsplit(".", 1)
        c = comps.get(cname)
        if c is None:
            raise ValueError(
                f"sensitivities: {nm!r}: no component named {cname!r} "
                f"(components: {sorted(comps)}).")
        kind = c["kind"]
        if kind == "switch" and fld == "duty":
            out.append(_Param(nm, cname, "duty", float("nan"),
                              bit=int(builder.switch_index_of(cname))))
            continue
        spec = _FIELDS.get(kind)
        if spec is None or fld not in spec[0]:
            have = ("'duty' (of its gate schedule)" if kind == "switch"
                    else f"{list(spec[0])}" if spec else "none")
            raise _unsup(
                "sensitivity_parameter",
                f"sensitivities: {nm!r}: field {fld!r} of {cname!r} "
                f"({kind}) is not a differentiable parameter of the exact "
                f"path (differentiable fields: {have}). Integer, "
                "topology and table parameters, switch conductances and "
                "Newton-device parameters are refused.",
                device=cname, kind=fld)
        upd = getattr(builder, spec[1])
        out.append(_Param(nm, cname, kind, float(c["params"][spec[2]]),
                          setter=lambda v, _u=upd, _n=cname: _u(_n, v)))
    return out


def _duty_period(sf, bit, t_end, name, nsw=0):
    """Trailing-edge PWM of one switch bit: rising edges on a fixed grid
    of period T, falling edges at rise + D T. Returns (T, D); refuses any
    other schedule (the duty would not be a timing parameter with
    dtau/dD = T)."""
    rises, falls, t = [], [], 0.0
    prev_m = sf(0.0)
    prev = bool(prev_m.get(bit))
    others = False
    while len(rises) < 3 or t < t_end:
        e = float(sf.next_edge_after(t))
        if not math.isfinite(e) or e <= t or len(rises) + len(falls) > 200000:
            break
        m = sf(e)
        # another switch with edges (a dead-time pair, a bridge leg): the
        # duty may move them too, and which ones is not in the schedule
        others = others or any(bool(m.get(i)) != bool(prev_m.get(i))
                               for i in range(nsw) if i != bit)
        prev_m = m
        now = bool(m.get(bit))
        if now != prev:
            (rises if now else falls).append(e)
            prev = now
        t = e
    bad = others or len(rises) < 2 or not falls
    if not bad:
        T = rises[1] - rises[0]
        r = np.asarray(rises)
        bad = not (T > 0 and np.all(np.abs(np.diff(r) - T) <= 1e-9 * T))
    if not bad:
        f = np.asarray(falls)
        k = np.floor((f - r[0]) / T + 1e-9)
        D = (f - r[0] - k * T) / T
        bad = not (np.all(np.abs(D - D[0]) <= 1e-9) and 0 < D[0] < 1)
    if bad:
        raise _unsup(
            "sensitivity_parameter",
            f"sensitivities: {name!r}: the gate schedule of this switch is "
            "not a fixed-period trailing-edge PWM driving this switch "
            "alone (rising edges every T, falling edges at rise + D T, no "
            "other switch with edges: a dead-time pair or a bridge would "
            "move the other edges too), so its duty is not a parameter "
            "whose edge moves by T per unit. Build it with "
            "make_pwm_switch_fn.", device=name.rsplit(".", 1)[0],
            kind="duty")
    return float(T), float(D[0])


# ----------------------------------------------------------------------
# Assembly snapshots and their exact derivatives
# ----------------------------------------------------------------------

def _snapshot(builder, masks, h_max, ic):
    """Per mask: ``Ab`` (n+1, n+1) and the node-voltage recovery
    ``R`` (n_nodes, n+1) with ``v = R . xb``; plus the reduced initial
    state when ``ic``. One flat vector per mask (and one for ic)."""
    from . import PwlStateSpaceCache
    from . import _pulsim
    cache = PwlStateSpaceCache(builder.graph, builder.pool)
    cache.build_lazy(h_max)
    ad = _pulsim._NativeCircuitBuilderAdapter(builder, cache, None)
    nn = int(builder.graph.num_nodes)
    out = {}
    for key, m in masks.items():
        ad.set_mask(m)
        n = int(ad.n_state)
        Ab = np.zeros((n + 1, n + 1))
        Ab[:n, :n] = np.asarray(ad.A_matrix(), dtype=float)
        Ab[:n, n] = np.asarray(ad.b_vector(0.0), dtype=float)
        R = np.empty((nn, n + 1))
        z = np.zeros(n)
        for node in range(nn):
            v0 = float(ad.node_pair_voltage(node, -1, 0.0, z))
            for j in range(n):
                e = z.copy()
                e[j] = 1.0
                R[node, j] = float(ad.node_pair_voltage(node, -1, 0.0, e)) - v0
            R[node, n] = v0
        out[key] = np.concatenate([Ab.ravel(), R.ravel()])
    if ic:
        ad.set_mask(next(iter(masks.values())))
        x0, _adj = ad.reduced_initial_state(builder)
        out["__ic__"] = np.asarray(x0, dtype=float)
    return out


def _harmonic(a, b, delta):
    """Derivative of a Moebius function from ``a = f(p+d) - f(p)``,
    ``b = f(p-d) - f(p)``; plain central difference where the two are
    not of opposite sign (rounding noise of an unaffected entry)."""
    out = (a - b) / (2 * delta)
    ok = a * b < 0
    out[ok] = 2 * a[ok] * b[ok] / (delta * (b[ok] - a[ok]))
    return out


def _param_derivative(builder, prm, masks, h_max, ic, base, n1):
    v0 = prm.value
    if not (math.isfinite(v0) and v0 != 0.0):
        # the steps are relative (v0/8, v0/16): at 0 they are 0 and the
        # derivative 0/0 (a NaN that passed the check below)
        raise _unsup(
            "sensitivity_parameter",
            f"sensitivities: {prm.name!r} = {v0!r}; its derivative is taken "
            "from relative steps, which need a finite non-zero value",
            device=prm.comp, kind=prm.name.rsplit(".", 1)[1])
    snaps = {}
    try:
        for f in (1 / 8, -1 / 8, 1 / 16, -1 / 16):
            prm.setter(v0 * (1 + f))
            snaps[f] = _snapshot(builder, masks, h_max, ic)
    finally:
        prm.setter(v0)
    out = {}
    for key, f0 in base.items():
        d1 = _harmonic(snaps[1 / 8][key] - f0, snaps[-1 / 8][key] - f0,
                       v0 / 8)
        d2 = _harmonic(snaps[1 / 16][key] - f0, snaps[-1 / 16][key] - f0,
                       v0 / 16)
        # per row of Ab / of the recovery (one scale for the whole
        # vector let a 1e12 1/s entry hide a non-Moebius 1 V entry)
        w = n1 if key != "__ic__" else f0.size
        F0, D1, D2 = (np.abs(v.reshape(-1, w)) for v in (f0, d1, d2))
        # assembly rounding: 1024 u of the row (Schur complements of
        # g_on / g_off stamps cancel; 64 u refused a series 20 mOhm)
        floor = 1024 * _U * F0.max(axis=1) / (abs(v0) / 16)
        dev = np.abs(d1 - d2).reshape(-1, w).max(axis=1)
        lim = _MOEBIUS_RTOL * D1.max(axis=1) + floor
        del D2
        if not np.all(dev <= lim):      # a NaN is refused too
            j = int(np.argmax(dev - lim))
            dev, scale = float(dev[j]), float(D1.max(axis=1)[j])
            raise _unsup(
                "sensitivity_parameter",
                f"sensitivities: {prm.name!r} does not enter the state space "
                "as a linear-fractional function (the derivative from two "
                f"step sizes differs by {dev:.3e} against {scale:.3e} in "
                f"row {j} of {key}), so it cannot be differentiated "
                "exactly from assemblies.", device=prm.comp,
                kind=prm.name.rsplit(".", 1)[1])
        out[key] = d1
    return out


# ----------------------------------------------------------------------
# Per-mask propagator with its parameter derivatives
# ----------------------------------------------------------------------

class _Mode:
    def __init__(self, label, Ab, dAb, dsc, h_max):
        self.label = label
        self.Ab = Ab
        self.dAb = dAb                      # (n_p, n1, n1)
        self.n1 = Ab.shape[0]
        self.dsc = dsc
        As = dsc[:, None] * Ab / dsc[None, :]
        self.As = As
        self.dAs = dsc[None, :, None] * dAb / dsc[None, None, :]
        self.tier = 3
        self.reason = ""
        self._pade = {}
        try:
            # graded order (largest rows first): LAPACK's QR deflates from
            # the bottom, and with the 5e12 1/s idle-mode row last the
            # -425 1/s eigenvalue of the DCM buck was off by 2.7e-4
            # (u ||A||), 1e-9 of v_C per step
            order = np.argsort(-np.abs(As).sum(axis=1), kind="stable")
            mu, Vp = np.linalg.eig(As[np.ix_(order, order)])
            V = np.empty_like(Vp)
            V[order] = Vp
            kappa = float(np.linalg.cond(V))
        except np.linalg.LinAlgError:
            mu, V, kappa = None, None, math.inf
        self.kappa = kappa
        if math.isfinite(kappa) and _U * kappa <= _TIER_BOUND:
            self.tier = 1
            self.mu, self.V = mu, V
            self.Vi = np.linalg.inv(V)
            self.G = self.Vi @ self.dAs @ V          # (n_p, n1, n1)
            return
        nA = float(np.abs(As).sum(axis=0).max())
        nah = nA * h_max
        if _U * nah <= _TIER_BOUND:
            self.tier = 2
            # balance the coupling blocks (a similarity, undone below):
            # an unbalanced dA/dp would set the Pade scaling and cost
            # the accuracy of every block
            nd = np.abs(self.dAs).sum(axis=1).max(axis=1)
            self.cs = np.where(nd > 0, nA / np.where(nd > 0, nd, 1.0), 1.0)
            return
        self.reason = (f"kappa2(V) = {kappa:.3e} (near-defective basis) and "
                       f"||A h||_1 = {nah:.3e} (stiff): neither propagator "
                       "is accurate for this mode")

    # tier 1 ----------------------------------------------------------
    def modal(self, xb, Sb):
        return (self.Vi @ (self.dsc * xb),
                self.Vi @ (self.dsc[:, None] * Sb))

    def modal_advance(self, z, Z, s):
        e = np.exp(self.mu * s)
        F = s * _dd_exp(self.mu[:, None] * s, self.mu[None, :] * s)
        return e * z, e[:, None] * Z + np.einsum("pij,j->ip", F * self.G, z)

    def phys(self, z, Z):
        return ((self.V @ z).real / self.dsc,
                (self.V @ Z).real / self.dsc[:, None])

    # tier 2 ----------------------------------------------------------
    def _block(self, s, integral=False):
        key = (s, integral)
        E = self._pade.get(key)
        if E is None:
            n1, n_p = self.n1, self.dAs.shape[0]
            N = (1 + n_p) * n1
            Z = np.zeros((2 * N if integral else N,) * 2)
            for j in range(1 + n_p):
                Z[j * n1:(j + 1) * n1, j * n1:(j + 1) * n1] = self.As
            for j in range(n_p):
                Z[(j + 1) * n1:(j + 2) * n1, :n1] = self.dAs[j] * self.cs[j]
            if integral:
                Z[N:, :N] = np.eye(N)
            E = _expm(Z * s)
            if len(self._pade) > 64:
                self._pade.clear()
            self._pade[key] = E
        return E

    def advance(self, xb, Sb, s):
        """``(xb(s), Sb(s))`` from ``(xb, Sb)``."""
        if self.tier == 1:
            z, Z = self.modal(xb, Sb)
            return self.phys(*self.modal_advance(z, Z, s))
        n1, n_p = self.n1, Sb.shape[1]
        y = np.concatenate([self.dsc * xb,
                            (self.dsc[:, None] * Sb * self.cs).T.ravel()])
        y = self._block(s) @ y
        return (y[:n1] / self.dsc,
                y[n1:].reshape(n_p, n1).T / self.dsc[:, None] / self.cs)

    def integrals(self, xb, Sb, s):
        """``(int_0^s xb, int_0^s Sb)``, tier 2."""
        n1, n_p = self.n1, Sb.shape[1]
        N = (1 + n_p) * n1
        y = np.concatenate([self.dsc * xb,
                            (self.dsc[:, None] * Sb * self.cs).T.ravel(),
                            np.zeros(N)])
        y = self._block(s, True) @ y
        q = y[N:]
        return (q[:n1] / self.dsc,
                q[n1:].reshape(n_p, n1).T / self.dsc[:, None] / self.cs)


# ----------------------------------------------------------------------
# Runs
# ----------------------------------------------------------------------

class _ConstSchedule:
    def __init__(self, n):
        from . import SwitchStateMask
        self._m = SwitchStateMask(n)

    def __call__(self, t):
        return self._m

    def next_edge_after(self, t):
        return math.inf


def _check_circuit(builder, switch_fn, b_extra_fn):
    from . import _pulsim
    _pulsim._require_supported(builder.graph, builder.pool, "dsed")
    if b_extra_fn is not None:
        raise _unsup("b_extra_fn", "sensitivities: a b_extra_fn is an "
                     "opaque callback between the parameters and the "
                     "circuit; its derivative is unknown.")
    if getattr(builder, "_c_blocks", None):
        raise _unsup("c_blocks", "sensitivities: C blocks drive the "
                     "circuit through an opaque callback.")
    for c in builder.components():
        if c["kind"] not in ("resistor", "inductor", "capacitor",
                             "voltage_source", "current_source", "switch",
                             "diode"):
            raise _unsup(
                "sensitivities",
                f"sensitivities: component {c['name']!r} ({c['kind']}) is "
                "not covered by the exact sensitivity path of this slice "
                "(linear R, L, C, DC sources, controlled switches with a "
                "schedule, PWL diodes). Time-varying sources, controlled "
                "sources and Newton devices wait for the unified device "
                "interface.", device=c["name"],
                kind=builder.pool.kind_of(c["branch_id"]).name)
    _n, _nd, controlled = _pulsim._switch_census(builder.graph,
                                                 builder.pool)
    if switch_fn is None:
        if len(controlled):
            raise _unsup("switch_fn", "sensitivities: the circuit has "
                         "controlled switches and no switch_fn.")
        return _ConstSchedule(builder.graph.num_switches)
    from . import SwitchSchedule
    if not hasattr(switch_fn, "next_edge_after") or (
            isinstance(switch_fn, SwitchSchedule)
            and not switch_fn.has_edges):
        raise _unsup(
            "switch_fn", "sensitivities: the gate schedule cannot name its "
            "edges (an opaque Python callable): its edges would be found by "
            "sampling and their instants are not a function of the "
            "parameters. Use make_pwm_switch_fn & co. or give the callable "
            "a next_edge_after(t).")
    return switch_fn


@dataclass
class _Traj:
    times: np.ndarray              # (N,)
    states: np.ndarray             # (N, n) left limits
    masks: list                    # mask of the interval starting at k
    mask_left: list                # mask under which sample k was reached
    events: dict                   # k -> list of event dicts
    native: dict
    n_steps: int


def _run(builder, sf, x0, t_end, dt_max, dt_init, ic_from_builder):
    from . import PwlStateSpaceCache
    from . import _pulsim
    cache = PwlStateSpaceCache(builder.graph, builder.pool)
    cache.build_lazy(dt_max)
    ad = _pulsim._NativeCircuitBuilderAdapter(builder, cache, None)
    m0 = sf(0.0)
    ad.set_mask(m0)
    if ic_from_builder:
        x0, _adj = ad.reduced_initial_state(builder)
    x0 = np.asarray(x0, dtype=float)
    if ad.has_dynamic_sources:
        raise _unsup("sensitivities", "sensitivities: the circuit has "
                     "time-varying sources; this slice covers DC sources.")
    r = _pulsim.run_ped_native_builder(
        ad, sf, x0, t_end, rtol=1e-11, atol=1e-14, dt_init=dt_init,
        dt_max=dt_max, store_every=1, enable_event_projection=False,
        event_log=True)
    if int(r["n_exact_steps"]) != int(r["n_accept"]):
        raise _unsup(
            "sensitivities", "sensitivities: the exact stepper took "
            f"{int(r['n_accept']) - int(r['n_exact_steps'])} numeric (RK45) "
            "step(s): a mode has no closed-form propagator there, so the "
            "trajectory (and its events) is not the exact map this "
            "derivative belongs to.")
    times = np.asarray(r["times"], dtype=float)
    states = np.asarray(r["states"], dtype=float).reshape(len(times), -1)
    log = list(r["event_log"])
    events: dict = {}
    for e in log:
        k = int(np.argmin(np.abs(times - e["t"])))
        if abs(times[k] - e["t"]) > 1e-12 * max(abs(e["t"]), t_end):
            raise RuntimeError(
                f"sensitivities: event {e['name']!r} at t = {e['t']!r} has "
                "no sample of its own in the trajectory.")
        events.setdefault(k, []).append(dict(
            t=float(e["t"]), name=str(e["name"]),
            type=int(e["predicate_type"]), old=e["old_mask"],
            new=e["new_mask"]))
    masks, left = [], []
    cur = log[0]["old_mask"] if log else m0
    for k in range(len(times)):
        left.append(cur)
        for e in events.get(k, ()):
            cur = e["new"]
        masks.append(cur)
    return _Traj(times, states, masks, left, events, r, int(r["n_accept"])), \
        x0


# ----------------------------------------------------------------------
# The sensitivity computation
# ----------------------------------------------------------------------

class _Context:
    """Masks, propagators, exact derivatives, predicate rows."""

    def __init__(self, builder, params, traj, ic_from_builder, h_max):
        from . import PwlStateSpaceCache
        from .periodic_steady_state import _labels
        self.builder = builder
        self.params = params
        self.comps = list(builder.components())
        self.mask_by_key = {}
        for m in traj.masks + traj.mask_left:
            self.mask_by_key.setdefault(str(m), m)
        base = _snapshot(builder, self.mask_by_key, h_max, ic_from_builder)
        n_p = len(params)
        derivs = [None] * n_p
        self.n_assemblies = 1
        for j, prm in enumerate(params):
            if prm.kind != "duty":
                derivs[j] = _param_derivative(
                    builder, prm, self.mask_by_key, h_max, ic_from_builder,
                    base, len(traj.states[0]) + 1)
                self.n_assemblies += 4
        m0 = traj.masks[0]
        cache = PwlStateSpaceCache(builder.graph, builder.pool)
        cache.build_lazy(h_max)
        self.labels, devs = _labels(builder, cache, m0, self.comps)
        n = len(self.labels)
        self.n = n
        mass = np.ones(n)
        vals = {c["name"]: c["params"] for c in self.comps}
        for i, d in enumerate(devs):
            if len(d) == 1 and d[0] in vals:
                pr = vals[d[0]]
                mass[i] = float(pr.get("C_farads", pr.get("L_henries", 1.0)))
        dsc = np.ones(n + 1)
        dsc[:n] = np.sqrt(mass)
        nn = int(builder.graph.num_nodes)
        n1 = n + 1
        hmax = {}
        for k in range(len(traj.times) - 1):
            key = str(traj.masks[k])
            hmax[key] = max(hmax.get(key, 0.0),
                            traj.times[k + 1] - traj.times[k])
        self.modes, self.rec, self.drec = {}, {}, {}
        for key in self.mask_by_key:
            f0 = base[key]
            Ab = f0[:n1 * n1].reshape(n1, n1)
            self.rec[key] = f0[n1 * n1:].reshape(nn, n1)
            dAb = np.zeros((n_p, n1, n1))
            dR = np.zeros((n_p, nn, n1))
            for j in range(n_p):
                if derivs[j] is not None:
                    d = derivs[j][key]
                    dAb[j] = d[:n1 * n1].reshape(n1, n1)
                    dR[j] = d[n1 * n1:].reshape(nn, n1)
            self.drec[key] = dR
            d = dsc.copy()
            A = Ab[:n, :n]
            try:
                xeq = -np.linalg.solve(A, Ab[:n, n])
                sc = float(np.linalg.norm(d[:n] * xeq))
                if math.isfinite(sc) and sc > 0:
                    d[n] = 1.0 / sc
            except np.linalg.LinAlgError:
                pass
            self.modes[key] = _Mode(key, Ab, dAb, d, hmax.get(key, 0.0))
        self.S0 = np.zeros((n, n_p))
        if ic_from_builder:
            for j in range(n_p):
                if derivs[j] is not None:
                    self.S0[:, j] = derivs[j]["__ic__"]
        self.diodes = {c["branch_id"]: c for c in self.comps
                       if c["kind"] == "diode"}

    def f(self, key, x):
        Ab = self.modes[key].Ab
        return Ab[:-1, :-1] @ x + Ab[:-1, -1]

    def pair_row(self, key, a, k):
        """``(c, dc)``: ``v_a - v_k = c . xb``, ``dc`` its p-derivative."""
        R, dR = self.rec[key], self.drec[key]
        c = (R[a] if a >= 0 else 0.0) - (R[k] if k >= 0 else 0.0)
        dc = (dR[:, a] if a >= 0 else 0.0) - (dR[:, k] if k >= 0 else 0.0)
        return np.broadcast_to(c, (self.n + 1,)).astype(float), \
            np.broadcast_to(dc, (len(self.params), self.n + 1)).astype(float)

    def diode_of(self, name):
        m = _BRANCH_RE.search(name)
        if m and int(m.group(1)) in self.diodes:
            return self.diodes[int(m.group(1))]
        raise _unsup("event", f"sensitivities: event {name!r} is a state "
                     "event whose predicate the sensitivity path cannot "
                     "read, so its time derivative cannot be formed.")


def _jump(ctx, ev, x, S, params):
    """``(S+, dtau/dp)`` at one logged event (cascade merged by the
    engine: f- under the old mask, f+ under the final one)."""
    ko, kn = str(ev["old"]), str(ev["new"])
    fm, fp = ctx.f(ko, x), ctx.f(kn, x)
    n_p = len(params)
    dtau = np.zeros(n_p)
    if ev["type"] == _GATE:
        for j, prm in enumerate(params):
            if prm.kind == "duty" and ev["old"].get(prm.bit) \
                    and not ev["new"].get(prm.bit):
                dtau[j] = prm.period
    elif ev["type"] in (_DIODE_ON, _DIODE_OFF):
        c = ctx.diode_of(ev["name"])
        a, k = (int(v) for v in c["nodes"])
        row, drow = ctx.pair_row(ko, a, k)
        grad = row[:-1]
        den = float(grad @ fm)
        d = ctx.modes[ko].dsc[:-1]      # the cosine in energy coordinates
        if abs(den) <= _GRAZE * float(np.linalg.norm(grad / d)
                                      * np.linalg.norm(fm * d)):
            raise _unsup(
                "grazing_event",
                f"sensitivities: grazing event: {ev['name']} at t = "
                f"{ev['t']!r} is tangent to its switching surface "
                f"(grad(g).f = {den:.3e}); the trajectory is not "
                "differentiable in the parameters there (a parameter "
                "change on one side removes the event).",
                device=c["name"], kind="Diode")
        g_p = drow @ np.append(x, 1.0)
        dtau = -(grad @ S + g_p) / den
    else:
        raise _unsup("event", f"sensitivities: event {ev['name']!r} "
                     f"(type {ev['type']}) has no time derivative on the "
                     "exact path.")
    return S + np.outer(fm - fp, dtau), dtau


def _propagate(ctx, traj, S0):
    """Left-limit sensitivities at every sample, right limits at event
    samples, ``dtau/dp`` per event; checks every step against the
    engine."""
    params = ctx.params
    N = len(traj.times)
    n, n_p = ctx.n, S0.shape[1]
    S_all = np.empty((N, n, n_p))
    S_right: dict = {}
    dtaus: list = []
    S = S0.copy()
    scale = np.abs(traj.states).max(axis=0) + 1e-12
    n_sens = 0
    for k in range(N):
        x = traj.states[k]
        S_all[k] = S
        for ev in traj.events.get(k, ()):
            S, dtau = _jump(ctx, ev, x, S, params)
            dtaus.append(dict(ev, k=k, dtau=dtau))
        if k in traj.events:
            S_right[k] = S
        if k == N - 1:
            break
        h = traj.times[k + 1] - traj.times[k]
        mode = ctx.modes[str(traj.masks[k])]
        if mode.tier == 3:
            raise _unsup("sensitivities", f"sensitivities: mode {mode.label}"
                         f": {mode.reason}.")
        Sb = np.vstack([S, np.zeros((1, n_p))])
        xb1, Sb1 = mode.advance(np.append(x, 1.0), Sb, h)
        n_sens += n_p
        err = np.abs(xb1[:n] - traj.states[k + 1]) / scale
        if float(err.max()) > _STEP_RTOL:
            raise _unsup(
                "sensitivities",
                f"sensitivities: the step {traj.times[k]!r} -> "
                f"{traj.times[k + 1]!r} under {mode.label} is not reproduced "
                f"by the exact propagator (relative {float(err.max()):.2e}): "
                "the engine took a numeric step there, so the trajectory is "
                "not the exact map this derivative belongs to.")
        S = Sb1[:n]
    return S_all, S_right, dtaus, n_sens


# ----------------------------------------------------------------------
# Results
# ----------------------------------------------------------------------

class KpiValue:
    """A KPI and its gradient. Unpacks as ``value, grad``; ``grad`` is
    a dict by parameter name. ``non_unique`` marks a peak / ripple whose
    extremum is attained at two separate instants (not differentiable:
    the gradient given is that of the first)."""

    def __init__(self, value, grad, non_unique=False, where=None):
        self.value = float(value)
        self.grad = dict(grad)
        self.non_unique = bool(non_unique)
        self.where = where

    def __iter__(self):
        return iter((self.value, self.grad))

    def __repr__(self):
        return (f"KpiValue(value={self.value!r}, grad={self.grad!r}, "
                f"non_unique={self.non_unique})")


_KINDS = ("final", "avg", "rms", "energy", "peak", "valley", "ripple")


@dataclass
class SensitivityResult:
    """``dx/dp`` along an exact-path run.

    ``states`` is ``(n_samples, n_state, n_params)``: the left limit at
    every sample (as ``res.states``); ``params`` the names in column
    order; ``labels`` the state names; ``event_dtau`` the event instants'
    derivatives (``name``, ``t``, ``dtau`` by parameter); ``flags`` the
    near misses (see the module text); ``counters`` the cost."""

    params: list
    labels: list
    times: np.ndarray
    states: np.ndarray
    event_dtau: list
    flags: list
    counters: dict
    _ctx: Any = field(default=None, repr=False)
    _traj: Any = field(default=None, repr=False)
    _S_right: dict = field(default_factory=dict, repr=False)
    #: an orbit run (its two ends are one point)
    periodic: bool = False

    # -- signals ---------------------------------------------------------
    def _signal(self, name):
        """``key -> (c, dc)``: the signal is ``c . xb`` under each mask."""
        ctx = self._ctx
        n = ctx.n
        lab = None
        if name in ctx.labels:
            lab = ctx.labels.index(name)
        elif f"i({name})" in ctx.labels:
            lab = ctx.labels.index(f"i({name})")
        if lab is not None:
            c = np.zeros(n + 1)
            c[lab] = 1.0
            dc = np.zeros((len(self.params), n + 1))
            return lambda key: (c, dc)
        try:
            node = int(ctx.builder.node_id_of(name))
        except Exception:  # noqa: BLE001 — not a node
            node = -1
        if node < 0:
            raise ValueError(
                f"sensitivity: signal {name!r} is not a state "
                f"({ctx.labels}), an inductor or a node name.")
        return lambda key: ctx.pair_row(key, node, -1)

    def _col(self, param):
        try:
            return self.params.index(param)
        except ValueError:
            raise ValueError(f"sensitivity: {param!r} was not requested "
                             f"(requested: {self.params}).") from None

    def sens(self, signal, param):
        """``d signal / d param`` at every sample (left limits, as
        ``res.states``)."""
        j = self._col(param)
        row = self._signal(signal)
        tr = self._traj
        out = np.empty(len(tr.times))
        for k in range(len(tr.times)):
            c, dc = row(str(tr.mask_left[k]))
            out[k] = c[:-1] @ self.states[k, :, j] \
                + dc[j] @ np.append(tr.states[k], 1.0)
        return out

    # -- KPIs ------------------------------------------------------------
    def kpi(self, signal, kind="avg", *, t_from=None, t_to=None,
            with_grad=True):
        """KPI of a signal over ``[t_from, t_to]`` (default the whole
        run) with its exact gradient. ``kind``: ``final``, ``avg``
        (time-weighted), ``rms``, ``energy`` (the integral), ``peak``,
        ``valley``, ``ripple`` (peak - valley). Integrals are closed
        forms of the exact trajectory, with the event-time terms
        ``(y- - y+) dtau/dp`` of a signal that jumps at an event."""
        if kind not in _KINDS:
            raise ValueError(f"kpi: kind {kind!r}; use one of {_KINDS}.")
        tr = self._traj
        t0 = float(tr.times[0] if t_from is None else t_from)
        t1 = float(tr.times[-1] if t_to is None else t_to)
        if not (tr.times[0] <= t0 < t1 <= tr.times[-1] * (1 + 1e-12)):
            raise ValueError(f"kpi: window [{t0!r}, {t1!r}] is not inside "
                             f"the run [{tr.times[0]!r}, {tr.times[-1]!r}].")
        t1 = min(t1, float(tr.times[-1]))
        row = self._signal(signal)
        tol = 1e-12 * max(abs(t1), t1 - t0)
        for ev in self.event_dtau:
            if not np.any(ev["dtau"]) or not (
                    (t0 > tr.times[0] and abs(ev["t"] - t0) <= tol)
                    or abs(ev["t"] - t1) <= tol):
                continue
            xb = np.append(tr.states[ev["k"]], 1.0)
            if row(str(ev["old"]))[0] @ xb != row(str(ev["new"]))[0] @ xb:
                raise _unsup(
                    kind, f"kpi: the window edge t = {ev['t']!r} sits on "
                    f"the event {ev['name']!r}, which moves with the "
                    f"parameters, and {signal!r} jumps there: the KPI is "
                    "one-sided (not differentiable). Move the window edge.")
        val, grad, nu, where = _kpi(self, row, kind, t0, t1)
        if not with_grad:
            return val
        return KpiValue(val, dict(zip(self.params, grad)), nu, where)


def _pieces(sr, t0, t1):
    """Intervals overlapping ``[t0, t1]``: ``(k, s0, s1, mode, xb, Sb)``
    with ``(xb, Sb)`` the right limit at sample ``k``."""
    tr, ctx = sr._traj, sr._ctx
    n_p = len(sr.params)
    t = tr.times
    k0 = max(0, int(np.searchsorted(t, t0, side="right")) - 1)
    for k in range(k0, len(t) - 1):
        if t[k] >= t1:
            break
        s0 = max(0.0, t0 - t[k])
        s1 = min(t[k + 1], t1) - t[k]
        if s1 <= s0:
            continue
        S = sr._S_right.get(k, sr.states[k])
        yield (k, s0, s1, ctx.modes[str(tr.masks[k])],
               np.append(tr.states[k], 1.0),
               np.vstack([S, np.zeros((1, n_p))]))


def _event_terms(sr, row, t0, t1, power):
    """``sum (y-^q - y+^q) dtau/dp`` over events strictly inside."""
    out = np.zeros(len(sr.params))
    tr = sr._traj
    for ev in sr.event_dtau:
        if not (t0 < ev["t"] < t1):
            continue
        xb = np.append(tr.states[ev["k"]], 1.0)
        ym = row(str(ev["old"]))[0] @ xb
        yp = row(str(ev["new"]))[0] @ xb
        out += (ym ** power - yp ** power) * ev["dtau"]
    return out


def _kpi(sr, row, kind, t0, t1):
    n_p = len(sr.params)
    W = t1 - t0
    if kind == "final":
        tr = sr._traj
        if t1 >= tr.times[-1]:
            k = len(tr.times) - 1
            c, dc = row(str(tr.mask_left[k]))
            xb = np.append(tr.states[k], 1.0)
            return (float(c @ xb),
                    c[:-1] @ sr.states[k] + dc @ xb, False, t1)
        k, s0, _s1, mode, xb, Sb = next(_pieces(sr, t1, tr.times[-1]))
        x1, S1 = mode.advance(xb, Sb, s0)
        c, dc = row(str(tr.masks[k]))
        return float(c @ x1), c @ S1 + dc @ x1, False, t1
    if kind in ("peak", "valley", "ripple"):
        return _extrema(sr, row, kind, t0, t1)
    acc = 0.0
    Ig = np.zeros(n_p)
    power = 2 if kind == "rms" else 1
    for k, s0, s1, mode, xb, Sb in _pieces(sr, t0, t1):
        c, dc = row(mode.label)
        L = s1 - s0
        if mode.tier != 1:
            if power == 2:
                raise _unsup("rms", "kpi: the RMS gradient over a "
                             f"near-defective mode ({mode.label}) is not "
                             "available in this slice (closed form needs "
                             "the eigenbasis).")
            x0b, S0b = mode.advance(xb, Sb, s0) if s0 > 0 else (xb, Sb)
            ix, iS = mode.integrals(x0b, S0b, L)
            acc += float(c @ ix)
            Ig += c @ iS + dc @ ix
            continue
        z, Z = mode.modal(xb, Sb)
        if s0 > 0:
            z, Z = mode.modal_advance(z, Z, s0)
        mu, G = mode.mu, mode.G
        r = (c / mode.dsc) @ mode.V
        q = (dc / mode.dsc[None, :]) @ mode.V
        alpha = r * z
        beta = r[None, :] * Z.T + q * z[None, :]
        gamma = r[None, :, None] * G * z[None, None, :]
        if power == 1:
            I1 = L * _phi1(mu * L)
            IF = L * L * _dd_phi1(mu[:, None] * L, mu[None, :] * L)
            acc += float((alpha * I1).sum().real)
            Ig += (beta @ I1 + (gamma * IF[None]).sum(axis=(1, 2))).real
        else:
            mk = mu[:, None] + mu[None, :]
            I1kk = L * _phi1(mk * L)
            acc += float((alpha[:, None] * alpha[None, :] * I1kk).sum().real)
            P3 = L * L * _dd_phi1(mk[:, :, None] * L, mk[:, None, :] * L)
            t1_ = np.einsum("k,pi,ki->p", alpha, beta, I1kk)
            t2_ = np.einsum("k,pij,kij->p", alpha, gamma, P3)
            Ig += 2 * (t1_ + t2_).real
    Ig += _event_terms(sr, row, t0, t1, power)
    if kind == "energy":
        return acc, Ig, False, None
    if kind == "avg":
        return acc / W, Ig / W, False, None
    rms = math.sqrt(max(acc / W, 0.0))
    return rms, Ig / (2 * W * rms) if rms > 0 else Ig * 0.0, False, None


def _extrema(sr, row, kind, t0, t1, K=16):
    """Peak / valley by a sub-grid scan refined by Newton on ``y'``;
    the gradient is ``dy/dp`` at the extremum (plus ``y' dtau/dp`` when
    it sits on an event instant)."""
    cands = []          # (y, t, grad)
    tr = sr._traj
    ev_at = {}
    for ev in sr.event_dtau:
        if ev["k"] in ev_at and not np.array_equal(ev_at[ev["k"]]["dtau"],
                                                   ev["dtau"]):
            ev_at[ev["k"]] = dict(ev, dtau=None)    # two different instants
        else:
            ev_at.setdefault(ev["k"], ev)
    for k, s0, s1, mode, xb, Sb in _pieces(sr, t0, t1):
        if mode.tier != 1:
            raise _unsup(kind, f"kpi: {kind} over a near-defective mode "
                         f"({mode.label}) is not available in this slice.")
        c, dc = row(mode.label)
        z, Z = mode.modal(xb, Sb)
        r = (c / mode.dsc) @ mode.V
        alpha = r * z
        mu = mode.mu
        # at least eight points per period of the fastest oscillation
        w = float(np.abs(mu.imag).max())
        Kp = max(K, int(math.ceil((s1 - s0) * w * 8 / (2 * math.pi))))
        if Kp > 1 << 16:
            raise _unsup(kind, f"kpi: {kind} over a mode ringing "
                         f"{Kp // 8} times in one step is not searched.")
        ss = np.linspace(s0, s1, Kp + 1)

        def y(s):
            return float((alpha * np.exp(mu * s)).sum().real)

        def dy(s, order=1):
            return float((alpha * mu ** order * np.exp(mu * s)).sum().real)

        ys = np.array([y(s) for s in ss])
        yscale = float(np.abs(ys).max()) or 1.0
        pts = [s0, s1]
        for kind_ in (("peak", "valley") if kind == "ripple" else (kind,)):
            sg = 1.0 if kind_ == "peak" else -1.0
            # sub-grid maxima, the end cells included (a peak inside the
            # first or last cell has its sub-grid maximum on the end
            # point); Newton on y' within the neighbouring cells, kept
            # only when it converges inside the piece
            for i in range(Kp + 1):
                lo, hi = max(i - 1, 0), min(i + 1, Kp)
                if sg * ys[i] < sg * ys[lo] or sg * ys[i] < sg * ys[hi]:
                    continue
                a, b_ = ss[lo], ss[hi]
                if sg * dy(a) >= 0 >= sg * dy(b_) and a < b_:
                    # bracketed: bisection on y' to float resolution (a
                    # stiff mode's e^{mu s} sends Newton out of the cell)
                    for _ in range(200):
                        m_ = 0.5 * (a + b_)
                        if m_ in (a, b_):
                            break
                        if sg * dy(m_) >= 0:
                            a = m_
                        else:
                            b_ = m_
                    s = 0.5 * (a + b_)
                else:
                    s = ss[i]
                    for _ in range(50):
                        d2 = dy(s, 2)
                        if d2 == 0:
                            break
                        s_new = min(max(s - dy(s) / d2, ss[lo]), ss[hi])
                        done = abs(s_new - s) <= max(4 * _U * abs(s),
                                                     1e-13 * (s1 - s0))
                        s = s_new
                        if done:
                            break
                # a stationary point is a candidate; a sub-grid maximum
                # strictly inside with none found is a failed search
                if abs(dy(s)) * (s1 - s0) <= 1e-9 * yscale or (
                        a < s < b_ and b_ - a <= 4 * _U * max(abs(s), 1.0)):
                    if s0 < s < s1:
                        pts.append(s)
                elif 0 < i < Kp and sg * ys[i] > sg * ys[lo] \
                        and sg * ys[i] > sg * ys[hi]:
                    raise _unsup(
                        kind, f"kpi: the {kind_} search did not converge "
                        f"in the step at t = {tr.times[k]!r}; the "
                        "extremum (and its gradient) is not located.")
        for s in pts:
            x1, S1 = mode.advance(xb, Sb, s)
            g = c @ S1 + dc @ x1
            t = tr.times[k] + s
            if s == 0.0 and k in ev_at and t0 < t:
                g = g + (c[:-1] @ sr._ctx.f(mode.label, x1[:-1])) \
                    * _one_dtau(ev_at[k], kind)
            cands.append((float(c @ x1), t, g, k, s))
            if s == s1 and (k + 1) in ev_at and tr.times[k + 1] < t1:
                key = str(tr.masks[k])
                cands[-1] = (cands[-1][0], t,
                             g + (c[:-1] @ sr._ctx.f(key, x1[:-1]))
                             * _one_dtau(ev_at[k + 1], kind), k, s)
    ys = np.array([cd[0] for cd in cands])
    span = float(ys.max() - ys.min()) or max(abs(float(ys.max())), 1.0)
    out = {}
    for kind_ in (("peak", "valley") if kind == "ripple" else (kind,)):
        sg = 1.0 if kind_ == "peak" else -1.0
        i = int(np.argmax(sg * ys))
        tie = [cd for cd in cands
               if sg * (ys[i] - cd[0]) <= 1e-9 * span
               and abs(cd[1] - cands[i][1]) > 1e-9 * (t1 - t0)
               # the two ends of an orbit are one point
               and not (sr.periodic and abs(abs(cd[1] - cands[i][1])
                                            - (t1 - t0)) <= 1e-9 * (t1 - t0)
                        and t0 == tr.times[0] and t1 == tr.times[-1])]
        out[kind_] = (cands[i][0], cands[i][2], bool(tie), cands[i][1])
    if kind == "ripple":
        p_, v_ = out["peak"], out["valley"]
        return (p_[0] - v_[0], p_[1] - v_[1], p_[2] or v_[2],
                (p_[3], v_[3]))
    return out[kind]


def _one_dtau(ev, kind):
    if ev["dtau"] is None:
        raise _unsup(kind, f"kpi: the {kind} sits on two events of "
                     f"different time derivatives at t = {ev['t']!r}: it "
                     "is not differentiable there.")
    return ev["dtau"]


def _near_misses(ctx, traj, K=8):
    """Near misses of the PWL diodes' predicates.

    Along each stretch in which a diode keeps its state, its armed
    predicate ``g`` (turn-off while on, turn-on while off) is sampled on
    a sub-grid of every step. A near miss is a point where ``|g|`` comes
    within ``_NEAR_MISS`` of the largest ``|g|`` of that predicate over
    the run without crossing: an interior local minimum, or a stretch
    end the diode did not cause itself (a gate edge that switched it, the
    CCM valley at the DCM boundary). A stretch that starts or ends at the
    diode's own event sits on ``g = 0`` by construction and is not one.
    """
    flags = []
    for bid, c in ctx.diodes.items():
        try:
            bit = int(ctx.builder.switch_index_of(c["name"]))
        except Exception:  # noqa: BLE001 — not a switch bit
            continue
        a, k_ = (int(v) for v in c["nodes"])
        prm = c["params"]
        V_f, V_th = float(prm.get("V_f", 0.0)), float(prm["V_th"])
        tag = f"(branch #{bid})"
        stretches, cur = [], None
        for k in range(len(traj.times) - 1):
            key = str(traj.masks[k])
            on = bool(traj.masks[k].get(bit))
            evs = traj.events.get(k, ())
            own = any(tag in e["name"] and e["type"] != _GATE for e in evs)
            if cur is None or cur["on"] != on:
                if cur is not None:
                    cur["own_end"] = own
                    stretches.append(cur)
                cur = dict(on=on, own_start=own or k == 0, pts=[],
                           mins=[])
            mode = ctx.modes[key]
            row = ctx.pair_row(key, a, k_)[0]
            thr = V_f if on else (V_th if V_f == 0.0 else max(V_th, V_f))
            h = traj.times[k + 1] - traj.times[k]
            xb = np.append(traj.states[k], 1.0)
            if mode.tier == 1:
                w = float(np.abs(mode.mu.imag).max())
                Kp = max(K, int(math.ceil(h * w * 8 / (2 * math.pi))))
                ss = np.linspace(0, h, Kp + 1)
                z = mode.Vi @ (mode.dsc * xb)
                r = (row / mode.dsc) @ mode.V
                a_ = r * z
                g = (np.exp(np.outer(ss, mode.mu)) @ a_).real - thr
                # the sampled |g| misses a sharp interior minimum: refine
                # each sub-grid minimum by Newton on g'
                ag_ = np.abs(g)
                for i in range(1, Kp):
                    if not (ag_[i] <= ag_[i - 1] and ag_[i] <= ag_[i + 1]):
                        continue
                    s = ss[i]
                    for _ in range(50):
                        d1 = float((a_ * mode.mu * np.exp(mode.mu * s)).sum().real)
                        d2 = float((a_ * mode.mu ** 2
                                    * np.exp(mode.mu * s)).sum().real)
                        if d2 == 0:
                            break
                        s_new = min(max(s - d1 / d2, ss[i - 1]), ss[i + 1])
                        if abs(s_new - s) <= 1e-13 * h:
                            s = s_new
                            break
                        s = s_new
                    gm = float((a_ * np.exp(mode.mu * s)).sum().real) - thr
                    if np.sign(gm) == np.sign(g[i]):
                        cur["mins"].append((traj.times[k] + s, gm))
            else:
                # near-defective mode: the block exponential (no S needed)
                ss = np.linspace(0, h, 4 * K + 1)
                S0 = np.zeros((len(xb), len(ctx.params)))
                g = np.array([row @ mode.advance(xb, S0, s)[0]
                              for s in ss]) - thr
            cur["pts"].extend(zip(traj.times[k] + ss, g))
        if cur is not None:
            # the end of the run counts: an event just before it is
            # one a parameter change creates (the CCM valley of an orbit
            # sits on the section)
            cur["own_end"] = False
            stretches.append(cur)
        for on in (True, False):
            st = [x for x in stretches if x["on"] == on and x["pts"]]
            if not st:
                continue
            gmax = max(abs(g) for x in st for _t, g in x["pts"]) or 1.0
            tol = _NEAR_MISS * gmax
            for x in st:
                t = np.array([q[0] for q in x["pts"]])
                g = np.array([q[1] for q in x["pts"]])
                ag = np.abs(g)
                hits = [i for i in range(1, len(g) - 1)
                        if ag[i] <= tol and ag[i] < ag[i - 1]
                        and ag[i] <= ag[i + 1]
                        and np.sign(g[i - 1]) == np.sign(g[i + 1])]
                if not x["own_start"] and ag[0] <= tol:
                    hits.append(0)
                if not x["own_end"] and ag[-1] <= tol:
                    hits.append(len(g) - 1)
                found = [(float(t[i]), float(g[i])) for i in hits]
                for tm, gm in x["mins"]:
                    if abs(gm) <= tol and all(
                            abs(tm - tf) > 1e-9 * (t[-1] - t[0] or 1.0)
                            for tf, _g in found):
                        found.append((tm, gm))
                for tm, gm in sorted(found):
                    flags.append(dict(kind="near_miss", device=c["name"],
                                      t=tm, margin=gm,
                                      state="on" if on else "off"))
    return flags


def _compute(ctx, traj, S0=None):
    params = ctx.params
    S0 = ctx.S0 if S0 is None else S0
    S_all, S_right, dtaus, n_sens = _propagate(ctx, traj, S0)
    flags = _near_misses(ctx, traj)
    if flags:
        f = flags[0]
        warnings.warn(
            f"sensitivities: {len(flags)} near miss(es): the predicate of "
            f"{f['device']} comes within {abs(f['margin']):.3e} V of its "
            f"threshold at t = {f['t']:.6g} s without switching. The "
            "gradient is exact for this event pattern; a parameter change "
            "that closes the margin adds an event and changes it (see "
            "result.sensitivity.flags).", UserWarning, stacklevel=3)
    counters = dict(n_engine_runs=1, n_engine_steps=traj.n_steps,
                    n_sens_propagations=n_sens,
                    n_assemblies=ctx.n_assemblies,
                    tiers={k: m.tier for k, m in ctx.modes.items()})
    S_all.setflags(write=False)
    return SensitivityResult(
        params=[p.name for p in params], labels=list(ctx.labels),
        times=traj.times, states=S_all,
        event_dtau=[dict(name=d["name"], t=d["t"], k=d["k"], old=d["old"],
                         new=d["new"], dtau=d["dtau"]) for d in dtaus],
        flags=flags, counters=counters, _ctx=ctx, _traj=traj,
        _S_right=S_right)


def _prepare(builder, names, sf, t_end):
    params = _resolve(builder, names, sf)
    for prm in params:
        if prm.kind == "duty":
            prm.period, prm.value = _duty_period(
                sf, prm.bit, t_end, prm.name,
                int(builder.graph.num_switches))
    return params


# ----------------------------------------------------------------------
# Entry points
# ----------------------------------------------------------------------

def simulate_sensitivities(builder, t_end, *, sensitivities, dt=None,
                           t_start=0.0, switch_fn=None, initial_state=None,
                           ic_from_builder=False, dt_init=None,
                           b_extra_fn=None, integrator=None):
    """The ``simulate(engine='dsed', sensitivities=[...])`` path: one run
    of the exact stepper (events located, no event projection: the map
    ``periodic_steady_state`` differentiates) and the sensitivities of
    every sample. Returns the ``simulate`` result with ``.sensitivity``
    (a :class:`SensitivityResult`), ``.sens`` and ``.kpi``."""
    from ._dsed_dispatch import _result_from_native
    if integrator not in (None, "auto", "rk45"):
        raise _unsup("integrator", f"sensitivities: integrator="
                     f"{integrator!r}; the sensitivities are those of the "
                     "exact stepper (integrator='auto' or 'rk45').")
    sf = _check_circuit(builder, switch_fn, b_extra_fn)
    window = float(t_end) - float(t_start)
    params = _prepare(builder, sensitivities, sf, window)
    dt_max = float(dt) if dt else 1e-5
    if initial_state is None and not ic_from_builder:
        from . import _pulsim
        from . import PwlStateSpaceCache
        cache = PwlStateSpaceCache(builder.graph, builder.pool)
        cache.build_lazy(dt_max)
        ad = _pulsim._NativeCircuitBuilderAdapter(builder, cache, None)
        ad.set_mask(sf(0.0))
        initial_state = np.zeros(int(ad.n_state))
    traj, _x0 = _run(builder, sf, initial_state, window, dt_max,
                     dt_init if dt_init else 1e-9, ic_from_builder)
    sr = _compute(_Context(builder, params, traj, ic_from_builder, dt_max),
                  traj)
    res = _result_from_native(traj.native, t_start)
    sr.times = traj.times + float(t_start)
    res.sensitivity = sr
    res.sens = sr.sens
    res.kpi = sr.kpi
    return res


def value_and_grad(objective, params: Sequence[str]):
    """``f(x) -> (value, gradient)`` for ``scipy.optimize.minimize(...,
    jac=True)``. ``objective(values)`` gets a dict ``{name: value}`` and
    returns a :class:`KpiValue` (or ``(value, {name: grad})``) whose
    gradient covers every name in ``params``; ``x`` is in the order of
    ``params``. No finite difference is ever taken: a missing gradient
    is an error."""
    names = list(params)

    def f(x):
        x = np.asarray(x, dtype=float).reshape(-1)
        if x.size != len(names):
            raise ValueError(f"value_and_grad: x has {x.size} entries for "
                             f"{len(names)} parameters {names}.")
        out = objective(dict(zip(names, (float(v) for v in x))))
        val, grad = out
        missing = [n for n in names if n not in grad]
        if missing:
            raise ValueError(f"value_and_grad: the objective returned no "
                             f"gradient for {missing}.")
        return float(val), np.array([float(grad[n]) for n in names])

    return f


def orbit_sensitivities(res, solver, names, n_steps=50):
    """Steady-state gradients (D5) on a solved orbit: one more period run
    from ``x*`` (``res.n_period_runs`` counts it), ``dP/dp = S(T)`` with
    ``S(0) = 0``, ``dx*/dp = (I - M)^-1 dP/dp``, and the orbit's
    sensitivity with ``S(0) = dx*/dp`` for orbit KPIs. Sets
    ``res.dx0`` and ``res.sensitivity``."""
    builder, sf, T = solver.builder, solver.sf, solver.T
    params = _prepare(builder, names, sf, T)
    traj, _x0 = _run(builder, sf, res.x0, T, T / n_steps, T * 1e-6, False)
    solver.n_runs += 1
    solver.n_steps += traj.n_steps
    res.n_period_runs, res.n_engine_steps = solver.n_runs, solver.n_steps
    ctx = _Context(builder, params, traj, False, T / n_steps)
    if ctx.labels != list(res.state_labels):
        raise RuntimeError("sensitivities: state order differs between the "
                           f"solver ({res.state_labels}) and the sensitivity "
                           f"run ({ctx.labels}).")
    n = ctx.n
    # the dP/dp run (T/n_steps) and the solver's run (dt_max = T) must be
    # one map: the same events, and x(T) = x*
    ref = [(e["name"], e["t"]) for e in solver.orbit_traj.events]
    got = [(e["name"], e["t"]) for k in sorted(traj.events)
           for e in traj.events[k] if e["type"] != _GATE]
    ref = [e for e in ref if e[0] != "gate_edge"]
    if [e[0] for e in ref] != [e[0] for e in got] or any(
            abs(a[1] - b[1]) > 1e-9 * T for a, b in zip(ref, got)):
        raise _unsup("sensitivities", "sensitivities: the orbit's "
                     "sensitivity period does not reproduce the solver's "
                     f"events ({ref} against {got}); the gradient would mix "
                     "two maps.")
    x0 = np.asarray(res.x0, dtype=float)
    sc = np.abs(traj.states).max(axis=0) + 1e-300
    per = float((np.abs(traj.states[-1] - x0) / sc).max())
    if per > 1e-7:
        raise _unsup("sensitivities", "sensitivities: the sensitivity "
                     f"period moves the orbit by {per:.3e} of the state "
                     "scale; it is not the solver's map.")
    S_all, _r, _d, _n = _propagate(ctx, traj, np.zeros((n, len(params))))
    dPdp = S_all[-1]
    dx = np.linalg.solve(np.eye(n) - res.monodromy, dPdp)
    sr = _compute(ctx, traj, dx)
    sr.periodic = True
    sr.counters["n_sens_propagations"] += _n
    res.dPdp = {p.name: dPdp[:, j].copy() for j, p in enumerate(params)}
    res.dx0 = {p.name: dx[:, j].copy() for j, p in enumerate(params)}
    res.sensitivity = sr
    return res
