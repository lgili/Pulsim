"""The design KPIs of one steady period from the kernel's period map.

``_pulsim.dsed_period_map`` runs one period of the exact stepper with
the GIL released and returns, per affine observable (a source, inductor
or branch current, a node or branch voltage), the closed-form integrals
``int y``, ``int y^2``, ``int y_a y_b`` and the extrema with their
instants (``core/include/pulsim/dsed/period_integrals.hpp``). This
module turns them into the rows :func:`pulsim.device_loss_summary` and
the evaluator would have produced from a sampled run, so an evaluation
holds the GIL only for scalar arithmetic
(add-design-optimization-workflow, kernel item 3).

The kernel path covers R, L, C, DC voltage sources, switches and PWL
diodes, with ``switch_specs`` as the only loss option (a scalar ``Tj``).
Anything else returns a reason and the evaluator takes the sampled
route; nothing is approximated.
"""

from __future__ import annotations

import math

import numpy as np

_KINDS = ("resistor", "inductor", "capacitor", "voltage_source", "switch",
          "diode")


class Spec:
    """Observables of a circuit for ``dsed_period_map``."""

    def __init__(self):
        self.obs: list = []
        self.pairs: list = []
        self.idx: dict = {}
        self.semis: list = []        # (name, kind, bit)
        self.sources: dict = {}      # name -> V

    def add(self, key, w, bit=-1, g_on=1.0, g_off=1.0, v_on=0.0,
            extrema=False):
        w = [(int(k), float(c)) for k, c in w if int(k) >= 0]
        self.idx[key] = len(self.obs)
        self.obs.append(([k for k, _ in w], [c for _, c in w], int(bit),
                         float(g_on), float(g_off), float(v_on),
                         bool(extrema)))


def spec_for(b, output, loss_options, extra_nodes=(), on_state=()):
    """A :class:`Spec`, or a string naming why the kernel path does not
    apply (the evaluator then samples the period). ``on_state`` names
    devices whose ON-state current (0 while off) joins the observables:
    the conduction tables of the junction fixed point read it."""
    from ._result_views import switch_bits_by_branch
    extra = sorted(k for k, v in loss_options.items()
                   if v and k != "switch_specs")
    if extra:
        return f"loss options {extra} are computed on a sampled period"
    kinds = {c["name"]: c["kind"] for c in b.components()}
    for name, spec in (loss_options.get("switch_specs") or {}).items():
        kind = kinds.get(str(name))
        if kind == "diode" and _diode_recovery_only(spec) is not None:
            raise ValueError(f"switch_specs[{name!r}] names a diode: "
                             + _diode_recovery_only(spec))
        if kind not in ("switch", "diode"):
            return (f"switch_specs entry {name!r} is not a switch or a "
                    "diode by name")
        tj = spec.get("Tj")
        if tj is not None and not isinstance(tj, (int, float)):
            return "a switch_specs Tj that is not a number"
    s = Spec()
    bits = switch_bits_by_branch(b)
    for c in b.components():
        kind, name = c["kind"], c["name"]
        if kind not in _KINDS:
            return f"{kind} {name!r} is not covered by the period integrals"
        p = c["params"]
        nodes = tuple(int(x) for x in c.get("nodes", ()))
        bid = int(c["branch_id"])
        if kind == "voltage_source":
            col = b.pool.branch_var_id_for_source(bid, b.graph)
            s.add(("i", name), [(col, 1.0)])
            s.sources[name] = float(p["V"])
        elif kind == "resistor":
            R = float(p.get("R_ohms", math.nan))
            if not (R > 0 and math.isfinite(R)):
                return f"resistor {name!r} with R = {R!r}"
            s.add(("v", name), [(nodes[0], 1.0), (nodes[1], -1.0)])
        elif kind == "inductor":
            col = b.pool.branch_var_id_for_inductor(bid, b.graph)
            s.add(("i", name), [(col, 1.0)], extrema=True)
        elif kind in ("switch", "diode"):
            w = [(nodes[0], 1.0), (nodes[1], -1.0)]
            s.add(("v", name), w, extrema=True)
            if p.get("branch_form"):
                # its current is a solved unknown in both modes (R_on = 0 legal)
                iw = [(b.pool.branch_var_id_for_switch(bid, b.graph), 1.0)]
                s.add(("i", name), iw, extrema=True)
                if name in on_state:
                    s.add(("ion", name), iw, bits[bid], 1.0, 0.0, extrema=True)
            else:
                g_on, g_off = float(p["g_on"]), float(p["g_off"])
                if not (math.isfinite(g_on) and math.isfinite(g_off)):
                    return f"{kind} {name!r} without finite conductances"
                vf = float(p.get("V_f", 0.0)) if kind == "diode" else 0.0
                s.add(("i", name), w, bits[bid], g_on, g_off, vf, extrema=True)
                if name in on_state:
                    s.add(("ion", name), w, bits[bid], g_on, 0.0, vf,
                          extrema=True)
            s.pairs.append((s.idx[("i", name)], s.idx[("v", name)]))
            s.semis.append((name, kind, bits[bid]))
    for name in ((output,) if output is not None else ()) + tuple(extra_nodes):
        try:
            k = int(b.node_id_of(name))
        except IndexError as e:    # a configuration error, named
            raise KeyError(f"node {name!r} (output= or a Kpi): no such "
                           "node") from e
        s.add(("node", name), [(k, 1.0)], extrema=True)
    # every node's extrema: the implausible-voltage check of a sampled run
    s.n_full = len(b.state_var_names())
    s.n_nodes = int(b.graph.num_nodes)
    for k in range(s.n_nodes):
        s.add(("node#", k), [(k, 1.0)], extrema=True)
    return s


class Period:
    """One kernel period: integrals, extrema and event values."""

    def __init__(self, spec, d, T):
        self.spec = spec
        self.T = float(T)
        self.st = np.asarray(d["stats"], dtype=float)
        self.cross = list(d["cross"])
        self.events = d["events"]
        self.y_left = np.asarray(d["y_left"], dtype=float)
        self.y_right = np.asarray(d["y_right"], dtype=float)
        self.y_start = np.asarray(d["y_start"], dtype=float)
        self.y_end = np.asarray(d["y_end"], dtype=float)
        self.first_mask, self.last_mask = d["first_mask"], d["last_mask"]

    def _j(self, key):
        return self.spec.idx[key]

    def mean(self, key):
        return float(self.st[self._j(key), 0]) / self.T

    def rms(self, key):
        return math.sqrt(max(float(self.st[self._j(key), 1]) / self.T, 0.0))

    def energy2(self, key):
        return float(self.st[self._j(key), 1])

    def p2p(self, key):
        r = self.st[self._j(key)]
        return float(r[2] - r[4])

    def peak_abs(self, key):
        r = self.st[self._j(key)]
        return float(max(abs(r[2]), abs(r[4])))

    def node_extrema_trace(self):
        """``(times, states)``: two full-width rows holding every node's
        largest and smallest value over the period, for
        ``find_implausible_voltage`` (which reads magnitudes per node)."""
        X = np.zeros((2, self.spec.n_full))
        for k in range(self.spec.n_nodes):
            r = self.st[self._j(("node#", k))]
            X[0, k], X[1, k] = r[2], r[4]
        return [0.0, self.T], X

    def edges(self, name, bit):
        """``(closed, v, i)`` at the left and right limit of every edge of
        a switch bit in ``[0, T)``: the section edge (``T-`` to ``0+``)
        first, then the events strictly inside the period (an event at
        ``T`` is the section edge)."""
        jv, ji = self._j(("v", name)), self._j(("i", name))
        tol = 1e-12 * self.T
        seq = []
        lb, fb = bool(self.last_mask.get(bit)), bool(self.first_mask.get(bit))
        if lb != fb:
            seq += [(lb, self.y_end[jv], self.y_end[ji]),
                    (fb, self.y_start[jv], self.y_start[ji])]
        for k, e in enumerate(self.events):
            o, n = bool(e["old_mask"].get(bit)), bool(e["new_mask"].get(bit))
            if o != n and tol < float(e["t"]) < self.T - tol:
                seq += [(o, self.y_left[k, jv], self.y_left[k, ji]),
                        (n, self.y_right[k, jv], self.y_right[k, ji])]
        for a, c in zip(seq[1::2], seq[2::2]):
            if a[0] != c[0]:
                raise RuntimeError(f"period map: the edges of {name!r} do "
                                   "not alternate")
        return seq


def loss_rows(b, kp, loss_options):
    """``{name: row}`` with the fields of :func:`device_loss_summary`
    the evaluator reads (``kind``, ``P_avg``, ``i_peak``, ``i_rms``,
    ``P_sw_avg`` and the switching bookkeeping)."""
    from .losses import _edge_law, _switch_switching_loss
    T = kp.T
    specs = loss_options.get("switch_specs") or {}
    comps = {c["name"]: c for c in b.components()}
    rows = {}
    for name, c in comps.items():
        if c["kind"] == "resistor":
            rows[name] = {"kind": "resistor", "name": name,
                          "P_avg": kp.energy2(("v", name))
                          / float(c["params"]["R_ohms"]) / T}
    for name, kind, bit in kp.spec.semis:
        q = kp.spec.pairs.index((kp.spec.idx[("i", name)],
                                 kp.spec.idx[("v", name)]))
        row = {"kind": kind, "name": name, "P_avg": kp.cross[q] / T,
               "i_peak": kp.peak_abs(("i", name)),
               "i_rms": kp.rms(("i", name))}
        spec = specs.get(name)
        if spec is not None:
            seq = kp.edges(name, bit)
            if not seq:
                seq = [(bool(kp.first_mask.get(bit)), 0.0, 0.0)]
            law = _edge_law(kind, comps[name]["params"])
            sw = _table_switching_loss(seq, spec, law)
            if sw is None and kind == "diode":
                raise RuntimeError(f"switch_specs[{name!r}]: a non-finite "
                                   "commutation point of a diode")
            if sw is not None:
                sw["P_sw_avg"] = sw["E_sw_total"] / T
                sw["f_sw_estimate"] = (sw["n_turn_on_events"]
                                       + sw["n_turn_off_events"]) / (2.0 * T)
                row.update(sw)
                rows[name] = row
                continue
            cl = np.array([s[0] for s in seq], dtype=bool)
            v = np.array([s[1] for s in seq], dtype=float)
            i = np.array([s[2] for s in seq], dtype=float)
            # the same energy model, read at the exact edge limits
            sw = _switch_switching_loss(cl, np.zeros(cl.size), v, i, spec,
                                        law=law, name=name)
            sw["P_sw_avg"] = sw["E_sw_total"] / T
            sw["f_sw_estimate"] = (sw["n_turn_on_events"]
                                   + sw["n_turn_off_events"]) / (2.0 * T)
            row.update(sw)
        rows[name] = row
    return rows


def _table_switching_loss(seq, spec, law):
    """:func:`pulsim.losses._switch_switching_loss` on an edge sequence
    for a spec whose energies are all 3-D tables at a scalar ``Tj``,
    in scalars: the same edges, the same limits (turn-on: blocking ``v``
    before, load ``i`` after; turn-off: the reverse), each point read by
    the kernel table and summed as numpy sums them -- bit for bit (a
    test compares the two). ``None`` for any other spec, or a
    non-finite point (the general path refuses it)."""
    from .loss_tables import LossTable
    on_t, off_t, tj = (spec.get("E_on_table"), spec.get("E_off_table"),
                       spec.get("Tj"))
    if (on_t is None and off_t is None) or type(tj) not in (int, float) \
            or not math.isfinite(tj):
        return None
    for t, ref, curve in ((on_t, "E_on_ref", "E_on_curve"),
                          (off_t, "E_off_ref", "E_off_curve")):
        if t is None and (float(spec.get(ref, 0.0)) > 0
                          or spec.get(curve) is not None):
            return None        # a mixed spec: the general path
        if t is not None and not isinstance(t, LossTable):
            return None
    on, off = commutations(seq, law)
    tj = float(tj)

    def total(table, pts):
        if table is None or not pts:
            return 0.0
        if not all(math.isfinite(v) and math.isfinite(i) for v, i, _ in pts):
            raise _NonFinite
        f = table._kernel()
        e = [f(v, i, tj) for v, i, soft in pts if not soft]
        if len(e) < 8:         # numpy's pairwise sum is a plain loop here
            acc = 0.0
            for x in e:
                acc += x
            return float(acc)
        return float(np.sum(e))
    try:
        e_on, e_off = total(on_t, on), total(off_t, off)
    except _NonFinite:
        return None
    return {"E_sw_on_total": e_on, "E_sw_off_total": e_off,
            "E_sw_total": e_on + e_off, "n_turn_on_events": len(on),
            "n_turn_off_events": len(off)}


class _NonFinite(Exception):
    pass


def commutations(seq, law):
    """``(on, off)``: the ``(|v|, |i|, soft)`` of every turn-on and
    turn-off of an edge sequence (:meth:`Period.edges`), by the reading
    rule -- a turn-on reads the blocking ``v`` before and the load ``i``
    after, a turn-off the reverse; with ``law = (gated, g_on, g_off)``
    (``losses._edge_law``) a switch's edge whose two readings have
    opposite signs, the current above ``(g_off + eps g_on)|v|``, is soft
    (no energy). The rule is the
    kernel's (``thermal::commutation_point``), the one the transient
    engine reads with ``set_device_losses``: one implementation for both
    routes."""
    from ._pulsim import _commutation_points
    return _commutation_points([(bool(c), float(v), float(i))
                                for c, v, i in seq], bool(law[0]),
                               float(law[1]), float(law[2]))


def _diode_recovery_only(spec):
    """Why a ``switch_specs`` entry cannot be a diode's recovery table,
    or None: a diode has no commanded turn-on, so only ``E_off_table``
    (its recovery energy, read by the turn-off row) and ``Tj``."""
    from .loss_tables import LossTable
    extra = sorted(set(spec) - {"E_off_table", "Tj"})
    if extra or not isinstance(spec.get("E_off_table"), LossTable):
        return ("a diode's entry takes E_off_table (a LossTable: the "
                "recovery energy, read at each turn-off) and Tj only"
                + (f"; got {extra}" if extra else ""))
    return None
