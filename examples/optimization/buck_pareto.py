"""Efficiency against box volume of a synchronous buck: a Pareto front.

48 V -> 12 V (D = 0.25, open loop), 1.2 ohm load (10 A), over the
switching frequency, the inductance, the output capacitance and a
choice of two MOSFETs. Every design is one call of
``pulsim.design.Evaluator.evaluate``: the periodic steady state by
Newton shooting on the exact stepper, the KPIs on one steady period.

READ THIS BEFORE QUOTING A NUMBER FROM IT
-----------------------------------------
* The two MOSFETs are SYNTHETIC ILLUSTRATIONS ("SYN-A", "SYN-B"):
  plausible on-resistances and straight-line switching energies made
  up for this example, flagged ``provenance.method: synthetic``. They
  are not vendor data and no published part has them. Pulsim's own
  shipped parts are synthetic too (``pulsim.lib.part`` refuses them by
  number unless ``allow_synthetic=True``). For real numbers import the
  vendor's PLECS thermal file (``pulsim.lib.import_plecs_xml``).
* The volume law is ILLUSTRATIVE, not a magnetics or thermal design:
  inductor ``k_L (L I_pk I_rms)^(3/4)`` (area-product scaling),
  capacitor ``k_C C V^2``, heatsink ``k_hs / R_th,sa`` with the
  largest ``R_th,sa`` that keeps ``T_j <= 125 C`` at ``T_amb = 40 C``.
  The constants below are assumptions, stated as such.
* The switches are resistor pairs with ``R_on`` at 125 C; the
  high-side switching energy comes from the parts' tables read at
  125 C; the low-side switch commutates at near-zero voltage in a
  synchronous buck and its switching loss is neglected (assumption).
  There is no dead time and no body diode.

What IS checked: ``python/tests/test_example_buck_pareto.py`` runs 32
designs and checks the status counts, that the front is non-empty
and that every member satisfies the constraints. The printed front
of a 512-design run is an OBSERVATION of this model.

    python examples/optimization/buck_pareto.py [n_designs] [--lhs] [--process]

``--process`` evaluates the Latin-hypercube sample on a process pool
(one worker per core but one): the evaluators pickle.
"""

from __future__ import annotations

import functools
import sys

import numpy as np

import pulsim as p
from pulsim.design import (Constraint, Evaluator, Kpi, SteadySpec,
                           as_constraints, as_objectives)
from pulsim.lib.parts import part_from_mapping

VIN, D, R_LOAD = 48.0, 0.25, 1.2
T_J_MAX, T_AMB = 125.0, 40.0
RIPPLE_MAX = 0.01 * 12.0                 # 1 % of 12 V
DCR = 3e-3                               # inductor resistance (assumption)

# Volume constants: ASSUMPTIONS for this illustration.
K_L = 3.0e-4     # m^3 / J^0.75: ~2 cm^3 for 10 uH at 12 A peak, 10 A rms
K_C = 7.0e-5     # m^3 / (F V^2): ~1 cm^3 for 100 uF at 12 V
K_HS = 3.0e-4    # m^3 K / W: ~300 cm^3 for a 1 K/W natural-convection sink
R_TH_SA_MIN = 0.05   # K/W: below this no plausible sink exists (assumption)

BOUNDS = {"f_sw": (50e3, 500e3), "L": (2.2e-6, 100e-6),
          "C": (22e-6, 470e-6)}
PART_NAMES = ("SYN-A", "SYN-B")


def _synthetic_mosfet(name, r_on_25, e_on, e_off, r_jc):
    """A made-up 100 V MOSFET: E = e * (v/48) * (i/10) * (1 + 0.003 (Tj-25))."""
    v, i, tj = [24.0, 48.0, 80.0], [0.0, 10.0, 100.0], [25.0, 125.0]

    def table(e):
        return [e * (vv / 48.0) * (ii / 10.0) * (1 + 0.003 * (t - 25.0))
                for vv in v for ii in i for t in tj]
    return part_from_mapping({
        "class": "mosfet", "vendor": "Illustration", "part": name,
        "provenance": {"source": "synthetic", "method": "synthetic",
                       "source_ref": "examples/optimization/buck_pareto.py",
                       "retrieved": "2026-11-13",
                       "note": "made up for this example; not vendor data"},
        "R_ds_on_25c": r_on_25, "R_ds_on_temp_coef": 0.005,
        "R_th_jc": r_jc,
        "Eon_table": {"v": v, "i": i, "tj": tj, "energy": table(e_on)},
        "Eoff_table": {"v": v, "i": i, "tj": tj, "energy": table(e_off)},
    }, f"synthetic {name}")


# Low on-resistance, large die (slow, lossy edges) against the reverse.
PARTS = {
    "SYN-A": _synthetic_mosfet("SYN-A", 4e-3, 24e-6, 12e-6, 0.5),
    "SYN-B": _synthetic_mosfet("SYN-B", 10e-3, 8e-6, 4e-6, 1.0),
}


def r_on(part, t_j):
    c = part.conduction_spec()
    return c["R_on"] * (1.0 + c["R_on_temp_coef"] * (t_j - 25.0))


def build(design):
    """The circuit and its gate schedule for one design."""
    part = PARTS[design["part"]]
    g_on = 1.0 / r_on(part, T_J_MAX)
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", VIN)
    b.add_switch("Q1", "vin", "sw", g_on, 1e-9)
    b.add_switch("Q2", "sw", "gnd", g_on, 1e-9)
    b.add_inductor("L", "sw", "x", design["L"])
    b.add_resistor("R_dcr", "x", "vout", DCR)
    b.add_capacitor("C", "vout", "gnd", design["C"])
    b.add_resistor("R_load", "vout", "gnd", R_LOAD)
    sf = p.make_dead_time_pwm_pair_fn(
        design["f_sw"], D, b.switch_index_of("Q1"), b.switch_index_of("Q2"),
        b.graph.num_switches, 0.0)
    return b, sf


def build_circuit(part_name):
    """The circuit form: ONE immutable circuit per part (its switch
    conductance); a design is ``with_value`` of ``L.L`` and ``C.C`` of
    it, shared by every worker thread, with no rebuild."""
    b, _ = build({"part": part_name, "L": 10e-6, "C": 100e-6, "f_sw": 1e5})
    return b.build()


def schedule(design):
    """The gate schedule of a design (``Q1`` is bit 0, ``Q2`` bit 1)."""
    return p.make_dead_time_pwm_pair_fn(design["f_sw"], D, 0, 1, 2, 0.0)


def kpis_of(part_name):
    """The two design KPIs as formulas (``Kpi.formula``): declarative,
    so they pickle and the kernel batch computes them with no Python
    per design (DECISIONS 26).

    ``r_th_sa``: the largest sink-to-ambient resistance that keeps both
    switches at or below 125 C on one sink -- 85 K over the total
    semiconductor loss, minus the hotter device's junction-to-case drop.
    ``volume``: box volume (m^3) of the illustrative law, heatsink sized
    by ``r_th_sa`` (floored at ``R_TH_SA_MIN``: the ``r_th_sa``
    constraint flags a design below it)."""
    r_jc = float(PARTS[part_name].params["R_th_jc"])
    return [
        Kpi.formula("r_th_sa", f"({T_J_MAX!r} - {T_AMB!r} - {r_jc!r} * "
                    "max(loss.Q1, loss.Q2)) / (loss.Q1 + loss.Q2)"),
        Kpi.formula("volume", f"{K_L!r} * (L * i_peak.L * i_rms.L) ** 0.75"
                    f" + {K_C!r} * C * v_out ** 2"
                    f" + {K_HS!r} / max(r_th_sa, {R_TH_SA_MIN!r})"),
    ]


def switching_period(design):
    return 1.0 / design["f_sw"]


def make_evaluator(form="circuit"):
    """One evaluator per part. ``form='circuit'`` (default): one shared
    immutable circuit per part, a design being ``with_value`` of it;
    ``'factory'``: ``build`` per design. Same records. Everything is a
    module-level function or a value (``Constraint``), so the evaluator
    pickles and ``evaluate_many(backend='process')`` runs it (each
    worker builds the part's circuit once). With the circuit form and
    formula KPIs, ``evaluate_many`` runs the kernel batch: every design
    in one GIL-free kernel call, on ``workers`` native threads."""
    t_switch_specs = {
        name: {"Q1": part.switch_spec(Tj=T_J_MAX)} for name, part in
        PARTS.items()}

    def source(name):
        if form == "factory":
            return dict(build=build)
        return dict(circuit=functools.partial(build_circuit, name),
                    fields={"L": "L.L", "C": "C.C"}, schedule=schedule)
    # One evaluator per part: the switch's loss table belongs to the
    # options, so it is part of every record's options digest.
    return {name: Evaluator(
        **source(name), period=switching_period, loads=("R_load",),
        output="vout",
        kpis=kpis_of(name),
        constraints=[Constraint.max("ripple_v", RIPPLE_MAX),
                     Constraint.min("r_th_sa", R_TH_SA_MIN)],
        steady=SteadySpec(method="pss"),
        loss_options={"switch_specs": t_switch_specs[name]})
        for name in PARTS}


def latin_hypercube(n, seed=0):
    """n designs, log-uniform strata in f_sw, L, C; the part alternates
    on a shuffled half/half split."""
    rng = np.random.default_rng(seed)
    out = [dict() for _ in range(n)]
    for key, (lo, hi) in BOUNDS.items():
        u = (rng.permutation(n) + rng.random(n)) / n
        for d, x in zip(out, np.exp(np.log(lo) + u * (np.log(hi) - np.log(lo)))):
            d[key] = float(x)
    parts = rng.permutation(np.arange(n) % 2)
    for d, k in zip(out, parts):
        d["part"] = PART_NAMES[int(k)]
    return out


def evaluate_all(designs, workers=1, form="circuit", backend="thread"):
    evs = make_evaluator(form)
    recs = [None] * len(designs)
    for name, ev in evs.items():
        idx = [i for i, d in enumerate(designs) if d["part"] == name]
        for i, r in zip(idx, ev.evaluate_many([designs[i] for i in idx],
                                              workers=workers,
                                              backend=backend)):
            recs[i] = r
    return recs


OBJECTIVES = ("efficiency", "volume")
CONSTRAINTS = ("ripple_v", "r_th_sa")


def non_dominated(F):
    """Indices of the non-dominated rows of F (all columns minimised)."""
    F = np.asarray(F, dtype=float)
    keep = []
    for i in range(len(F)):
        dom = np.all(F <= F[i], axis=1) & np.any(F < F[i], axis=1)
        if not dom.any():
            keep.append(i)
    return keep


def pareto(records):
    """The feasible non-dominated records (max efficiency, min volume)."""
    ok = [r for r in records if r.status == "ok"]
    F = [as_objectives(r, OBJECTIVES, maximise=("efficiency",)) for r in ok]
    return [ok[i] for i in non_dominated(F)] if ok else []


def status_counts(records):
    return {s: sum(r.status == s for r in records)
            for s in ("ok", "refused", "failed", "infeasible")}


def run_lhs(n=512, seed=0, workers=1, backend="thread"):
    designs = latin_hypercube(n, seed)
    recs = evaluate_all(designs, workers=workers, backend=backend)
    return recs, pareto(recs)


def run_pymoo(n_gen=20, pop=40, seed=0):    # pragma: no cover - optional
    """NSGA-II with pymoo (optional extra ``pulsim[opt]``)."""
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.core.problem import ElementwiseProblem
    from pymoo.optimize import minimize

    evs = make_evaluator()
    lo = np.log([BOUNDS[k][0] for k in BOUNDS])
    hi = np.log([BOUNDS[k][1] for k in BOUNDS])
    records = []

    class Buck(ElementwiseProblem):
        def __init__(self):
            super().__init__(n_var=4, n_obj=2, n_ieq_constr=2,
                             xl=np.r_[lo, 0.0], xu=np.r_[hi, 1.0])

        def _evaluate(self, x, out, *a, **k):
            d = {key: float(np.exp(v)) for key, v in zip(BOUNDS, x[:3])}
            d["part"] = PART_NAMES[int(x[3] >= 0.5)]
            r = evs[d["part"]].evaluate(d)
            records.append(r)
            out["F"] = as_objectives(r, OBJECTIVES, maximise=("efficiency",))
            out["G"] = as_constraints(r, CONSTRAINTS)

    minimize(Buck(), NSGA2(pop_size=pop), ("n_gen", n_gen), seed=seed,
             verbose=False)
    return records, pareto(records)


def report(records, front):
    c = status_counts(records)
    print(f"designs: {len(records)}   " + "   ".join(
        f"{k}: {v}" for k, v in c.items()))
    print("SYNTHETIC parts and an ILLUSTRATIVE volume law: an observation "
          "of this model, not a design recommendation.")
    print(f"front: {len(front)} designs")
    print("  f_sw/kHz   L/uH   C/uF  part   eff/%   volume/cm3  ripple/mV")
    for r in sorted(front, key=lambda r: r.kpis["volume"]):
        d, k = r.design, r.kpis
        print(f"  {d['f_sw'] / 1e3:8.1f} {d['L'] * 1e6:6.1f} {d['C'] * 1e6:6.1f}"
              f"  {d['part']}  {100 * k['efficiency']:6.2f}  "
              f"{k['volume'] * 1e6:10.1f}  {k['ripple_v'] * 1e3:8.1f}")


def plot(front, path="buck_pareto.png"):    # pragma: no cover - optional
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, m in zip(PART_NAMES, "os"):
        pts = [(r.kpis["volume"] * 1e6, 100 * r.kpis["efficiency"])
               for r in front if r.design["part"] == name]
        if pts:
            ax.plot(*zip(*sorted(pts)), m, label=name)
    ax.set_xlabel("box volume / cm^3 (illustrative law)")
    ax.set_ylabel("efficiency / %")
    ax.set_title("SYNTHETIC parts - illustration, not vendor data",
                 color="crimson")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    return path


def main(argv):
    """NSGA-II when pymoo imports (and ``--lhs`` is not given), else a
    Latin-hypercube sample of ``n`` designs (default 512)."""
    n = int(argv[1]) if len(argv) > 1 and argv[1].isdigit() else 512
    try:
        if "--lhs" in argv or "--process" in argv:
            raise ImportError
        import pymoo  # noqa: F401
        records, front = run_pymoo()
    except ImportError:
        records, front = (run_lhs(n, workers=None, backend="process")
                          if "--process" in argv else run_lhs(n))
    report(records, front)
    try:
        print("plot:", plot(front))
    except ImportError:
        pass
    return records, front


if __name__ == "__main__":
    main(sys.argv)
