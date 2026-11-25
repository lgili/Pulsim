#!/usr/bin/env python3
"""The variable-step engine's results against a STORED baseline (the
comparison is ``pwl_identity``'s: digests of every array bit for bit and
every stats key, recorded at the base commit, compared at the tip).

    trbdf2_identity.py record|compare DIR --goldens   # golden_baseline.py's trbdf2 cases
    trbdf2_identity.py record|compare DIR --reference [--circuits GLOB,...] [--modes '',lcp,shadow]
    trbdf2_identity.py record|compare DIR --extra     # examples, stateful, ticks + monitor,
                                                       # resume (two split points)

The refactor-unified-transient-stepper identity gate since the legacy loops
were deleted (DECISIONS 20). Bit identity is a LOCAL gate (this machine,
this build).
"""

from __future__ import annotations

import argparse
import fnmatch
import sys
import types
from pathlib import Path

import pulsim as p

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(HERE.parent)]
import pwl_identity as pt  # noqa: E402


def golden_cases():
    sys.path[:0] = [str(ROOT / "openspec/changes/update-linear-time-assembly/probes")]
    import golden_baseline as g  # noqa: PLC0415
    for name, (factory, kw) in g.CASES.items():
        if kw.get("engine") != "trbdf2":
            continue

        def fn(factory=factory, kw=kw):
            b, sw = factory()
            return p.simulate(b, switch_fn=sw, **kw)
        yield name, fn


def reference_cases(patterns, modes):
    import differential_runner as dr  # noqa: PLC0415
    import reference_status as rs  # noqa: PLC0415
    import v1_reader  # noqa: PLC0415
    eng, kw0 = rs.CONFIGS["trbdf2"]
    for rel, entry in sorted(dr.load_triage().items()):
        if entry["status"] != "runs" or not any(fnmatch.fnmatch(rel, q) for q in patterns):
            continue
        case = v1_reader.load(ROOT / "benchmarks" / rel)
        for mode in modes:
            kw = dict(kw0, mode_selection=mode) if mode else dict(kw0)

            def fn(case=case, kw=kw):
                with rs.extra_kwargs(kw):
                    return case.run(eng)
            yield f"trbdf2{'+' + mode if mode else ''} {rel}", fn


def _buck():
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "in", "gnd", 24.0)
    b.add_switch("Q", "in", "sw", 1e2, 1e-9)
    b.add_diode("D", "gnd", "sw", 1e2, 1e-9)
    b.add_inductor("L", "sw", "out", 1e-3)
    b.add_capacitor("C", "out", "gnd", 47e-6)
    b.add_resistor("R", "out", "gnd", 10.0)
    return b


def _pwm(b):
    return p.make_pwm_switch_fn(20e3, 0.4, b.switch_index_of("Q"), b.graph.num_switches)


def _ticks_and_monitor():
    b = _buck()
    lp = p.control.bind_pi_to_switch(
        b, pi=p.PIController(Kp=0.02, Ki=20.0, output_min=0.05, output_max=0.95),
        measured=lambda x, i=b.node_id_of("out"): x[i], setpoint=12.0,
        switch="Q", freq=20e3)
    seen = []
    r = p.simulate(b, 3e-3, engine="trbdf2", closed_loops=[lp],
                   step_observer=p.control.Monitor(lambda t, x: seen.append(t), 1e-4 / 3))
    return types.SimpleNamespace(result=pt.fingerprint(r), monitor_seen=seen)


def _resume(split, t_end=2e-3):
    b = _buck()
    a = p.simulate(b, split, engine="trbdf2", switch_fn=_pwm(b))
    r = p.simulate(b, t_end, engine="trbdf2", switch_fn=_pwm(b),
                   resume_from=a.final_snapshot)
    return types.SimpleNamespace(first=pt.fingerprint(a), second=pt.fingerprint(r))


def extra_cases():
    sys.path[:0] = [str(ROOT / "python")]
    from tests import test_trbdf2_stateful_with_diodes as sd  # noqa: PLC0415
    from tests import test_trbdf2_turnoff_invariant as ti  # noqa: PLC0415
    for kind in ("linear", "atan", "gapped"):
        yield f"stateful rectifier {kind}", (
            lambda kind=kind: p.simulate(sd._rectifier(kind), t_end=3e-3, engine="trbdf2",
                                         dt=4e-6, rtol=1e-6, atol=1e-9))
    for name, make in sorted(ti.CASES.items()):
        def fn(make=make):   # test_trbdf2_turnoff_invariant's settings
            b, sw = make()
            kw = (dict(rtol=2e-5, atol=2e-8, t_end=40e-3) if sw is None else
                  dict(rtol=1e-10, atol=1e-13, t_end=1e-3, dt=ti.ts.T / 100))
            return p.simulate(b, engine="trbdf2", switch_fn=sw, **kw)
        yield f"turnoff {name}", fn
    yield "ticks + monitor (closed-loop buck)", _ticks_and_monitor
    for path in sorted((ROOT / "examples").glob("*.yaml")):
        def fn(path=path):
            lc = p.load_yaml_file(str(path))
            off = p.SwitchStateMask(lc.builder.graph.num_switches)
            return p.simulate(lc, t_end=min(lc.t_end, lc.t_start + 2e-3),
                              engine="trbdf2", switch_fn=lambda t: off)
        yield f"example {path.name}", fn
    # 1.02 ms + 1 ns: one nanosecond after a turn-off edge (the run ends
    # inside the post-edge restart); 1.0333 ms: between edges.
    for split in (1.02e-3 + 1e-9, 1.0333e-3):
        yield f"resume split={split:.7g}", (lambda split=split: _resume(split))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("mode", choices=("record", "compare"))
    ap.add_argument("dir", type=Path)
    ap.add_argument("--goldens", action="store_true")
    ap.add_argument("--reference", action="store_true")
    ap.add_argument("--extra", action="store_true")
    ap.add_argument("--circuits", default="*")
    ap.add_argument("--modes", default="")
    a = ap.parse_args(argv)
    cases = []
    if a.goldens:
        cases += list(golden_cases())
    if a.reference:
        cases += list(reference_cases(a.circuits.split(","), a.modes.split(",")))
    if a.extra:
        cases += list(extra_cases())
    return pt.record_or_compare(a.mode, a.dir / "trbdf2_identity.json", cases)


if __name__ == "__main__":
    sys.exit(main())
