#!/usr/bin/env python3
"""Golden checkpoints of the fixed-step engine (``engine='pwl'``).

Per circuit: exact integers (``n_steps``, ``n_commutations``) and a sha256
of the stored switch-state record, compared EXACTLY; the full state vector
at 8 instants, compared with ``|x - g| <= atol + rtol*|g|`` where ``atol``
is ``rtol`` times the largest magnitude of that state over the 8
checkpoints (a roundoff-level value in a large-scale state is not a
regression). ``rtol`` caps: 1e-9 without Newton devices, 1e-6 with them.
A bit-exact hash of floating-point output is never the key.

    goldens.py                    # compare, print the failures
    goldens.py --update-goldens   # rewrite benchmarks/goldens/pwl/*.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
import v1_reader  # noqa: E402
from analytic import BY_ID  # noqa: E402

GOLDEN_DIR = ROOT / "goldens" / "pwl"
N_CHECKPOINTS = 8
RTOL_LINEAR, RTOL_NEWTON = 1e-9, 1e-6

# v1 suite circuits pinned on pwl: one or more per topology family, both
# Level-1 Newton devices the reader maps. No closed loop (the control-tick
# change moves them) and nothing whose result depends on dsed.
TRIAGED = [
    "circuits/linear/rc_step.yaml",
    "circuits/linear/rlc_series_resonant.yaml",
    "circuits/linear/current_source_rc.yaml",
    "circuits/linear/two_stage_rc_lowpass.yaml",
    "circuits/diodes/diode_bridge_rectifier.yaml",
    "circuits/diodes/voltage_doubler.yaml",
    "circuits/diodes/back_to_back_diodes_sine.yaml",
    "circuits/magnetics/coupled_inductor_step.yaml",
    "circuits/magnetics/transformer_step_up.yaml",
    "circuits/magnetics/half_bridge_transformer_dc.yaml",
    "circuits/motors/motor_dc_brush_step_load.yaml",
    "circuits/motors/motor_pmsm_dq_open_loop.yaml",
    "circuits/resonance/driven_lc_series_resonance.yaml",
    "circuits/stress/stiff_rlc.yaml",
    "circuits/stress/periodic_rc_pwm.yaml",
    "circuits/switching/basic/mosfet_nmos_dc.yaml",
    "circuits/switching/buck/buck_mosfet_nonlinear.yaml",
    "circuits/switching/buck/buck_switching.yaml",
    "circuits/switching/buck/synchronous_buck.yaml",
    "circuits/switching/boost/boost_classic.yaml",
    "circuits/switching/buck_boost/cuk_converter.yaml",
    "circuits/switching/isolated/flyback_converter.yaml",
    "circuits/switching/inverter/half_bridge_inverter.yaml",
    "circuits/switching/multilevel/flying_capacitor_3level.yaml",
    "circuits/three_phase/three_phase_diode_rectifier.yaml",
]


def cases():
    """(golden name, run() -> result, state names, has Newton device)."""
    for cid, c in BY_ID.items():
        names = c.build(1.0)[0].state_var_names()
        yield f"analytic__{cid}", (lambda c=c: c.run("pwl")), names, cid == "shockley_op"
    for rel in TRIAGED:
        case = v1_reader.load(ROOT / rel)
        names = case.build()[0].state_var_names()
        newton = any("Newton" in n for n in case.notes)
        yield Path(rel).stem, (lambda case=case: case.run("pwl")), names, newton


def snapshot(res, names, newton):
    t = np.asarray(res.times, dtype=float)
    x = np.asarray(res.states, dtype=float)
    sw = np.ascontiguousarray(np.asarray(res.switch_states, dtype=bool))
    idx = np.unique(np.linspace(0, len(t) - 1, N_CHECKPOINTS + 1).round().astype(int)[1:])
    return {
        "n_steps": int(len(t)),
        "n_commutations": int(len(res.commutation_events)),
        "switch_digest": "sha256:" + hashlib.sha256(
            np.packbits(sw, axis=None).tobytes() + repr(sw.shape).encode()).hexdigest(),
        "checkpoints": [{"t": float(t[i]), "x": {n: float(x[i, k]) for k, n in enumerate(names)}}
                        for i in idx],
        "rtol": RTOL_NEWTON if newton else RTOL_LINEAR,
    }


def compare(golden: dict, got: dict) -> list:
    """Failure messages, empty when the run matches the golden."""
    for key in ("n_steps", "n_commutations", "switch_digest"):
        if golden[key] != got[key]:
            return [f"{key}: golden {golden[key]} != {got[key]}"]
    rtol = golden["rtol"]
    scale = {n: max(1e-6, *(abs(cp["x"][n]) for cp in golden["checkpoints"]))
             for n in golden["checkpoints"][0]["x"]}
    bad = []
    for g, h in zip(golden["checkpoints"], got["checkpoints"]):
        for n, gv in g["x"].items():
            hv = h["x"][n]
            if abs(hv - gv) > rtol * scale[n] + rtol * abs(gv):
                bad.append(f"t={g['t']:.6g} {n}: golden {gv!r} got {hv!r} "
                           f"(rel diff {abs(hv - gv) / max(abs(gv), 1e-300):.3e}, rtol {rtol:g})")
    return bad


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--update-goldens", action="store_true")
    args = ap.parse_args(argv)
    warnings.simplefilter("ignore")
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    fails = 0
    for name, run, names, newton in cases():
        got = snapshot(run(), names, newton)
        path = GOLDEN_DIR / f"{name}.json"
        if args.update_goldens:
            got["generated_on"] = f"{platform.system()}-{platform.machine()}"
            path.write_text(json.dumps(got, indent=1) + "\n", encoding="utf-8")
            continue
        bad = compare(json.loads(path.read_text(encoding="utf-8")), got)
        fails += bool(bad)
        print(f"{'FAIL' if bad else 'ok  '} {name}" + "".join(f"\n     {b}" for b in bad[:5]))
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
