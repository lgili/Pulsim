"""Bit-identity with the diagnostics OFF: runs the bookkeeping baseline
check (`update-fixed-step-loop-bookkeeping/probes/bookkeeping_baseline.py`)
with every `simulate` and `SimulationOptions` forced to
`resolution_diagnostics=False`.

    python diag_off_check.py BASELINE.npz
"""
import os
import runpy
import sys

import pulsim as p

_sim = p.simulate


def _off(*a, **k):
    k.setdefault("resolution_diagnostics", False)
    return _sim(*a, **k)


p.simulate = _off
_opts_init = p.SimulationOptions.__init__


def _opts(self, *a, **k):
    _opts_init(self, *a, **k)
    self.resolution_diagnostics = False


p.SimulationOptions.__init__ = _opts
here = os.path.dirname(os.path.abspath(__file__))
sys.argv = ["bookkeeping_baseline.py", "check", sys.argv[1]]
runpy.run_path(os.path.join(here, "..", "..", "update-fixed-step-loop-bookkeeping",
                            "probes", "bookkeeping_baseline.py"), run_name="__main__")
