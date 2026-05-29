"""Python reference implementation of the DSED engine — TEST-ONLY.

This package is the legacy Python implementation of the Path-Based
Event-Driven scheduler (PED). As of v1.6.3 it is **no longer part of
the public Pulsim API** — the production hot path is the C++ kernel
exposed via :func:`pulsim.simulate` with ``engine='dsed'``, and
users who need a custom-LTI escape hatch should use
:func:`pulsim.dsed.run_user_lti`.

The Python implementation is kept inside the test tree to support
two ongoing needs that the C++ port cannot satisfy alone:

1. **Cross-validation.** Running the same circuit through the C++
   scheduler and the Python reference and asserting bit-tolerant
   agreement catches binding bugs and algorithm drift between the
   two ports.
2. **Reference for documentation.** The "How Pulsim Works" Part II
   chapters (11–16) describe the DSED algorithm in step-by-step
   detail through this code. Removing it would orphan that pedagogy.

Layout mirrors what used to live under ``pulsim/dsed/``:

* ``rk45_dormand_prince`` — embedded 5(4) RK with Hermite interp.
* ``step_controller``     — PI step-size controller.
* ``event_predictor``     — Illinois / Brent root finder + cache.
* ``bdf2_integrator``     — implicit BDF2 with Crank–Nicolson bootstrap.
* ``stiffness_detector``  — eigenvalue-spread heuristic.
* ``scheduler``           — :class:`PEDSimulator` (RK45-only).
* ``scheduler_bdf2``      — :class:`PEDSimulatorBDF2` (BDF2-only).
* ``scheduler_auto``      — :class:`PEDSimulatorAuto` (RK45 ↔ BDF2).

**Do not depend on this package from production code.** It is
importable from tests via::

    from tests._dsed_reference import PEDSimulator, PIController

(or equivalent absolute import depending on the test layout).
The public Pulsim API never imports anything from here.
"""

from .rk45_dormand_prince import RK45State, step as rk45_step, interpolate
from .step_controller import PIController
from .event_predictor import (
    EventPredictor,
    EventPredicate,
    illinois,
    brent_fallback,
)
from .scheduler import PEDSimulator, PEDResult, EventRecord
from .bdf2_integrator import BDF2State, BDF2PIController, bdf2_step
from .stiffness_detector import StiffnessDetector, IntegratorChoice
from .scheduler_bdf2 import PEDSimulatorBDF2
from .scheduler_auto import (
    PEDSimulatorAuto,
    PEDResultAuto,
    AutoDispatchEventRecord,
)

__all__ = [
    # RK45 PED scheduler.
    "PEDSimulator",
    "PEDResult",
    "EventRecord",
    "PIController",
    "EventPredictor",
    "EventPredicate",
    "RK45State",
    "rk45_step",
    "interpolate",
    "illinois",
    "brent_fallback",
    # BDF2 + stiffness detection primitives.
    "BDF2State",
    "BDF2PIController",
    "bdf2_step",
    "StiffnessDetector",
    "IntegratorChoice",
    # BDF2 PED scheduler.
    "PEDSimulatorBDF2",
    # Auto-dispatch RK45 ↔ BDF2 scheduler.
    "PEDSimulatorAuto",
    "PEDResultAuto",
    "AutoDispatchEventRecord",
]
