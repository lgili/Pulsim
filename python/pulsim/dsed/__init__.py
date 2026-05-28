"""Pulsim Path-Based Event-Driven (PED) engine — Python facade.

Path-Based Event-Driven simulation is Pulsim's alternative to the
default fixed-step trapezoidal + PWL cache loop. The PED engine:

* Predicts the next event time analytically (gate edge, diode
  commutation, voltage threshold, custom predicate) — no aliasing.
* Integrates with adaptive step size between events (DOPRI5 + PI
  controller).
* Handles mask transitions instantaneously at predicted times.

Captured benchmark (Gate 1 / Gate 2 — see notes/GATE1_RESULTS.md
and notes/GATE2_PROGRESS.md):

* Buck CCM 24V→12V, 100 kHz, 5 ms window:
    RMSE on V_out = 0.0057 % vs fixed-step trapezoidal reference
    Wall-clock = 0.46× of fixed-step trap (1.85× speedup asymptote)

The current ``v2.0.0`` PED Python facade exposes:

* :class:`PEDSimulator` — outer scheduler loop (the main entry).
* :class:`PIController` — adaptive step controller (Söderlind 2002).
* :class:`EventPredictor` — predicate registry + Illinois root-finder.
* :func:`illinois` and :func:`brent_fallback` — root-finders.
* :mod:`rk45_dormand_prince` — DOPRI5 with FSAL.

Quick example::

    import numpy as np
    from pulsim.dsed import PEDSimulator, PIController

    class LinearRC:
        '''dx/dt = -x / (R·C) for a discharging RC.'''
        R, C = 1.0, 1.0
        _mask = True
        def current_mask(self): return self._mask
        def set_mask(self, m): self._mask = m
        def rhs(self, t, x): return np.array([-x[0] / (self.R * self.C)])

    sys = LinearRC()
    sf = lambda t: True  # constant mask
    sim = PEDSimulator(sys, sf, controller=PIController(), dt_max=0.1)
    res = sim.simulate(np.array([1.0]), t_end=5.0)
    print(f"x(5) = {res.states[-1][0]:.6f}  (expected exp(-5) = "
          f"{np.exp(-5):.6f})")

For higher-level usage from a :class:`pulsim.CircuitBuilder`, see
:func:`pulsim.simulate` with ``engine='dsed'``.

See also
--------

* ``notes/DSED_FOUNDATIONS.md`` — algorithmic background.
* ``notes/GATE1_RESULTS.md``    — Python prototype validation.
* ``notes/GATE2_PROGRESS.md``   — C++ port + integration progress.
* ``openspec/changes/add-path-based-dsed-engine/`` — full proposal.
* ``core/include/pulsim/dsed/``  — C++23 production headers.
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
from .scheduler_auto import PEDSimulatorAuto, PEDResultAuto, AutoDispatchEventRecord

__all__ = [
    # RK45 PED scheduler (Gate 2)
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
    # BDF2 + stiffness detection primitives (Gate 4)
    "BDF2State",
    "BDF2PIController",
    "bdf2_step",
    "StiffnessDetector",
    "IntegratorChoice",
    # BDF2 PED scheduler (Gate 4.C)
    "PEDSimulatorBDF2",
    # Auto-dispatch RK45↔BDF2 scheduler (Gate 4.D)
    "PEDSimulatorAuto",
    "PEDResultAuto",
    "AutoDispatchEventRecord",
]
