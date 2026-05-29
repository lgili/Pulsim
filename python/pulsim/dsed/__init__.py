"""Path-Based Event-Driven (DSED) public surface.

The hot path is the C++ scheduler exposed via :func:`pulsim.simulate`
with ``engine='dsed'``. For workflows that need to drive the
scheduler from a hand-rolled LTI system (custom physics, system-ID
benchmarks, FMU integrations), :func:`run_user_lti` is the
documented escape hatch — a thin wrapper around the native
``run_ped_native`` / ``run_bdf2_native`` / ``run_auto_native``
scheduler entry points that validates the system protocol at the
call site so the user gets a clear ``TypeError`` instead of a
cryptic pybind11 dispatch error.

The submodule also exposes :class:`CircuitBuilderAdapter` for
advanced users who want the same `(A, b)` extraction logic the
dispatcher uses internally (Bridge.10 path) — most users should
prefer :func:`pulsim.simulate` instead.

**Removed in v1.6.3:** the pure-Python schedulers (``PEDSimulator``,
``PEDSimulatorBDF2``, ``PEDSimulatorAuto``, ``PIController``,
``EventPredictor``, ``BDF2State``, ``RK45State``, ``StiffnessDetector``,
and helper functions) were moved to the test tree as a cross-validation
reference. They are no longer importable from ``pulsim.dsed``. The
canonical user-facing replacement is :func:`run_user_lti`; if you were
sub-classing the Python schedulers for custom step controllers /
integrators, please open an issue describing the use case so we can
expose the right extension hooks from the C++ side.

See ``docs/how-pulsim-works/11-dsed-engine-overview.md`` for
algorithm details.
"""
from ._builder_bridge import CircuitBuilderAdapter
from .run_user_lti import run_user_lti

__all__ = [
    "run_user_lti",
    "CircuitBuilderAdapter",
]
