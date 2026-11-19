"""add-kernel-simulation-object Stage A: what one ``simulate()`` call
will do, resolved ONCE above the engine dispatch.

``simulate()`` is ``_plan_run`` (keywords -> :class:`SimPlan`), then
``_run_pwl`` / ``_run_trbdf2`` / ``_run_dsed`` (each reads only the
plan), then ``_finalize`` (the result's handles and warnings). A keyword,
an initial state, a route or a drive that is resolved below the dispatch
applies to one engine only: that is the defect class this layout exists
to prevent (see docs/internals/run-planning.md).

The plan is frozen; the objects it holds (the builder, the control plan,
the user's callables) are not, and Stage A keeps today's order of every
side effect on them (DECISIONS 23).

Not ``pulsim.RunPlan``: that public name is the circuit record a result
carries (``result.plan``: circuit, ties, preflight); this plan holds it as
``circuit_plan``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional


@dataclass(frozen=True)
class Diagnostic:
    """One planning fact a caller may want to read (Stage B routes the
    warnings through these; Stage A records the route's reasons)."""
    code: str
    message: str


@dataclass(frozen=True)
class NewtonOptions:
    max_iterations: Optional[int] = None
    tol_dx: Optional[float] = None
    tol_res: Optional[float] = None
    line_search: Optional[bool] = None
    lm: Optional[bool] = None
    roundoff_budget: Optional[int] = None
    nonlinear_refresh: Optional[bool] = None


@dataclass(frozen=True)
class EventOptions:
    max_iterations: Optional[int] = None
    strict: bool = False
    mode_selection: Optional[str] = None
    mode_auto: Optional[dict] = None   # the 'auto' record, when resolved
    on_impulse: str = "record"
    discontinuity_restart: Optional[bool] = None
    substep_state_correction: Optional[bool] = None


@dataclass(frozen=True)
class FixedStep:
    """engine 'pwl': the grid, and the kernel options it resolved."""
    dt: float
    store_every: Optional[int]
    max_dt_halvings: Optional[int]
    run_options: Any            # pulsim.RunOptions (sim::RunOptions)
    via_simulation: bool        # through sim::Simulation (else the raw driver)


@dataclass(frozen=True)
class VariableStep:
    """engine 'trbdf2': dt is the step ceiling (None: the kernel picks)."""
    dt: Optional[float]
    rtol: Optional[float]
    atol: Optional[float]
    dt_init: Optional[float]
    integrator: Optional[str]
    run_options: Any


@dataclass(frozen=True)
class ExactStep:
    """engine 'dsed' (canonical 'exact'): dt is the step cap dt_max."""
    dt: Optional[float]
    rtol: Optional[float]
    atol: Optional[float]
    dt_init: Optional[float]
    integrator: Optional[str]
    stiffness_threshold: Optional[float]
    h_bdf2: Optional[float]
    sensitivities: Any
    allow_implicit_segments: Optional[bool]


@dataclass(frozen=True)
class Initial:
    """Where the run starts: an explicit / builder (i0=, c0=) state, a
    snapshot, or the DC operating point (pwl)."""
    state: Any = None
    from_builder: bool = False
    resume_from: Any = None
    start_from_dc_op: bool = False


@dataclass(frozen=True)
class Drive:
    """What drives the circuit, composed above the dispatch (the control
    plan's gates, graphs and ticks; the user's callables merged into it).
    ``user_*`` are the callables as passed (motor-trace attachment walks
    them); the others are the composed ones the engine receives."""
    control: Any = None
    switch_fn: Any = None
    b_extra_fn: Any = None
    step_observer: Any = None
    should_continue: Any = None
    closed_loops: Any = None
    user_step_observer: Any = None
    user_b_extra_fn: Any = None
    user_closed_loops: Any = None
    mmc_arms: Any = None
    on_cache: Any = None
    live_stream: Any = None
    progress: Any = False


@dataclass(frozen=True)
class SimPlan:
    """One run, resolved. ``resolved_from`` has one entry per
    ``simulate()`` keyword: 'explicit' | 'solver' | 'yaml' | 'default'."""
    builder: Any
    caller_builder: Any
    circuit: Any                     # the run's pulsim.Circuit (None: a stand-in)
    circuit_plan: Any                # pulsim.RunPlan or None
    preflight: Any
    kernel_run: Any                  # sweep(circuit, ...): the run already made
    recipe: Any                      # sweep(circuit, ...): the recipe to fill
    engine: str
    engine_asked: str
    route_reasons: tuple
    features: frozenset
    t_start: float
    t_end: float
    step: Any                        # FixedStep | VariableStep | ExactStep
    newton: NewtonOptions
    events: EventOptions
    initial: Initial
    drive: Drive
    device_path: str
    voltage_sanity_factor: Optional[float]
    auto_regularize: bool
    diagnostics: tuple = ()
    resolved_from: Mapping[str, str] = field(default_factory=dict)
