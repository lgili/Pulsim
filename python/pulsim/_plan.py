"""add-kernel-simulation-object Stage A: what one ``simulate()`` call
will do, resolved ONCE above the engine dispatch.

``simulate()`` is ``_plan_run`` (keywords -> :class:`SimPlan`), then
``_run_pwl`` / ``_run_trbdf2`` / ``_run_dsed`` (each reads only the
plan), then ``_finalize`` (the result's handles and warnings). A keyword,
an initial state, a route or a drive that is resolved below the dispatch
applies to one engine only: that is the defect class this layout exists
to prevent (see docs/internals/run-planning.md).

The plan is immutable: every record here is a ``typing.NamedTuple`` (a
tuple: no ``__dict__``, a field cannot be set or deleted; assigning one
raises ``AttributeError``). Not frozen dataclasses: their ``__init__``
pays one ``object.__setattr__`` per field, ~500 bytecodes per call on a
path parameter sweeps call thousands of times (task 7.2). The objects it
holds (the builder, the control plan, the user's callables) are not
immutable, and Stage A keeps today's order of every side effect on them
(DECISIONS 23).

Not ``pulsim.RunPlan``: that public name is the circuit record a result
carries (``result.plan``: circuit, ties, preflight); this plan holds it as
``circuit_plan``.
"""

from __future__ import annotations

from typing import Any, Iterator, Mapping, NamedTuple, Optional


# The records' constructor on simulate()'s path: ``new(Record, (v, ...))``
# with every field, in order (tuple.__new__: no Python frame, no keyword
# dict; test_run_plan checks the arity of every call). ``Record(...)``
# works too, with keywords and defaults.
new = tuple.__new__


# The fixed-step defaults of discontinuity_restart and of the step split
# (enable_substep_state_correction), resolved in _plan_run when the call
# leaves them None: the restart rule first (OFF under mmc_arms, with a
# diagnostic), then the split, which defaults ON only where the rule is on.
# True since the number-changing release (update-number-changing-release
# task 3.1); discontinuity_restart=False restores the old fixed step.
DEFAULT_DISCONTINUITY_RESTART = True
DEFAULT_SUBSTEP_STATE_CORRECTION = True

# THE run defaults the number-changing release flipped that simulate()
# resolves when the call leaves the keyword None (one place: the plan).
# mode_selection: 'auto' on every engine (mode_selection='legacy' restores
# the settle); device_path: 'interface' on pwl / trbdf2 (device_path=
# 'legacy' restores the hand stamps; the exact engine stamps its own);
# output_rtol: the exact engine's output-grid rule where the engine picks
# the samples (no dt / store_every / recorders / sensitivities, not
# integrator='bdf2'); output_rtol=0 restores the 10 us default grid.
DEFAULT_MODE_SELECTION = "auto"
DEFAULT_DEVICE_PATH = "interface"
DEFAULT_OUTPUT_RTOL = 1e-5

# simulate(undriven=None): the policy for controlled switches nothing drives
# (no switch_fn, no closed loop's gate), read only when the call leaves it
# None. None = today's per-engine behaviour (pwl and the exact engine run
# them all CLOSED with a warning; trbdf2 refuses them by name; 'auto' without
# dt refuses, with dt it is pwl). 'refuse' | 'open' | 'closed' hold on every
# engine and through 'auto'. 'refuse' since the number-changing release
# (add-kernel-simulation-object DECISIONS 26, task 11.3); undriven='closed'
# restores the old fixed-step / exact behaviour (without its warning).
DEFAULT_UNDRIVEN = "refuse"

# The engines' remaining position-dependent rules made uniform, one switch
# (DECISIONS 26): the count checks (store_every, max_event_iterations,
# max_dt_halvings) with one message on every engine that reads them, before
# the mmc_arms combinations; on_impulse='warn'/'error' refused by name where
# the run cannot detect an impulse (the exact engine's integrator='bdf2');
# the exact engine's impulse refusal names the device kind as the kernel's.
# True since the number-changing release (refusal order and messages only:
# no number depends on it; False is the pre-release behaviour, no keyword).
UNIFORM_ENGINE_RULES = True


# plan.features: the keywords that put a feature in the run (set, not
# default), plus "builder_ics" when the builder's i0= / c0= seed the state.
FEATURE_KW = ("switch_fn", "b_extra_fn", "step_observer", "closed_loops",
              "controller_period", "should_continue", "live_stream",
              "mmc_arms", "on_cache", "progress", "initial_state",
              "resume_from", "start_from_dc_op", "sensitivities",
              "store_every")  # (recorders= runs through simulate_with_recorders)


class ResolvedFrom(Mapping):
    """``plan.resolved_from``: keyword -> 'explicit' | 'solver' | 'yaml' |
    'resume_from' | 'default', computed when read, so a run that never
    reads it pays one tuple. ``args``: the values as passed (captured at
    call time), in the order of ``defaults`` (simulate()'s parameters after
    ``builder``). 'explicit' = not the signature's default (an explicit
    default reads 'default'); 'yaml' / 'solver' = filled from a
    LoadedCircuit / SolverOptions; 'resume_from' = ``t_start`` taken from
    the snapshot's clock (`mark`)."""

    __slots__ = ("_args", "_defaults", "_marks", "_d")

    def __init__(self, args: tuple, defaults: dict):
        self._args, self._defaults, self._marks, self._d = args, defaults, {}, None

    def mark(self, key: str, source: str) -> None:
        self._marks[key] = source
        self._d = None

    def _dict(self) -> dict:
        if self._d is None:
            out = {}
            for (k, d), v in zip(self._defaults.items(), self._args,
                                 strict=True):
                out[k] = ("default" if v is d or (
                    d is not None and type(v) is type(d) and v == d)
                    else "explicit")
            out.update(self._marks)
            self._d = out
        return self._d

    def __getitem__(self, key: str) -> str:
        if self._d is None:
            # one key, as _dict() reads it, without building all of them (the
            # exact engine reads 'output_rtol' on every run: ~700 bytecodes)
            src = self._marks.get(key)
            if src is not None:
                return src
            defaults = self._defaults
            if key in defaults and len(defaults) == len(self._args):
                d = defaults[key]
                v = self._args[list(defaults).index(key)]
                return ("default" if v is d or (
                    d is not None and type(v) is type(d) and v == d)
                    else "explicit")
        return self._dict()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._dict())

    def __len__(self) -> int:
        return len(self._dict())

    def __repr__(self) -> str:
        return f"ResolvedFrom({self._dict()!r})"


class Diagnostic(NamedTuple):
    """One planning fact a caller may want to read (Stage B routes the
    warnings through these; Stage A records the route's reasons)."""
    code: str
    message: str


class NewtonOptions(NamedTuple):
    max_iterations: Optional[int] = None
    tol_dx: Optional[float] = None
    tol_res: Optional[float] = None
    line_search: Optional[bool] = None
    lm: Optional[bool] = None
    roundoff_budget: Optional[int] = None
    nonlinear_refresh: Optional[bool] = None


class EventOptions(NamedTuple):
    max_iterations: Optional[int] = None
    strict: bool = False
    mode_selection: Optional[str] = None
    mode_auto: Optional[dict] = None   # the 'auto' record, when resolved
    on_impulse: str = "record"
    discontinuity_restart: Optional[bool] = None
    substep_state_correction: Optional[bool] = None


class FixedStep(NamedTuple):
    """engine 'pwl': the grid, and the kernel options it resolved."""
    dt: float
    store_every: Optional[int]
    max_dt_halvings: Optional[int]
    run_options: Any            # pulsim.RunOptions (sim::RunOptions)
    via_simulation: bool        # through sim::Simulation (else the raw driver)


class VariableStep(NamedTuple):
    """engine 'trbdf2': dt is the step ceiling (None: the kernel picks)."""
    dt: Optional[float]
    rtol: Optional[float]
    atol: Optional[float]
    dt_init: Optional[float]
    integrator: Optional[str]
    run_options: Any


class ExactStep(NamedTuple):
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
    output_rtol: Optional[float]     # the output-grid rule (None: today's samples)


class Initial(NamedTuple):
    """Where the run starts: an explicit / builder (i0=, c0=) state, a
    snapshot, or the DC operating point (pwl)."""
    state: Any = None
    from_builder: bool = False
    resume_from: Any = None
    start_from_dc_op: bool = False


class Drive(NamedTuple):
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


class SimPlan(NamedTuple):
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
    t_start: float
    t_end: float
    step: Any                        # FixedStep | VariableStep | ExactStep
    newton: NewtonOptions
    events: EventOptions
    initial: Initial
    drive: Drive
    device_path: str
    device_path_enum: Any            # pulsim.DevicePath
    voltage_sanity_factor: Optional[float]
    auto_regularize: bool
    resolved_from: Mapping[str, str]
    # None, or (policy, switch names): an undriven= policy that applied
    # (controlled switches nothing drives), decided above the dispatch.
    undriven: Any = None

    @property
    def diagnostics(self) -> tuple:
        """The planning facts (the route's reasons, then the undriven
        policy that applied), built when read from ``route_reasons`` (a
        tuple of strings, fixed at call time) and ``undriven``."""
        out = tuple(Diagnostic("route", str(r)) for r in self.route_reasons)
        if self.undriven is not None:
            pol, names = self.undriven
            out += (Diagnostic("undriven", (
                f"undriven={pol!r}: {len(names)} controlled switch(es) with "
                f"no driver run {'OPEN' if pol == 'open' else 'CLOSED'}: "
                + ", ".join(names))),)
        return out

    @property
    def features(self) -> frozenset:
        """The features in play: FEATURE_KW keywords that are set, and
        "builder_ics" when the builder's i0= / c0= seed the state."""
        out = {k for k in FEATURE_KW
               if self.resolved_from.get(k, "default") != "default"}
        if self.initial.from_builder:
            out.add("builder_ics")
        return frozenset(out)


class RunOutcome(NamedTuple):
    """What an engine runner hands ``_finalize``: the result and the drive
    as the run finally composed it (pwl composes its cache hooks, the
    default switch_fn and the live stream's cancel hook at run time, and
    the kernel object's builder is the one the result reads)."""
    result: Any
    builder: Any
    switch_fn: Any = None
    b_extra_fn: Any = None
    step_observer: Any = None
    should_continue: Any = None
    stats: Any = None               # trbdf2's stats dict
    cxx_chain: Any = None           # pwl: the chain fast path's C++ chain
    mmc_finalize: Any = None        # pwl: the MMC arms' end-of-run fold
