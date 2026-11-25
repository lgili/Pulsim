# Hand-written stub overlay for ``pulsim._pulsim``.
#
# ``scripts/generate_stubs.py`` merges this file into the generated
# ``python/pulsim/_pulsim.pyi``. It declares the members that the pure-
# Python layer attaches to native classes at import time
# (``python/pulsim/__init__.py``: ``SimulationResult.v = _result_v`` ...)
# or per instance (``res.engine_used = ...`` inside ``simulate()``).
# pybind11-stubgen either cannot see them (instance attributes) or sees
# them without types (plain Python functions), so a type checker would
# otherwise reject ``res.engine_used`` or infer ``Any`` for ``res.v(...)``.
#
# Merge rules:
# - every member here REPLACES the generated member of the same name in
#   the same class (all overloads of it), and is appended at the end of
#   the class under a marker comment;
# - a function here without a docstring inherits the runtime docstring
#   that stubgen emitted for the member it replaces, so the stub keeps
#   one source of documentation (the code);
# - top-level imports are added to the stub when missing.
#
# Keep the parameter NAMES and ORDER identical to the Python functions:
# python/tests/test_stubs.py compares them with ``inspect.signature``.

import typing

import numpy

class CircuitBuilder:
    def _resolve_alias(self, name: str) -> tuple[str, str] | None: ...
    def _snapshot(self) -> Circuit:
        """``build()`` without the side-state check (``simulate`` keeps the builder's Python side state beside it)."""
    def set_device_losses(self, device: str, *, junction: str | None = None, Tj: float | None = None, heat_node: str | None = None, E_on: typing.Any = None, E_off: typing.Any = None, V_on: typing.Any = None, conduction: str = "electrical") -> CircuitBuilder: ...

class SimulationResult:
    engine_used: str
    """Engine that produced this result (for example ``"pwl"``,
    ``"trbdf2"`` or ``"dsed"``). Set by :func:`pulsim.simulate`."""
    engine_route_reason: str | None
    """Why ``engine="auto"`` did not take the variable-step path, or
    ``None`` when it did (or when the engine was chosen explicitly)."""
    recorded: dict[str, typing.Any]
    """``simulate(recorders=[...])`` only: ``pulsim.Recording`` per
    recorder name (kernel probes and exact windowed statistics)."""
    recorder_feed: dict[str, typing.Any]
    """``simulate(recorders=[...])`` only: the feed that ran (``mode``:
    ``'in-loop'`` with ``n_steps`` / ``n_events`` / ``n_jumps`` /
    ``n_dropped``, or ``'stored-arrays'`` with ``why``)."""
    step_loop_python_callbacks: tuple[str, ...]
    """Names of the Python callables :func:`pulsim.simulate` installed in
    the engine's step loop (``"switch_fn"``, ``"b_extra_fn"``,
    ``"step_observer"``, ``"should_continue"``, ``"closed_loops"``);
    empty when the loop ran fully native."""
    diagnostics: dict[str, typing.Any]
    """Complementarity mode-selection counters of a ``pwl`` run
    (``'mode_shadow'``, ``'mode_selection'``, ``'mode_guard'``): a copy of
    ``mode_diagnostics``. Set by :func:`pulsim.simulate` on ``pwl``. With
    ``mode_selection='auto'`` it also holds ``'mode_selection_auto'``
    (``requested``, ``chosen`` ``'lcp'`` / ``'legacy'``, ``reason``), on
    every engine. On every engine ``'messages'`` holds the warnings the run
    emitted, in order: ``(code, message, category)`` records."""
    control_tick_counts: list[int]
    """Controller ticks per tick source, in registration order (closed
    loops, then ``step_observer``): ``ceil(span / T_j)`` each. Set by
    :func:`pulsim.simulate` on ``pwl`` and ``trbdf2``."""
    control_tick_jitter_max: float
    """Largest lateness of a controller tick [s]: below ``dt`` on
    ``pwl`` (0 when every period is a multiple of ``dt``), 0 on
    ``trbdf2``."""
    @typing.overload
    def v(self, name: str, t: None = None) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]]: ...
    @typing.overload
    def v(self, name: str, t: int) -> float: ...
    @typing.overload
    def v(self, name: str, t: slice | typing.Sequence[int] | numpy.ndarray[typing.Any, typing.Any]) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]]: ...
    @typing.overload
    def i(self, name: str, t: None = None, *, port: int | None = None) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]]: ...
    @typing.overload
    def i(self, name: str, t: int, *, port: int | None = None) -> float: ...
    @typing.overload
    def i(self, name: str, t: slice | typing.Sequence[int] | numpy.ndarray[typing.Any, typing.Any], *, port: int | None = None) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]]: ...
    @property
    def switch_states(self) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.bool_]] | None: ...
    @property
    def switch_states_unsolved(self) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.bool_]] | None: ...
    def power(self, device_name: str) -> float: ...
    def signal(self, name: str) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]]: ...
    def signals(self) -> list[str]: ...
    def currents(self, *, skip_unsupported: bool = True) -> dict[str, numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]]]: ...
    def loss_events(self, device: str) -> numpy.ndarray[typing.Any, typing.Any]: ...
    def device_energy(self, device: str) -> dict[str, float]: ...
    def plot(self, *signals: str, save: str | None = None, show: bool | None = None, **kwargs: typing.Any) -> typing.Any: ...
    def flux(self, name: str) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]]: ...
    def b_field(self, name: str) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]]: ...
    def h_field(self, name: str) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]]: ...
    @property
    def plan(self) -> typing.Any: ...
    @property
    def circuit(self) -> Circuit | None: ...
