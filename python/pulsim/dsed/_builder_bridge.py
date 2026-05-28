"""CircuitBuilder ↔ PED scheduler adapter (DSED bridge).

This module wires :class:`pulsim.CircuitBuilder` through the C++
``PwlStateSpaceCache.compute_lti_state_space(mask)`` method into the
PED scheduler (:class:`pulsim.dsed.PEDSimulatorAuto`).

**Status:** the Python side (this file + the dispatch in
``pulsim._dsed_dispatch``) is COMPLETE. The C++ side
(``compute_lti_state_space``) is currently a stub that raises
``std::logic_error`` — the real implementation is Phase 5.1 of the
bridge work documented in ``notes/DSED_BRIDGE_DESIGN.md``.

When the C++ method lands, this module's :class:`CircuitBuilderAdapter`
needs no changes — the per-mask `(A, b)` cache populates lazily on
first access via the binding.

The adapter satisfies BOTH PED scheduler contracts:

* ``HasLTIPerMode``: ``A_matrix()`` + ``b_vector(t)`` (used by BDF2)
* ``HasRhs``: ``rhs(t, x)`` (used by RK45 / DOPRI5)
* ``HasMaskInterface``: ``current_mask()`` + ``set_mask(mask)``

so :class:`PEDSimulatorAuto` can dispatch RK45 ↔ BDF2 per mask
transparently.

See ``notes/DSED_BRIDGE_DESIGN.md`` for the full architecture +
algorithm + work breakdown.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np


class CircuitBuilderAdapter:
    """Adapter exposing a Pulsim :class:`CircuitBuilder` as a PED-
    scheduler-compatible LTI system.

    Construction is cheap — the per-mask state-space extraction is
    deferred to the first time each mask is queried via the
    underlying C++ ``cache.compute_lti_state_space(mask)`` call.
    Results are cached in ``_mask_cache`` so subsequent visits to
    the same mask hit a Python dict lookup, not C++ MNA assembly.

    Parameters
    ----------
    builder
        A populated :class:`pulsim.CircuitBuilder`.
    cache
        A :class:`pulsim.PwlStateSpaceCache` initialised with
        ``cache.build(dt)`` (the dt value is unused by the
        continuous-time extractor but the cache must be built so
        ``cache.compute_lti_state_space(mask)`` has something to
        recover from).
    switch_fn
        Callable ``t -> SwitchStateMask`` controlling which mask is
        active at each time. Must satisfy the PED scheduler's
        ``next_edge_after(t)`` contract for the gate-edge fast path.

    Notes
    -----
    The bridge currently supports **LTI circuits only** — circuits
    with nonlinear devices (diodes, MOSFETs, IGBTs, saturable
    inductors) will get a ``NotImplementedError`` from the
    underlying C++ ``compute_lti_state_space()`` because the PED
    engine doesn't model per-operating-point linearization. For DCM
    and body-diode commutation use ``pulsim.dsed.PEDSimulatorAuto``
    directly with a user-defined LTI system (see
    ``prototype/dsed/buck_dcm_model.py`` for an example).
    """

    def __init__(
        self,
        builder: Any,                                # pulsim.CircuitBuilder
        cache: Any,                                  # pulsim.PwlStateSpaceCache
        switch_fn: Callable[[float], Any],
        b_extra_fn: Optional[Callable[[float], "np.ndarray"]] = None,
    ) -> None:
        self.builder = builder
        self.cache = cache
        self.switch_fn = switch_fn
        # User-supplied callback returning a full-MNA-size b_extra
        # vector at time t. Plumbed in from `_dsed_dispatch` so the
        # user's `simulate(..., b_extra_fn=lambda t: ...)` keeps the
        # same semantics as the PWL engine (Bridge.7).
        self.b_extra_fn = b_extra_fn
        # Lazy per-mask cache. Key: mask (hashable); Value: tuple
        #   (A, b_constant, idx, is_cap, b_projection)
        self._mask_cache: dict[Any, tuple[np.ndarray, np.ndarray,
                                              list[int], list[bool],
                                              np.ndarray]] = {}
        # The "current" mask is set by the scheduler via set_mask();
        # initialized lazily on the first switch_fn call.
        self._current_mask: Optional[Any] = None
        # Whether the builder has any time-varying source. Lazy: the
        # first b_vector(t) call probes for non-zero output at t=0
        # AND t=1µs; if both zero on the probes AND there's no user
        # b_extra_fn, we mark the circuit "DC-only" and skip the
        # overlay in the hot path.
        self._has_dynamic_sources: Optional[bool] = None

    # ------------------------------------------------------------------
    # Lazy mask resolution
    # ------------------------------------------------------------------

    def _resolve_current(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (A, b_constant, b_projection) for the current mask,
        caching the result on first access."""
        if self._current_mask is None:
            raise RuntimeError(
                "CircuitBuilderAdapter: current_mask not set. "
                "The scheduler should call set_mask() before any "
                "rhs/A_matrix/b_vector query."
            )
        entry = self._mask_cache.get(self._current_mask)
        if entry is None:
            # Delegate to C++ — may throw NotImplementedError today
            # (the stub) or std::runtime_error on nonlinear topologies.
            try:
                A, b, idx, is_cap, B = \
                    self.cache.compute_lti_state_space(self._current_mask)
            except RuntimeError as e:
                # Convert C++'s std::logic_error / std::runtime_error
                # to a Python NotImplementedError with context.
                raise NotImplementedError(
                    f"CircuitBuilderAdapter: cannot extract LTI "
                    f"state-space for mask {self._current_mask}: {e}. "
                    f"See notes/DSED_BRIDGE_DESIGN.md for the bridge "
                    f"implementation status."
                ) from e
            entry = (np.asarray(A, dtype=float),
                     np.asarray(b, dtype=float),
                     list(idx),
                     list(is_cap),
                     np.asarray(B, dtype=float))
            self._mask_cache[self._current_mask] = entry
        return entry[0], entry[1], entry[4]

    @property
    def n_state(self) -> int:
        """Number of continuous-time state variables. Lazy — triggers
        a mask resolution on the first call."""
        A, _, _ = self._resolve_current()
        return A.shape[0]

    # ------------------------------------------------------------------
    # PEDSimulatorAuto / PEDSimulator contracts
    # ------------------------------------------------------------------

    def current_mask(self) -> Any:
        return self._current_mask

    def set_mask(self, mask: Any) -> None:
        self._current_mask = mask

    def A_matrix(self) -> np.ndarray:
        """Continuous-time A matrix for the currently-active mask."""
        A, _, _ = self._resolve_current()
        return A

    def _detect_dynamic_sources(self) -> bool:
        """Probe the builder for any time-varying source (sine/PWM/
        pulse). Returns True if compute_time_varying_b_extra is non-
        zero at t=0 OR t=1e-6 (covers DC offsets + AC). Also True if
        the user supplied a b_extra_fn callback."""
        if self.b_extra_fn is not None:
            return True
        # Probe at two times; if both are all-zero, the builder has
        # no time-varying source.
        try:
            b0 = np.asarray(
                self.builder.compute_time_varying_b_extra(0.0))
            b1 = np.asarray(
                self.builder.compute_time_varying_b_extra(1e-6))
        except AttributeError:
            # Older pulsim build without the binding — fall back to DC
            return False
        return bool(np.any(b0 != 0.0) or np.any(b1 != 0.0))

    def b_vector(self, t: float) -> np.ndarray:
        """Continuous-time b vector for the currently-active mask
        at time ``t``.

        Computes ``b(t) = b_constant + B · (b_sources(t) +
        b_user(t))`` where:
          * ``b_constant`` = DC sources projected via the extractor.
          * ``B`` = full-MNA → state projection (Bridge.6).
          * ``b_sources(t)`` = sum of sine/PWM/pulse contributions,
            computed via ``builder.compute_time_varying_b_extra(t)``.
          * ``b_user(t)`` = optional user-supplied ``b_extra_fn(t)``.

        Fast path: when the circuit has no time-varying sources and
        no user callback, returns ``b_constant`` directly (no B · 0
        product on the hot path).
        """
        _, b_const, B = self._resolve_current()

        # Cache the "is anything time-varying?" decision after the
        # first call.
        if self._has_dynamic_sources is None:
            self._has_dynamic_sources = self._detect_dynamic_sources()

        if not self._has_dynamic_sources:
            return b_const

        # Time-varying overlay
        b_extra = np.asarray(
            self.builder.compute_time_varying_b_extra(t),
            dtype=float)
        if self.b_extra_fn is not None:
            user_b = np.asarray(self.b_extra_fn(t), dtype=float)
            if user_b.shape != b_extra.shape:
                raise ValueError(
                    f"b_extra_fn returned shape {user_b.shape}; "
                    f"expected {b_extra.shape} (full-MNA size).")
            b_extra = b_extra + user_b

        return b_const + B @ b_extra

    def rhs(self, t: float, x: np.ndarray) -> np.ndarray:
        """Continuous-time RHS: dx/dt = A·x + b(t)."""
        A = self.A_matrix()
        return A @ x + self.b_vector(t)


__all__ = ["CircuitBuilderAdapter"]
