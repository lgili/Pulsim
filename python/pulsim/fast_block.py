"""PSIM / PLECS "C block" workalike via Numba JIT.

User writes a Python function with the body of a control law (state
mutation + scalar output), decorates it with :func:`fast_block`, and
gets a Numba-compiled native callable suitable for plugging into
``step_observer=`` or ``MixedDomainBlockChain``. The compiled function
runs at C speed without the user installing a C compiler, dealing
with cross-platform build systems, or accepting the security risk of
runtime ``cc`` invocation.

Why not literal C?

* PSIM's "Custom C Block" and PLECS's "C-Script Block" exist because
  the host simulators are written in C++ and the only language that
  fits the runtime is C. Pulsim is dual-language (C++ kernel + Python
  control), so the analogous slot is "fast Python" — and Numba turns
  numerically-flavoured Python into LLVM-compiled machine code that
  matches hand-written C for the kinds of loops control engineers
  actually write (10–50 floating-point ops, no Python objects).
* No compiler toolchain on the user's machine; no MSVC vs clang
  vs gcc dance; no `.so` loading or symbol lookups. ``pip install
  pulsim[fast]`` and it works.

Authoring contract:

* Signature: ``f(*scalar_args, state)`` — last positional argument is
  a 1-D ``numpy.ndarray`` of dtype ``float64`` holding the block's
  persistent state. Mutate ``state`` in-place; return the scalar
  output.
* Body: any Numba-supported Python (arithmetic, ``np.*`` reductions,
  control flow, ``math.*``). No Python objects, no dicts of Python
  values, no string formatting. Numba's "@njit nopython" rules apply.
* First call triggers JIT compilation (~0.3–1 s); subsequent calls
  hit the cache.

Example::

    @pulsim.fast_block
    def pi(error, dt, Kp, Ki, state):
        # state[0] = integrator
        state[0] += Ki * dt * error
        return Kp * error + state[0]

    state = pi.make_state()           # zeros, sized via `n_states=...`
                                       # or default 1
    def observer(t, x):
        err = setpoint - x[v_idx]
        u = pi(err, DT, KP, KI, state)
        ...                           # apply u to switch_fn

Limitations:

* Numba supports only a subset of Python. Strings, dicts, generators,
  exceptions other than ``ValueError``/``AssertionError`` are out.
* First-call JIT cost is non-trivial — call ``pi.warm_up()`` once
  at sim-build time if you need predictable latency.
* The block runs INSIDE a Python step observer, so each call pays
  the Python → Numba boundary (≈ 1 µs). Acceptable for kHz-class
  control loops; for MHz-class hot paths, wire the C++ kernel
  directly via ``pulsim.control`` blocks (already JIT-compiled to
  C++ at chain build time).
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np


__all__ = ["fast_block", "FastBlock", "is_available"]


_NUMBA = None  # lazy import — keeps importing pulsim cheap


def is_available() -> bool:
    """Return ``True`` iff numba is installed and importable.

    Use to gate optional code paths::

        if pulsim.fast_block.is_available():
            controller = pulsim.fast_block(my_pi)
        else:
            controller = my_pi   # pure-Python fallback
    """
    global _NUMBA
    if _NUMBA is None:
        try:
            import numba as _nb
            _NUMBA = _nb
        except ImportError:
            _NUMBA = False  # cached negative
    return _NUMBA is not False


def _require_numba():
    if not is_available():
        raise ImportError(
            "pulsim.fast_block requires numba — install with "
            "`pip install pulsim[fast]` (or `pip install numba`). "
            "If you don't need JIT, your function works fine as a "
            "plain Python callable in `step_observer=`.")
    return _NUMBA


class FastBlock:
    """A JIT-compiled callable returned by :func:`fast_block`.

    Wraps the underlying ``numba.njit`` function with a couple of
    ergonomic helpers (``make_state``, ``warm_up``) without changing
    the call semantics — calling the instance forwards to the
    compiled function with zero overhead beyond the Numba boundary.
    """

    def __init__(self, compiled: Callable[..., Any],
                  py_func: Callable[..., Any],
                  n_states: int):
        self._compiled = compiled
        self._py_func = py_func
        self._n_states = int(n_states)
        # Match the wrapped function's __name__/__doc__/__qualname__
        # so tracebacks and ``help()`` are useful. We keep only the
        # subset of attributes that doesn't trip read-only descriptors
        # on the class (e.g. ``__module__`` is a slot on this type).
        for attr in ("__doc__", "__name__", "__qualname__"):
            try:
                setattr(self, attr, getattr(py_func, attr))
            except (AttributeError, TypeError):
                pass

    def __call__(self, *args, **kwargs):
        return self._compiled(*args, **kwargs)

    def make_state(self) -> np.ndarray:
        """Allocate a zero-initialised state vector of the size
        declared on the decorator (``n_states=``, default 1)."""
        return np.zeros(self._n_states, dtype=np.float64)

    def warm_up(self, *args) -> None:
        """Trigger JIT compilation eagerly so the first hot-path call
        doesn't pay the ~0.3–1 s LLVM cost. Pass a representative set
        of argument values (the same dtypes you'll use at runtime);
        Numba's specialisation is per-signature.

        When called with no args, attempts a heuristic dispatch using
        the declared state size and float scalars for every other
        parameter — works for the common ``f(*floats, state)`` shape."""
        if args:
            self._compiled(*args)
            return
        # Heuristic: introspect the wrapped Python function's signature
        # and pass float zeros for non-state args + a state vector.
        import inspect
        sig = inspect.signature(self._py_func)
        params = list(sig.parameters)
        if not params:
            return  # zero-arg block: nothing to specialise
        sample = []
        for name in params:
            if name == "state":
                sample.append(self.make_state())
            else:
                sample.append(0.0)
        self._compiled(*sample)

    @property
    def py_func(self) -> Callable[..., Any]:
        """The original undecorated Python function — useful for
        unit-testing the control law without JIT in the path."""
        return self._py_func


def fast_block(func: Callable[..., Any] | None = None,
               *,
               n_states: int = 1,
               cache: bool = True,
               parallel: bool = False) -> Any:
    """Decorator: turn a Python control function into a Numba-JIT
    :class:`FastBlock`.

    Usable both bare and parametrised::

        @fast_block
        def pi(err, dt, Kp, Ki, state): ...     # n_states=1 default

        @fast_block(n_states=3, parallel=False)
        def lqr(*states, state): ...

    Parameters
    ----------
    func
        The Python function to compile. When ``None``, returns a
        decorator; otherwise applies directly.
    n_states
        Size of the state vector ``FastBlock.make_state()`` will
        allocate. Doesn't affect the compiled function — it only
        sizes the helper. Default 1.
    cache
        Forward to ``numba.njit(cache=...)``. Default ``True`` —
        compiled code is persisted to ``__pycache__`` so the
        second run of a script is instant.
    parallel
        Forward to ``numba.njit(parallel=...)``. Default ``False`` —
        control loops are typically too small to amortise the
        parallel-launch overhead; turn on only for blocks doing
        thousands of vector ops per call.

    Returns
    -------
    FastBlock | Callable[[Callable], FastBlock]
        The compiled block, or a decorator factory if ``func`` is
        ``None``.
    """
    def _wrap(f: Callable[..., Any]) -> FastBlock:
        nb = _require_numba()
        compiled = nb.njit(cache=cache, parallel=parallel)(f)
        return FastBlock(compiled, f, n_states)

    if func is None:
        return _wrap
    return _wrap(func)
