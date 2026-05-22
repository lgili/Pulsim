"""Pulsim v2 — live data streaming + cancellation.

Foundation for the live scope GUI. The simulation pushes per-step
samples into a thread-safe queue; one or more consumers (typically
the GUI in the main thread) pull them and update plots.

Architecture:

   ┌─────────────────┐     step_observer(t, x)
   │  simulate()     │ ─────────────────────────► LiveStream
   │  (worker thread)│                              │  batches every
   │                 │ ◄── should_continue() ───── │  N steps, pushes
   │                 │                              │  to thread-safe
   └─────────────────┘                              │  queue
                                                    ▼
                                     ┌──────────────────────────┐
                                     │  consumer (main thread)  │
                                     │  (GUI scope / logger / …)│
                                     └──────────────────────────┘

Typical usage:

    import threading
    import pulsim.v2 as p

    stream = p.LiveStream(batch_size=100, max_queue=50)

    def run_sim():
        return p.simulate(b, t_end=10.0, dt=1e-7,
                            step_observer=stream.step_observer,
                            should_continue=stream.should_continue)

    worker = threading.Thread(target=run_sim, daemon=True)
    worker.start()

    # Main thread: pull batches as they arrive, update GUI.
    while worker.is_alive() or not stream.empty():
        batch = stream.get_batch(timeout=0.1)
        if batch is not None:
            t_arr, x_arr = batch
            # update GUI...
            if user_clicked_stop:
                stream.stop()

    worker.join()

The same `LiveStream` instance can be passed both as `step_observer`
and as `should_continue` — see the ``step_observer`` and
``should_continue`` properties.

If you don't run the simulation in a worker thread, you can still
use ``LiveStream`` as a pure-Python batched recorder — the queue
fills up during ``simulate(...)`` and you drain it after. But the
INTERACTIVE use case (stop on demand) requires the worker-thread
pattern shown above.
"""

from __future__ import annotations

import queue
import threading
from typing import Callable, List, Optional, Tuple

import numpy as np


__all__ = ["LiveStream"]


class LiveStream:
    """Thread-safe streaming output for a running simulation.

    Parameters
    ----------
    batch_size
        How many simulation steps to accumulate before pushing to
        the queue. Larger batches = lower Python overhead per step
        but coarser GUI update rate. Sensible defaults: 100 (fast
        scope update) → 1000 (low overhead).
    max_queue
        Maximum number of batches queued before back-pressure. When
        the queue is full and a new batch arrives, the OLDEST batch
        is dropped (live-scope semantics: the consumer always sees
        the most recent data). Defaults to 200 batches.
    """

    def __init__(self, *, batch_size: int = 100,
                  max_queue: int = 200):
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        self.batch_size = int(batch_size)
        self._queue: "queue.Queue[Tuple[np.ndarray, np.ndarray]]" = \
            queue.Queue(maxsize=int(max_queue))
        self._batch_t: List[float] = []
        self._batch_x: List[np.ndarray] = []
        self._stop_event = threading.Event()
        # Optional per-batch hook (called inline on the kernel
        # thread — keep it cheap).
        self._inline_consumer: Optional[
            Callable[[np.ndarray, np.ndarray], None]] = None
        # Stats.
        self._n_steps_received = 0
        self._n_batches_emitted = 0
        self._n_batches_dropped = 0

    # ------------------------------------------------------------------
    # Public callbacks passed into simulate(...)
    # ------------------------------------------------------------------

    @property
    def step_observer(self):
        """The `step_observer(t, x)` callback to pass to
        ``simulate(...)``. Captures a closure over this instance.

        Cheap: appends to a per-batch buffer; only flushes (pushes
        a numpy array onto the queue) every `batch_size` steps.
        """
        def obs(t: float, x) -> None:
            # NOTE: x is a borrowed reference into the C++ kernel's
            # state vector. We MUST copy it before stashing — the
            # next kernel step would otherwise overwrite our data.
            self._batch_t.append(float(t))
            self._batch_x.append(np.asarray(x).copy())
            self._n_steps_received += 1
            if len(self._batch_t) >= self.batch_size:
                self._flush()
        return obs

    @property
    def should_continue(self):
        """The `should_continue() -> bool` callback for
        ``simulate(...)``. Returns False once :meth:`stop` is
        called. The kernel checks at the top of each step."""
        return lambda: not self._stop_event.is_set()

    # ------------------------------------------------------------------
    # Consumer API
    # ------------------------------------------------------------------

    def get_batch(self, timeout: Optional[float] = None
                    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Pull one batch from the queue. Returns ``(times, states)``
        where times has shape ``(N,)`` and states has shape
        ``(N, state_size)``. Returns ``None`` if no batch arrives
        within `timeout` seconds (use ``timeout=None`` to block
        indefinitely)."""
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def empty(self) -> bool:
        """True if the queue currently has no pending batches."""
        return self._queue.empty()

    def qsize(self) -> int:
        """Number of batches currently queued."""
        return self._queue.qsize()

    def add_inline_consumer(
        self,
        callback: Callable[[np.ndarray, np.ndarray], None]) -> None:
        """Register a callback invoked INLINE on each flush — runs
        on the kernel thread. Use for cheap operations (e.g.
        printing progress). For full GUI updates, pull from the
        queue on the main thread instead.
        """
        self._inline_consumer = callback

    def stop(self) -> None:
        """Signal the simulation to stop at the next step boundary.
        The simulation's `simulate(...)` call returns whatever
        partial result it has accumulated."""
        self._stop_event.set()

    def stopped(self) -> bool:
        """True if :meth:`stop` was called."""
        return self._stop_event.is_set()

    # ------------------------------------------------------------------
    # Flushing (final cleanup after the simulation ends)
    # ------------------------------------------------------------------

    def flush_pending(self) -> None:
        """Flush any partially-accumulated batch to the queue.
        Call once at the end of the simulation to make sure
        consumers see ALL samples even when the last batch wasn't
        full."""
        if self._batch_t:
            self._flush()

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    @property
    def n_steps_received(self) -> int:
        return self._n_steps_received

    @property
    def n_batches_emitted(self) -> int:
        return self._n_batches_emitted

    @property
    def n_batches_dropped(self) -> int:
        return self._n_batches_dropped

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _flush(self) -> None:
        t_arr = np.array(self._batch_t, dtype=float)
        x_arr = np.array(self._batch_x, dtype=float)
        if self._inline_consumer is not None:
            try:
                self._inline_consumer(t_arr, x_arr)
            except Exception:  # noqa: BLE001
                # Don't let consumer bugs kill the simulation.
                pass
        try:
            self._queue.put_nowait((t_arr, x_arr))
            self._n_batches_emitted += 1
        except queue.Full:
            # Drop the OLDEST batch, push the newest — live-scope
            # semantics (consumer wants the most recent data).
            try:
                self._queue.get_nowait()
                self._queue.put_nowait((t_arr, x_arr))
                self._n_batches_dropped += 1
                self._n_batches_emitted += 1
            except (queue.Empty, queue.Full):
                self._n_batches_dropped += 1
        self._batch_t.clear()
        self._batch_x.clear()
