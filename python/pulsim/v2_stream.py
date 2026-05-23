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


__all__ = ["LiveStream", "NativeLiveStream"]


class NativeLiveStream:
    """Live data stream backed by a C++ lock-free ring buffer.

    This is the high-performance alternative to :class:`LiveStream` for
    the live-scope use case. Instead of going through a Python
    ``step_observer`` callback every kernel step (with GIL acquire/
    release) and a ``queue.Queue`` for batching, the kernel writes
    samples directly to a preallocated numpy array shared with Python.
    The GUI reads from that array as a numpy view — zero copies, zero
    Python in the kernel hot loop.

    Architecture::

        ┌──────────────────────────┐
        │  C++ run_transient_      │
        │  with_chain (no GIL)     │
        │                          │  every `decimate` steps:
        │   chain.step(t, x)       │    ring.t_data[head%cap] = t
        │   if(++c >= decimate):   │    memcpy(ring.x_data + ..., x)
        │     ring.push(t, x)      │    head.store(h+1, release)
        │   if ring.stopped():     │
        │     break  ───────────────────────────────────►
        └──────────────────────────┘                    │
                                                        │ atomic
                ┌───────────────────────────────────────┘
                │
        ┌───────▼──────────────────┐
        │  Python LiveScope tick   │   head = ring.get_head()  [atomic acq]
        │  (60 Hz, GIL-held)       │   slice = t_buf[last:head]
        │                          │   xs    = x_buf[last:head]
        │   stream.get_new_samples │   redraw, advance last
        └──────────────────────────┘

    The ring is a single-producer, single-consumer (SPSC) lock-free
    queue. Synchronisation is provided by a single ``std::atomic<int64_t>``
    write head with ``memory_order_release`` on push and
    ``memory_order_acquire`` on read. The reader is guaranteed to see
    all data writes that happened before the head increment.

    Drop-in compatible with :class:`LiveStream`'s consumer surface for
    use with :class:`LiveScope`: it exposes ``stop()``,
    ``n_steps_received``, ``n_steps_kept``, etc. — but the GUI
    additionally checks for an ``is_native`` flag and switches to
    ``get_new_samples()`` instead of queue draining.

    Parameters
    ----------
    capacity, default 1_000_000
        Number of (t, x) slots in the ring buffer. At ``decimate=100``
        and the default 100 ns kernel dt, that covers ~10 s of visual
        sample history before the head wraps and oldest samples are
        overwritten. Memory: ``(1 + state_size) * 8 * capacity`` bytes
        (e.g. ~80 MB for capacity=1M with a 10-state circuit).
    decimate, default 100
        Kernel-side decimation factor. Every ``decimate`` steps the
        kernel pushes one sample. Use 1 to capture every step, higher
        values for finer-grained downsampling at the source (no Python
        overhead).
    """

    is_native = True

    def __init__(self, *, capacity: int = 1_000_000,
                  decimate: int = 100):
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        if decimate < 1:
            raise ValueError("decimate must be >= 1")
        self._capacity = int(capacity)
        self._decimate = int(decimate)
        self._ring = None                  # set by attach()
        self._state_size = -1
        self._last_read = 0                # reader's monotonic counter
        self._dropped_samples = 0          # samples we missed (ring overflow)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def attach(self, state_size: int) -> None:
        """Allocate the ring buffer for the given state vector size.
        Called automatically by ``simulate(live_stream=…)`` before the
        kernel runs."""
        from ._pulsim import v2_kernel as _k  # type: ignore[import-not-found]
        self._state_size = int(state_size)
        self._ring = _k.LiveRingHandle(
            capacity=self._capacity,
            state_size=self._state_size,
            decimate=self._decimate)

    @property
    def native_ring(self):
        """The underlying C++ ``LiveRingHandle`` (or None if not
        attached). Passed to ``run_transient_with_chain(live_ring=…)``."""
        return self._ring

    @property
    def attached(self) -> bool:
        return self._ring is not None

    # ------------------------------------------------------------------
    # Consumer API (used by LiveScope)
    # ------------------------------------------------------------------

    def get_new_samples(self):
        """Return ``(t_view, x_view)`` of new samples written since the
        last call. Both are zero-copy numpy slices of the ring (unless
        the read wraps the ring boundary, in which case a single
        ``np.concatenate`` is done — still one allocation, no per-sample
        cost). Returns ``None`` if no new samples.
        """
        if self._ring is None:
            return None
        head = self._ring.get_head()
        if head <= self._last_read:
            return None
        # If the kernel got more than ``capacity`` ahead of us, the
        # oldest samples were overwritten — we accept the loss and
        # only see the latest ``capacity`` samples.
        gap = head - self._last_read
        if gap > self._capacity:
            self._dropped_samples += (gap - self._capacity)
            self._last_read = head - self._capacity
        t_buf = self._ring.t_buf
        x_buf = self._ring.x_buf
        start_slot = int(self._last_read % self._capacity)
        end_slot = int(head % self._capacity)
        if end_slot > start_slot:
            t_view = t_buf[start_slot:end_slot]
            x_view = x_buf[start_slot:end_slot]
        elif end_slot == 0:
            # Exact wrap to start — tail to end of buffer.
            t_view = t_buf[start_slot:]
            x_view = x_buf[start_slot:]
        else:
            # Wraparound — one concat (still cheap relative to the
            # per-tick budget).
            t_view = np.concatenate(
                [t_buf[start_slot:], t_buf[:end_slot]])
            x_view = np.concatenate(
                [x_buf[start_slot:], x_buf[:end_slot]])
        self._last_read = head
        return t_view, x_view

    def stop(self) -> None:
        """Signal the kernel to stop at the next step boundary."""
        if self._ring is not None:
            self._ring.request_stop()

    def stopped(self) -> bool:
        if self._ring is None:
            return False
        return self._ring.stopped()

    @property
    def should_continue(self):
        """For symmetry with :class:`LiveStream`. The C++
        run_transient_with_chain already wires the ring's stop flag
        into the kernel's should_continue check — this is the
        explicit Python-side hook for callers that don't use the
        chain path."""
        return lambda: not self.stopped()

    # ------------------------------------------------------------------
    # Stats — names compatible with LiveStream so LiveScope can poll
    # them generically.
    # ------------------------------------------------------------------

    @property
    def n_steps_received(self) -> int:
        if self._ring is None:
            return 0
        return self._ring.get_head() * self._decimate

    @property
    def n_steps_kept(self) -> int:
        if self._ring is None:
            return 0
        return self._ring.get_head()

    @property
    def n_batches_emitted(self) -> int:
        # Ring-mode doesn't use batches; report the number of fresh
        # samples since attach as a useful proxy.
        return self.n_steps_kept

    @property
    def n_batches_dropped(self) -> int:
        return self._dropped_samples

    def qsize(self) -> int:
        if self._ring is None:
            return 0
        return int(self._ring.get_head() - self._last_read)

    def empty(self) -> bool:
        return self.qsize() == 0

    def flush_pending(self) -> None:
        # No-op — the ring is always "flushed" by definition (writes
        # are visible to the reader as soon as the atomic head moves).
        pass

    # Misc compatibility shims (some scope code calls these).
    def add_inline_consumer(self, _callback) -> None:  # noqa: D401
        raise NotImplementedError(
            "NativeLiveStream doesn't support inline consumers — the "
            "kernel writes directly to shared memory.")


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
                  max_queue: int = 200,
                  downsample: int = 1):
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if downsample < 1:
            raise ValueError("downsample must be >= 1")
        self.batch_size = int(batch_size)
        self.downsample = int(downsample)
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
        self._n_steps_kept = 0
        self._n_batches_emitted = 0
        self._n_batches_dropped = 0
        # Downsample counter — appends only when (counter % downsample == 0).
        self._ds_counter = 0

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
            self._n_steps_received += 1
            self._ds_counter += 1
            if self._ds_counter < self.downsample:
                return     # skip — downsampling
            self._ds_counter = 0
            self._batch_t.append(float(t))
            self._batch_x.append(np.asarray(x).copy())
            self._n_steps_kept += 1
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
        """Total kernel steps observed (before downsampling)."""
        return self._n_steps_received

    @property
    def n_steps_kept(self) -> int:
        """Steps actually pushed to the queue (after downsampling)."""
        return self._n_steps_kept

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
