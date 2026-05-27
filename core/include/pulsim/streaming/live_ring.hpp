#pragma once

// =============================================================================
// Pulsim — Layer 5+: live streaming ring buffer
// =============================================================================
//
// Backs ``pulsim.NativeLiveStream`` on the Python side. Goals:
//
//   * **GIL-free push.** ``run_transient`` runs inside
//     ``py::gil_scoped_release``, so the per-step push must NOT touch
//     the Python interpreter. We use raw doubles + ``std::atomic`` —
//     no PyObject* anywhere.
//   * **Zero-copy read.** The Python side reads the ring as a numpy
//     view (``stream.native_ring.t_buf``) without copying. pybind11
//     ``py::array_t`` over the same memory.
//   * **Single-producer single-consumer.** One simulation thread pushes,
//     one GUI thread polls. No locks needed — head_/tail bookkeeping
//     is via atomics with release/acquire semantics.
//   * **Lossy overflow.** If the consumer falls behind by more than
//     ``capacity`` samples, the OLDEST samples get overwritten silently.
//     Production GUIs poll at 60 Hz; a ``capacity`` of 1M samples at
//     decimate=100 covers ~30 s at 100 kHz visual rate before any
//     loss. The consumer tracks ``head - last_read`` to detect &
//     report drops.
//   * **Cooperative stop.** The simulation polls ``stopped()`` once
//     per step. GUI calls ``request_stop()`` from any thread and the
//     simulation exits cleanly at the next step boundary.
//
// THE per-step hot path is:
//
//     if (ring->should_decimate())  // ~5 ns
//         ring->push(t, x_data, n);  // ~50 ns memcpy + atomic incr
//     if (ring->stopped())           // ~2 ns relaxed load
//         break;
//
// Total overhead vs no-streaming: < 100 ns/step. A 100 kHz sim
// (dt=10 µs) loses < 1 % to streaming overhead.

#include "pulsim/numeric/types.hpp"

#include <atomic>
#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace pulsim::streaming {

// -----------------------------------------------------------------------------
// LiveRing — fixed-capacity SPSC ring buffer for ``(t, x)`` samples.
//
// Memory layout
// -------------
//     t_buf_: capacity × Real
//     x_buf_: capacity × state_size × Real  (row-major, slot-major)
//
// The ``i``-th slot for the row covers ``[i*state_size, (i+1)*state_size)``
// in ``x_buf_``. Slot indices wrap mod ``capacity``.
//
// Ordering guarantees
// -------------------
// ``head_`` is incremented with ``memory_order_release`` *after* the
// payload write. The consumer reads ``head_`` with
// ``memory_order_acquire`` before reading the slot. Together these
// guarantee the consumer never sees a partially-written slot.
//
// Single-producer single-consumer only. Multiple producers would
// race on slot ownership; multiple consumers can each read the same
// slots independently if they each keep their own ``last_read``
// counter — but we don't bother to support that.
// -----------------------------------------------------------------------------
class LiveRing {
public:
    LiveRing(std::size_t capacity,
             std::size_t state_size,
             std::size_t decimate)
        : capacity_(capacity),
          state_size_(state_size),
          decimate_(decimate == 0 ? 1 : decimate),
          t_buf_(capacity),
          x_buf_(capacity * state_size),
          head_(0),
          step_counter_(0),
          stopped_(false) {
        if (capacity_ == 0) {
            throw std::invalid_argument(
                "LiveRing: capacity must be > 0");
        }
        if (state_size_ == 0) {
            throw std::invalid_argument(
                "LiveRing: state_size must be > 0");
        }
    }

    // No copy / no move — the buffers are large and the atomics
    // would need explicit handling. Stable address is also part of
    // the contract with the Python numpy view.
    LiveRing(const LiveRing&) = delete;
    LiveRing& operator=(const LiveRing&) = delete;
    LiveRing(LiveRing&&) = delete;
    LiveRing& operator=(LiveRing&&) = delete;

    // ------------------------------------------------------------------
    // Producer-side API (called from the simulation thread, GIL-free).
    // ------------------------------------------------------------------

    /// Returns true on every ``decimate``-th call and advances the
    /// internal counter. Use as a cheap gate before ``push`` so that
    /// the kernel skips the payload copy when this sample is
    /// decimated out.
    bool should_decimate() noexcept {
        const std::size_t c =
            step_counter_.fetch_add(1, std::memory_order_relaxed);
        return (c % decimate_) == 0;
    }

    /// Write one (t, x) sample. ``x_data`` must point to at least
    /// ``state_size`` Reals. Thread-safe wrt the consumer; NOT safe
    /// against concurrent producers.
    void push(Real t, const Real* x_data, std::size_t n) noexcept {
        // Truncate if the caller passes a state vector larger than
        // ours. (Trim is intentional — a partial write would corrupt
        // the consumer's view.)
        const std::size_t copy_n =
            (n < state_size_) ? n : state_size_;
        // Use a relaxed load for the head's current value, then
        // release-publish after the payload write below. The
        // consumer pairs this with an acquire load.
        const std::size_t h =
            head_.load(std::memory_order_relaxed);
        const std::size_t slot = h % capacity_;
        t_buf_[slot] = t;
        Real* dst = x_buf_.data() + slot * state_size_;
        std::memcpy(dst, x_data, copy_n * sizeof(Real));
        // Zero-fill any tail bytes if the source was short. Cheap
        // and avoids surprising stale data on the consumer side.
        if (copy_n < state_size_) {
            std::memset(dst + copy_n, 0,
                        (state_size_ - copy_n) * sizeof(Real));
        }
        head_.store(h + 1, std::memory_order_release);
    }

    // ------------------------------------------------------------------
    // Consumer-side API (called from the GUI thread holding the GIL).
    // ------------------------------------------------------------------

    /// Monotonic count of pushed samples. Wraps semantically into
    /// slot index via ``% capacity``. ``head() - last_read`` is the
    /// number of unread samples (subject to overflow truncation).
    [[nodiscard]] std::size_t head() const noexcept {
        return head_.load(std::memory_order_acquire);
    }

    // ------------------------------------------------------------------
    // Cooperative-stop API (either side may call).
    // ------------------------------------------------------------------

    void request_stop() noexcept {
        stopped_.store(true, std::memory_order_release);
    }

    [[nodiscard]] bool stopped() const noexcept {
        return stopped_.load(std::memory_order_acquire);
    }

    // ------------------------------------------------------------------
    // Sizing accessors (immutable post-construction).
    // ------------------------------------------------------------------
    [[nodiscard]] std::size_t capacity() const noexcept {
        return capacity_;
    }
    [[nodiscard]] std::size_t state_size() const noexcept {
        return state_size_;
    }
    [[nodiscard]] std::size_t decimate() const noexcept {
        return decimate_;
    }

    // ------------------------------------------------------------------
    // Raw buffer access — for pybind11 ``py::array_t`` views. The
    // pointers are stable for the lifetime of the LiveRing because
    // we deleted copy/move; numpy can borrow without copying.
    // ------------------------------------------------------------------
    [[nodiscard]] Real* t_data() noexcept { return t_buf_.data(); }
    [[nodiscard]] const Real* t_data() const noexcept {
        return t_buf_.data();
    }
    [[nodiscard]] Real* x_data() noexcept { return x_buf_.data(); }
    [[nodiscard]] const Real* x_data() const noexcept {
        return x_buf_.data();
    }

private:
    const std::size_t capacity_;
    const std::size_t state_size_;
    const std::size_t decimate_;

    std::vector<Real> t_buf_;
    std::vector<Real> x_buf_;

    std::atomic<std::size_t> head_;
    std::atomic<std::size_t> step_counter_;
    std::atomic<bool> stopped_;
};

}  // namespace pulsim::streaming
