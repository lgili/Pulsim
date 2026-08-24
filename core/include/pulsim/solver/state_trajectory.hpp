#pragma once

// =============================================================================
// Pulsim — Layer 5: StateTrajectory (contiguous waveform storage)
// =============================================================================
//
// v2.0 Phase 1, audit finding `waveform-storage-vector-of-vectors`
// (HIGH, CONFIRMED).
//
// The v1.x `SimulationResult::states` was a `std::vector<Vector>` —
// ONE heap block per recorded sample. A 10^7-step run of a
// 100-state converter meant 10^7 allocations plus 8 GB of
// fragmented blocks, and every `res.states` access from Python
// converted the entire history into a fresh list of 10^7 small
// ndarrays (a second full copy, per access).
//
// StateTrajectory stores the same data as ONE row-major buffer of
// `n_samples × n_state` reals:
//   * a single allocation for the whole run when `reserve()` is
//     called with the state size known (`run_transient` does);
//   * per-sample recording is a contiguous append — no allocator
//     round-trip, no pointer chasing on readback;
//   * `data()/rows()/cols()` expose the buffer directly, so the
//     Python binding hands out a ZERO-COPY 2-D numpy view instead
//     of copying the run.
//
// The READ API is deliberately source-compatible with the old
// vector-of-vectors: `traj[k]`, `size()`, `empty()`, `front()`,
// `back()` and range-for all work as before. The difference is
// that element access yields an `Eigen::Map<const Vector>` BY
// VALUE rather than a `const Vector&`, so bind with `const auto&`
// / `const Vector&` (both extend the temporary's lifetime) — never
// with a non-const `auto&`.
//
// Ragged trajectories were silently representable before (nothing
// stopped a caller pushing states of differing length); here a
// mismatched push throws, since a contiguous buffer has one row
// stride by construction.

#include "pulsim/numeric/dense.hpp"
#include "pulsim/numeric/types.hpp"

#include <cstddef>
#include <format>
#include <initializer_list>
#include <iterator>
#include <stdexcept>
#include <vector>

namespace pulsim::solver {

class StateTrajectory {
public:
    /// One recorded sample, as a read-only view into the buffer.
    /// Returned BY VALUE (a Map is a pointer + size — copying it
    /// copies no data).
    using ConstStateRef = Eigen::Map<const Vector>;
    using value_type    = ConstStateRef;

    // -------------------------------------------------------------
    // Sizing
    // -------------------------------------------------------------

    /// Declare the per-sample state size up front. Lets `reserve`
    /// allocate the ENTIRE run in one shot. Optional — the first
    /// `push_back` infers it — but calling it first is what turns
    /// recording into a zero-allocation append.
    ///
    /// Throws if samples are already stored under a different size.
    void set_state_size(Index n) {
        if (n < 0) {
            throw std::invalid_argument(
                "StateTrajectory::set_state_size: negative size");
        }
        if (n_samples_ > 0 && n != n_state_) {
            throw std::invalid_argument(std::format(
                "StateTrajectory::set_state_size: {} samples already "
                "stored with state size {}; cannot change to {}",
                n_samples_, n_state_, n));
        }
        n_state_ = n;
        apply_reserve_();
    }

    /// Pre-allocate room for `n` samples. With the state size known
    /// (see `set_state_size`) this is the run's ONE allocation.
    void reserve(Size n) {
        reserve_samples_ = n;
        apply_reserve_();
    }

    // -------------------------------------------------------------
    // Recording
    // -------------------------------------------------------------

    /// Append one sample. The state size is taken from the first
    /// push if it was not declared; later pushes must match.
    void push_back(const Vector& x) {
        if (n_samples_ == 0 && n_state_ == 0) {
            n_state_ = static_cast<Index>(x.size());
            apply_reserve_();
        }
        if (static_cast<Index>(x.size()) != n_state_) {
            throw std::invalid_argument(std::format(
                "StateTrajectory::push_back: state size {} does not "
                "match the trajectory's {} (a contiguous trajectory "
                "has one row stride)",
                x.size(), n_state_));
        }
        // insert (not resize+assign) so the appended region is
        // never zero-filled first; with capacity reserved this is a
        // pure copy into the tail.
        data_.insert(data_.end(), x.data(), x.data() + n_state_);
        ++n_samples_;
    }

    void clear() noexcept {
        data_.clear();
        n_samples_ = 0;
    }

    /// Replace the trajectory with `samples` (all of which must
    /// share a state size). Keeps the `traj = {x0, x1, ...}`
    /// construction idiom the vector-of-vectors allowed, which
    /// tests and small fixtures use to build a trajectory by hand.
    void assign(std::initializer_list<Vector> samples) {
        clear();
        n_state_ = samples.size() > 0
            ? static_cast<Index>(samples.begin()->size())
            : Index{0};
        reserve(samples.size());
        for (const auto& s : samples) {
            push_back(s);
        }
    }

    StateTrajectory& operator=(std::initializer_list<Vector> samples) {
        assign(samples);
        return *this;
    }

    // -------------------------------------------------------------
    // Read access (source-compatible with vector<Vector>)
    // -------------------------------------------------------------

    [[nodiscard]] ConstStateRef operator[](Size k) const noexcept {
        return ConstStateRef(row_ptr_(k), n_state_);
    }

    [[nodiscard]] ConstStateRef at(Size k) const {
        if (k >= n_samples_) {
            throw std::out_of_range(std::format(
                "StateTrajectory::at: sample {} of {}", k, n_samples_));
        }
        return (*this)[k];
    }

    [[nodiscard]] ConstStateRef front() const noexcept {
        return (*this)[0];
    }

    [[nodiscard]] ConstStateRef back() const noexcept {
        return (*this)[n_samples_ - 1];
    }

    [[nodiscard]] Size size() const noexcept { return n_samples_; }
    [[nodiscard]] bool empty() const noexcept { return n_samples_ == 0; }

    /// Samples that fit without reallocating.
    ///
    /// Before the row width is known (`reserve` called with neither
    /// `set_state_size` nor a push yet) no bytes exist to count, so
    /// this reports the PENDING reservation — the allocation
    /// happens as soon as the width arrives. Once the width is
    /// known it reports the real buffer capacity, which may be
    /// SMALLER than a very large requested reservation (see
    /// `kEagerReserveByteCap`).
    [[nodiscard]] Size capacity() const noexcept {
        if (n_state_ <= 0) {
            return reserve_samples_;
        }
        return static_cast<Size>(
            data_.capacity() / static_cast<std::size_t>(n_state_));
    }

    // -------------------------------------------------------------
    // Raw buffer — the zero-copy numpy view's backing store
    // -------------------------------------------------------------

    /// Row-major base pointer (`rows() × cols()` reals). Null only
    /// when nothing has been recorded.
    [[nodiscard]] const Real* data() const noexcept {
        return data_.data();
    }
    [[nodiscard]] Size  rows() const noexcept { return n_samples_; }
    [[nodiscard]] Index cols() const noexcept { return n_state_; }

    /// Total bytes held by the sample buffer.
    [[nodiscard]] std::size_t bytes() const noexcept {
        return data_.size() * sizeof(Real);
    }

    // -------------------------------------------------------------
    // Iteration — proxy iterator yielding ConstStateRef by value.
    // Range-for over `const auto&` binds to the temporary and works
    // exactly as it did over vector<Vector>.
    // -------------------------------------------------------------

    class const_iterator {
    public:
        using iterator_category = std::input_iterator_tag;
        using value_type        = ConstStateRef;
        using reference         = ConstStateRef;   // proxy: by value
        using pointer           = void;
        using difference_type   = std::ptrdiff_t;

        const_iterator() = default;
        const_iterator(const StateTrajectory* t, Size k)
            : traj_{t}, k_{k} {}

        [[nodiscard]] reference operator*() const { return (*traj_)[k_]; }
        const_iterator& operator++() { ++k_; return *this; }
        const_iterator operator++(int) {
            const_iterator tmp = *this; ++k_; return tmp;
        }
        const_iterator& operator--() { --k_; return *this; }
        [[nodiscard]] reference operator[](difference_type d) const {
            return (*traj_)[k_ + static_cast<Size>(d)];
        }
        const_iterator& operator+=(difference_type d) {
            k_ = static_cast<Size>(static_cast<difference_type>(k_) + d);
            return *this;
        }
        [[nodiscard]] friend const_iterator operator+(
            const_iterator it, difference_type d) { it += d; return it; }
        [[nodiscard]] friend difference_type operator-(
            const const_iterator& a, const const_iterator& b) {
            return static_cast<difference_type>(a.k_) -
                   static_cast<difference_type>(b.k_);
        }
        [[nodiscard]] friend bool operator==(
            const const_iterator& a, const const_iterator& b) noexcept {
            return a.k_ == b.k_ && a.traj_ == b.traj_;
        }
        [[nodiscard]] friend bool operator!=(
            const const_iterator& a, const const_iterator& b) noexcept {
            return !(a == b);
        }

    private:
        const StateTrajectory* traj_ = nullptr;
        Size k_ = 0;
    };

    [[nodiscard]] const_iterator begin() const noexcept {
        return {this, 0};
    }
    [[nodiscard]] const_iterator end() const noexcept {
        return {this, n_samples_};
    }
    [[nodiscard]] const_iterator cbegin() const noexcept { return begin(); }
    [[nodiscard]] const_iterator cend() const noexcept { return end(); }

    /// Upper bound on the EAGER (pre-run) reservation. Runs whose
    /// full trace exceeds this still record everything — the buffer
    /// simply grows on demand instead of being committed up front.
    /// 256 MiB covers e.g. 10^6 samples x 32 states comfortably.
    static constexpr std::size_t kEagerReserveByteCap =
        std::size_t{256} << 20;

private:
    [[nodiscard]] const Real* row_ptr_(Size k) const noexcept {
        return data_.data() +
               static_cast<std::size_t>(k) *
                   static_cast<std::size_t>(n_state_);
    }

    /// Reserve the flat buffer once BOTH the sample count hint and
    /// the state size are known. Idempotent; never shrinks.
    ///
    /// The eager reservation is CAPPED (adversarial-review finding
    /// contig-02): `expected_sample_count() * state_size * 8 B` can
    /// be many gigabytes for a long high-fidelity run, and
    /// committing that in one block before the first step would
    /// turn a run the user might cancel early (`should_continue`,
    /// live streaming) — or simply mis-specify — into an immediate
    /// bad_alloc. Beyond the cap the buffer grows geometrically
    /// like any std::vector: still amortized O(1) per sample and
    /// still contiguous, just not a single up-front commitment.
    /// Every ordinary run fits under the cap and gets exactly one
    /// allocation.
    void apply_reserve_() {
        if (reserve_samples_ == 0 || n_state_ <= 0) {
            return;
        }
        const auto row = static_cast<std::size_t>(n_state_);
        const auto want = static_cast<std::size_t>(reserve_samples_) * row;
        const std::size_t cap_reals = kEagerReserveByteCap / sizeof(Real);
        const std::size_t take = want < cap_reals ? want : cap_reals;
        if (data_.capacity() < take) {
            data_.reserve(take);
        }
    }

    std::vector<Real> data_;      // row-major, n_samples_ × n_state_
    Index n_state_        = 0;
    Size  n_samples_      = 0;
    Size  reserve_samples_ = 0;
};

}  // namespace pulsim::solver
