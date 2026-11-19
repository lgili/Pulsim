#pragma once

// add-probes-and-recorders: a recorder evaluates probes on the accepted
// steps of a run and keeps what it was asked to keep:
//
//   keep      every accepted sample, every n-th one, or none
//   events    at every instant the feeder reports a right limit: the
//             instant and BOTH limits of every probe
//   windows   per declared window [t0, t1] and/or per period
//             [offset + kT, offset + (k+1)T]: integral, average, RMS,
//             minimum and maximum with the times they occur
//
// The signal model is the engines' dense output: between two accepted
// samples a probe is the straight line from the START of the step (the
// right limit where an event fired at its left end, else the previous
// sample) to the sample at its end (the left limit there). Every
// statistic is exact for that line: windows are clipped exactly at t0 /
// t1 by interpolating on the step that straddles them, the integral of a
// piece is h*(a + b)/2 and the integral of its square h*(a^2 + ab + b^2)/3
// (no squared samples, no trapezoid of y^2). Clipping follows the limits:
// a window that starts exactly at an event starts from the right limit, a
// window that ends exactly at one ends on the left limit; min / max see
// both limits of an event inside the window. This is the arithmetic of
// pulsim.time_average / time_rms, so the two agree to round-off on the
// same samples. A nonlinear probe (v*i) is linear between its own samples,
// as it is when time_average is applied to the product trace.
//
// Feeding:
//   * `sample(t, readings)` once per accepted step; `event(readings)` after
//     it when the feeder has the right limit at that instant (resolution
//     Exact or Settled: the next line starts from it); `event_bracket` when
//     it only brackets the commutation within one step (WithinDt);
//   * in-loop from the step hooks: `record/loop_feed.hpp` (`LoopFeed`, the
//     route of Python's simulate() on pwl / trbdf2), or `attach` below
//     (accepted-step hook, state columns, no events);
//   * a kernel-side pass over stored arrays (`feed_table`): the exact
//     engine and the Python fallback.
// Nothing allocates per sample after `reserve` (period windows and kept
// rows are pre-sized); an event row allocates (events are not per step).

#include "pulsim/events/step_hooks.hpp"
#include "pulsim/numeric/dense.hpp"
#include "pulsim/numeric/types.hpp"
#include "pulsim/record/probe.hpp"
#include "pulsim/streaming/live_ring.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <format>
#include <limits>
#include <memory>
#include <span>
#include <type_traits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

// Forced inline on the per-step path (Recorder::sample and its helpers into
// LoopFeed's step): a call per helper was a third of a step's cost. Empty
// on MSVC, as the code base's other forced-inline macros. NOINLINE keeps
// the rare paths (events, jumps) out of the step's frame.
#if defined(__GNUC__) || defined(__clang__)
#define PULSIM_RECORD_INLINE __attribute__((always_inline)) inline
#define PULSIM_RECORD_NOINLINE __attribute__((noinline))
#elif defined(_MSC_VER)
#define PULSIM_RECORD_INLINE inline
#define PULSIM_RECORD_NOINLINE __declspec(noinline)
#else
#define PULSIM_RECORD_INLINE inline
#define PULSIM_RECORD_NOINLINE
#endif

namespace pulsim::record {

struct Window {
    Real t0;
    Real t1;
};

struct RecorderSpec {
    enum class Keep : std::uint8_t { Every, Decimate, None, Times };
    Keep keep = Keep::Every;
    Size every_n = 1;            // Decimate: keep sample 0, n, 2n, ...
    std::vector<Real> sample_times;  // Times (t_eval): increasing; one row each
    std::vector<Window> windows; // declared windows
    Real period = 0;             // > 0: one window per period
    Real offset = 0;             // first period starts here
};

/// One window's statistics for one probe.
struct WindowStat {
    Real t0 = 0, t1 = 0;
    Real integral = 0;
    Real sq3 = 0;  // 3 * integral of y^2
    Real min = std::numeric_limits<Real>::infinity();
    Real t_min = std::numeric_limits<Real>::quiet_NaN();
    Real max = -std::numeric_limits<Real>::infinity();
    Real t_max = std::numeric_limits<Real>::quiet_NaN();
    bool seen_t0 = false, seen_t1 = false;  // a piece started at t0 / ended at t1
    [[nodiscard]] bool complete() const noexcept { return seen_t0 && seen_t1; }
    [[nodiscard]] Real avg() const noexcept { return integral / (t1 - t0); }
    [[nodiscard]] Real rms() const noexcept {
        return std::sqrt(std::max(sq3 / (3 * (t1 - t0)), Real{0}));
    }
};

class Recorder {
public:
    Recorder(std::vector<Probe> probes, RecorderSpec spec)
        : probes_(std::move(probes)), spec_(std::move(spec)) {
        if (probes_.empty()) throw std::invalid_argument("recorder: no probes");
        if (spec_.keep == RecorderSpec::Keep::Decimate && spec_.every_n == 0) {
            throw std::invalid_argument("recorder: every_n must be >= 1");
        }
        if (!(spec_.period >= 0) || !std::isfinite(spec_.period) || !std::isfinite(spec_.offset)) {
            throw std::invalid_argument("recorder: period must be finite and >= 0, offset finite");
        }
        if (spec_.keep == RecorderSpec::Keep::Times) {
            const auto& ts = spec_.sample_times;
            for (Size k = 0; k < ts.size(); ++k) {
                if (!std::isfinite(ts[k]) || (k > 0 && !(ts[k] > ts[k - 1]))) {
                    throw std::invalid_argument(
                        "recorder: sample times must be finite and strictly increasing");
                }
            }
        }
        for (const auto& w : spec_.windows) {
            if (!(std::isfinite(w.t0) && std::isfinite(w.t1) && w.t1 > w.t0)) {
                throw std::invalid_argument(std::format(
                    "recorder: window [{}, {}] must be finite with t1 > t0", w.t0, w.t1));
            }
        }
        const Size np = probes_.size();
        declared_.resize(spec_.windows.size() * np);
        for (Size w = 0; w < spec_.windows.size(); ++w) {
            for (Size p = 0; p < np; ++p) {
                declared_[w * np + p].t0 = spec_.windows[w].t0;
                declared_[w * np + p].t1 = spec_.windows[w].t1;
            }
        }
        stats_ = !spec_.windows.empty() || spec_.period > 0;
        left_.resize(np);
        start_.resize(np);
        right_.resize(np);
        row_.resize(np);
    }

    /// Map every probe reading to a column of the feeder's reading row.
    /// Throws std::invalid_argument naming the first reading not offered.
    void bind(std::span<const std::string> reading_names) {
        Size max_atoms = 0;
        cols_.clear();
        offs_.clear();
        src_.clear();
        for (const auto& pr : probes_) {
            offs_.push_back(cols_.size());
            const Size first = cols_.size();
            for (const auto& a : pr.atoms()) {
                const auto it = std::find(reading_names.begin(), reading_names.end(), a);
                if (it == reading_names.end()) {
                    throw std::invalid_argument(std::format(
                        "probe \"{}\": reading \"{}\" is not available", pr.text(), a));
                }
                cols_.push_back(static_cast<Size>(it - reading_names.begin()));
            }
            // a plain probe IS one reading: a load and a store per sample
            src_.push_back(pr.plain() ? cols_[first] : kNotPlain);
            max_atoms = std::max(max_atoms, pr.atoms().size());
        }
        offs_.push_back(cols_.size());
        n_readings_ = reading_names.size();
        scratch_.resize(max_atoms);
        bound_ = true;
    }

    /// Pre-size the kept rows for `n_samples` accepted samples and the
    /// period windows up to `t_last`.
    void reserve(Size n_samples, Real t_last) {
        const Size np = probes_.size();
        Size rows = 0;
        if (spec_.keep == RecorderSpec::Keep::Every) rows = n_samples;
        if (spec_.keep == RecorderSpec::Keep::Decimate) rows = n_samples / spec_.every_n + 1;
        if (spec_.keep == RecorderSpec::Keep::Times) rows = spec_.sample_times.size();
        times_.reserve(rows);
        if (spec_.keep == RecorderSpec::Keep::Times) interpolated_.reserve(rows);
        values_.reserve(rows * np);
        if (spec_.period > 0 && t_last > spec_.offset) {
            const Real nf = std::floor((t_last - spec_.offset) / spec_.period) + 1;
            if (!(nf <= 1e8)) {
                throw std::invalid_argument(std::format(
                    "recorder: period {} gives more than 1e8 windows up to t = {}",
                    spec_.period, t_last));
            }
            const auto n = static_cast<Size>(nf);
            ensure_period(n + 2);  // piece() looks one window past t_last
        }
    }

    /// Pre-size the event rows for `n_events` events.
    void reserve_events(Size n_events) {
        const Size np = probes_.size();
        event_t_.reserve(n_events);
        event_t_right_.reserve(n_events);
        event_resolution_.reserve(n_events);
        event_left_.reserve(n_events * np);
        event_right_.reserve(n_events * np);
    }

    /// The left limit at `t`: the end of an accepted step.
    PULSIM_RECORD_INLINE void sample(Real t, std::span<const Real> readings) {
        require_bound(readings);
        if (n_samples_ > 0 && !(t >= t_prev_)) [[unlikely]] {
            bad_order(t);
        }
        eval_into(readings, left_);
        if (stats_ && n_samples_ > 0 && t > t_prev_) {
            if (spec_.windows.empty() && in_fast_window(t_prev_, t)) {
                accumulate_window(t_prev_, t);
            } else {
                piece(t_prev_, t);
            }
        }
        if (spec_.keep == RecorderSpec::Keep::Every) {
            keep_row(t, left_);
        } else if (spec_.keep == RecorderSpec::Keep::Times) {
            emit_times(t);
        } else if (spec_.keep == RecorderSpec::Keep::Decimate &&
                   n_samples_ % spec_.every_n == 0) {
            keep_row(t, left_);
        }
        const Size np = left_.size();
        Real* const st = start_.data();
        const Real* const lf = left_.data();
        for (Size p = 0; p < np; ++p) st[p] = lf[p];
        t_prev_ = t;
        ++n_samples_;
    }

    /// How an event row's right value was obtained: the engine's exact
    /// right limit, its settled post-event state at the instant, or a
    /// bracket within one step (`event_bracket`).
    enum class Resolution : std::uint8_t { Exact = 0, Settled = 1, WithinDt = 2 };
    /// The right limit at the last sample's instant (after its settle):
    /// the next step starts from it.
    void event(std::span<const Real> readings_right, Resolution res = Resolution::Exact) {
        require_bound(readings_right);
        if (n_samples_ == 0) throw std::logic_error("recorder: event before any sample");
        eval_into(readings_right, right_);
        push_event(t_prev_, res);
        start_ = right_;
    }

    /// A commutation the feeder only BRACKETS: the left limit is the last
    /// sample (at t_prev), `readings_right` the state at `t_right` > t_prev
    /// under the new mode (the fixed engine: the end of the first step
    /// under the new stamp). Reported as an event row with resolution
    /// `WithinDt`; the record's line is not changed (no right limit at the
    /// instant is fabricated). Call it before the sample at `t_right`.
    void event_bracket(Real t_right, std::span<const Real> readings_right) {
        require_bound(readings_right);
        if (n_samples_ == 0) throw std::logic_error("recorder: event before any sample");
        eval_into(readings_right, right_);
        push_event(t_right, Resolution::WithinDt);
    }

    /// End of the run: a requested time at the last sample takes its value
    /// (the right limit when an event fired there). Times past the last
    /// sample, or before the first, are not reached (`n_times_unreached`).
    void flush() {
        if (spec_.keep != RecorderSpec::Keep::Times) return;
        const auto& ts = spec_.sample_times;
        while (n_samples_ > 0 && next_time_ < ts.size() &&
               (ts[next_time_] == t_prev_ || close_to(ts[next_time_], t_prev_))) {
            keep_row(ts[next_time_], start_, false);
            ++next_time_;
        }
        unreached_ += ts.size() - next_time_;
        next_time_ = ts.size();
    }

    /// RingSink: every kept row (n_probes values) is also pushed into
    /// `ring` (its state size must be n_probes; its own decimation applies).
    void stream_to(streaming::LiveRing* ring) {
        if (ring != nullptr && ring->state_size() != probes_.size()) {
            throw std::invalid_argument(std::format(
                "recorder: ring state size {} != {} probes", ring->state_size(),
                probes_.size()));
        }
        ring_ = ring;
    }

    // ---- snapshot (task 0b.7): everything a resumed run continues from ----
    /// Append the recorder's run state to `out` (flat; see `restore`): the
    /// last sample and the line's start (a partial window ends there), the
    /// counters, kept rows, event rows and every window's statistics.
    void save(std::vector<Real>& out) const {
        const auto put = [&](Real v) { out.push_back(v); };
        const auto vec = [&](const auto& v) {
            put(static_cast<Real>(v.size()));
            out.insert(out.end(), v.begin(), v.end());   // one pass (kept rows are long)
        };
        put(static_cast<Real>(probes_.size()));
        put(t_prev_);
        put(static_cast<Real>(n_samples_));
        put(static_cast<Real>(next_time_));
        put(static_cast<Real>(unreached_));
        vec(start_);
        vec(left_);
        vec(times_);
        vec(values_);
        vec(interpolated_);
        vec(event_t_);
        vec(event_t_right_);
        vec(event_resolution_);
        vec(event_left_);
        vec(event_right_);
        for (const auto* ws : {&declared_, &periodic_}) {
            put(static_cast<Real>(ws->size()));
            for (const auto& w : *ws) {
                for (const Real v : {w.t0, w.t1, w.integral, w.sq3, w.min, w.t_min, w.max,
                                     w.t_max, Real(w.seen_t0), Real(w.seen_t1)}) {
                    put(v);
                }
            }
        }
    }
    /// Read what `save` wrote from `in` at `pos` (advanced past it). Throws
    /// std::invalid_argument when it does not fit this recorder.
    void restore(std::span<const Real> in, Size& pos) {
        const auto get = [&]() -> Real {
            if (pos >= in.size()) throw std::invalid_argument("recorder state: truncated");
            return in[pos++];
        };
        const auto count = [&]() {
            const Real v = get();
            if (!(v >= 0 && v <= static_cast<Real>(in.size()))) {
                throw std::invalid_argument("recorder state: bad length");
            }
            return static_cast<Size>(v);
        };
        const auto vec = [&](auto& v) {
            using T = typename std::remove_reference_t<decltype(v)>::value_type;
            const Size n = count();
            v.resize(n);
            for (Size k = 0; k < n; ++k) v[k] = static_cast<T>(get());
        };
        if (count() != probes_.size()) {
            throw std::invalid_argument("recorder state: another number of probes");
        }
        const auto counter = [&]() {   // a count, not a length: no size bound
            const Real v = get();
            if (!(v >= 0 && v <= Real{9007199254740992.0} && v == std::floor(v))) {
                throw std::invalid_argument("recorder state: bad counter");
            }
            return static_cast<Size>(v);
        };
        t_prev_ = get();
        n_samples_ = counter();
        next_time_ = counter();
        unreached_ = counter();
        vec(start_);
        vec(left_);
        if (start_.size() != probes_.size() || left_.size() != probes_.size()) {
            throw std::invalid_argument("recorder state: bad line");
        }
        vec(times_);
        vec(values_);
        vec(interpolated_);
        vec(event_t_);
        vec(event_t_right_);
        vec(event_resolution_);
        vec(event_left_);
        vec(event_right_);
        for (auto* ws : {&declared_, &periodic_}) {
            const Size n = count();
            if (ws == &declared_ && n != declared_.size()) {
                throw std::invalid_argument("recorder state: other declared windows");
            }
            ws->resize(n);
            kc_np_ = kNoWindow;   // the period fast path re-finds its window
            for (auto& w : *ws) {
                w.t0 = get(); w.t1 = get(); w.integral = get(); w.sq3 = get();
                w.min = get(); w.t_min = get(); w.max = get(); w.t_max = get();
                w.seen_t0 = get() != 0; w.seen_t1 = get() != 0;
            }
        }
    }

    // ---- results ----
    [[nodiscard]] Size n_probes() const noexcept { return probes_.size(); }
    [[nodiscard]] const std::vector<Probe>& probes() const noexcept { return probes_; }
    [[nodiscard]] const RecorderSpec& spec() const noexcept { return spec_; }
    [[nodiscard]] Size n_samples() const noexcept { return n_samples_; }
    /// Kept rows: `times()[r]`, `values()[r * n_probes() + p]`.
    [[nodiscard]] const std::vector<Real>& times() const noexcept { return times_; }
    [[nodiscard]] const std::vector<Real>& values() const noexcept { return values_; }
    /// Times: per kept row, 1 when it lies strictly inside a step (linear
    /// on the dense-output line), 0 when it is a sample.
    [[nodiscard]] const std::vector<std::uint8_t>& interpolated() const noexcept {
        return interpolated_;
    }
    [[nodiscard]] Size n_times_unreached() const noexcept { return unreached_; }
    [[nodiscard]] const std::vector<Real>& event_times() const noexcept { return event_t_; }
    [[nodiscard]] const std::vector<Real>& event_left() const noexcept { return event_left_; }
    [[nodiscard]] const std::vector<Real>& event_right() const noexcept { return event_right_; }
    /// Per event row: where the right value was taken (`t_right == t` but
    /// for a bracket) and how (`Resolution`).
    [[nodiscard]] const std::vector<Real>& event_t_right() const noexcept { return event_t_right_; }
    [[nodiscard]] const std::vector<std::uint8_t>& event_resolution() const noexcept {
        return event_resolution_;
    }
    /// Declared windows: `[w * n_probes() + p]`.
    [[nodiscard]] const std::vector<WindowStat>& windows() const noexcept { return declared_; }
    /// Period windows k = 0, 1, ... (complete or not): `[k * n_probes() + p]`.
    [[nodiscard]] const std::vector<WindowStat>& periods() const noexcept { return periodic_; }

private:
    std::vector<Probe> probes_;
    RecorderSpec spec_;
    static constexpr Size kNotPlain = static_cast<Size>(-1);
    static constexpr Size kNoWindow = static_cast<Size>(-1);
    std::vector<Size> cols_, offs_;
    std::vector<Size> src_;   // per probe: its reading for a plain probe, else kNotPlain
    Size n_readings_ = 0;
    bool bound_ = false;
    bool stats_ = false;      // declared windows or periods
    // Period fast path: the window holding the last piece's end (its
    // first entry, k * n_probes) and the open interval in which a piece
    // can only touch that window, without a snap (see piece()).
    Size kc_np_ = kNoWindow;
    Real fast_lo_ = 0, fast_hi_ = 0;
    std::vector<Real> scratch_, left_, start_, right_, row_;
    Real t_prev_ = 0;
    Size n_samples_ = 0;
    std::vector<Real> times_, values_;
    std::vector<Real> event_t_, event_left_, event_right_, event_t_right_;
    std::vector<std::uint8_t> event_resolution_;
    std::vector<WindowStat> declared_, periodic_;
    std::vector<std::uint8_t> interpolated_;
    Size next_time_ = 0, unreached_ = 0;
    streaming::LiveRing* ring_ = nullptr;

    void push_event(Real t_right, Resolution res) {
        event_t_.push_back(t_prev_);
        event_t_right_.push_back(t_right);
        event_resolution_.push_back(static_cast<std::uint8_t>(res));
        event_left_.insert(event_left_.end(), left_.begin(), left_.end());
        event_right_.insert(event_right_.end(), right_.begin(), right_.end());
    }

    PULSIM_RECORD_INLINE void keep_row(Real t, const std::vector<Real>& v, bool interp = false) {
        times_.push_back(t);
        for (const Real x : v) values_.push_back(x);
        if (spec_.keep == RecorderSpec::Keep::Times) interpolated_.push_back(interp ? 1 : 0);
        if (ring_ != nullptr && ring_->should_decimate()) ring_->push(t, v.data(), v.size());
    }

    // Requested times in [t_prev_, t): on the line from start_ (the right
    // limit at t_prev_ when an event fired there) to left_. A time at t
    // waits for the next sample or flush(), which know whether t is an event.
    void emit_times(Real t) {
        const auto& ts = spec_.sample_times;
        while (next_time_ < ts.size() && ts[next_time_] < t && !close_to(ts[next_time_], t)) {
            const Real tau = ts[next_time_++];
            if (n_samples_ == 0 || tau < t_prev_) {
                if (!(n_samples_ > 0 && close_to(tau, t_prev_))) { ++unreached_; continue; }
            }
            if (tau == t_prev_ || close_to(tau, t_prev_)) {
                keep_row(tau, start_, false);
                continue;
            }
            for (Size p = 0; p < probes_.size(); ++p) {
                row_[p] = lerp(t_prev_, start_[p], t, left_[p], tau);
            }
            keep_row(tau, row_, true);
        }
    }

    PULSIM_RECORD_INLINE void require_bound(std::span<const Real> readings) const {
        if (!bound_ || readings.size() != n_readings_) [[unlikely]] {
            bad_feed(readings.size());
        }
    }
    [[noreturn]] void bad_feed(Size n) const {
        if (!bound_) throw std::logic_error("recorder: bind() before feeding");
        throw std::invalid_argument(
            std::format("recorder: {} readings fed, {} bound", n, n_readings_));
    }
    [[noreturn]] void bad_order(Real t) const {
        throw std::invalid_argument(std::format(
            "recorder: sample at t = {} before the previous one at {}", t, t_prev_));
    }

    PULSIM_RECORD_INLINE void eval_into(std::span<const Real> readings, std::vector<Real>& out) {
        const Size np = probes_.size();
        for (Size p = 0; p < np; ++p) {
            if (src_[p] != kNotPlain) {
                out[p] = readings[src_[p]];
                continue;
            }
            const Size n = offs_[p + 1] - offs_[p];
            for (Size k = 0; k < n; ++k) scratch_[k] = readings[cols_[offs_[p] + k]];
            out[p] = probes_[p].eval(std::span<const Real>(scratch_.data(), n));
        }
    }

    void ensure_period(Size n) {
        const Size np = probes_.size();
        const Size have = periodic_.size() / np;
        if (n <= have) return;
        periodic_.resize(n * np);
        for (Size k = have; k < n; ++k) {
            // multiplied, not accumulated: the edges are the same on every call
            const Real t0 = spec_.offset + static_cast<Real>(k) * spec_.period;
            const Real t1 = spec_.offset + static_cast<Real>(k + 1) * spec_.period;
            for (Size p = 0; p < np; ++p) {
                periodic_[k * np + p].t0 = t0;
                periodic_[k * np + p].t1 = t1;
            }
        }
    }

    // pulsim.stats._lerp, operation for operation
    static Real lerp(Real ta, Real ya, Real tb, Real yb, Real t) noexcept {
        return ya + (yb - ya) * ((t - ta) / (tb - ta));
    }

    // Within 4 ulps an edge IS the sample: `3 * 0.1` is one ulp above a
    // run that ended at 0.3, and a period edge one ulp before an event
    // would start on the left limit. Pieces arrive in time order, so an
    // edge snaps to the first sample near it before any piece uses it.
    static bool close_to(Real a, Real b) noexcept {
        return std::abs(a - b) <= 4 * std::numeric_limits<Real>::epsilon() *
                                      std::max(std::abs(a), std::abs(b));
    }
    static void snap(Real& edge, Real ta, Real tb) noexcept {
        if (edge != ta && close_to(edge, ta)) edge = ta;
        else if (edge != tb && close_to(edge, tb)) edge = tb;
    }

    // The piece (ta, tb] runs from start_ (at ta) to left_ (at tb).
    void accumulate(WindowStat& w, Real ta, Real tb, Real ya, Real yb) const noexcept {
        snap(w.t0, ta, tb);
        snap(w.t1, ta, tb);
        const Real lo = std::max(w.t0, ta), hi = std::min(w.t1, tb);
        if (!(hi > lo)) return;
        const Real a = lo == ta ? ya : lerp(ta, ya, tb, yb, lo);
        const Real b = hi == tb ? yb : lerp(ta, ya, tb, yb, hi);
        const Real h = hi - lo;
        w.integral += 0.5 * (a + b) * h;
        w.sq3 += (a * a + a * b + b * b) * h;
        if (a < w.min) { w.min = a; w.t_min = lo; }
        if (b < w.min) { w.min = b; w.t_min = hi; }
        if (a > w.max) { w.max = a; w.t_max = lo; }
        if (b > w.max) { w.max = b; w.t_max = hi; }
        w.seen_t0 = w.seen_t0 || lo == w.t0;
        w.seen_t1 = w.seen_t1 || hi == w.t1;
    }

    // Period fast path: the piece lies inside one window, farther than the
    // margin from its edges, so piece()'s general loop would skip every
    // other window, snap nothing and clip nothing (lo = ta, hi = tb): the
    // same operations as accumulate() on that window alone.
    [[nodiscard]] PULSIM_RECORD_INLINE bool in_fast_window(Real ta, Real tb) const noexcept {
        return kc_np_ != kNoWindow && ta > fast_lo_ && tb < fast_hi_;
    }
    PULSIM_RECORD_INLINE void accumulate_window(Real ta, Real tb) noexcept {
        const Size np = probes_.size();
        const Real h = tb - ta;
        WindowStat* const ws = periodic_.data() + kc_np_;
        const Real* const st = start_.data();
        const Real* const lf = left_.data();
        for (Size p = 0; p < np; ++p) {
            WindowStat& w = ws[p];
            const Real a = st[p], b = lf[p];
            w.integral += 0.5 * (a + b) * h;
            w.sq3 += (a * a + a * b + b * b) * h;
            if (a < w.min) { w.min = a; w.t_min = ta; }
            if (b < w.min) { w.min = b; w.t_min = tb; }
            if (a > w.max) { w.max = a; w.t_max = ta; }
            if (b > w.max) { w.max = b; w.t_max = tb; }
        }
    }

    void piece(Real ta, Real tb) {
        const Size np = probes_.size();
        for (Size w = 0; w < spec_.windows.size(); ++w) {
            for (Size p = 0; p < np; ++p) {
                accumulate(declared_[w * np + p], ta, tb, start_[p], left_[p]);
            }
        }
        if (spec_.period <= 0 || tb <= spec_.offset) return;
        if (in_fast_window(ta, tb)) {
            accumulate_window(ta, tb);
            return;
        }
        const Real T = spec_.period;
        const Real ka = std::max(Real{0}, std::floor((ta - spec_.offset) / T));
        const auto k0 = static_cast<Size>(ka);
        // one window back / forward: floor() of a quotient can land one off
        // the window whose multiplied edges hold the time
        const auto k1 = static_cast<Size>(std::floor((tb - spec_.offset) / T)) + 1;
        ensure_period(k1 + 1);
        for (Size k = k0 == 0 ? 0 : k0 - 1; k <= k1; ++k) {
            // a window wholly outside the piece (and not within the 4-ulp
            // snap of its ends) takes nothing: skip it before the snaps
            const WindowStat& w0 = periodic_[k * np];
            if ((w0.t1 < ta && !close_to(w0.t1, ta)) || (w0.t0 > tb && !close_to(w0.t0, tb))) {
                continue;
            }
            for (Size p = 0; p < np; ++p) {
                accumulate(periodic_[k * np + p], ta, tb, start_[p], left_[p]);
            }
        }
        // The window that holds tb strictly inside it is the fast path's
        // next window. Margin: 64 ulps of its edges, beyond the 4-ulp snap
        // of this window and of its neighbours' edges (each within a snap of
        // the shared multiplied edge), so a piece inside the margins touches
        // no other window and moves no edge. Only when the windows a piece
        // there could ask for (k1 + 1 <= kc + 3) exist already: the fast path
        // never grows the window table.
        kc_np_ = kNoWindow;
        for (Size k = k0 == 0 ? 0 : k0 - 1; k <= k1; ++k) {
            const WindowStat& w0 = periodic_[k * np];
            if (!(w0.t0 < tb && tb < w0.t1)) continue;
            const Real m = 64 * std::numeric_limits<Real>::epsilon() *
                           std::max(std::abs(w0.t0), std::abs(w0.t1));
            if (w0.t0 + m < w0.t1 - m && periodic_.size() >= (k + 3) * np) {
                kc_np_ = k * np;
                fast_lo_ = w0.t0 + m;
                fast_hi_ = w0.t1 - m;
            }
            break;
        }
    }
};

/// Kernel-side pass over a stored record: `readings` is row-major
/// (n_rows x n_readings), row k the left limit at `t[k]`; `event_rows[j]`
/// (strictly increasing) carries the right limit `right` row j at that
/// instant (the exact engine's stored right limits).
inline void feed_table(Recorder& rec, std::span<const Real> t, std::span<const Real> readings,
                       Size n_readings, std::span<const Size> event_rows,
                       std::span<const Real> right) {
    const Size n = t.size();
    if (readings.size() != n * n_readings || right.size() != event_rows.size() * n_readings) {
        throw std::invalid_argument("feed_table: reading arrays do not match their row counts");
    }
    rec.reserve(n, n == 0 ? Real{0} : t[n - 1]);
    Size e = 0;
    for (Size k = 0; k < n; ++k) {
        rec.sample(t[k], readings.subspan(k * n_readings, n_readings));
        if (e < event_rows.size() && event_rows[e] == k) {
            rec.event(right.subspan(e * n_readings, n_readings));
            ++e;
        }
    }
    if (e != event_rows.size()) {
        throw std::invalid_argument("feed_table: event rows must be increasing row indices");
    }
    rec.flush();
}

/// In-loop feeding from the engines' accepted-step hook. `columns[j]` is
/// the state index of reading j (-1 = ground, 0 V). The hook carries no
/// start state, so the caller seeds the record with the run's initial
/// sample (`rec.sample(t_start, ...)`) when it knows it; otherwise the
/// record starts at the end of the first accepted step. No right limits
/// (the hook has none): see DECISIONS "requests to the stepper track".
/// With `Keep::Times` the caller calls `rec.flush()` after the run.
inline void attach(events::StepHooks& hooks, Recorder& rec, std::vector<Index> columns) {
    auto buf = std::make_shared<std::vector<Real>>(columns.size());
    hooks.on_accept([&rec, buf, cols = std::move(columns)](
                        Real, Real t, const Vector& x, const topology::SwitchStateMask&) {
        for (Size j = 0; j < cols.size(); ++j) {
            (*buf)[j] = cols[j] < 0 ? Real{0} : x[cols[j]];
        }
        rec.sample(t, *buf);
    });
}

}  // namespace pulsim::record
