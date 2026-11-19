#pragma once

// =============================================================================
// Pulsim — in-loop feeding of recorders from the step hooks (task 0b.6)
// =============================================================================
//
// `LoopFeed` owns a run's recorders and feeds them from `StepHooks` while
// the engine runs, so nothing is read back from the stored arrays: a run
// with `store_every = n` (or one that stores nothing a recorder needs)
// still gives statistics over EVERY accepted step.
//
// Readings are functions of the state vector a hook delivers:
//   reading = (Σ c_k x[col_k]) / div            (linear: v(n), v(a,b), a
//                                                state column, a resistor)
//           = lin_a(x) · lin_b(x)               (product: p() of those)
// A reading that is not of this form (a switched device's law under the
// mask, a Newton device) is not offered here; the caller keeps the
// stored-array pass for that run (and says so).
//
// Per accepted step [t0, t] (`StepHooks::step`):
//   * the first step seeds the record with the start state at t0;
//   * a start state that differs from the previous step's end (bitwise in
//     x, or in the mask) is a jump the engine made at t0: an event row with
//     the start state as its right limit (resolution `Settled`, `n_jumps`),
//     and the line of the step starts from it;
//   * the end state is the sample at t.
// Commutations (`StepHooks::on_event`):
//   * fixed engine (`Events::Bracket`): the pair brackets the commutation
//     within one step; recorded with `event_bracket` (left at the step's
//     start, right at its end, resolution `WithinDt`; the line is not moved);
//   * variable-step engine (`Events::Settled`): the right side is the
//     settled post-event state at the instant: an event row there, the next
//     line starts from it. Before the step hook that lands t (a diode
//     landing) it waits for that sample; after it (t == the last sample) it
//     applies at once; before a hook step that ALSO covers the committed
//     post-event probe (a landed gate edge: [t0, t + dtp] as one step) the
//     event's left side is the sample at t. A step solved under a new mode
//     with no settled report at its start is bracketed [t0, t] (`WithinDt`).
// Reports the feed cannot place are counted (`n_dropped`), never invented.
// A pure observer: the callbacks read their arguments only. No allocation
// per step for keep = none / decimate-free statistics once `reserve` has
// sized the period windows; kept rows and event rows grow their vectors.

#include "pulsim/events/step_hooks.hpp"
#include "pulsim/numeric/dense.hpp"
#include "pulsim/numeric/types.hpp"
#include "pulsim/pwl/device_pool.hpp"
#include "pulsim/pwl/device_readings.hpp"
#include "pulsim/pwl/history_state.hpp"
#include "pulsim/pwl/pwl_branch_law.hpp"
#include "pulsim/pwl/stateful_histories.hpp"
#include "pulsim/record/recorder.hpp"
#include "pulsim/topology/graph.hpp"
#include "pulsim/topology/switch_state.hpp"

#include <cstdint>
#include <cstring>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace pulsim::record {

struct LinearReading {
    std::vector<Index> cols;  ///< state columns (no ground: the caller drops it)
    std::vector<Real> coef;
    Real div = Real{1};

    [[nodiscard]] Real eval(const Vector& x) const noexcept {
        Real s = Real{0};
        for (Size k = 0; k < cols.size(); ++k) s += coef[k] * x[cols[k]];
        return div == Real{1} ? s : s / div;
    }
};

/// A device's current (or absorbed power) read under the mode the engine
/// COMMITTED, at the committed state (item 1, DECISIONS 20):
///   Pwl        PWL switch / diode: pwl::PwlBranchLaw under the stamped mask
///              (the loss tracker's law);
///   Capacitor  the companion's committed current (`HistoryState` i_prev:
///              the current the solver enforced, so KCL holds at the sample
///              and the trapezoid of i is C·Δv on a trapezoidal step); at a
///              settled event's right side, the post-event probe's own
///              companion current (as `refresh_derivatives` computes it);
///   Static     a Newton branch's law (Shockley / smooth diode, MOSFET,
///              IGBT level 1: pwl::static_reading);
///   Stateful   the summed readings of a stateful device's branch
///              (saturable inductor, nonlinear capacitor, Lauritzen diode,
///              IGBT tail: pwl::for_each_stateful_reading, committed rates);
///   Waveform / Constant   a current source's value at t.
struct DeviceCurrent {
    enum class Law : std::uint8_t { None, Pwl, Capacitor, Static, Stateful, Waveform, Constant };
    Law law = Law::None;
    Index branch = -1;
    bool power = false;     ///< p(): v(from, to)·i, or Σ v_pin·i_pin of a Newton law
    pwl::PwlBranchLaw pwl{};
    Real value = 0;         ///< unused (kept for the binding's tuple)
    Index entry = -1;       ///< Capacitor: its HistoryState entry (bound at the run's start)
};

struct StateReading {
    LinearReading a;
    bool product = false;
    LinearReading b;  ///< used when `product`
    DeviceCurrent dev;  ///< law != None: a device reading (a = v(from, to) for power)

    [[nodiscard]] Real eval(const Vector& x) const noexcept {
        const Real va = a.eval(x);
        return product ? va * b.eval(x) : va;
    }
};

class LoopFeed {
public:
    LoopFeed(std::vector<std::string> reading_names, std::vector<StateReading> readings,
             std::vector<Recorder> recorders)
        : names_(std::move(reading_names)), readings_(std::move(readings)),
          recs_(std::move(recorders)) {
        if (names_.size() != readings_.size()) {
            throw std::invalid_argument("LoopFeed: one reading per name");
        }
        for (const auto& r : readings_) {
            if (r.a.cols.size() != r.a.coef.size() || r.b.cols.size() != r.b.coef.size()) {
                throw std::invalid_argument("LoopFeed: a reading's columns and coefficients differ");
            }
        }
        for (auto& r : recs_) r.bind(names_);
        row_.resize(names_.size());
    }

    LoopFeed(const LoopFeed&) = delete;
    LoopFeed& operator=(const LoopFeed&) = delete;

    /// The circuit a device reading reads (graph, pool: they must outlive
    /// the runs). Resolves each device reading's law from the pool's kind;
    /// throws std::invalid_argument naming a reading no law covers.
    void set_circuit(const topology::Graph& graph, const pwl::DevicePool& pool) {
        using SK = pwl::DevicePool::StoredKind;
        using L = DeviceCurrent::Law;
        for (Size j = 0; j < readings_.size(); ++j) {
            auto& d = readings_[j].dev;
            if (d.branch < 0) continue;
            if (d.branch >= graph.num_branches() || !pool.is_registered(d.branch)) {
                throw std::invalid_argument("LoopFeed: reading " + names_[j] +
                                            " names no device of this circuit");
            }
            switch (pool.kind_of(d.branch)) {
            case SK::Switch:
            case SK::Diode:
                d.law = L::Pwl;
                d.pwl = pwl::PwlBranchLaw::of(graph, pool, d.branch);
                break;
            case SK::Capacitor: d.law = L::Capacitor; break;
            case SK::NonlinearDiode:
            case SK::ShockleyDiode:
            case SK::MosfetLevel1:
            case SK::IgbtLevel1: d.law = L::Static; break;
            case SK::SaturableInductor:
            case SK::NonlinearCapacitor:
            case SK::LauritzenDiode: d.law = L::Stateful; break;
            case SK::SineCurrentSource:
            case SK::PulseCurrentSource:
            case SK::PwlCurrentSource: d.law = L::Waveform; break;
            case SK::CurrentSource: d.law = L::Constant; break;
            default:
                throw std::invalid_argument(
                    "LoopFeed: reading " + names_[j] + ": a " +
                    std::string(pwl::stored_kind_name(pool.kind_of(d.branch))) +
                    " has no in-loop current law");
            }
            d.entry = -1;
        }
        graph_ = &graph;
        pool_ = &pool;
    }

    /// Pre-size kept rows for `n_samples` and the period windows up to t_last.
    void reserve(Size n_samples, Real t_last, Size n_events = 0) {
        for (auto& r : recs_) {
            r.reserve(n_samples, t_last);
            if (n_events > 0) r.reserve_events(n_events);
        }
    }

    /// How the engine reports a commutation: the fixed engine BRACKETS it
    /// (`on_event` pairs a step-start state with the end of the first step
    /// under the new stamp: `event_bracket`, resolution WithinDt); the
    /// variable-step engine SETTLES it at the instant (`on_event`'s right
    /// side is the settled post-event state: an event row at t with
    /// resolution Settled, and the next step's line starts from it).
    enum class Events : std::uint8_t { Bracket, Settled };

    /// Register on `hooks` (which must outlive the run, and this object it).
    void attach(events::StepHooks& hooks, Events kind) {
        if (attached_) throw std::logic_error("LoopFeed: already attached (every sample would repeat)");
        view_of_ = &hooks;
        hooks.on_start([this](const Vector& x0, const topology::SwitchStateMask& m0) {
            on_start(x0, m0);
        });
        hooks.on_step([this](Real t0, const Vector& xs, const topology::SwitchStateMask& ms,
                             Real t, const Vector& xe, const topology::SwitchStateMask& me) {
            on_step(t0, xs, ms, t, xe, me);
        });
        settled_ = kind == Events::Settled;
        hooks.on_event([this](Real t, const Vector& xl, const topology::SwitchStateMask& ml,
                              const Vector& xr, const topology::SwitchStateMask& mr) {
            on_event(t, xl, ml, xr, mr);
        });
        pending_.resize(names_.size());
        pending_left_.resize(names_.size());
        attached_ = true;
    }

    /// End of the run (`Keep::Times` rows at the last sample).
    void finish() {
        for (auto& r : recs_) r.flush();
    }

    /// `attach` on the feed's own hooks (the Python binding's route: the
    /// hooks then live exactly as long as the feed).
    void attach_own(bool bracket_events) {
        attach(hooks_, bracket_events ? Events::Bracket : Events::Settled);
    }
    [[nodiscard]] events::StepHooks& hooks() noexcept { return hooks_; }
    [[nodiscard]] bool attached() const noexcept { return attached_; }
    [[nodiscard]] Size n_steps() const noexcept { return n_steps_; }
    [[nodiscard]] Size n_jumps() const noexcept { return n_jumps_; }
    [[nodiscard]] Size n_events() const noexcept { return n_events_; }
    /// `on_event` reports the feed could not place (an instant that is not
    /// the last or the next sample): counted, never fabricated.
    [[nodiscard]] Size n_dropped() const noexcept { return n_dropped_; }
    [[nodiscard]] const std::vector<Recorder>& recorders() const noexcept { return recs_; }

    /// Task 0b.7: the feed's run state at the end of a run (it goes into
    /// the final snapshot): the last sample, its mask, a settled event at
    /// it, the counters, and every recorder's state. Flat, versioned.
    [[nodiscard]] std::vector<Real> save() const {
        if (has_pending_) {
            // an event reported for a sample the run never reached: a
            // snapshot cannot carry it without inventing its left side
            throw std::logic_error("LoopFeed: an event report is pending at the run's end");
        }
        std::vector<Real> out{Real{1}, static_cast<Real>(recs_.size()),
                              static_cast<Real>(names_.size()), Real(started_),
                              Real(event_at_last_), t_last_, static_cast<Real>(n_steps_),
                              static_cast<Real>(n_jumps_), static_cast<Real>(n_events_),
                              static_cast<Real>(n_dropped_),
                              static_cast<Real>(x_last_.size())};
        for (Index k = 0; k < x_last_.size(); ++k) out.push_back(x_last_[k]);
        out.push_back(static_cast<Real>(mask_last_.size()));
        for (Size k = 0; k < mask_last_.size(); ++k) out.push_back(Real(mask_last_.get(k)));
        for (const auto& r : recs_) r.save(out);
        return out;
    }

    /// Continue from `save()`'s output (before the resumed run). Throws
    /// std::invalid_argument when it was saved by another set of recorders.
    void restore(std::span<const Real> in) {
        Size pos = 0;
        const auto get = [&]() -> Real {
            if (pos >= in.size()) throw std::invalid_argument("recorder state: truncated");
            return in[pos++];
        };
        if (get() != Real{1}) throw std::invalid_argument("recorder state: unknown version");
        if (static_cast<Size>(get()) != recs_.size() ||
            static_cast<Size>(get()) != names_.size()) {
            throw std::invalid_argument("recorder state: saved by other recorders");
        }
        started_ = get() != 0;
        event_at_last_ = get() != 0;
        t_last_ = get();
        n_steps_ = static_cast<Size>(get());
        n_jumps_ = static_cast<Size>(get());
        n_events_ = static_cast<Size>(get());
        n_dropped_ = static_cast<Size>(get());
        const auto nx = static_cast<Index>(get());
        if (nx < 0 || static_cast<Size>(nx) > in.size()) {
            throw std::invalid_argument("recorder state: bad state size");
        }
        x_last_.resize(nx);
        for (Index k = 0; k < nx; ++k) x_last_[k] = get();
        const auto nm = static_cast<Size>(get());
        if (nm > in.size()) throw std::invalid_argument("recorder state: bad mask size");
        mask_last_ = topology::SwitchStateMask(nm);
        for (Size k = 0; k < nm; ++k) mask_last_.set(k, get() != 0);
        for (auto& r : recs_) r.restore(in, pos);
        if (pos != in.size()) throw std::invalid_argument("recorder state: trailing data");
        has_pending_ = false;
        restored_ = true;
    }
    /// The feed continues a run from a snapshot's recorder state.
    [[nodiscard]] bool restored() const noexcept { return restored_; }

private:
    std::vector<std::string> names_;
    std::vector<StateReading> readings_;
    std::vector<Recorder> recs_;
    events::StepHooks hooks_;
    std::vector<Real> row_, pending_, pending_left_;
    Vector x_last_, x_seed_;
    std::vector<Real> seed_;
    bool seeded_ = false, restored_ = false;
    const topology::Graph* graph_ = nullptr;
    const pwl::DevicePool* pool_ = nullptr;
    const events::StepHooks* view_of_ = nullptr;
    topology::SwitchStateMask mask_last_, mask_pending_;
    bool started_ = false, has_pending_ = false, attached_ = false, settled_ = false;
    bool event_at_last_ = false;  // a settled event row at the last sample
    Real t_last_ = 0, t_pending_ = 0;
    Size n_steps_ = 0, n_jumps_ = 0, n_events_ = 0, n_dropped_ = 0;

    // A reading at (t, x) under `mask`. `probe_right`: x is a settled
    // post-event probe (TR-BDF2), the histories are still the left side's.
    void eval(Real t, const Vector& x, const topology::SwitchStateMask& mask,
              std::vector<Real>& out, bool probe_right = false) {
        for (Size j = 0; j < readings_.size(); ++j) {
            const auto& r = readings_[j];
            out[j] = r.dev.law == DeviceCurrent::Law::None
                         ? r.eval(x)
                         : device(readings_[j], t, x, mask, probe_right);
        }
    }

    Real device(StateReading& r, Real t, const Vector& x, const topology::SwitchStateMask& mask,
                bool probe_right) {
        using L = DeviceCurrent::Law;
        auto& d = r.dev;
        const auto& view = view_of_->run_view();
        Real i = 0;
        switch (d.law) {
        case L::Pwl: i = d.pwl.i(x, d.pwl.on(mask)); break;
        case L::Capacitor: {
            if (view.history == nullptr) {
                throw std::logic_error("LoopFeed: the engine gave no companion history");
            }
            const auto& es = view.history->entries();
            if (d.entry < 0 || static_cast<Size>(d.entry) >= es.size() ||
                es[static_cast<Size>(d.entry)].branch_id != d.branch) {
                d.entry = -1;
                for (Size k = 0; k < es.size(); ++k) {
                    if (es[k].branch_id == d.branch) d.entry = static_cast<Index>(k);
                }
                if (d.entry < 0) throw std::logic_error("LoopFeed: capacitor not in the history");
            }
            const auto& e = es[static_cast<Size>(d.entry)];
            if (probe_right && view.probe_dt != nullptr && *view.probe_dt > Real{0}) {
                const Real v = stamping::read_node_voltage(x, e.from) -
                               stamping::read_node_voltage(x, e.to);
                i = models::Capacitor::g_eq(*view.probe_dt, models::Capacitor::Params{e.C_or_L}) *
                        (v - e.v_prev) - e.i_prev;
            } else {
                i = e.i_prev;
            }
            break;
        }
        case L::Static: {
            pwl::DeviceReading dr;
            if (!pwl::static_reading(*graph_, *pool_, x, d.branch, dr)) return 0;
            return d.power ? dr.power : dr.i_pin[0];
        }
        case L::Stateful: {
            if (view.stateful == nullptr) {
                throw std::logic_error("LoopFeed: the engine gave no stateful histories");
            }
            Real sum = 0;
            pwl::for_each_stateful_reading(*view.stateful, x, [&](const pwl::DeviceReading& dr) {
                if (dr.branch_id == d.branch) sum += d.power ? dr.power : dr.i_pin[0];
            });
            return sum;
        }
        case L::Waveform: i = pool_->current_waveform_value(d.branch, t); break;
        case L::Constant: i = pool_->current_source_params(d.branch).I; break;
        case L::None: break;
        }
        return d.power ? r.a.eval(x) * i : i;
    }

    void on_start(const Vector& x0, const topology::SwitchStateMask& m0) {
        // bounds of every column a reading reads, once per run (the
        // per-step reads are unchecked)
        const auto in = [&](Index c) { return c >= -1 && c < x0.size(); };
        for (Size j = 0; j < readings_.size(); ++j) {
            const auto& r = readings_[j];
            bool ok = in(r.dev.pwl.var) && in(r.dev.pwl.from) && in(r.dev.pwl.to);
            for (const Index c : r.a.cols) ok = ok && c >= 0 && c < x0.size();
            for (const Index c : r.b.cols) ok = ok && c >= 0 && c < x0.size();
            if (!ok) {
                throw std::invalid_argument("LoopFeed: reading " + names_[j] +
                                            " reads a column outside the state vector");
            }
        }
        for (const auto& r : readings_) {
            if (r.dev.branch >= 0 && r.dev.law == DeviceCurrent::Law::None) {
                throw std::logic_error("LoopFeed: device readings need set_circuit() first");
            }
        }
        // the histories at the start (a capacitor's starting companion
        // current): the first step's start reading when it starts from x0
        seed_.resize(names_.size());
        eval(0, x0, m0, seed_);
        x_seed_ = x0;
        seeded_ = true;
    }

    static bool same(const Vector& a, const Vector& b) noexcept {
        if (a.size() != b.size()) return false;
        if (a.size() == 0 ||
            std::memcmp(a.data(), b.data(), sizeof(Real) * static_cast<Size>(a.size())) == 0) {
            return true;   // the usual case, bit for bit
        }
        for (Index k = 0; k < a.size(); ++k) {
            if (!(a[k] == b[k]) && !(a[k] != a[k] && b[k] != b[k])) return false;
        }
        return true;
    }

    void settled_event(const std::vector<Real>& right) {
        for (auto& r : recs_) r.event(right, Recorder::Resolution::Settled);
        event_at_last_ = true;
        ++n_events_;
    }

    void on_event(Real t, const Vector& xl, const topology::SwitchStateMask& ml,
                  const Vector& xr, const topology::SwitchStateMask& mr) {
        if (settled_ && !started_) {
            ++n_dropped_;
            return;
        }
        if (settled_ && t == t_last_) {  // after the sample at t
            eval(t, xr, mr, row_, true);
            settled_event(row_);
            mask_last_ = mr;  // the mode the next step runs in
            return;
        }
        if (has_pending_) ++n_dropped_;  // two reports for one step: keep the last
        // fixed engine: xr is the end of the step being committed (the
        // histories are already its); variable step: a settled probe
        eval(t, xr, mr, pending_, settled_);
        if (settled_) eval(t, xl, ml, pending_left_);
        mask_pending_ = mr;
        t_pending_ = t;
        has_pending_ = true;
    }

    void on_step(Real t0, const Vector& xs, const topology::SwitchStateMask& ms, Real t,
                 const Vector& xe, const topology::SwitchStateMask& me) {
        if (!started_) {
            eval(t0, xs, ms, row_);
            if (seeded_ && same(xs, x_seed_)) {
                // a reading of the histories at the run's start
                for (Size j = 0; j < readings_.size(); ++j) {
                    const auto law = readings_[j].dev.law;
                    if (law == DeviceCurrent::Law::Capacitor || law == DeviceCurrent::Law::Stateful) {
                        row_[j] = seed_[j];
                    }
                }
            }
            for (auto& r : recs_) r.sample(t0, row_);
            x_last_ = xe;  // sizes the buffer once
            t_last_ = t0;
            started_ = true;
        } else if (settled_ && has_pending_ && t_last_ < t_pending_ && t_pending_ < t) {
            // trbdf2 committed [t0, t_ev] and then the post-event probe
            // [t_ev, t] as ONE hook step (a landed gate edge / breakpoint
            // with refresh_after_event): the left limit at t_ev is the
            // event's committed side, the settled right side starts the line
            has_pending_ = false;
            for (auto& r : recs_) r.sample(t_pending_, pending_left_);
            settled_event(pending_);
            mask_last_ = mask_pending_;
        } else if (!event_at_last_ && (!(ms == mask_last_) || !same(xs, x_last_))) {
            // the engine restarted from another state than it committed
            eval(t0, xs, ms, row_);
            for (auto& r : recs_) r.event(row_, Recorder::Resolution::Settled);
            ++n_jumps_;
        }
        const bool had_event = event_at_last_;
        event_at_last_ = false;
        eval(t, xe, me, row_);
        if (has_pending_ && !settled_) {
            // the bracket's right side is this step's end (xr was xe), read
            // at t: a source's value belongs to the end, not to the report
            for (auto& r : recs_) r.event_bracket(t, row_);
            has_pending_ = false;
            ++n_events_;
        } else if (settled_ && !had_event && n_steps_ > 0 && !(me == mask_last_)) {
            // the variable-step engine solved this step under a new mode
            // without a settled report at its start: bracket [t0, t]
            for (auto& r : recs_) r.event_bracket(t, row_);
            ++n_events_;
        }
        if (settled_ && has_pending_ && t_pending_ == t) {
            // an event at this step's end reported before the step hook: the
            // engine may have replaced the derivative slots with the probe's
            // since (TR-BDF2, no commit across t_end / a tick); the sample's
            // history readings are the ones taken at the report
            for (Size j = 0; j < readings_.size(); ++j) {
                const auto law = readings_[j].dev.law;
                if (law == DeviceCurrent::Law::Capacitor || law == DeviceCurrent::Law::Stateful) {
                    row_[j] = pending_left_[j];
                }
            }
        }
        for (auto& r : recs_) r.sample(t, row_);
        x_last_ = xe;  // same size: no allocation
        mask_last_ = me;
        t_last_ = t;
        ++n_steps_;
        if (has_pending_ && settled_) {
            has_pending_ = false;
            if (t_pending_ == t) {
                settled_event(pending_);
                mask_last_ = mask_pending_;
            } else {
                ++n_dropped_;
            }
        }
    }
};

}  // namespace pulsim::record
