// =============================================================================
// Pulsim — pybind11 bindings for the DSED schedulers (Bridge.10)
// =============================================================================
//
// Wraps the three C++ scheduler templates (PEDSimulator [DOPRI5],
// PEDSimulatorBDF2, PEDSimulatorAuto) so Python callers go through the
// native inner loop instead of the pure-Python port. The Python-side
// `system` and `switch_fn` are still invoked via callbacks (the C++
// scheduler holds a `py::object` and calls `.attr("A_matrix")()` etc.),
// so RHS / b_vector queries still pay the GIL acquire on every step —
// but the scheduler control flow, step controllers, Hermite interpolation,
// LU solves, and event predicates all run natively.
//
// Captured speedup on buck CCM (5 ms, 100 kHz, see
// notes/DSED_BRIDGE_DESIGN.md §11):
//
//   * Python scheduler   : ~60 µs / step (1007 steps, 61 ms wall)
//   * C++ scheduler here : ~10–15 µs / step (4–6× speedup)
//   * Fully-native (no callbacks) : ~5 µs / step — see follow-up
//
// MaskT handling: we wrap Python masks in a thin `PyMask` struct
// (struct so ADL finds our `mode_id_of(PyMask)` overload inside
// `scheduler_auto.hpp`).
//
// Result types: PEDResult / PEDResultAuto are returned as Python
// dataclass-like dicts so callers don't need a custom .py shim.

#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/typing.h>

#include <array>
#include <memory>
#include <cstddef>
#include <unordered_map>
#include <utility>
#include <vector>

#include "pulsim/blockchain/chain.hpp"
#include "pulsim/builder/circuit_builder.hpp"
#include "pulsim/dsed/builder_adapter.hpp"
#include "pulsim/pwl/diode_event_state.hpp"
#include "pulsim/pwl/row_names.hpp"
#include "pulsim/dsed/scheduler.hpp"
#include "pulsim/dsed/scheduler_bdf2.hpp"
#include "pulsim/dsed/scheduler_auto.hpp"
#include "pulsim/dsed/period_integrals.hpp"
#include "pulsim/dsed/loss_port.hpp"
#include "pulsim/dsed/loop_feed_adapter.hpp"
#include "pulsim/dsed/segment_integrals.hpp"
#include "pulsim/record/loop_feed.hpp"
#include "pulsim/dsed/periodic_orbit.hpp"
#include "pulsim/dsed/design_batch.hpp"
#include "pulsim/dsed/period_thermal.hpp"
#include "pulsim/ir/circuit.hpp"
#include "pulsim/analysis/cancellation.hpp"
#include "pulsim/dsed/step_controller.hpp"
#include "pulsim/dsed/stiffness_detector.hpp"
#include "pulsim/numeric/dense.hpp"
#include "pulsim/numeric/types.hpp"
#include "pulsim/pwl/cache.hpp"
#include "pulsim/solver/result.hpp"
#include "pulsim/dsed/time_eps.hpp"
#include "pulsim/events/mode_selector.hpp"
#include "pulsim/sources/native_pwm_switch_fn.hpp"
#include "pulsim/sources/switch_schedule.hpp"
#include "pulsim/topology/switch_state.hpp"
#include "switch_fn_adapter.hpp"

#include <cstdint>
#include <optional>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace {
// `resume=None` reads Optional in the signature (add-ai-usage-guide 1.4).
// typing::Optional accepts any object, so the type is checked here: a
// wrong one is a TypeError, as the plain `SolverSnapshot*` argument gave.
// The pointer lives as long as the argument object (the call's duration).
const pulsim::solver::SolverSnapshot* snapshot_or_null(
    const py::typing::Optional<pulsim::solver::SolverSnapshot>& arg) {
    if (arg.is_none()) return nullptr;
    if (!py::isinstance<pulsim::solver::SolverSnapshot>(arg))
        throw py::type_error("resume: expected a SolverSnapshot or None");
    return &arg.cast<const pulsim::solver::SolverSnapshot&>();
}
}  // namespace

namespace pulsim_dsed_bindings {

using pulsim::Real;
using pulsim::Vector;
using pulsim::DenseMatrix;

// ---------------------------------------------------------------------------
// PyMask: thin wrapper around py::object so ADL finds our `mode_id_of`
// overload when the C++ scheduler templates expand.
// ---------------------------------------------------------------------------
struct PyMask {
    py::object obj;

    bool operator==(const PyMask& other) const {
        if (obj.is(other.obj)) return true;
        try {
            return obj.equal(other.obj);
        } catch (...) {
            return false;
        }
    }
    bool operator!=(const PyMask& other) const {
        return !(*this == other);
    }
};

// Phase-0 fix #5: the old free-function here truncated py::hash
// (64-bit Py_hash_t) to int32 for PEDSimulatorAuto's per-mode
// integrator cache — two distinct masks colliding in 32 bits
// silently reused the wrong cached λ_max. PySystem now owns an
// injective interning `mode_id()` (full-hash bucketed, equality-
// checked) that `resolve_mode_id` prefers; no stateless hash
// shim remains.

// ---------------------------------------------------------------------------
// PySystem: adapter that holds a Python `system` object and forwards
// `A_matrix()`, `b_vector(t)`, `rhs(t, x)`, `current_mask()`, `set_mask(m)`
// through pybind11. Matches the `HasLTIPerMode` concept.
// ---------------------------------------------------------------------------
class PySystem {
public:
    explicit PySystem(py::object obj)
        : obj_(std::move(obj)),
          A_attr_(obj_.attr("A_matrix")),
          b_attr_(obj_.attr("b_vector")),
          rhs_attr_(obj_.attr("rhs")),
          mask_attr_(obj_.attr("current_mask")),
          set_mask_attr_(obj_.attr("set_mask")) {}

    // Returns DenseMatrix by value (Eigen copy is cheap for small n_state).
    DenseMatrix A_matrix() const {
        return A_attr_().cast<DenseMatrix>();
    }

    Vector b_vector(Real t) const {
        return b_attr_(t).cast<Vector>();
    }

    Vector rhs(Real t, const Vector& x) const {
        return rhs_attr_(t, x).cast<Vector>();
    }

    PyMask current_mask() const {
        return PyMask{mask_attr_()};
    }

    void set_mask(const PyMask& m) {
        set_mask_attr_(m.obj);
    }

    /// Injective, dense mode id for `m` (Phase-0 fix #5). Buckets by
    /// the FULL 64-bit Python hash, then equality-checks within the
    /// bucket, so distinct masks can never alias an id. Preferred
    /// over the (removed) truncating hash shim by `resolve_mode_id`.
    [[nodiscard]] int mode_id(const PyMask& m) const {
        std::size_t h;
        try {
            h = static_cast<std::size_t>(py::hash(m.obj));
        } catch (...) {
            // Unhashable mask object — fall back to identity. Two
            // views of the same underlying object still match via
            // the equality scan below when hashing is available;
            // identity is the last resort.
            h = reinterpret_cast<std::uintptr_t>(m.obj.ptr());
        }
        auto& bucket = mode_id_buckets_[h];
        for (const auto& [seen, id] : bucket) {
            if (seen == m) return id;
        }
        const int id = next_mode_id_++;
        bucket.emplace_back(m, id);
        return id;
    }

private:
    py::object obj_;
    // Pre-resolved attributes to skip the attr-lookup cost per call.
    py::object A_attr_;
    py::object b_attr_;
    py::object rhs_attr_;
    py::object mask_attr_;
    py::object set_mask_attr_;
    // Phase-0 fix #5: full-hash → [(mask, dense id)] intern table.
    mutable std::unordered_map<std::size_t,
        std::vector<std::pair<PyMask, int>>> mode_id_buckets_;
    mutable int next_mode_id_ = 0;
};

// ---------------------------------------------------------------------------
// Gate-edge search for a switch_fn with no `next_edge_after`.
//
// The schedulers fall back to landing every dt_max/10 (the "poll") when
// the switch_fn cannot name its next edge, and used to take whatever
// mask they found there: every gate edge moved onto the poll grid. At
// the default dt_max = 1e-5 that is a 1 us grid, which deleted a 100 ns
// dead time outright (a synchronous buck's body diodes never conducted)
// and quantised the duty (D = 0.25 at 100 kHz ran as 0.30: 14.28 V
// against 11.49 V).
//
// The search below samples the mask at the end of the poll window and,
// when it changed, bisects (mask evaluations only, no solves) down to
// the time resolution of `time_margin`, returning the first instant
// that already carries the new mask. A pulse that begins AND ends
// inside one poll window is still invisible to it; the dispatch warns
// about that.
// ---------------------------------------------------------------------------
template <class MaskFn>
class EdgeSearch {
public:
    void set_poll(Real poll) noexcept { poll_ = poll; }

    template <class Eval>
    [[nodiscard]] Real next_edge(Real t, Eval&& eval) const {
        if (!(poll_ > Real{0})) {
            return std::numeric_limits<Real>::infinity();
        }
        // The mask is a function of t: the first change after any
        // t' in [from_, edge_) is still edge_.
        if (valid_ && t >= from_ && t < edge_) return edge_;
        // The scheduler sampled the mask just past its last event;
        // match it so an edge-inclusive-left switch_fn is not seen
        // as changing again at t itself.
        // The window end is probed where the scheduler samples after
        // landing, advance_past(hi): landing there next (the usual
        // case), that probe IS m0, one Python call per window less.
        // (Probing AT hi instead broke the common `(t % T) < D * T`
        // switch_fn, whose fmod reads the wrong side at k * T.)
        const MaskFn m0 = (last_ && t == last_hi_)
            ? *last_ : eval(pulsim::dsed::advance_past(t));
        Real lo = t;
        Real hi = t + poll_;
        last_hi_ = hi;
        last_ = eval(pulsim::dsed::advance_past(hi));
        // No change in the window: the plain poll boundary, uncached,
        // exactly as before the search existed (a constant switch_fn,
        // e.g. the default all-closed mask, keeps its step grid).
        if (*last_ == m0) return hi;
        // An edge ON the poll boundary (a PWM whose edges sit on the
        // poll grid) costs one evaluation instead of ~50: the search
        // made the 100 kHz buck of test_dsed_without_scipy 3.5x
        // slower (19 -> 68 ms) without it.
        if (!(eval(hi - pulsim::dsed::time_margin(hi)) == m0)) {
            for (int it = 0; it < 200; ++it) {
                if (hi - lo <= pulsim::dsed::time_margin(hi)) break;
                const Real mid = lo + Real{0.5} * (hi - lo);
                if (!(mid > lo && mid < hi)) break;
                if (eval(mid) == m0) {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
        }
        from_ = t;
        edge_ = hi;
        valid_ = true;
        return hi;
    }

private:
    Real poll_ = Real{0};
    mutable std::optional<MaskFn> last_;
    mutable Real last_hi_ = Real{0};
    mutable bool valid_ = false;
    mutable Real from_ = Real{0};
    mutable Real edge_ = Real{0};
};

// ---------------------------------------------------------------------------
// PySwitchFn: adapter that holds a Python `switch_fn` callable and an
// optional `next_edge_after(t)` method. Returns PyMask from operator().
// Without `next_edge_after`, edges are bisected inside each poll
// window once `set_poll` was called (see EdgeSearch).
// ---------------------------------------------------------------------------
class PySwitchFn {
public:
    explicit PySwitchFn(py::object fn)
        : fn_(std::move(fn)) {
        if (py::hasattr(fn_, "next_edge_after")) {
            nea_attr_ = fn_.attr("next_edge_after");
            has_nea_ = true;
        }
    }

    PyMask operator()(Real t) const {
        return PyMask{fn_(t)};
    }

    Real next_edge_after(Real t) const {
        if (has_nea_) {
            return nea_attr_(t).cast<Real>();
        }
        return search_.next_edge(
            t, [this](Real tau) { return (*this)(tau); });
    }

    void set_poll(Real poll) noexcept { search_.set_poll(poll); }

    /// Whether next_edge_after names real edges (the selector's probe
    /// horizon); the EdgeSearch fallback returns poll boundaries.
    [[nodiscard]] bool has_edge_oracle() const noexcept { return has_nea_; }

private:
    py::object fn_;
    py::object nea_attr_;
    bool has_nea_ = false;
    EdgeSearch<PyMask> search_;
};

/// The poll window the schedulers use for a switch_fn without an edge
/// oracle (scheduler.hpp / scheduler_auto.hpp / scheduler_bdf2.hpp:
/// one tenth of their step ceiling).
[[nodiscard]] inline Real poll_window(Real step_ceiling) noexcept {
    return step_ceiling * Real{0.1};
}

// ---------------------------------------------------------------------------
// PEDResult / PEDResultAuto → Python dict converters.
//
// We return a dict so callers can index by name;
// `_dsed_dispatch._exact_result` fills a SimulationResult from it.
// ---------------------------------------------------------------------------

/// The mask each stored sample is reconstructed and recorded under:
/// the STAMPED mask, i.e. the one in force over the step that ended
/// at that sample (PEDResult::step_masks). Falls back to the
/// post-event sample_masks for a result built without step masks.
template <class ResultT>
const auto& stamped_masks_of(const ResultT& res) {
    return res.step_masks.size() == res.times.size() ? res.step_masks
                                                     : res.sample_masks;
}

/// v2.0 Phase 3 item 5 — reconstruct the FULL MNA trajectory from
/// the reduced samples through the adapter's per-mask recovery map.
/// Samples are grouped by mask so a run that visits M modes pays M
/// mask resolutions, not one per sample. Returns an empty array if
/// the result carries no per-sample masks (a non-builder path).
///
/// Each sample is reconstructed through its STAMPED mask (the mask of
/// the step that produced it), from the state that step ended in. At
/// a sample where an event fired that is the pre-event (left-limit)
/// state, the same convention the kernel engines use: a TR-BDF2
/// landing sample is a solve under the mask from before the flip, and
/// the new mode first appears on the next sample. Recording the
/// post-event mask instead booked one whole step per event to the
/// wrong mode (a DCM boost diode's duty read 0.200 against a true
/// 0.178 at the default dt_max).
template <class AdapterT, class ResultT>
py::array_t<double> reconstruct_full_states(
    AdapterT& adapter, const ResultT& res) {
    const auto n = res.times.size();
    const auto& masks = stamped_masks_of(res);
    if (n == 0 || masks.size() != n) {
        return py::array_t<double>(py::ssize_t{0});
    }
    const bool have_pre =
        res.step_masks.size() == n
        && res.pre_event_index.size() == res.pre_event_states.size();
    std::size_t next_pre = 0;
    bool pre = false;   // sample i is an event's LEFT limit
    auto state_of = [&](std::size_t i) -> const pulsim::Vector& {
        pre = false;
        if (have_pre && next_pre < res.pre_event_index.size()
            && res.pre_event_index[next_pre] == i) {
            pre = true;
            return res.pre_event_states[next_pre++];
        }
        return res.states[i];
    };
    // A left-limit sample reads its sources' left limit too: a source
    // corner is an event, and an instant edge there (PWM, pulse) would
    // otherwise pair the old mode with the new source value. Half the
    // last step lies inside the piece that step covered. The final
    // sample is the left limit of t_end too: a corner exactly at t_end
    // never fires (the run ends), so its new piece is never integrated.
    auto left_by = [&](std::size_t i) {
        return (pre || i + 1 == n) && i > 0
            ? Real{0.5} * (res.times[i] - res.times[i - 1]) : Real{0};
    };
    // Probe the width from the first sample.
    adapter.set_mask(masks[0]);
    const pulsim::Vector first =
        adapter.recover_full(res.times[0], state_of(0));
    const auto n_mna = static_cast<py::ssize_t>(first.size());
    py::array_t<double> out(
        {static_cast<py::ssize_t>(n), n_mna});
    auto* ptr = out.mutable_data();
    for (py::ssize_t j = 0; j < n_mna; ++j) {
        ptr[j] = first[j];
    }
    for (std::size_t i = 1; i < n; ++i) {
        if (!(masks[i] == masks[i - 1])) {
            adapter.set_mask(masks[i]);
        }
        double* row = ptr + static_cast<py::ssize_t>(i) * n_mna;
        if constexpr (requires {
                          adapter.recover_full_into(
                              Real{0}, state_of(i),
                              Eigen::Map<pulsim::Vector>(row, n_mna), Real{0});
                      }) {
            const pulsim::Vector& xi = state_of(i);
            adapter.recover_full_into(
                res.times[i], xi,
                Eigen::Map<pulsim::Vector>(row, n_mna), left_by(i));
        } else if constexpr (requires {
                          adapter.recover_full_into(
                              Real{0}, state_of(i),
                              Eigen::Map<pulsim::Vector>(row, n_mna));
                      }) {
            adapter.recover_full_into(
                res.times[i], state_of(i),
                Eigen::Map<pulsim::Vector>(row, n_mna));
        } else {
            const pulsim::Vector full =
                adapter.recover_full(res.times[i], state_of(i));
            for (py::ssize_t j = 0; j < n_mna; ++j) {
                row[j] = full[j];
            }
        }
    }
    return out;
}

/// The RIGHT limit of every sample where an event changed the state
/// or the mask: the full MNA state rebuilt through the POST-event mask
/// (sample_masks[k]) from the post-event integrator state (states[k]).
/// It is where the NEXT step starts. Written into `d` as
/// `right_limit_index` (sample indices, increasing),
/// `right_limit_states` ((m, n_mna)) and `right_limit_switch_states`
/// ((m, n_switches) bool).
///
/// Why: a stored DSED event sample is the LEFT limit (the state the
/// step that ended there produced, reconstruct_full_states), so a
/// switched branch's current jumps between that sample and the start
/// of the next step. A trapezoid drawn from the left limit spread a
/// freewheel diode's 0 -> I_L turn-on jump as a ramp over the whole
/// following step (1 us by default): a DCM boost diode's average
/// current read 0.0234 A against 0.0535 A. Integrating each step from
/// the right limit of its start sample to the left limit of its end
/// sample closes that charge exactly.
/// Dependent states (add-exact-propagator-strategy DECISIONS): the
/// impulsive re-initialisations of the run as `d["reinit_records"]`, a
/// list of dicts with the fixed engine's ReinitRecord fields; `sample`
/// is the stored sample at the record's time (its left limit), or -1.
template <class AdapterT, class ResultT>
void reinit_records_into(AdapterT& adapter, const ResultT& res, py::dict& d) {
    py::list out;
    if constexpr (requires { adapter.take_reinit_log(); }) {
        for (const auto& r : adapter.take_reinit_log()) {
            long sample = -1;
            const auto it = std::lower_bound(res.times.begin(), res.times.end(),
                                             r.t - 4 * std::numeric_limits<Real>::epsilon()
                                                       * std::max(Real{1}, std::abs(r.t)));
            if (it != res.times.end()
                && std::abs(*it - r.t)
                       <= 4 * std::numeric_limits<Real>::epsilon()
                              * std::max(Real{1}, std::abs(r.t))) {
                sample = static_cast<long>(it - res.times.begin());
            }
            py::dict rd;
            rd["t"] = r.t;
            rd["kind"] = r.kind;
            rd["devices"] = r.devices;
            rd["states"] = r.states;
            rd["lambda"] = r.lambda;
            rd["dW"] = r.dW;
            rd["projection_gap"] = r.projection_gap;
            rd["sample"] = sample;
            out.append(std::move(rd));
        }
    }
    d["reinit_records"] = std::move(out);
}

template <class AdapterT, class ResultT>
void reconstruct_right_limits(AdapterT& adapter, const ResultT& res,
                              py::dict& d) {
    const auto n = res.times.size();
    std::vector<std::size_t> idx;
    using MaskT = typename decltype(res.sample_masks)::value_type;
    if constexpr (!std::is_same_v<MaskT, PyMask>) {
        if (n > 0 && res.sample_masks.size() == n
            && res.step_masks.size() == n) {
            std::size_t next_pre = 0;
            for (std::size_t i = 0; i < n; ++i) {
                bool fired = false;
                while (next_pre < res.pre_event_index.size()
                       && res.pre_event_index[next_pre] < i) {
                    ++next_pre;
                }
                // A fired event that left the state untouched (a
                // same-mask gate edge is dropped inside fire_event_,
                // and a source-only circuit "fires" its t_end edge on
                // every step) has right limit == left limit: skip it,
                // or the record doubles the state storage for nothing.
                if (next_pre < res.pre_event_index.size()
                    && res.pre_event_index[next_pre] == i
                    && next_pre < res.pre_event_states.size()) {
                    const auto& xp = res.pre_event_states[next_pre];
                    const auto& xs = res.states[i];
                    fired = xp.size() != xs.size()
                            || (xp.array() != xs.array()).any();
                    // A source corner leaves the reduced state alone but
                    // moves the source-driven nodes: the sample reads the
                    // sources' left limit (reconstruct_full_states), so
                    // the right limit differs and is recorded too.
                    if constexpr (requires {
                                      adapter.recover_full_into(
                                          Real{0}, xs,
                                          Eigen::Map<pulsim::Vector>(nullptr, 0),
                                          Real{0});
                                  }) {
                        if (!fired && i > 0
                            && res.step_masks[i] == res.sample_masks[i]) {
                            adapter.set_mask(res.sample_masks[i]);
                            const pulsim::Vector r =
                                adapter.recover_full(res.times[i], xs);
                            pulsim::Vector l(r.size());
                            adapter.recover_full_into(
                                res.times[i], xs, l,
                                Real{0.5} * (res.times[i] - res.times[i - 1]));
                            fired = (l.array() != r.array()).any();
                        }
                    }
                }
                if (fired || !(res.step_masks[i] == res.sample_masks[i])) {
                    idx.push_back(i);
                }
            }
        }
    }
    const auto m = static_cast<py::ssize_t>(idx.size());
    py::array_t<std::int64_t> idx_arr(m);
    auto* ip = idx_arr.mutable_data();
    for (py::ssize_t j = 0; j < m; ++j) {
        ip[j] = static_cast<std::int64_t>(idx[static_cast<std::size_t>(j)]);
    }
    d["right_limit_index"] = std::move(idx_arr);
    if (m == 0) {
        d["right_limit_states"] =
            py::array_t<double>(std::vector<py::ssize_t>{0, 0});
        d["right_limit_switch_states"] =
            py::array_t<bool>(std::vector<py::ssize_t>{0, 0});
        return;
    }
    if constexpr (!std::is_same_v<MaskT, PyMask>) {
        const auto& m0 = res.sample_masks[idx[0]];
        adapter.set_mask(m0);
        const pulsim::Vector first =
            adapter.recover_full(res.times[idx[0]], res.states[idx[0]]);
        const auto n_mna = static_cast<py::ssize_t>(first.size());
        const auto n_sw = static_cast<py::ssize_t>(m0.size());
        py::array_t<double> st(std::vector<py::ssize_t>{m, n_mna});
        py::array_t<bool> sw(std::vector<py::ssize_t>{m, n_sw});
        auto* sp = st.mutable_data();
        auto* wp = sw.mutable_data();
        for (py::ssize_t j = 0; j < m; ++j) {
            const auto i = idx[static_cast<std::size_t>(j)];
            const auto& mask = res.sample_masks[i];
            if (j > 0) {
                adapter.set_mask(mask);
            }
            const pulsim::Vector full = (j == 0)
                ? first
                : adapter.recover_full(res.times[i], res.states[i]);
            for (py::ssize_t c = 0; c < n_mna; ++c) {
                sp[j * n_mna + c] = full[c];
            }
            for (py::ssize_t c = 0; c < n_sw; ++c) {
                wp[j * n_sw + c] = mask.get(static_cast<pulsim::Size>(c));
            }
        }
        d["right_limit_states"] = std::move(st);
        d["right_limit_switch_states"] = std::move(sw);
    }
}

/// Per-sample switch / diode mask as an (n_samples, n_switches) bool
/// array — the same record `SimulationResult.switch_states` carries
/// on the pwl / trbdf2 engines: row k is the STAMPED mask, the one in
/// force over (t[k-1], t[k]]. Every DSED sample's full MNA state is
/// reconstructed THROUGH that mask (reconstruct_full_states), so
/// i = g(mask)·v holds at every sample by construction. Only the
/// native SwitchStateMask path has a typed mask; the PyMask path (no
/// PWL diodes, reduced states only) returns an empty array.
template <class ResultT>
py::array_t<bool> sample_masks_to_array(const ResultT& res) {
    using MaskT = typename decltype(res.sample_masks)::value_type;
    if constexpr (std::is_same_v<MaskT, PyMask>) {
        return py::array_t<bool>(std::vector<py::ssize_t>{0, 0});
    } else {
        const auto& masks = stamped_masks_of(res);
        const auto n = static_cast<py::ssize_t>(masks.size());
        if (n == 0 || masks.size() != res.times.size()) {
            return py::array_t<bool>(std::vector<py::ssize_t>{0, 0});
        }
        const auto n_sw =
            static_cast<py::ssize_t>(masks[0].size());
        py::array_t<bool> out(std::vector<py::ssize_t>{n, n_sw});
        auto* ptr = out.mutable_data();
        for (py::ssize_t i = 0; i < n; ++i) {
            const auto& m = masks[static_cast<std::size_t>(i)];
            for (py::ssize_t j = 0; j < n_sw; ++j) {
                ptr[i * n_sw + j] = m.get(static_cast<pulsim::Size>(j));
            }
        }
        return out;
    }
}

/// Event times and predicate types as arrays (cheap, always present).
template <class ResultT>
void event_arrays(const ResultT& res, py::dict& d) {
    const auto m = static_cast<py::ssize_t>(res.event_log.size());
    py::array_t<double> et(m);
    py::array_t<int> ty(m);
    for (py::ssize_t i = 0; i < m; ++i) {
        const auto& e = res.event_log[static_cast<std::size_t>(i)];
        et.mutable_data()[i] = e.t;
        ty.mutable_data()[i] = static_cast<int>(e.type);
    }
    d["event_times"] = std::move(et);
    d["event_types"] = std::move(ty);
}

template <class MaskT>
py::dict ped_result_to_dict(
    const pulsim::dsed::PEDResult<MaskT>& res, bool event_log = true) {
    // Convert times + states to numpy arrays
    const auto n_steps = static_cast<py::ssize_t>(res.times.size());
    const auto n_state = res.states.empty()
        ? py::ssize_t{0}
        : static_cast<py::ssize_t>(res.states[0].size());

    py::array_t<double> times_arr(n_steps);
    py::array_t<double> states_arr({n_steps, n_state});
    auto* times_ptr = times_arr.mutable_data();
    auto* states_ptr = states_arr.mutable_data();
    for (py::ssize_t i = 0; i < n_steps; ++i) {
        times_ptr[i] = res.times[i];
        for (py::ssize_t j = 0; j < n_state; ++j) {
            states_ptr[i * n_state + j] = res.states[i][j];
        }
    }

    py::dict d;
    d["times"] = std::move(times_arr);
    d["states"] = std::move(states_arr);
    d["switch_states"] = sample_masks_to_array(res);
    d["n_accept"] = res.n_accept;
    d["n_reject"] = res.n_reject;
    d["n_events"] = res.n_events;
    d["n_exact_steps"] = res.n_exact_steps;
    d["n_output_samples"] = res.n_output_samples;
    d["n_predicate_evals"] = res.n_predicate_evals;
    d["cpu_time_seconds"] = res.cpu_time_seconds;
    event_arrays(res, d);
    // Event log: skip the MaskT objects (they're py::object wrappers
    // around Python objects that we just hand back). event_log=False
    // (the simulate() path) skips the per-event dicts and their two
    // mask objects each: a 2000-period buck has 4000 events.
    py::list events;
    for (const auto& e : res.event_log) {
        if (!event_log) break;
        py::dict ed;
        ed["t"] = e.t;
        ed["name"] = e.name;
        ed["predicate_type"] = static_cast<int>(e.type);
        // Cast MaskT — PyMask carries a raw py::object; the
        // SwitchStateMask path goes through pybind11 auto-cast since
        // the type is bound in `bindings.cpp`.
        if constexpr (std::is_same_v<MaskT, PyMask>) {
            ed["old_mask"] = e.old_mask.obj;
            ed["new_mask"] = e.new_mask.obj;
        } else {
            ed["old_mask"] = py::cast(e.old_mask);
            ed["new_mask"] = py::cast(e.new_mask);
        }
        events.append(ed);
    }
    // Patch the (non-PyMask) overload's mask serialisation
    if constexpr (!std::is_same_v<MaskT, PyMask>) {
        // Empty `event_log` already populated above without mask
        // entries; nothing extra to do here.
    }
    d["event_log"] = events;
    return d;
}

template <class MaskT>
py::dict ped_result_auto_to_dict(
    const pulsim::dsed::PEDResultAuto<MaskT>& res, bool event_log = true) {
    const auto n_steps = static_cast<py::ssize_t>(res.times.size());
    const auto n_state = res.states.empty()
        ? py::ssize_t{0}
        : static_cast<py::ssize_t>(res.states[0].size());

    py::array_t<double> times_arr(n_steps);
    py::array_t<double> states_arr({n_steps, n_state});
    auto* times_ptr = times_arr.mutable_data();
    auto* states_ptr = states_arr.mutable_data();
    for (py::ssize_t i = 0; i < n_steps; ++i) {
        times_ptr[i] = res.times[i];
        for (py::ssize_t j = 0; j < n_state; ++j) {
            states_ptr[i * n_state + j] = res.states[i][j];
        }
    }

    py::dict d;
    d["times"] = std::move(times_arr);
    d["states"] = std::move(states_arr);
    d["switch_states"] = sample_masks_to_array(res);
    d["n_accept"] = res.n_accept;
    d["n_reject"] = res.n_reject;
    d["n_events"] = res.n_events;
    d["n_rk45_steps"] = res.n_rk45_steps;
    d["n_bdf2_steps"] = res.n_bdf2_steps;
    d["n_exact_steps"] = res.n_exact_steps;
    d["cpu_time_seconds"] = res.cpu_time_seconds;
    event_arrays(res, d);
    py::list events;
    for (const auto& e : res.event_log) {
        if (!event_log) break;
        py::dict ed;
        ed["t"] = e.t;
        ed["name"] = e.name;
        ed["predicate_type"] = static_cast<int>(e.type);
        ed["integrator_used"] = static_cast<int>(e.integrator_used);
        if constexpr (std::is_same_v<MaskT, PyMask>) {
            ed["old_mask"] = e.old_mask.obj;
            ed["new_mask"] = e.new_mask.obj;
        } else {
            ed["old_mask"] = py::cast(e.old_mask);
            ed["new_mask"] = py::cast(e.new_mask);
        }
        events.append(ed);
    }
    d["event_log"] = events;
    return d;
}

// ---------------------------------------------------------------------------
// Module init — registers three free functions:
//   run_ped_native(system, switch_fn, x0, t_end, rtol, atol, dt_init, dt_max)
//   run_bdf2_native(system, switch_fn, x0, t_end, h_fixed)
//   run_auto_native(system, switch_fn, x0, t_end, rtol, atol, dt_init,
//                    dt_max, h_bdf2, stiffness_threshold)
// ---------------------------------------------------------------------------
/// D10 — the exact propagator's per-mask tier record and counters.
template <class AdapterT>
py::dict exact_report_dict(const AdapterT& adapter,
                           std::size_t n_predicate_evals) {
    py::list masks, routes;
    for (const auto& r : adapter.exact_report()) {
        py::list bits;
        for (pulsim::Size i = 0; i < r.mask.size(); ++i) {
            bits.append(static_cast<bool>(r.mask.get(i)));
        }
        py::dict row;
        row["mask"] = bits;
        row["tier"] = r.tier;
        row["kappa"] = r.kappa;
        row["norm_Ah"] = r.norm_Ah;
        row["reason"] = r.reason;
        row["defect"] = r.defect;
        row["backward_error"] = r.backward;
        row["refined"] = r.refined;
        row["balanced"] = r.balanced;
        row["pade_defect"] = r.pade_defect;
        masks.append(row);
        // add-exact-propagator-strategy 15.2 (D13): the per-mode kernel and its counters
        py::dict rt;
        rt["mask"] = bits;
        // tier 3 (no exact step: the hand-over 'cost', or D2's refusal) is the implicit kernel
        rt["kernel"] = r.tier == 3 ? std::string{"implicit"}
                       : r.route.kernel.empty() ? std::string{"dense"} : r.route.kernel;
        rt["reason"] = !r.route.reason.empty() ? r.route.reason
                       : r.tier == 3 ? r.reason
                       : r.route.kernel.empty() ? std::string{"n < route_min_n"} : std::string{};
        rt["n"] = r.n_aug;
        rt["nnz"] = r.nnz_aug;
        rt["norm_Ah1"] = r.norm_Ah;
        rt["matvecs_planned"] = r.route.matvecs;
        rt["dense_setup_flops"] = r.route.dense_setup;
        rt["action_step_flops"] = r.route.action_step;
        rt["action_steps"] = r.action_steps;
        rt["action_matvecs"] = r.action_matvecs;
        routes.append(rt);
    }
    py::dict d;
    d["masks"] = masks;
    d["routes"] = routes;
    d["n_eig"] = adapter.n_eig();
    d["build_seconds"] = adapter.build_seconds();
    d["n_pade"] = adapter.n_pade();
    d["n_phi_cache_hits"] = adapter.n_pade_hits();
    d["n_predicate_evals"] = n_predicate_evals;
    // add-exact-propagator-strategy 10.4: the fixed-step form's counters
    d["n_fixed_steps"] = adapter.n_fixed_steps();
    d["n_phi_gamma_builds"] = adapter.n_phi_gamma_builds();
    d["n_phi_gamma_refused"] = adapter.n_phi_gamma_refused();
    d["fixed_step_flops"] = adapter.fixed_step_flops();
    d["phi_gamma_bytes"] = adapter.phi_gamma_bytes();
    return d;
}

/// Task 5.1: the circuit's passivity class at its diode ports (D3).
inline py::dict passivity_dict(const pulsim::events::PassivityClass& c) {
    py::dict d;
    d["class"] = std::string(c.name());
    d["device"] = c.passive() ? c.nonreciprocal_device : c.device;
    d["kind"] = c.passive() ? c.nonreciprocal_kind : c.kind;
    return d;
}

/// add-complementarity-mode-selection — result.diagnostics['mode_shadow'].
inline py::dict mode_shadow_dict(const pulsim::dsed::ModeShadowReport& r,
                                 const std::vector<std::string>& names,
                                 std::size_t port_problems) {
    auto bits = [](const std::vector<char>& v) {
        py::tuple t(v.size());
        for (std::size_t i = 0; i < v.size(); ++i) t[i] = py::bool_(v[i] != 0);
        return t;
    };
    py::list dis;
    for (const auto& e : r.disagreements) {
        py::dict d;
        d["t"] = e.t;
        d["settle_mode"] = bits(e.settle_mode);
        d["lcp_mode"] = bits(e.lcp_mode);
        d["previous_mode"] = bits(e.previous_mode);
        d["settle_rounds"] = e.settle_rounds;
        d["pivots"] = e.pivots;
        d["ties"] = e.ties;
        d["status"] = pulsim::events::to_string(e.status);
        d["lcp_current"] = e.lcp_current;
        d["lcp_gap"] = e.lcp_gap;
        auto verdict = [](int c) -> py::object {
            return c < 0 ? py::object(py::none()) : py::object(py::bool_(c != 0));
        };
        d["settle_consistent"] = verdict(e.settle_consistent);
        d["lcp_consistent"] = verdict(e.lcp_consistent);
        d["settle_violation"] = e.settle_violation;
        d["lcp_violation"] = e.lcp_violation;
        dis.append(d);
    }
    py::dict d;
    d["diodes"] = names;
    d["h_e"] = r.h_e;
    d["events"] = r.lcp_events;
    d["disagreements"] = dis;
    d["lcp_count"] = r.lcps;
    d["lcp_pivots_total"] = r.pivots_total;
    d["max_pivots_per_event"] = r.max_pivots_per_event;
    d["ties"] = r.ties;
    d["widenings"] = r.widenings;
    d["leak_pins"] = r.leak_pins;
    d["pin_judgements"] = r.pin_judgements;
    d["unfixed"] = r.unfixed;
    d["capped"] = r.capped;
    d["enumerations"] = r.enumerations;
    d["nonunique_modes"] = r.nonunique_modes;
    d["hysteresis_released"] = r.released;
    d["settle_rounds_total"] = r.settle_rounds_total;
    d["port_model_problems"] = port_problems;
    d["errors"] = r.errors;
    d["last_error"] = r.last_error;
    d["rounding_ties"] = r.rounding_ties;
    return d;
}

void init_module(py::module_& m) {
    // A stored sample broke the PWL diode law (PEDSimulator's end-of-run
    // check): typed, with the diode and the first sample's time.
    static py::object diode_law_exc = py::exception<void>(
        m, "DiodeModeInconsistentError", PyExc_RuntimeError);
    diode_law_exc.attr("device") = "";
    diode_law_exc.attr("t_failed") = 0.0;
    py::register_exception_translator([](std::exception_ptr pe) {
        if (!pe) return;
        try {
            std::rethrow_exception(pe);
        } catch (const pulsim::dsed::DiodeLawViolation& e) {
            py::object inst = diode_law_exc(e.what());
            inst.attr("device") = e.diode();
            inst.attr("t_failed") = e.time();
            PyErr_SetObject(diode_law_exc.ptr(), inst.ptr());
        }
    });

    // diagnostics['output_grid'] of an exact run under the output rule, from
    // the run's raw keys (_dsed_dispatch._result_from_native): built here so a
    // simulate() call pays one call for it, not ~160 bytecodes (the per-call
    // overhead ratchet, test_simulate_overhead). `ncol`: the stored columns of
    // the full state (0 without it); `t_start`: the run's clock offset.
    m.def("_output_grid_dict",
        [](const py::dict& nr, py::ssize_t ncol, double t_start) {
            const auto vec = [&](const char* k) {
                return nr.contains(k) ? nr[k].cast<std::vector<double>>()
                                      : std::vector<double>{};
            };
            const auto tail = [&](const std::vector<double>& v) {
                py::array_t<double> a(ncol > 0 ? ncol : 0);
                auto* p = a.mutable_data();
                const bool fits = ncol > 0 && static_cast<py::ssize_t>(v.size()) >= ncol;
                for (py::ssize_t j = 0; j < ncol; ++j)
                    p[j] = fits ? v[v.size() - static_cast<std::size_t>(ncol) + j] : 0.0;
                return a;
            };
            py::dict og;
            og["n_output_capped"] = py::int_(nr["n_output_capped"]);
            og["n_output_unbounded"] = py::int_(nr["n_output_unbounded"]);
            og["n_output_samples"] = nr.contains("n_output_samples")
                ? py::int_(nr["n_output_samples"]) : py::int_(0);
            og["n_output_floored"] = nr.contains("n_output_floored")
                ? py::int_(nr["n_output_floored"]) : py::int_(0);
            // of them, the calls certified only by the basis-inconsistency floor (F-X-8)
            og["n_output_inconsistent"] = nr.contains("n_output_inconsistent")
                ? py::int_(nr["n_output_inconsistent"]) : py::int_(0);
            const auto fr = vec("output_floor_row");
            double fmax = 0.0;   // numpy's max: a NaN propagates
            for (std::size_t j = 0; j < fr.size(); ++j) {
                if (j == 0 || std::isnan(fr[j]) || fr[j] > fmax) fmax = fr[j];
                if (std::isnan(fmax)) break;
            }
            og["output_floor_max"] = fmax;
            og["output_floor_by_column"] = tail(fr);
            // the event projections, as _dsed_dispatch._EventProjections defines them (the
            // run's records are an (n, 4) array [t, amplitude, row (-1: none), tau])
            const long long n_proj = nr.contains("n_projections")
                ? nr["n_projections"].cast<long long>() : 0;
            og["n_projections"] = py::int_(n_proj);
            const auto pm = vec("projection_max_row");
            og["projection_max_by_column"] = tail(pm);
            // a row of the recovered vector that is not a stored column (or none): None
            const long long first = static_cast<long long>(pm.size()) - ncol;
            const bool mapped = ncol > 0 && static_cast<long long>(pm.size()) >= ncol;
            py::list recs;
            if (nr.contains("projection_records")) {
                const auto a = py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(
                    nr["projection_records"]);
                const auto n = a && a.ndim() == 2 && a.shape(1) == 4 ? a.shape(0) : 0;
                const double* p = n ? a.data() : nullptr;
                for (py::ssize_t k = 0; k < n; ++k, p += 4) {
                    const auto r = static_cast<long long>(p[2]);
                    py::dict d;
                    d["t"] = p[0] + t_start;
                    d["amplitude"] = p[1];
                    d["tau"] = p[3];
                    d["column"] = (mapped && r >= first) ? py::object(py::int_(r - first))
                                                         : py::object(py::none());
                    recs.append(d);
                }
            }
            og["capped"] = py::bool_(n_proj > static_cast<long long>(py::len(recs)));
            og["projections"] = recs;
            const double tb = nr["output_budget_t"].cast<double>();
            og["output_budget_t"] = std::isnan(tb) ? py::object(py::none())
                                                   : py::object(py::float_(tb + t_start));
            return og;
        },
        py::arg("native_res"), py::arg("ncol"), py::arg("t_start"),
        "diagnostics['output_grid'] of an exact run under the output rule "
        "(internal: _dsed_dispatch._result_from_native).");

    // add-complementarity-mode-selection task 2: the dense Lemke solver,
    // for the oracle comparison against tests/oracles/lcp_reference.py.
    m.def("_lcp_lemke",
        [](const pulsim::DenseMatrix& M, const Vector& q) {
            pulsim::events::LemkeSolver solver;
            Vector z, w;
            const auto o = solver.solve(M, q, z, w);
            return py::make_tuple(z, w, o.pivots,
                                  pulsim::events::to_string(o.status));
        },
        py::arg("M"), py::arg("q"),
        "Lemke with the lexicographic ratio test on w = M z + q "
        "(add-complementarity-mode-selection D4). Returns "
        "(z, w, pivots, status).");
    m.def("_min_source_period",
        [](const pulsim::topology::Graph& g, const pulsim::pwl::DevicePool& pool) {
            using K = pulsim::pwl::DevicePool::StoredKind;
            Real T = std::numeric_limits<Real>::infinity();
            for (pulsim::Index b = 0; b < g.num_branches(); ++b) {
                K k;
                try { k = pool.kind_of(b); } catch (const std::out_of_range&) { continue; }
                Real f = 0;
                if (k == K::SineVoltageSource) f = pool.sine_voltage_source_params(b).frequency;
                else if (k == K::PWMVoltageSource) f = pool.pwm_voltage_source_params(b).frequency;
                if (f > 0) T = std::min(T, Real{1} / f);
            }
            return T;
        },
        py::arg("graph"), py::arg("pool"),
        "Shortest period of the sine / PWM sources (inf when none): T_ref "
        "of the shadow LCP's event step (add-complementarity-mode-selection D5).");
    m.def("_lcp_enumerate",
        [](const pulsim::DenseMatrix& M, const Vector& q) {
            Vector z, w;
            const int n = pulsim::events::enumerate_lcp(M, q, z, w);
            return py::make_tuple(n, z, w);
        },
        py::arg("M"), py::arg("q"),
        "2^k enumeration (k <= 12): (number of solutions, z, w) with the "
        "fewest basic z.");

    using pulsim::dsed::PEDSimulator;
    using pulsim::dsed::PEDSimulatorBDF2;
    using pulsim::dsed::PEDSimulatorAuto;
    using pulsim::dsed::PIController;
    using pulsim::dsed::EventPredictor;
    using pulsim::dsed::StiffnessDetector;

    m.def("run_ped_native",
        [](py::object system_obj,
           py::object switch_fn_obj,
           Vector x0,
           Real t_end,
           Real rtol,
           Real atol,
           Real dt_init,
           Real dt_max,
           std::size_t store_every) {
            PySystem sys(std::move(system_obj));
            PySwitchFn sf(std::move(switch_fn_obj));
            sf.set_poll(poll_window(dt_max));
            PIController ctrl(rtol, atol);
            EventPredictor pred;
            PEDSimulator<PySystem, PySwitchFn> sim(
                sys, std::move(sf), std::move(ctrl),
                std::move(pred), dt_init, dt_max,
                store_every);
            auto res = sim.simulate(x0, t_end);
            auto d = ped_result_to_dict(res);
            d["n_rk45_steps"] = res.n_accept - res.n_exact_steps;   // its non-exact steps are DOPRI5 (F-3-6)
            return d;
        },
        py::arg("system"),
        py::arg("switch_fn"),
        py::arg("x0"),
        py::arg("t_end"),
        py::arg("rtol")       = Real{1e-6},
        py::arg("atol")       = Real{1e-9},
        py::arg("dt_init")    = Real{1e-9},
        py::arg("dt_max")     = Real{1e-5},
        py::arg("store_every") = std::size_t{1},
        "Native PEDSimulator (DOPRI5 + adaptive PI + event scan) "
        "running the C++ scheduler inner loop. `system` and "
        "`switch_fn` are still invoked via Python callbacks per step "
        "(GIL overhead), but the scheduler control flow runs natively. "
        "Returns a dict with times, states, n_accept, n_reject, "
        "n_events, cpu_time_seconds, event_log.");

    m.def("run_bdf2_native",
        [](py::object system_obj,
           py::object switch_fn_obj,
           Vector x0,
           Real t_end,
           Real h_fixed,
           std::size_t store_every) {
            PySystem sys(std::move(system_obj));
            PySwitchFn sf(std::move(switch_fn_obj));
            sf.set_poll(poll_window(h_fixed));
            PEDSimulatorBDF2<PySystem, PySwitchFn> sim(
                sys, std::move(sf), h_fixed, store_every);
            auto res = sim.simulate(x0, t_end);
            return ped_result_to_dict(res);
        },
        py::arg("system"),
        py::arg("switch_fn"),
        py::arg("x0"),
        py::arg("t_end"),
        py::arg("h_fixed")    = Real{1e-6},
        py::arg("store_every") = std::size_t{1},
        "Native PEDSimulatorBDF2 (implicit BDF2 + Crank-Nicolson "
        "bootstrap) running the C++ scheduler inner loop.");

    m.def("run_auto_native",
        [](py::object system_obj,
           py::object switch_fn_obj,
           Vector x0,
           Real t_end,
           Real rtol,
           Real atol,
           Real dt_init,
           Real dt_max,
           Real h_bdf2,
           Real stiffness_threshold,
           std::size_t store_every) {
            PySystem sys(std::move(system_obj));
            PySwitchFn sf(std::move(switch_fn_obj));
            sf.set_poll(poll_window(dt_max));
            PIController ctrl(rtol, atol);
            StiffnessDetector det(stiffness_threshold);
            PEDSimulatorAuto<PySystem, PySwitchFn> sim(
                sys, std::move(sf), std::move(ctrl),
                std::move(det), dt_init, dt_max, h_bdf2,
                store_every);
            auto res = sim.simulate(x0, t_end);
            return ped_result_auto_to_dict(res);
        },
        py::arg("system"),
        py::arg("switch_fn"),
        py::arg("x0"),
        py::arg("t_end"),
        py::arg("rtol")                = Real{1e-6},
        py::arg("atol")                = Real{1e-9},
        py::arg("dt_init")             = Real{1e-9},
        py::arg("dt_max")              = Real{1e-5},
        py::arg("h_bdf2")              = Real{1e-6},
        py::arg("stiffness_threshold") = Real{10.0},
        py::arg("store_every")          = std::size_t{1},
        "Native PEDSimulatorAuto (per-mode RK45↔BDF2 dispatch via "
        "stiffness detector) running the C++ scheduler inner loop. "
        "Returns a dict that additionally includes n_rk45_steps + "
        "n_bdf2_steps for each mode-segment.");

    // -------------------------------------------------------------------
    // Bridge.11 — NativeCircuitBuilderAdapter + native-typed schedulers.
    //
    // The schedulers above (`run_ped_native` / `run_bdf2_native` /
    // `run_auto_native`) take a generic `system` Python object and
    // pay GIL roundtrips on every A_matrix / b_vector / rhs call (~22
    // µs/step on buck CCM). The variants below take a native C++
    // `NativeCircuitBuilderAdapter` and a SwitchStateMask-returning
    // switch_fn; the scheduler inner loop calls C++ → C++ for RHS
    // (no GIL, no marshalling), only paying the GIL cost at gate
    // edges when the Python switch_fn is invoked.
    //
    // Expected speedup vs Bridge.10: ~5-10× per-step → buck CCM 5 ms
    // down from ~22 ms wall-clock to ~2-5 ms.
    // -------------------------------------------------------------------

    using pulsim::dsed::NativeCircuitBuilderAdapter;
    using SSM = pulsim::topology::SwitchStateMask;
    using NativeAdapter = NativeCircuitBuilderAdapter<SSM>;

    // --- PySwitchFnSSM: switch_fn adapter with Bridge.12 native fast path ---
    //
    // The C++ scheduler template (`operator()(t)` + `next_edge_after(t)`)
    // is satisfied by THIS class. At construction we check whether the
    // user's `py::object` is actually a pybind11-bound NativePwm2Switch
    // or NativeMultiMaskPwm — if so, we store a C++ pointer to the
    // underlying object and call its native methods directly. Otherwise
    // (Python class with __call__ / next_edge_after) we fall back to
    // the GIL-acquiring path.
    //
    // The py::object `fn_` keeps the underlying native instance alive
    // as long as PySwitchFnSSM is alive (pybind11 ref-count).
    struct PySwitchFnSSM {
        explicit PySwitchFnSSM(py::object fn)
            : fn_{std::move(fn)} {
            // The native schedules (SwitchSchedule, NativePwm2Switch,
            // NativeMultiMaskPwm) are recognised by the adapter every
            // engine binding shares; they run without the GIL.
            if (py::isinstance<pulsim::sources::SwitchSchedule>(fn_)
                || py::isinstance<pulsim::sources::NativePwm2Switch>(fn_)
                || py::isinstance<pulsim::sources::NativeMultiMaskPwm>(fn_)) {
                native_ = pulsim_bindings_common::adapt_switch_fn(fn_);
                is_native_ = true;
            } else if (py::hasattr(fn_, "next_edge_after")) {
                // Generic Python callable: pre-resolve the attribute
                // to avoid the hasattr cost per call.
                nea_attr_ = fn_.attr("next_edge_after");
                has_nea_  = true;
            }
        }

        [[nodiscard]] SSM operator()(Real t) const {
            SSM m = [&] {
                if (is_native_) return native_.fn(t);
                // Generic Python fallback (Bridge.11 path).
                py::gil_scoped_acquire gil;
                if (n_python_calls) ++*n_python_calls;
                return fn_(t).cast<SSM>();
            }();
            if (check_width_) {
                pulsim::topology::require_mask_width(m, width_);
            }
            return m;
        }

        /// The circuit's switch count; a mask of another width is refused.
        void set_width(pulsim::Size n) noexcept { width_ = n; check_width_ = true; }

        [[nodiscard]] Real next_edge_after(Real t) const {
            if (is_native_ && native_.next_edge) return native_.next_edge(t);
            if (has_nea_) {
                py::gil_scoped_acquire gil;
                if (n_python_calls) ++*n_python_calls;
                return nea_attr_(t).cast<Real>();
            }
            // No oracle: bisect inside the poll window (EdgeSearch).
            return search_.next_edge(
                t, [this](Real tau) { return (*this)(tau); });
        }

        void set_poll(Real poll) noexcept { search_.set_poll(poll); }

        /// Whether next_edge_after names real edges (the selector's probe
        /// horizon); the EdgeSearch fallback returns poll boundaries.
        [[nodiscard]] bool has_edge_oracle() const noexcept {
            return (is_native_ && native_.next_edge) || has_nea_;
        }

        // dsed_period_map: Python calls of this schedule (owned by the
        // caller's frame; null elsewhere).
        std::size_t* n_python_calls = nullptr;
        py::object fn_;
        pulsim_bindings_common::SwitchFnAdapter native_;
        bool       is_native_ = false;
        py::object nea_attr_;
        bool       has_nea_ = false;
        EdgeSearch<SSM> search_;
        bool check_width_ = false;
        pulsim::Size width_ = 0;
    };

    py::class_<NativeAdapter>(m, "_NativeCircuitBuilderAdapter",
        "Bridge.11 — native C++ CircuitBuilderAdapter. Wraps a "
        "(CircuitBuilder, PwlStateSpaceCache) pair so the C++ "
        "schedulers can call A_matrix / b_vector / rhs without GIL "
        "roundtrips. Time-varying source overlay (sine/PWM/pulse) "
        "is computed natively via the same compute_*_b_extra "
        "helpers run_transient uses; an optional Python b_extra_fn "
        "callback adds onto that result.")
        .def(py::init([](
                const pulsim::builder::CircuitBuilder& builder,
                pulsim::pwl::PwlStateSpaceCache& cache,
                py::object b_extra_fn, bool loss_port) {
                std::function<Vector(Real)> cpp_b_extra;
                if (!b_extra_fn.is_none()) {
                    // Wrap the Python callable into std::function. The
                    // GIL cost only matters if the user actually
                    // provides a callback — most circuits use the
                    // auto-detected sine/PWM/pulse path.
                    cpp_b_extra = [cb = std::move(b_extra_fn)](Real t) {
                        py::gil_scoped_acquire gil;
                        return cb(t).cast<Vector>();
                    };
                }
                return std::make_unique<NativeAdapter>(
                    builder.graph(), builder.pool(), cache,
                    std::move(cpp_b_extra), loss_port);
            }),
            py::arg("builder"),
            py::arg("cache"),
            py::arg("b_extra_fn") = py::none(),
            // the loss port is honoured by run_ped_native_builder only
            py::arg("loss_port") = false,
            // the adapter holds the builder's graph/pool and the cache by reference
            py::keep_alive<1, 2>(), py::keep_alive<1, 3>())
        .def("set_refuse_implicit", &NativeAdapter::set_refuse_implicit,
             py::arg("on"),
             "add-exact-propagator-strategy 13.2: refuse a segment with no "
             "exact step (engine='exact' without allow_implicit_segments).")
        .def("set_mask", &NativeAdapter::set_mask,
            "Set the active mask. Subsequent A_matrix / b_vector / "
            "rhs queries resolve against this mask (cached lazily).")
        .def("current_mask",
            [](const NativeAdapter& self) { return self.current_mask(); },
            "Return the currently-active mask.")
        .def("A_matrix",
            [](const NativeAdapter& self) {
                // Return DenseMatrix by value (Eigen copy is cheap for
                // small n_state). Used by Python tests / diagnostics.
                return self.A_matrix();
            },
            "Continuous-time A matrix for the active mask.")
        .def("energy_scale",
            [](const NativeAdapter& self) { return self.energy_scale(); },
            "D's diagonal (sqrt of each state's C or L) for the active mask: "
            "the energy coordinates the exact propagator's bounds are stated in.")
        .def("n_constraints", &NativeAdapter::n_constraints,
            "Rows of the dependent-state constraint K x = k on the active "
            "mask (a capacitor loop or an inductor cut-set; 0 otherwise).")
        .def("b_vector",
            &NativeAdapter::b_vector,
            py::arg("t"),
            "Continuous-time b vector at time `t` (DC + sine + PWM + "
            "pulse + user-supplied overlay).")
        .def("rhs",
            &NativeAdapter::rhs,
            py::arg("t"), py::arg("x"),
            "Continuous-time RHS: dx/dt = A·x + b(t).")
        .def("node_pair_voltage",
            [](const NativeAdapter& self, pulsim::Index from, pulsim::Index to, Real t,
               const pulsim::Vector& x) { return self.node_pair_voltage(from, to, t, x); },
            py::arg("from_node"), py::arg("to_node"),
            py::arg("t"), py::arg("x"),
            "v(from_node) − v(to_node) recovered from the reduced "
            "state `x` under the active mask (−1 is ground). The "
            "quantity a PWL diode predicate evaluates; affine in "
            "`x`, so its gradient is the predicate's recovery row "
            "(used by pulsim.periodic_steady_state for saltation).")
        .def_property_readonly("n_state",
            &NativeAdapter::n_state,
            "Number of continuous-time state variables for the "
            "currently-active mask. Triggers a mask resolution if "
            "not already cached.")
        .def_property_readonly("has_dynamic_sources",
            &NativeAdapter::has_dynamic_sources,
            "Whether any time-varying source is present. Determined "
            "once at construction by probing compute_*_b_extra.")
        .def("reduced_initial_state",
            [](const NativeAdapter& ad,
               const pulsim::builder::CircuitBuilder& b) {
                return ad.reduced_initial_state(b.initial_conditions());
            },
            py::arg("builder"),
            "add-exact-propagator-strategy D5: the reduced state of the "
            "current mask from a full-MNA initial vector "
            "(the builder's c0= / i0= / set_initial), and the largest capacitor-voltage "
            "adjustment the charge-conserving projection made (0 when the "
            "declared values are consistent).")
        .def("reduced_state_from_full",
            [](const NativeAdapter& ad, const Vector& x) {
                // A full-MNA vector (a DC operating point) read the way the
                // builder's c0= / i0= are: every capacitor's branch voltage,
                // every inductor's current, projected onto the current
                // mask's reduced state.
                const auto& g = ad.graph();
                const auto& pool = ad.pool();
                if (x.size() != pool.state_size(g))
                    throw std::invalid_argument("reduced_state_from_full: size mismatch");
                using K = pulsim::pwl::DevicePool::StoredKind;
                std::unordered_map<pulsim::Index, Real> ics;
                auto v = [&](pulsim::Index n) { return n >= 0 ? x[n] : Real{0}; };
                for (pulsim::Index bid = 0; bid < g.num_branches(); ++bid) {
                    if (!pool.is_registered(bid)) continue;
                    const auto k = pool.kind_of(bid);
                    if (k == K::Capacitor)
                        ics[bid] = v(g.branch(bid).from) - v(g.branch(bid).to);
                    else if (k == K::Inductor)
                        ics[bid] = x[pool.branch_var_id_for_inductor(bid, g)];
                }
                return ad.reduced_initial_state(ics);
            },
            py::arg("x_full"),
            "The reduced state of the current mask from a full-MNA vector "
            "(e.g. a DC operating point) and the largest capacitor-voltage "
            "adjustment its projection needed (0 when consistent).")
        .def("_mode_select_at",
            [](const NativeAdapter& ad, const pulsim::topology::SwitchStateMask& mask,
               Real t, const Vector& x_full, Real h_e, Real tol) {
                // add-complementarity-mode-selection task 3: the port model
                // and the selector at one held state (full-MNA vector at
                // t⁻), previous mode = the diode bits of `mask`.
                pulsim::pwl::DiodeEventState census(ad.graph(), ad.pool());
                std::vector<pulsim::events::DiodePort> ports;
                std::vector<std::pair<pulsim::Index, pulsim::Index>> nodes;
                std::vector<char> prev;
                std::vector<std::string> names;
                for (const auto& d : census.entries()) {
                    ports.push_back({Real{1} / d.g_on, d.g_off, d.V_f, d.V_th});
                    nodes.emplace_back(d.from, d.to);
                    prev.push_back(mask.get(d.switch_idx) ? 1 : 0);
                    names.push_back(pulsim::pwl::branch_label(ad.graph(), d.branch_id));
                }
                pulsim::DenseMatrix Z, H;
                Vector u, q;
                ad.diode_port_problem_full(mask, t, x_full, h_e, nodes, Z, u);
                pulsim::events::ModeSelector::hybrid_lcp(Z, u, ports, prev, H, q);
                pulsim::events::ModeSelector sel;
                sel.tol = tol;
                const auto sc = ad.port_scale_full(x_full);
                std::vector<pulsim::Size> sidx;
                for (const auto& d : census.entries()) sidx.push_back(d.switch_idx);
                const auto out = sel.select(ports, prev,
                    [&](Real h, const std::vector<char>& anchor, pulsim::DenseMatrix& Zh, Vector& uh) {
                        auto m = mask;   // posed in the anchor mode (DECISIONS #30)
                        for (std::size_t i = 0; i < sidx.size(); ++i) m.set(sidx[i], anchor[i] != 0);
                        ad.diode_port_problem_full(m, t, x_full, h, nodes, Zh, uh);
                    }, h_e, sc);
                py::dict d;
                d["diodes"] = names;
                py::tuple on(out.on.size());
                for (std::size_t i = 0; i < out.on.size(); ++i) on[i] = py::bool_(out.on[i] != 0);
                d["mode"] = on;
                d["status"] = pulsim::events::to_string(out.status);
                d["pivots"] = out.pivots;
                d["pivots_total"] = out.pivots_total;
                d["ties"] = out.ties;
                d["widenings"] = out.widenings;
                d["leak_pins"] = out.leak_pins;
                d["pin_judgements"] = out.pin_judgements;
                d["unfixed"] = out.unfixed;
                d["capped"] = out.capped;
                d["released"] = out.released;
                d["first_pass_ties"] = out.first_pass_ties;
                d["tie_currents"] = out.tie_currents;
                d["Z"] = Z; d["u"] = u; d["H"] = H; d["q"] = q;
                d["current"] = sel.last_current();
                d["gap"] = sel.last_gap();
                d["vscale"] = sc.vscale; d["iscale"] = sc.iscale; d["c_max"] = sc.c_max;
                return d;
            },
            py::arg("mask"), py::arg("t"), py::arg("x_full"), py::arg("h_e"),
            py::arg("tol") = Real{1e-9},
            "add-complementarity-mode-selection: the diode-port LCP of the "
            "backward-Euler companion at step h_e from the full-MNA state "
            "x_full held at t, posed in the previous mode (the diode bits "
            "of `mask`), and the selector's answer. Test/probe surface.")
        .def_property_readonly("num_cached_masks",
            &NativeAdapter::num_cached_masks,
            "Number of distinct masks resolved (and thus cached) "
            "so far. Useful for tests + diagnostics.");

    m.def("run_ped_native_builder",
        [](NativeAdapter& adapter,
           py::object switch_fn_obj,
           Vector x0,
           Real t_end,
           Real rtol,
           Real atol,
           Real dt_init,
           Real dt_max,
           std::size_t store_every,
           bool enable_event_projection,
           bool event_log,
           Real mode_shadow_h_e,
           Real mode_lcp_h_e,
           Real mode_guard_h_e,
           bool mode_guard_refuse,
           int settle_round_cap, Real diode_law_rel,
           std::vector<Real> tick_periods,
           std::vector<Real> tick_offsets,
           std::vector<py::object> tick_fns,
           std::vector<bool> tick_monitors,
           Real output_rtol, std::size_t output_cap,
           py::typing::Optional<pulsim::solver::SolverSnapshot> resume_arg,
           std::vector<Real> tick_origins,
           std::vector<std::uint64_t> tick_next_k,
           py::object loop_feed,
           Real t_offset, std::size_t output_budget, bool fixed_step) {
            const pulsim::solver::SolverSnapshot* resume = snapshot_or_null(resume_arg);
            PySwitchFnSSM sf(std::move(switch_fn_obj));
            // add-unified-event-queue group 3: control ticks on the
            // scheduler's queue; each gets the full MNA state.
            if (tick_fns.size() != tick_periods.size()
                || (!tick_offsets.empty() && tick_offsets.size() != tick_periods.size())) {
                throw std::invalid_argument(
                    "run_ped_native_builder: tick_fns, tick_periods and tick_offsets "
                    "must have the same length");
            }
            std::vector<std::function<void(Real, const Vector&)>> tick_cbs;
            for (const auto& ob : tick_fns) {
                if (py::isinstance<pulsim::blockchain::ChainTick>(ob)) {   // control graph: C++
                    tick_cbs.push_back(ob.cast<pulsim::blockchain::ChainTick&>().fn);
                } else {
                    tick_cbs.push_back(ob.cast<std::function<void(Real, const Vector&)>>());
                }
            }
            sf.set_width(static_cast<pulsim::Size>(adapter.graph().num_switches()));
            sf.set_poll(poll_window(dt_max));
            PIController ctrl(rtol, atol);
            EventPredictor pred;

            // ---- v2.0 Phase 3: auto-derive diode predicates ------
            //
            // The census is the SAME walk run_transient does
            // (DiodeEventState), so the two engines cannot disagree
            // about which switches are diodes. For each diode, two
            // predicates on v_D reconstructed through the recovery
            // map: turn-ON armed while the bit is OFF
            // (g = v_D − V_on), turn-OFF armed while ON
            // (g = v_D − V_f — the i_D = g_on·(v_D − V_f) zero-cross),
            // both SwitchedDiode::crossing_signal. Firing
            // flips the bit inside the scheduler; the zero-time
            // cascade settles instantaneous consequences (a buck's
            // freewheel diode at gate-off).
            std::vector<pulsim::dsed::SchedulerDiode> sched_diodes;
            std::vector<pulsim::Size> diode_sw;   // census order (the snapshot's mask)
            pulsim::topology::SwitchStateMask diode_owned{0};
            {
                pulsim::pwl::DiodeEventState census(
                    adapter.graph(), adapter.pool());
                if (census.num_diodes() > 0) {
                    diode_owned = census.diode_owned_bits();
                    // An OFF ideal diode across a path of closed ideal elements
                    // (a body diode across its closed switch) has v_D = 0
                    // structurally and every ON candidate is a loop: its turn-on
                    // predicate is held below zero while that holds (it crossed
                    // on rounding: the PSFB's D_LS_B chattered 32 times in 2e-10 s).
                    std::shared_ptr<pulsim::pwl::BranchStructureChecker> shorted;
                    try {
                        shorted = std::make_shared<pulsim::pwl::BranchStructureChecker>(
                            adapter.graph(), adapter.pool(), "dsed");
                        if (!shorted->active()) shorted.reset();
                    } catch (const std::exception&) {
                        shorted.reset();   // compute_lti_state_space refuses it by name
                    }
                    for (const auto& d : census.entries()) {
                        diode_sw.push_back(d.switch_idx);
                        std::string label =
                            pulsim::pwl::branch_label(
                                adapter.graph(), d.branch_id);
                        const NativeAdapter* ad = &adapter;
                        const pulsim::Index from = d.from;
                        const pulsim::Index to = d.to;
                        // a branch-form diode's ON law is its solved current, R_on = 0
                        // or not (SchedulerDiode::reads_current): V_f + i with the
                        // census's g_on = 1, so the signal and its bands are amperes
                        const pulsim::Index var = d.var;
                        const Real R_on_b = d.var >= 0
                            ? adapter.pool().branch_R_on(d.branch_id) : Real{0};
                        const auto prm = d.params();
                        const auto chk = var >= 0 && R_on_b == Real{0} ? shorted : nullptr;
                        const pulsim::Size sw = d.switch_idx;
                        pred.add_predicate(pulsim::dsed::EventPredicate{
                            .name = label + " turns ON",
                            .fn = [ad, from, to, prm, chk, sw](
                                      Real t, const Vector& x) {
                                if (chk) {
                                    auto m = ad->current_mask();
                                    m.set(sw, true);
                                    try {
                                        // the bit closes a loop iff it adds one: a new loop
                                        // passes through the added branch, whichever diode
                                        // the checker blames (the PSFB's D3 across D1, D2,
                                        // D4 ON was not blamed: D4 was, and D3 chattered)
                                        const std::size_t with = chk->check(m, true).loop_diodes.size();
                                        m.set(sw, false);
                                        const std::size_t without = chk->check(m, true).loop_diodes.size();
                                        // a loop through a source is a real bias: the law
                                        // decides (refused by name as ideal_voltage_loop);
                                        // the projection's rounding (2e-14 V at mid_b) is
                                        // not bounded by the node pair's own scale
                                        if (with > without
                                            && std::abs(ad->node_pair_voltage(from, to, t, x))
                                                   <= Real{1e-9} * ad->port_scale(t, x).vscale)
                                            return Real{-1};
                                    } catch (const std::exception&) {
                                        // a floating node of the candidate: the law decides
                                    }
                                }
                                return pulsim::models::SwitchedDiode::
                                    crossing_signal(
                                        false,
                                        ad->node_pair_voltage(
                                            from, to, t, x),
                                        prm);
                            },
                            .type = pulsim::dsed::PredicateType::DiodeOn,
                            .priority = pulsim::dsed::
                                default_priority(
                                    pulsim::dsed::PredicateType::DiodeOn),
                            .aux_index =
                                static_cast<int>(d.switch_idx),
                            .required_bit = 0,
                        });
                        pred.add_predicate(pulsim::dsed::EventPredicate{
                            .name = label + " turns OFF",
                            // an ideal (branch-form) diode's ON law reads its
                            // solved current: V_f + i (g_on = 1, the census's
                            // law_voltage)
                            .fn = [ad, from, to, var, prm](
                                      Real t, const Vector& x) {
                                return pulsim::models::SwitchedDiode::
                                    crossing_signal(
                                        true,
                                        var >= 0
                                            ? prm.V_f + ad->node_pair_voltage(
                                                  var, -1, t, x)
                                            : ad->node_pair_voltage(
                                                  from, to, t, x),
                                        prm);
                            },
                            .type = pulsim::dsed::PredicateType::DiodeOff,
                            .priority = pulsim::dsed::
                                default_priority(
                                    pulsim::dsed::PredicateType::DiodeOff),
                            .aux_index =
                                static_cast<int>(d.switch_idx),
                            .required_bit = 1,
                        });
                        sched_diodes.push_back(
                            pulsim::dsed::SchedulerDiode{
                                d.switch_idx, d.from, d.to,
                                prm.g_on, d.g_off, d.V_th, d.V_f,
                                std::move(label), d.var, R_on_b});
                    }
                }
            }

            PEDSimulator<NativeAdapter, PySwitchFnSSM> sim(
                adapter, std::move(sf), std::move(ctrl),
                std::move(pred), dt_init, dt_max, store_every);
            if (!sched_diodes.empty()) {
                const auto n_sw = static_cast<pulsim::Size>(
                    adapter.graph().num_switches());
                sim.enable_diode_commutation(
                    diode_owned,
                    pulsim::topology::SwitchStateMask(n_sw),
                    std::move(sched_diodes));
            }
            sim.set_event_projection(enable_event_projection);
            sim.set_settle_round_cap(settle_round_cap);
            sim.set_diode_law_check(diode_law_rel);
            if (output_rtol > Real{0}) sim.set_output_rule(output_rtol, atol, output_cap, output_budget);
            sim.set_fixed_step(fixed_step);
            if (!tick_cbs.empty()) {
                pulsim::events::TickPlan plan;
                for (std::size_t j = 0; j < tick_periods.size(); ++j) {
                    plan.rates.push_back(pulsim::events::Rate{
                        tick_periods[j], tick_offsets.empty() ? Real{0} : tick_offsets[j]});
                    // a resume continues the first run's grid (its origin and index)
                    if (j < tick_origins.size()) plan.rates.back().origin = tick_origins[j];
                    if (j < tick_next_k.size()) plan.rates.back().k0 = tick_next_k[j];
                }
                plan.monitor = tick_monitors;   // empty = every rate a ControlTick
                plan.fn = [&adapter, &tick_cbs](std::uint32_t j, const pulsim::events::TickInfo&,
                                                Real t, const Vector& x) {
                    const Vector full = adapter.recover_full(t, x);
                    tick_cbs[j](t, full);
                };
                sim.set_ticks(std::move(plan));
            }
            std::vector<std::string> diode_names;
            if (mode_shadow_h_e > Real{0}) {
                pulsim::pwl::DiodeEventState census(adapter.graph(), adapter.pool());
                for (const auto& d : census.entries())
                    diode_names.push_back(pulsim::pwl::branch_label(adapter.graph(), d.branch_id));
                if (!diode_names.empty()) sim.enable_mode_shadow(mode_shadow_h_e);
            }
            // add-complementarity-mode-selection slice 2: the LCP decides
            // (mode_selection='lcp'), or guards the settle (default)
            const bool any_diode = diode_owned.count() > 0;
            if (any_diode && mode_lcp_h_e > Real{0}) {
                sim.enable_mode_lcp(mode_lcp_h_e, adapter.passivity());
            } else if (any_diode && mode_guard_h_e > Real{0}) {
                sim.enable_mode_guard(mode_guard_h_e, adapter.passivity(), mode_guard_refuse);
            }
            if (any_diode && diode_names.empty()) {
                pulsim::pwl::DiodeEventState census(adapter.graph(), adapter.pool());
                for (const auto& dd : census.entries())
                    diode_names.push_back(pulsim::pwl::branch_label(adapter.graph(), dd.branch_id));
            }
            // add-in-kernel-event-losses 3.2: the loss port on the step hooks
            std::optional<pulsim::dsed::ExactLossPort<NativeAdapter>> loss;
            pulsim::events::StepHooks loss_hooks;
            if (!adapter.pool().device_losses().empty()) {
                loss.emplace(adapter);
                loss->attach(loss_hooks);
                sim.set_hooks(&loss_hooks);
                if (loss->has_heat()) {   // the junction loop: the step's heat into the state
                    sim.set_state_kick([&loss](pulsim::Vector& x, pulsim::Vector* xl) {
                        return loss->kick(x, xl);
                    });
                }
            }
            // add-probes-and-recorders R5 / 0b.11: the recorders fed in the loop
            // (record::LoopFeed through dsed/loop_feed_adapter.hpp: full MNA
            // states, events at their exact instants). Not with the junction
            // loop: its kick moves the state after the step's hooks, so the
            // feed's sample would miss the step's heat (the stored arrays have it)
            std::optional<pulsim::dsed::ExactFeedAdapter<NativeAdapter>> feed_ad;
            // F-B2-1: the recorders' pieces in closed form (segment_integrals.hpp);
            // the feed outlives this call, so its pointer is cleared on every exit
            std::optional<pulsim::dsed::ExactSegmentSource<NativeAdapter>> seg_src;
            struct SegGuard {
                pulsim::record::LoopFeed* f = nullptr;
                ~SegGuard() { if (f != nullptr) f->set_segment_source(nullptr); }
            } seg_guard;
            std::string feed_refused;
            if (!loop_feed.is_none()) {
                if (loss && loss->has_heat()) {
                    feed_refused = "the junction temperature loop corrects the state after "
                                   "each step's hooks (dsed/loss_port.hpp kick), so an in-loop "
                                   "sample would miss the step's heat";
                } else {
                    auto& f = loop_feed.cast<pulsim::record::LoopFeed&>();
                    f.set_circuit(adapter.graph(), adapter.pool());
                    feed_ad.emplace(adapter, f.hooks(),
                                    resume != nullptr ? resume->t : Real{0});
                    feed_ad->attach(loss_hooks);
                    sim.set_hooks(&loss_hooks);
                    seg_src.emplace(adapter, f);
                    seg_src->set_extrema(output_rtol > Real{0});   // F-Y-3: rule on only
                    f.set_segment_source(&*seg_src);
                    seg_guard.f = &f;
                    feed_ad->set_segment(&*seg_src);
                }
            }
            // session 24: resume at a snapshot's committed instant
            using ResumeT = pulsim::dsed::ExactResumeState<pulsim::topology::SwitchStateMask>;
            std::optional<ResumeT> rs;
            const auto n_sw = static_cast<pulsim::Size>(adapter.graph().num_switches());
            // the circuit's device-state layout, as the pwl / trbdf2 snapshots
            // carry it (the exact engine runs no stateful device: the linear
            // companions' count only); a resume refuses another circuit's
            pulsim::device::DeviceStateLayout layout;
            layout.count[static_cast<std::size_t>(pulsim::device::StateBlock::LinearCompanion)] =
                static_cast<std::uint32_t>(adapter.pool().num_dynamic_branches());
            if (resume != nullptr) {
                if (!resume->dev_layout.empty()) {
                    const std::string diff = layout.first_difference(resume->layout());
                    if (!diff.empty() || resume->layout_signature
                                             != resume->layout().signature(resume->dev_layout.size())) {
                        throw std::invalid_argument(
                            "dsed(resume_from=...): the snapshot belongs to a different "
                            "circuit: " + (diff.empty() ? std::string("its device-state layout "
                                                                      "is corrupt (signature)")
                                                        : diff + " (this circuit, then the snapshot)"));
                    }
                }
                if (resume->controller.kind == pulsim::step::ControllerState::Kind::Exact) {
                    rs = ResumeT::unflatten(resume->controller.v);
                    if (rs->x.size() != x0.size()) {
                        throw std::invalid_argument(
                            "dsed(resume_from=...): the snapshot belongs to a different "
                            "circuit: its exact-engine state has "
                            + std::to_string(rs->x.size()) + " entries, this circuit's "
                            + std::to_string(x0.size()));
                    }
                } else {   // another engine's snapshot: x0 is its state, reduced
                    rs.emplace();
                    rs->t = resume->t;
                    rs->x = x0;
                    rs->decide = true;
                }
                if (resume->stamped_mask.size() != n_sw) {
                    throw std::invalid_argument(
                        "dsed(resume_from=...): the snapshot's stamped mask has "
                        + std::to_string(resume->stamped_mask.size())
                        + " bits, this circuit has " + std::to_string(n_sw)
                        + " switch-kind branches");
                }
                pulsim::topology::SwitchStateMask m(n_sw);
                for (pulsim::Size i = 0; i < n_sw; ++i) m.set(i, resume->stamped_mask[i]);
                rs->mask = m;
                rs->diode_bits = m;   // only the owned bits are read
                if (loss && !resume->loss_state.empty()) loss->restore(resume->loss_state);
            }
            // Release the GIL during simulate(); the C++ scheduler
            // only re-acquires it at gate edges to call switch_fn.
            py::gil_scoped_release release;
            auto res = sim.simulate(x0, t_end, rs ? &*rs : nullptr);
            py::gil_scoped_acquire acquire;
            auto d = ped_result_to_dict(res, event_log);
            d["n_rk45_steps"] = res.n_accept - res.n_exact_steps;   // its non-exact steps are DOPRI5 (F-3-6)
            {   // the end instant as a snapshot (pickle version 5; field 14 kind Exact)
                const auto& f = sim.final_state();
                pulsim::solver::SolverSnapshot snap;
                snap.t = f.t;
                if (!(adapter.current_mask() == f.mask)) adapter.set_mask(f.mask);
                snap.x = adapter.recover_full(f.t, f.x);
                snap.stamped_mask.resize(n_sw);
                for (pulsim::Size i = 0; i < n_sw; ++i) snap.stamped_mask[i] = f.mask.get(i);
                for (const auto sw : diode_sw) snap.mask.push_back(f.mask.get(sw));
                snap.controller.kind = pulsim::step::ControllerState::Kind::Exact;
                snap.controller.v = f.flatten();
                snap.dev_layout.assign(layout.count.begin(), layout.count.end());
                snap.layout_signature = layout.signature();
                snap.tick_origin = sim.tick_origins();
                snap.tick_next_k = sim.tick_next_k();
                if (loss) snap.loss_state = loss->save();
                snap.valid = true;
                d["final_snapshot"] = py::cast(std::move(snap));
            }
            if (loss) {
                struct {
                    std::vector<pulsim::thermal::DeviceLossRecord> device_losses;
                    Real heat_injected = 0, heat_pending = 0;
                } out;
                loss->write(out);
                // a fresh run with t_start runs on the engine's clock from 0 and
                // its times are relabelled by t_start: so are its loss events
                if (t_offset != Real{0}) {
                    for (auto& r : out.device_losses) {
                        for (auto& e : r.events) e.t += t_offset;
                    }
                }
                d["device_losses"] = py::cast(out.device_losses);
                d["heat_injected"] = out.heat_injected;
                d["heat_pending"] = out.heat_pending;
            }
            if (!feed_refused.empty()) d["loop_feed_refused"] = feed_refused;
            if (seg_src) {
                py::dict cf;
                cf["n_closed"] = seg_src->n_closed();
                cf["n_line"] = seg_src->n_line();
                cf["why"] = seg_src->why();
                if (output_rtol > Real{0}) {   // rule off: the PR head's keys only
                    cf["n_extrema_closed"] = seg_src->n_extrema_closed();
                    cf["n_extrema_ends"] = seg_src->n_extrema_ends();
                }
                d["recorder_closed_form"] = cf;
            }
            if (!tick_cbs.empty()) d["control_tick_counts"] = sim.tick_counts();
            if (output_rtol > Real{0}) {
                d["n_output_capped"] = sim.n_output_capped();
                d["n_output_unbounded"] = sim.n_output_unbounded();
                d["output_budget_t"] = sim.output_budget_t();
                d["n_output_floored"] = sim.n_output_floored();
                d["n_output_inconsistent"] = sim.n_output_inconsistent();
                const auto& fr = sim.output_floor_row();
                d["output_floor_row"] = std::vector<double>(fr.data(), fr.data() + fr.size());
            }
            {   // every event projection, rule on or off (F-SB-3 / F-R-3): one (n, 4)
                // array [t, amplitude, row (-1: none, rule off), tau]; Python builds the
                // records only when they are read. The per-row maximum exists under the rule
                d["n_projections"] = sim.n_projections();
                if (output_rtol > Real{0}) {
                    const auto pm = sim.projection_max();
                    d["projection_max_row"] = std::vector<double>(pm.data(), pm.data() + pm.size());
                }
                const auto recs = sim.projection_records();
                py::array_t<double> a({static_cast<py::ssize_t>(recs.size()), py::ssize_t{4}});
                auto m = a.mutable_unchecked<2>();
                for (std::size_t k = 0; k < recs.size(); ++k) {
                    const auto& r = recs[k];
                    const auto i = static_cast<py::ssize_t>(k);
                    m(i, 0) = r.t;
                    m(i, 1) = r.amplitude;
                    m(i, 2) = r.row == std::remove_cvref_t<decltype(sim)>::kNoProjectionRow
                                  ? -1.0 : static_cast<double>(r.row);
                    m(i, 3) = r.tau;
                }
                d["projection_records"] = a;
            }
            d["exact"] = exact_report_dict(adapter, res.n_predicate_evals);
            if (mode_shadow_h_e > Real{0}) {
                d["mode_shadow"] = mode_shadow_dict(res.mode_shadow, diode_names,
                                                    adapter.n_port_problems());
            }
            if (res.mode_shadow.decides) {
                auto ms = mode_shadow_dict(res.mode_shadow, diode_names,
                                           adapter.n_port_problems());
                ms["passivity"] = passivity_dict(adapter.passivity());
                d["mode_selection"] = ms;
            }
            if (res.mode_guard.enabled) {
                const auto& g = res.mode_guard;
                py::dict gd;
                gd["checks"] = g.checks;
                gd["unchecked"] = g.unchecked;
                gd["repairs"] = g.repairs;
                gd["settle_caps"] = g.settle_caps;
                gd["confirmed"] = g.confirmed;
                gd["repair_times"] = g.repair_times;
                gd["repair_diodes"] = g.repair_diodes;
                gd["repair_violations"] = g.repair_violations;
                gd["lcp"] = mode_shadow_dict(g.lcp, diode_names, adapter.n_port_problems());
                gd["passivity"] = passivity_dict(adapter.passivity());
                d["mode_guard"] = gd;
            }
            // v2.0 Phase 3 item 5 — the unified result: full MNA
            // trajectory reconstructed per recorded sample.
            d["states_full"] = reconstruct_full_states(adapter, res);
            reconstruct_right_limits(adapter, res, d);
            reinit_records_into(adapter, res, d);
            return d;
        },
        py::arg("adapter"),
        py::arg("switch_fn"),
        py::arg("x0"),
        py::arg("t_end"),
        py::arg("rtol")        = Real{1e-6},
        py::arg("atol")        = Real{1e-9},
        py::arg("dt_init")     = Real{1e-9},
        py::arg("dt_max")      = Real{1e-5},
        py::arg("store_every") = std::size_t{1},
        py::arg("enable_event_projection") = true,
        py::arg("event_log") = true,
        py::arg("mode_shadow_h_e") = Real{0},
        py::arg("mode_lcp_h_e") = Real{0},
        py::arg("mode_guard_h_e") = Real{0},
        py::arg("mode_guard_refuse") = false,
        py::arg("settle_round_cap") = 64,
        py::arg("diode_law_rel") = Real{1e-3},
        py::arg("tick_periods") = std::vector<Real>{},
        py::arg("tick_offsets") = std::vector<Real>{},
        py::arg("tick_fns") = std::vector<py::object>{},
        py::arg("tick_monitors") = std::vector<bool>{},
        py::arg("output_rtol") = Real{0},
        py::arg("output_cap") = std::size_t{4096},
        py::arg("resume") = py::none(),
        py::arg("tick_origins") = std::vector<Real>{},
        py::arg("tick_next_k") = std::vector<std::uint64_t>{},
        py::arg("loop_feed") = py::none(),
        py::arg("t_offset") = Real{0},
        py::arg("output_budget") = std::size_t{2000000},
        py::arg("fixed_step") = false,
        "Bridge.11 — native PEDSimulator (DOPRI5) over a "
        "NativeCircuitBuilderAdapter. RHS / b_vector / A_matrix all "
        "execute in C++ without GIL roundtrips; the only Python "
        "callback is `switch_fn` at gate edges (rare). v2.0 Phase 3: "
        "PWL diodes are commutated by auto-derived event predicates "
        "on the reconstructed branch voltage, and every event "
        "projects the state onto the new mode's slow manifold "
        "(consistent reinitialization; pass "
        "enable_event_projection=False for the raw pre-projection "
        "behaviour).");

    m.def("run_bdf2_native_builder",
        [](NativeAdapter& adapter,
           py::object switch_fn_obj,
           Vector x0,
           Real t_end,
           Real h_fixed,
           std::size_t store_every,
           bool event_log) {
            // no loss tracking here: an adapter built with loss_port refuses by name
            pulsim::refuse_device_losses(adapter.pool(), "dsed");
            PySwitchFnSSM sf(std::move(switch_fn_obj));
            sf.set_width(static_cast<pulsim::Size>(adapter.graph().num_switches()));
            sf.set_poll(poll_window(h_fixed));
            PEDSimulatorBDF2<NativeAdapter, PySwitchFnSSM> sim(
                adapter, std::move(sf), h_fixed, store_every);
            py::gil_scoped_release release;
            auto res = sim.simulate(x0, t_end);
            py::gil_scoped_acquire acquire;
            auto d = ped_result_to_dict(res, event_log);
            d["states_full"] = reconstruct_full_states(adapter, res);
            reconstruct_right_limits(adapter, res, d);
            reinit_records_into(adapter, res, d);
            return d;
        },
        py::arg("adapter"),
        py::arg("switch_fn"),
        py::arg("x0"),
        py::arg("t_end"),
        py::arg("h_fixed")     = Real{1e-6},
        py::arg("store_every") = std::size_t{1},
        py::arg("event_log") = true,
        "Bridge.11 — native PEDSimulatorBDF2 over a "
        "NativeCircuitBuilderAdapter (implicit BDF2 + CN bootstrap).");

    m.def("run_auto_native_builder",
        [](NativeAdapter& adapter,
           py::object switch_fn_obj,
           Vector x0,
           Real t_end,
           Real rtol,
           Real atol,
           Real dt_init,
           Real dt_max,
           Real h_bdf2,
           Real stiffness_threshold,
           std::size_t store_every,
           bool event_log,
           py::typing::Optional<pulsim::solver::SolverSnapshot> resume_arg,
           py::object loop_feed) {
            const pulsim::solver::SolverSnapshot* resume = snapshot_or_null(resume_arg);
            // no loss tracking here: an adapter built with loss_port refuses by name
            pulsim::refuse_device_losses(adapter.pool(), "dsed");
            PySwitchFnSSM sf(std::move(switch_fn_obj));
            sf.set_width(static_cast<pulsim::Size>(adapter.graph().num_switches()));
            sf.set_poll(poll_window(dt_max));
            PIController ctrl(rtol, atol);
            StiffnessDetector det(stiffness_threshold);
            PEDSimulatorAuto<NativeAdapter, PySwitchFnSSM> sim(
                adapter, std::move(sf), std::move(ctrl),
                std::move(det), dt_init, dt_max, h_bdf2,
                store_every);
            // task 20.2 (session 26): snapshot / resume on this integrator too
            using AutoT = pulsim::dsed::AutoResumeState<pulsim::topology::SwitchStateMask>;
            const auto n_sw = static_cast<pulsim::Size>(adapter.graph().num_switches());
            pulsim::device::DeviceStateLayout layout;
            layout.count[static_cast<std::size_t>(pulsim::device::StateBlock::LinearCompanion)] =
                static_cast<std::uint32_t>(adapter.pool().num_dynamic_branches());
            std::optional<AutoT> rs;
            if (resume != nullptr) {
                if (resume->controller.kind != pulsim::step::ControllerState::Kind::Exact) {
                    throw std::invalid_argument(
                        "dsed(integrator='auto', resume_from=...): only an exact-engine "
                        "snapshot of this integrator continues here");
                }
                if (!resume->dev_layout.empty()) {
                    const std::string diff = layout.first_difference(resume->layout());
                    if (!diff.empty() || resume->layout_signature
                                             != resume->layout().signature(resume->dev_layout.size())) {
                        throw std::invalid_argument(
                            "dsed(resume_from=...): the snapshot belongs to a different "
                            "circuit: " + (diff.empty() ? std::string("its device-state layout "
                                                                      "is corrupt (signature)")
                                                        : diff + " (this circuit, then the snapshot)"));
                    }
                }
                rs = AutoT::unflatten(resume->controller.v);
                if (rs->x.size() != x0.size() || resume->stamped_mask.size() != n_sw) {
                    throw std::invalid_argument(
                        "dsed(resume_from=...): the snapshot belongs to a different circuit "
                        "(its state or switch count differs)");
                }
                pulsim::topology::SwitchStateMask m(n_sw);
                for (pulsim::Size i = 0; i < n_sw; ++i) m.set(i, resume->stamped_mask[i]);
                rs->mask = m;
            }
            // the recorders in the loop, closed form per exact step (as on
            // run_ped_native_builder; the hooks only observe)
            pulsim::events::StepHooks hooks;
            std::optional<pulsim::dsed::ExactFeedAdapter<NativeAdapter>> feed_ad;
            std::optional<pulsim::dsed::ExactSegmentSource<NativeAdapter>> seg_src;
            struct SegGuard {
                pulsim::record::LoopFeed* f = nullptr;
                ~SegGuard() { if (f != nullptr) f->set_segment_source(nullptr); }
            } seg_guard;
            if (!loop_feed.is_none()) {
                auto& f = loop_feed.cast<pulsim::record::LoopFeed&>();
                f.set_circuit(adapter.graph(), adapter.pool());
                feed_ad.emplace(adapter, f.hooks(), rs ? rs->t : Real{0});
                feed_ad->attach(hooks);
                sim.set_hooks(&hooks);
                seg_src.emplace(adapter, f);
                f.set_segment_source(&*seg_src);
                seg_guard.f = &f;
                feed_ad->set_segment(&*seg_src);
            }
            py::gil_scoped_release release;
            auto res = sim.simulate(x0, t_end, rs ? &*rs : nullptr);
            py::gil_scoped_acquire acquire;
            auto d = ped_result_auto_to_dict(res, event_log);
            if (seg_src) {
                py::dict cf;
                cf["n_closed"] = seg_src->n_closed();
                cf["n_line"] = seg_src->n_line();
                cf["why"] = seg_src->why();
                d["recorder_closed_form"] = cf;
            }
            {   // the end instant as a snapshot (field 14 kind Exact, AutoResumeState)
                const auto& f = sim.final_state();
                pulsim::solver::SolverSnapshot snap;
                snap.t = f.t;
                if (!(adapter.current_mask() == f.mask)) adapter.set_mask(f.mask);
                snap.x = adapter.recover_full(f.t, f.x);
                snap.stamped_mask.resize(n_sw);
                for (pulsim::Size i = 0; i < n_sw; ++i) snap.stamped_mask[i] = f.mask.get(i);
                snap.controller.kind = pulsim::step::ControllerState::Kind::Exact;
                snap.controller.v = f.flatten();
                snap.dev_layout.assign(layout.count.begin(), layout.count.end());
                snap.layout_signature = layout.signature();
                snap.valid = true;
                d["final_snapshot"] = py::cast(std::move(snap));
            }
            d["exact"] = exact_report_dict(adapter, 0);
            d["states_full"] = reconstruct_full_states(adapter, res);
            reconstruct_right_limits(adapter, res, d);
            reinit_records_into(adapter, res, d);
            return d;
        },
        py::arg("adapter"),
        py::arg("switch_fn"),
        py::arg("x0"),
        py::arg("t_end"),
        py::arg("rtol")                = Real{1e-6},
        py::arg("atol")                = Real{1e-9},
        py::arg("dt_init")             = Real{1e-9},
        py::arg("dt_max")              = Real{1e-5},
        py::arg("h_bdf2")              = Real{1e-6},
        py::arg("stiffness_threshold") = Real{10.0},
        py::arg("store_every")          = std::size_t{1},
        py::arg("event_log") = true,
        py::arg("resume") = py::none(),
        py::arg("loop_feed") = py::none(),
        "Bridge.11 — native PEDSimulatorAuto over a "
        "NativeCircuitBuilderAdapter (per-mode RK45↔BDF2 dispatch).");

    // A non-owning view of a PySwitchFnSSM: copying it touches no Python
    // object, so a period run is set up with the GIL released.
    struct SwitchFnRef {
        const PySwitchFnSSM* p;
        [[nodiscard]] SSM operator()(Real t) const { return (*p)(t); }
        [[nodiscard]] Real next_edge_after(Real t) const {
            return p->next_edge_after(t);
        }
    };
    using pulsim::dsed::PeriodMapResult;
    using pulsim::dsed::PeriodObservable;
    using Pairs = std::vector<std::pair<int, int>>;
    // (indices, coefs, bit, g_on, g_off, v_on, extrema) per observable
    static const auto parse_observables = [](const py::list& observables,
                                             const Pairs& pairs) {
        std::vector<PeriodObservable> obs;
        for (const auto& item : observables) {
            const auto t = item.cast<py::tuple>();
            PeriodObservable o;
            const auto idx = t[0].cast<std::vector<pulsim::Index>>();
            const auto cf = t[1].cast<std::vector<Real>>();
            for (std::size_t i = 0; i < idx.size(); ++i) {
                o.w.emplace_back(idx[i], cf.at(i));
            }
            o.bit = t[2].cast<int>();
            o.g_on = t[3].cast<Real>();
            o.g_off = t[4].cast<Real>();
            o.v_on = t[5].cast<Real>();
            o.extrema = t[6].cast<bool>();
            obs.push_back(std::move(o));
        }
        for (const auto& [a, c] : pairs) {
            if (a < 0 || c < 0 || a >= static_cast<int>(obs.size())
                || c >= static_cast<int>(obs.size())) {
                throw py::value_error("dsed_period_map: pair index out "
                                      "of range");
            }
        }
        return obs;
    };
    // One period of the exact stepper from x0 (events located, no
    // projection) and its post-processing; no Python object is touched
    // except through a Python schedule, which takes the GIL itself.
    static const auto period_run = [](
            NativeAdapter& adapter, const PySwitchFnSSM& sf, const Vector& x0,
            Real period, const std::vector<PeriodObservable>& obs,
            const Pairs& pairs, bool monodromy, Real rtol, Real atol,
            Real mode_guard_h_e) {
        EventPredictor pred;
        auto diodes = pulsim::dsed::add_diode_predicates(adapter, pred);
        PEDSimulator<NativeAdapter, SwitchFnRef> sim(
            adapter, SwitchFnRef{&sf}, PIController(rtol, atol),
            std::move(pred), period * Real{1e-6}, period, 1);
        if (!diodes.list.empty()) {
            const auto n_sw = static_cast<pulsim::Size>(
                adapter.graph().num_switches());
            sim.enable_diode_commutation(
                diodes.owned, pulsim::topology::SwitchStateMask(n_sw),
                diodes.list);
            if (mode_guard_h_e > Real{0}) {
                // the kernel diode-mode guard, as the pss runs set it
                sim.enable_mode_guard(mode_guard_h_e, adapter.passivity());
            }
        }
        sim.set_event_projection(false);
        sim.set_diode_law_check(0.0);   // judged by the period map's own residual
        auto res = sim.simulate(x0, period);
        return pulsim::dsed::period_map_from_run(adapter, res, diodes, obs,
                                                 pairs, monodromy);
    };
    static const auto period_dict = [](const PeriodMapResult<SSM>& r,
                                       std::size_t n_python_calls,
                                       bool nogil) {
        py::dict d;
        d["x_end"] = r.x_end;
        d["monodromy"] = r.monodromy.size()
            ? py::object(py::cast(r.monodromy)) : py::object(py::none());
        d["grazing"] = r.grazing;
        py::list evs;
        for (const auto& e : r.events) {
            py::dict ed;
            ed["t"] = e.t;
            ed["name"] = e.name;
            ed["predicate_type"] = e.type;
            ed["old_mask"] = py::cast(e.old_mask);
            ed["new_mask"] = py::cast(e.new_mask);
            ed["x"] = e.x;
            evs.append(ed);
        }
        d["events"] = evs;
        d["first_mask"] = py::cast(r.first_mask);
        d["last_mask"] = py::cast(r.last_mask);
        d["scale"] = r.scale;
        d["n_accept"] = r.n_steps;
        d["n_exact_steps"] = r.n_exact_steps;
        d["python_callbacks"] = n_python_calls;
        d["nogil"] = nogil;
        d["integrals_ok"] = r.integrals_ok;
        d["integrals_reason"] = r.integrals_reason;
        const auto n_obs = static_cast<py::ssize_t>(r.stats.size());
        py::array_t<double> st(std::vector<py::ssize_t>{n_obs, 6});
        for (py::ssize_t j = 0; j < n_obs; ++j) {
            const auto& s = r.stats[static_cast<std::size_t>(j)];
            const double row[6] = {s.int1, s.int2, s.peak, s.t_peak,
                                   s.valley, s.t_valley};
            for (int c = 0; c < 6; ++c) st.mutable_at(j, c) = row[c];
        }
        d["stats"] = st;
        d["cross"] = r.cross;
        d["y_left"] = r.y_left;
        d["y_right"] = r.y_right;
        d["y_start"] = r.y_start;
        d["y_end"] = r.y_end;
        return d;
    };

    // add-design-optimization-workflow (kernel item 3): one period of
    // the exact stepper as a map, with M and the period integrals
    // (pulsim/dsed/period_integrals.hpp), the GIL released for the run
    // AND its post-processing.
    m.def("dsed_period_map",
        [](NativeAdapter& adapter, py::object switch_fn_obj, Vector x0,
           Real period, py::list observables,
           std::vector<std::pair<int, int>> pairs, bool monodromy,
           Real rtol, Real atol, Real mode_guard_h_e) {
            // no loss tracking here: an adapter built with loss_port refuses by name
            pulsim::refuse_device_losses(adapter.pool(), "dsed");
            if (adapter.has_dynamic_sources()) {
                throw py::value_error(
                    "dsed_period_map: time-varying sources are not "
                    "handled (their b(t) has no closed form here)");
            }
            const auto obs = parse_observables(observables, pairs);
            std::size_t n_python_calls = 0;
            PySwitchFnSSM sf(std::move(switch_fn_obj));
            sf.set_width(static_cast<pulsim::Size>(adapter.graph().num_switches()));
            sf.set_poll(poll_window(period));
            sf.n_python_calls = &n_python_calls;
            PeriodMapResult<SSM> r;
            bool nogil = false;
            {
                py::gil_scoped_release release;
                nogil = PyGILState_Check() == 0;
                r = period_run(adapter, sf, x0, period, obs, pairs,
                               monodromy, rtol, atol, mode_guard_h_e);
            }
            return period_dict(r, n_python_calls, nogil);
        },
        py::arg("adapter"), py::arg("switch_fn"), py::arg("x0"),
        py::arg("period"), py::arg("observables") = py::list(),
        py::arg("pairs") = std::vector<std::pair<int, int>>{},
        py::arg("monodromy") = true, py::arg("rtol") = Real{1e-11},
        py::arg("atol") = Real{1e-14},
        py::arg("mode_guard_h_e") = Real{0},
        "One period of the exact stepper from x0 (events located, no "
        "projection) with the GIL released: x(T), the events, the "
        "monodromy M = dP/dx0 and, per observable "
        "(indices, coefs, bit, g_on, g_off, v_on, extrema), the closed-form "
        "integrals stats[j] = (int y, int y^2, peak, t_peak, valley, "
        "t_valley) and cross[q] = int y_a*y_b per pair. integrals_ok is "
        "False with a reason when a mode has no modal form or a step "
        "was numeric. DC sources only.");

    // add-design-optimization-workflow DECISIONS 19: the whole Newton
    // shooting (pulsim/dsed/periodic_orbit.hpp) in one call, the GIL
    // released for every period run and every Newton step.
    m.def("dsed_periodic_orbit",
        [](NativeAdapter& adapter, py::object switch_fn_obj, Vector x0,
           Real period, py::list observables,
           std::vector<std::pair<int, int>> pairs, Real rtol,
           int max_period_runs, int n_relax, Real run_rtol, Real run_atol,
           Real mode_guard_h_e) {
            // no loss tracking here: an adapter built with loss_port refuses by name
            pulsim::refuse_device_losses(adapter.pool(), "dsed");
            if (adapter.has_dynamic_sources()) {
                throw py::value_error(
                    "dsed_periodic_orbit: time-varying sources are not "
                    "handled (their b(t) has no closed form here)");
            }
            const auto obs = parse_observables(observables, pairs);
            const std::vector<PeriodObservable> no_obs;
            const Pairs no_pairs;
            std::size_t n_python_calls = 0;
            PySwitchFnSSM sf(std::move(switch_fn_obj));
            sf.set_width(static_cast<pulsim::Size>(adapter.graph().num_switches()));
            sf.set_poll(poll_window(period));
            sf.n_python_calls = &n_python_calls;
            pulsim::dsed::OrbitOptions opt;
            opt.rtol = rtol;
            opt.max_period_runs = max_period_runs;
            opt.n_relax = n_relax;
            pulsim::dsed::OrbitResult<SSM> r;
            bool nogil = false;
            {
                py::gil_scoped_release release;
                nogil = PyGILState_Check() == 0;
                auto run = [&](const Vector& x, bool with_obs) {
                    try {
                        return period_run(adapter, sf, x, period,
                                          with_obs ? obs : no_obs,
                                          with_obs ? pairs : no_pairs, true,
                                          run_rtol, run_atol, mode_guard_h_e);
                    } catch (const pulsim::analysis::Cancelled&) {
                        throw;
                    } catch (const py::builtin_exception&) {
                        throw;
                    } catch (const std::runtime_error& e) {
                        // the engine's own non-convergence: a typed outcome
                        throw pulsim::dsed::PeriodRunFailed(e.what());
                    }
                };
                r = pulsim::dsed::periodic_orbit_newton<SSM>(
                    run, std::move(x0), period, opt);
            }
            py::dict d;
            d["status"] = pulsim::dsed::to_string(r.status);
            d["message"] = r.message;
            d["verify_budget"] = r.verify_budget;
            d["x0"] = r.x0;
            d["monodromy"] = r.monodromy.size()
                ? py::object(py::cast(r.monodromy)) : py::object(py::none());
            d["multipliers"] = r.multipliers;
            d["scale"] = r.scale;
            d["history"] = r.history;
            d["patterns"] = r.patterns;
            d["last_pattern"] = r.last_pattern;
            d["last_mu"] = r.last_mu;
            d["marginal_direction"] = r.marginal_direction;
            d["marginal_mu"] = r.marginal_mu;
            d["residual"] = r.residual;
            d["orbit_error_estimate"] = r.orbit_error_estimate;
            d["n_period_runs"] = r.n_runs;
            d["n_iterations"] = r.n_iterations;
            d["n_relax_rounds"] = r.n_relax_rounds;
            d["n_engine_steps"] = r.n_steps;
            d["python_callbacks"] = n_python_calls;
            d["nogil"] = nogil;
            d["period"] = r.has_period
                ? py::object(period_dict(r.period, n_python_calls, nogil))
                : py::object(py::none());
            return d;
        },
        py::arg("adapter"), py::arg("switch_fn"), py::arg("x0"),
        py::arg("period"), py::arg("observables") = py::list(),
        py::arg("pairs") = std::vector<std::pair<int, int>>{},
        py::arg("rtol") = Real{1e-9}, py::arg("max_period_runs") = 60,
        py::arg("n_relax") = 5, py::arg("run_rtol") = Real{1e-11},
        py::arg("run_atol") = Real{1e-14},
        py::arg("mode_guard_h_e") = Real{0},
        "Newton shooting for the periodic orbit on the exact stepper, the "
        "GIL released for the whole solve: the loop of "
        "pulsim.periodic_steady_state (scaled-norm residual, Armijo "
        "backtracking, relaxation rounds, stop on the Newton step, period-"
        "run budget). status is 'converged' or the typed outcome "
        "('budget', 'no_progress', 'singular', 'marginal', 'grazing', "
        "'run_failed', 'irreproducible'); 'period' is the verification "
        "period (dsed_period_map's dict, with the observables). DC "
        "sources only.");

    // add-design-optimization-workflow DECISIONS 28: one device's junction
    // temperature on a fixed periodic orbit (pulsim/dsed/period_thermal.hpp).
    m.def("_junction_fixed_point",
        [](std::vector<std::array<Real, 2>> on_pts,
           std::vector<std::array<Real, 2>> off_pts,
           const pulsim::thermal::LossTable3D* on,
           const pulsim::thermal::LossTable3D* off, Real period, Real p_fixed,
           Real r_th, Real t_amb, const pulsim::thermal::ConductionTable* cond,
           Real q1, Real q2, Real i_max) {
            namespace pt = pulsim::dsed::period_thermal;
            pt::DevicePower d;
            d.on_pts = std::move(on_pts);
            d.off_pts = std::move(off_pts);
            d.on = on;
            d.off = off;
            d.cond = cond;
            d.q1 = q1;
            d.q2 = q2;
            d.i_max = i_max;
            d.p_fixed = p_fixed;
            d.period = period;
            pt::Junction j;
            {
                py::gil_scoped_release nogil;
                j = pt::solve_junction(d, r_th, t_amb);
            }
            const auto parts = d.at(j.tj);
            py::dict out;
            out["runaway"] = j.runaway;
            out["tj"] = j.tj;
            out["p"] = j.p;
            out["residual"] = j.residual;
            out["slope"] = j.slope;
            out["evaluations"] = j.evaluations;
            out["e_on"] = parts[1];
            out["e_off"] = parts[2];
            out["p_cond_table"] = parts[3];
            return out;
        },
        py::arg("on_pts"), py::arg("off_pts"), py::arg("on").none(true),
        py::arg("off").none(true), py::arg("period"), py::arg("p_fixed"),
        py::arg("r_th"), py::arg("t_amb"), py::arg("cond").none(true),
        py::arg("q1") = 0.0, py::arg("q2") = 0.0, py::arg("i_max") = 0.0,
        "Smallest T_j >= t_amb with T_j = t_amb + r_th P(T_j), P the "
        "device's loss on a fixed orbit: p_fixed + (sum E_on + sum E_off) / "
        "period over the commutation points (|v|, |i|), plus a conduction "
        "table's (a q1 + b q2) / period. Returns runaway, tj, p, residual, "
        "slope (r_th dP/dT), evaluations, e_on, e_off, p_cond_table.");

    // add-design-optimization-workflow DECISIONS 26: the kernel batch
    // evaluator (pulsim/dsed/design_batch.hpp). The plan is parsed and the
    // schedules adapted with the GIL; the whole table then runs on native
    // threads with the GIL released; outputs are arrays.
    m.def("_design_batch",
        [](const pulsim::ir::Circuit& base, const py::dict& pd,
           const Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic,
                               Eigen::RowMajor>& table,
           const std::vector<Real>& periods, const py::list& schedules,
           int n_threads) {
            namespace dz = pulsim::dsed::design;
            dz::Plan plan;
            plan.names = pd["names"].cast<std::vector<std::string>>();
            plan.is_param = pd["is_param"].cast<std::vector<bool>>();
            plan.pairs = pd["pairs"].cast<Pairs>();
            plan.obs = parse_observables(pd["obs"].cast<py::list>(), plan.pairs);
            plan.slots = pd["slots"].cast<std::vector<std::string>>();
            for (const auto& it : pd["switching"].cast<py::list>()) {
                const auto t = it.cast<py::tuple>();
                dz::SwitchingEntry e;
                e.bit = t[0].cast<int>();
                e.obs_v = t[1].cast<int>();
                e.obs_i = t[2].cast<int>();
                if (!t[3].is_none()) e.on = std::make_shared<const pulsim::thermal::LossTable3D>(t[3].cast<const pulsim::thermal::LossTable3D&>());
                if (!t[4].is_none()) e.off = std::make_shared<const pulsim::thermal::LossTable3D>(t[4].cast<const pulsim::thermal::LossTable3D&>());
                e.tj = t[5].cast<Real>();
                e.edge = {t[6].cast<bool>(), t[7].cast<Real>(), t[8].cast<Real>()};
                plan.switching.push_back(std::move(e));
            }
            for (const auto& it : pd["code"].cast<py::list>()) {
                const auto t = it.cast<py::tuple>();
                plan.code.push_back({static_cast<dz::Op>(t[0].cast<int>()),
                                     t[1].cast<std::int32_t>(),
                                     t[2].cast<std::int32_t>(), t[3].cast<Real>()});
            }
            plan.n_regs = pd["n_regs"].cast<int>();
            plan.kpi_regs = pd["kpi_regs"].cast<std::vector<int>>();
            plan.loss_regs = pd["loss_regs"].cast<std::vector<int>>();
            plan.viol_regs = pd["viol_regs"].cast<std::vector<int>>();
            plan.energy_static = pd["energy_static"].cast<bool>();
            plan.p_in_reg = pd["p_in_reg"].cast<int>();
            plan.res_reg = pd["res_reg"].cast<int>();
            plan.p_table_reg = pd["p_table_reg"].cast<int>();
            plan.energy_tol = pd["energy_tol"].cast<Real>();
            plan.node_obs0 = pd["node_obs0"].cast<int>();
            plan.n_nodes = pd["n_nodes"].cast<int>();
            plan.n_full = pd["n_full"].cast<int>();
            plan.orbit.rtol = pd["rtol"].cast<Real>();
            plan.orbit.max_period_runs = pd["max_period_runs"].cast<int>();
            plan.orbit.n_relax = pd["n_relax"].cast<int>();
            plan.mode_guard = pd["mode_guard"].cast<bool>();
            plan.h_e_rel = pd["h_e_rel"].cast<Real>();
            const auto n = static_cast<std::size_t>(table.rows());
            if (static_cast<std::size_t>(table.cols()) != plan.names.size()
                || periods.size() != n || py::len(schedules) != n) {
                throw py::value_error("_design_batch: the table, periods and "
                                      "schedules do not match");
            }
            for (const auto& c : plan.code) {
                if (static_cast<int>(c.op) < 0 || static_cast<int>(c.op) > 24) {
                    throw py::value_error("_design_batch: unknown operation");
                }
            }
            std::vector<std::unique_ptr<PySwitchFnSSM>> own;
            std::vector<const PySwitchFnSSM*> ss;
            for (std::size_t i = 0; i < n; ++i) {
                own.push_back(std::make_unique<PySwitchFnSSM>(schedules[i]));
                if (!own.back()->is_native_ || !own.back()->native_.next_edge) {
                    throw py::value_error("_design_batch: every schedule must be "
                                          "a native schedule with edges");
                }
                own.back()->set_poll(poll_window(periods[i]));
                ss.push_back(own.back().get());
            }
            std::vector<std::size_t> calls(n, 0);   // Python calls, per design
            for (std::size_t i = 0; i < n; ++i) own[i]->n_python_calls = &calls[i];
            const Real run_rtol = pd["run_rtol"].cast<Real>();
            const Real run_atol = pd["run_atol"].cast<Real>();
            auto rp = [run_rtol, run_atol](auto& ad, const PySwitchFnSSM& sf,
                                           const Vector& x, Real T,
                                           const std::vector<PeriodObservable>& obs,
                                           const Pairs& pairs, Real h_e) {
                try {
                    return period_run(ad, sf, x, T, obs, pairs, true, run_rtol,
                                      run_atol, h_e);
                } catch (const pulsim::analysis::Cancelled&) {
                    throw;
                } catch (const std::runtime_error& e) {
                    throw pulsim::dsed::PeriodRunFailed(e.what());
                }
            };
            std::vector<dz::DesignOut> out;
            bool nogil = false;
            {
                py::gil_scoped_release release;
                nogil = PyGILState_Check() == 0;
                out = dz::evaluate_batch<PySwitchFnSSM>(
                    base, plan, table.data(), n, periods,
                    std::span<const PySwitchFnSSM* const>(ss), rp, n_threads);
            }
            auto mat = [&](auto get, std::size_t w) {
                py::array_t<double> a(std::vector<py::ssize_t>{
                    static_cast<py::ssize_t>(n), static_cast<py::ssize_t>(w)});
                auto m = a.mutable_unchecked<2>();
                for (std::size_t i = 0; i < n; ++i) {
                    const auto& v = get(out[i]);
                    for (std::size_t j = 0; j < w; ++j) {
                        m(static_cast<py::ssize_t>(i), static_cast<py::ssize_t>(j)) =
                            j < v.size() ? v[j] : std::numeric_limits<double>::quiet_NaN();
                    }
                }
                return a;
            };
            py::dict d;
            std::vector<int> status(n), runs(n), iters(n);
            std::vector<std::size_t> steps(n);
            std::vector<double> res(n), ptab(n), wall(n);
            std::vector<bool> applied(n);
            py::list reasons;
            std::uint64_t analyses = 0;
            for (std::size_t i = 0; i < n; ++i) {
                status[i] = static_cast<int>(out[i].status);
                runs[i] = out[i].n_runs;
                iters[i] = out[i].n_iterations;
                steps[i] = out[i].n_steps;
                res[i] = out[i].energy_res;
                ptab[i] = out[i].p_table;
                wall[i] = out[i].wall_s;
                applied[i] = out[i].energy_applied;
                reasons.append(out[i].reason);
                analyses += out[i].symbolic_analyses;
            }
            d["status"] = status;
            d["reasons"] = reasons;
            d["kpis"] = mat([](const dz::DesignOut& o) -> const auto& { return o.kpis; },
                            plan.kpi_regs.size());
            d["losses"] = mat([](const dz::DesignOut& o) -> const auto& { return o.losses; },
                              plan.loss_regs.size());
            d["viol"] = mat([](const dz::DesignOut& o) -> const auto& { return o.viol; },
                            plan.viol_regs.size());
            d["energy_applied"] = applied;
            d["energy_res"] = res;
            d["p_table"] = ptab;
            d["n_runs"] = runs;
            d["n_steps"] = steps;
            d["n_iterations"] = iters;
            d["wall_s"] = wall;
            d["symbolic_analyses"] = analyses;
            std::size_t n_calls = 0;
            for (auto c : calls) n_calls += c;
            d["python_calls"] = n_calls;
            d["nogil"] = nogil;
            // the periodicity check's samples: every design that reached
            // it, grouped by event names (first in input order), and the
            // designs with an event at the section (always checked)
            std::map<std::vector<std::string>, std::size_t> first;
            py::list groups, at_section;
            for (std::size_t i = 0; i < n; ++i) {
                if (out[i].x0.size() == 0) continue;
                std::vector<std::string> names;
                bool sec = false;
                for (const auto& [nm, f] : out[i].pattern) {
                    names.push_back(nm);
                    sec = sec || std::min(f, 1.0 - f) <= 1e-9;
                }
                if (sec) {
                    at_section.append(py::make_tuple(i, out[i].x0));
                } else if (first.emplace(names, i).second) {
                    groups.append(py::make_tuple(py::cast(names), i, out[i].x0));
                }
            }
            d["groups"] = groups;
            d["at_section"] = at_section;
            return d;
        },
        py::arg("base"), py::arg("plan"), py::arg("table"), py::arg("periods"),
        py::arg("schedules"), py::arg("n_threads") = 1,
        "pulsim.design's kernel batch: every design of `table` evaluated "
        "on the exact steady-state route with the GIL released "
        "(pulsim/dsed/design_batch.hpp). Internal.");
}

}  // namespace pulsim_dsed_bindings
