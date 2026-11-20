#pragma once

// =============================================================================
// Pulsim — add-in-kernel-event-losses: the run-time loss tracker
// =============================================================================
//
// One object per run, built from the pool's loss ports. An engine calls
// three things, and nothing else:
//
//   inject(b)                    every solve of a step adds the heat
//                                 power owed for that step to its RHS
//                                 (current into the deposit node);
//   accept(t0, t1, x0, x1, mask) once per ACCEPTED step: reads the
//                                 commutations by the reading rule
//                                 (thermal::commutation_point), the
//                                 conduction energy, and queues the heat;
//   advance(h_next)              turns the queue into the power of the
//                                 next step (energy / h_next).
//
// A step that is rolled back or re-tried never reaches accept(), and
// inject() reads only what the PREVIOUS accepted step queued, so a
// retry injects the same power again: nothing is counted twice and
// nothing is lost. The heat of a step is a rectangular power over the
// step that follows it (design "Deposition"); the ledger moves it from
// pending to injected when that step is accepted. On trapezoidal steps
// the capacitor's history carries half of each step's injection into
// the next one, so the network receives E over two steps (h/2 + h/2);
// the sum over a run is exact, a restart (BE/SDIRK) step is within one
// step's heat.
//
// Variable-step engines (trbdf2) call accept_step() in place of accept():
// the step's quadrature uses the integrator's own stage values and
// weights, the heat owed is consumed in proportion to the committed
// length (a landing shorter than the attempt keeps the rest pending), and
// event() books a commutation from the engine's left / right limits; the
// accept_step() whose mask is the event's right side then starts its
// quadrature from the right limit, so the flip is not read twice.
// They call advance(h, history): the power changes at the step start, and
// every capacitor that carries the change gets the right limit of its
// current in the history (DECISIONS 17) — the stages and the error
// estimate read that slot, and the left limit made every step boundary a
// jump the estimate could not shrink (DECISIONS 16).
//
// impulses(result.reinit_records): the energy of an impulsive
// re-initialisation (a capacitor loop closed, a cut-set opened, by an
// ideal element) is dissipated by the element the record is booked to
// (`cause`): it joins that device's heat as E_impulse, in the step that
// carried it. The engine calls it after its reinit.record().
//
// Engines without these calls refuse a circuit with loss ports by name
// (`require_supported`, feature "device_losses").

#include "pulsim/numeric/dense.hpp"
#include "pulsim/numeric/types.hpp"
#include "pulsim/pwl/device_pool.hpp"
#include "pulsim/pwl/history_state.hpp"
#include "pulsim/pwl/pwl_branch_law.hpp"
#include "pulsim/stamping/branch_coord.hpp"
#include "pulsim/thermal/device_losses.hpp"
#include "pulsim/topology/graph.hpp"
#include "pulsim/topology/switch_state.hpp"

#include <algorithm>
#include <array>
#include <format>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace pulsim::solver {

/// Release group B: the lines a fixed step taken by the restart rule is
/// integrated on, as the stored result keeps them (the step's start right
/// limit `x_start`, and per event inside the step its left limit and the
/// start of the restart sub-step after it). A restart (sub)step's start is
/// the value that makes the line's trapezoid the integrator's own
/// quadrature, 2·w1·X1 + (2·w2 − 1)·X2 (SDIRK(2,2): w = (1 − γ, γ)): a
/// quantity linear in the state integrates exactly as the step did, a
/// product to second order. A stage 2 in another mode than stage 1 (each
/// device changing once over the step): X1 held over w1·h, then X2 (an
/// extra row at a + w1·h), the same
/// quadrature with each point a solved state in its own mode. Empty: the
/// step's ends only.
struct StepLines {
    static constexpr Size kMax = 9;
    bool start = false;
    Vector x_start;
    topology::SwitchStateMask m_start;
    Size n = 0;
    std::array<Real, kMax> t{};
    std::array<Vector, kMax> left, right;
    std::array<topology::SwitchStateMask, kMax> m_left, m_right;
    std::array<bool, kMax> right_ok{};
    // the run's bookkeeping (run_transient): stage 1 of the last restart
    // (sub)step, whether its stage 2 was the backward-Euler re-take, and
    // whether the step is representable (false: the dt-retry ladder)
    Vector x1;
    topology::SwitchStateMask m1, m_a;   // stage 1's mode, the (sub)step start's
    bool be_kept = false;
    bool ok = true;
    // the step (its end time) the lines belong to: run_transient resets
    // them at a step's first restart, not every step
    Real owner = -std::numeric_limits<Real>::infinity();
    // F-4-4, filled only for an in-loop recorder (a step sink): per history
    // entry, a capacitor's companion current at the start right limit and
    // at each event's two limits — at a restart start the same combination
    // of the stage currents as the state, 2·w1·i(X1) + (1 − 2·w1)·i(X2), so
    // a reading is KCL-consistent there and its trapezoid is the step's
    // charge; `v0` the capacitor voltages a restart (sub)step starts from
    std::vector<Real> c_start, v0;
    std::array<std::vector<Real>, kMax> c_left, c_right;
    // the last restart (sub)step's end currents when its stage 1 was re-taken
    // from x0+ for the lines (an ideal element's jump): not the run's i_prev
    std::vector<Real> c_end;
    bool end_own = false;
    // F-SB-1: the RIGHT LIMIT of an event as a state value, x(t_e+) (the
    // result's right_limit_* / inner_events), apart from the quadrature
    // starts above (x_start, right[]: the lines' business, never a limit).
    // `rl_*`: run_restart's scratch; `rl_start*`: at the step start, set
    // only where something happens; `rl_right*`: per inner event; `held`:
    // the mixed step's held-pieces row (quadrature, not an event).
    Vector rl_x, rl_start;
    topology::SwitchStateMask rl_m, rl_m_start;
    // rl_left / rl_start_left: a continuity point (a located V_f = 0
    // crossing alone): the row is the left value
    bool rl_ok = false, rl_start_ok = false, rl_left = false, rl_start_left = false;
    std::array<Vector, kMax> rl_right;
    std::array<topology::SwitchStateMask, kMax> rl_m_right;
    std::array<bool, kMax> rl_right_ok{}, held{};
    void clear() noexcept {
        start = false;
        n = 0;
        end_own = false;
        rl_start_ok = false;
    }
    [[nodiscard]] bool empty() const noexcept { return !start && n == 0; }
};

class LossTracker {
public:
    LossTracker(const topology::Graph& graph, const pwl::DevicePool& pool,
                std::string_view engine) {
        const auto& specs = pool.device_losses();
        devs_.reserve(specs.size());
        for (const auto& s : specs) {
            Dev d;
            d.spec = s;
            d.branch = find_branch_(graph, s.device);
            if (d.branch < 0) {
                throw std::invalid_argument(std::format(
                    "device losses: '{}' is not a device of this circuit",
                    s.device));
            }
            using K = pwl::DevicePool::StoredKind;
            const auto k = pool.kind_of(d.branch);
            if (k != K::Switch && k != K::Diode) {
                throw std::invalid_argument(std::format(
                    "device losses: '{}' is a {}, not a PWL switch or diode",
                    s.device, pwl::stored_kind_name(k)));
            }
            d.law = pwl::PwlBranchLaw::of(graph, pool, d.branch);
            const auto& node_name = s.deposit_node();
            if (!node_name.empty()) {
                d.node = find_node_(graph, node_name);
                if (d.node < 0) {
                    throw std::invalid_argument(std::format(
                        "device losses: '{}' names thermal node '{}', which "
                        "is not a node of this circuit", s.device, node_name));
                }
                require_thermal_node_(graph, pool, s.device, node_name, d.node);
            }
            d.rec.device = s.device;
            d.rec.node = node_name;
            devs_.push_back(std::move(d));
        }
        (void)engine;
        bool deposits = false;
        for (const auto& d : devs_) deposits = deposits || d.node >= 0;
        if (deposits) classify_nodes_(graph, pool);
    }

    [[nodiscard]] bool active() const noexcept { return !devs_.empty(); }

    /// The state the run starts from and the mask it was stamped under.
    void begin(const Vector& x0, const topology::SwitchStateMask& m0) {
        for (auto& d : devs_) {
            d.on = m0.get(static_cast<Size>(d.law.bit));
            d.v = v_of_(d, x0);
            d.i = i_of_(d, x0, d.on);
            if (d.node >= 0 && d.spec.closed_loop()) {
                d.x_tj = stamping::read_node_voltage(x0, d.node);
            }
        }
    }

    /// Add the heat power owed for the step being solved: a current INTO
    /// the deposit node (the RHS sign of a current source leaving at it,
    /// sources/current_waveform_b_extra.hpp).
    void inject(Vector& b) const {
        for (const auto& n : now_) b[n.node] -= n.e / h_now_;
        for (const auto& q : pred_) b[q.node] -= q.e;   // W (advance(h, hist) only)
    }

    /// One accepted step from (t0, x0) to (t1, x1) under `mask`. `lines`
    /// (a restart-rule step, StepLines): the conduction energy is the
    /// integral of its lines; the commutation is read on the step's ends.
    void accept(Real t0, Real t1, const Vector& x1,
                const topology::SwitchStateMask& mask,
                const StepLines* lines = nullptr) {
        if (lines != nullptr && !lines->empty()) {
            accept_lines_(t0, t1, x1, mask, *lines);
            return;
        }
        const Real h = t1 - t0;
        // the power of this step has reached the network
        for (const auto& n : now_) injected_.add(n.e);
        now_.clear();
        for (auto& d : devs_) {
            const bool on1 = mask.get(static_cast<Size>(d.law.bit));
            const Real v1 = v_of_(d, x1);
            const Real i1 = i_of_(d, x1, on1);
            const Real tj0 = tj_of_(d);   // T_j(x⁻): before this step's state
            // Electrical dissipation of the step, as device_loss_summary
            // reads it (losses._branch_energy): one stamp over the step is
            // the exact integral of the product of its two lines; a step
            // that flips the stamp starts from a sample solved under the
            // other one, has no line, and keeps the trapezoid of the end
            // powers (the line product read a freewheel diode negative).
            const Real e_el = on1 == d.on
                ? h * (Real{2} * d.v * d.i + d.v * i1 + v1 * d.i +
                       Real{2} * v1 * i1) / Real{6}
                : h * (d.v * d.i + v1 * i1) / Real{2};
            d.e_el.add(e_el);
            Real heat = 0;
            if (d.spec.v_on) {
                // V_on(|i|, T_j)|i| on the on-state ends of the step,
                // trapezoidal; an OFF end carries no on-state current.
                const auto& c = *d.spec.v_on;
                const Real a = d.on ? c(std::abs(d.i), tj0) * std::abs(d.i) : 0;
                const Real tj1 = d.node >= 0 && d.spec.closed_loop()
                    ? stamping::read_node_voltage(x1, d.node) : tj0;
                const Real z = on1 ? c(std::abs(i1), tj1) * std::abs(i1) : 0;
                const Real e_t = h * (a + z) / Real{2};
                d.e_tab_last_ = e_t;
                d.e_tab.add(e_t);
                if (d.spec.conduction == thermal::ConductionHeat::Table) heat = e_t;
            }
            if (d.spec.conduction == thermal::ConductionHeat::Electrical) heat = e_el;
            d.last = {h, e_el, d.spec.v_on ? d.e_tab_last_ : Real{0}, false};
            if (on1 != d.on) {
                const auto pt = thermal::commutation_point(on1, d.v, d.i, v1, i1);
                const auto& tab = on1 ? d.spec.e_on : d.spec.e_off;
                const Real e = tab ? (*tab)(pt.v, pt.i, tj0) : Real{0};
                d.rec.events.push_back({t0, on1, pt.v, pt.i, tj0, e});
                (on1 ? d.e_on : d.e_off).add(e);
                ++(on1 ? d.rec.n_on : d.rec.n_off);
                heat += e;
            }
            if (d.node >= 0) add_(queue_, d.node, heat);
            d.on = on1;
            d.v = v1;
            d.i = i1;
            d.x_tj = d.node >= 0 && d.spec.closed_loop()
                ? stamping::read_node_voltage(x1, d.node) : d.x_tj;
        }
    }

private:
    /// accept() over a step's lines: per line from (a) to (b), the line
    /// product under one stamp, the trapezoid of the end powers across a
    /// stamp change (the rule of accept()); V_on tables trapezoidal.
    void accept_lines_(Real t0, Real t1, const Vector& x1,
                       const topology::SwitchStateMask& mask, const StepLines& L) {
        const Real h = t1 - t0;
        for (const auto& n : now_) injected_.add(n.e);
        now_.clear();
        for (auto& d : devs_) {
            const bool on1 = mask.get(static_cast<Size>(d.law.bit));
            const Real v1 = v_of_(d, x1);
            const Real i1 = i_of_(d, x1, on1);
            const Real tj0 = tj_of_(d);
            const bool closed = d.node >= 0 && d.spec.closed_loop();
            auto tj_at = [&](const Vector& x) {
                return closed ? stamping::read_node_voltage(x, d.node) : tj0;
            };
            Real e_el = 0, e_t = 0;
            auto line = [&](Real ha, Real va, Real ia, bool oa, Real tja,
                            Real vb, Real ib, bool ob, Real tjb) {
                e_el += oa == ob
                    ? ha * (Real{2} * va * ia + va * ib + vb * ia + Real{2} * vb * ib) / Real{6}
                    : ha * (va * ia + vb * ib) / Real{2};
                if (d.spec.v_on) {
                    const auto& c = *d.spec.v_on;
                    const Real a = oa ? c(std::abs(ia), tja) * std::abs(ia) : 0;
                    const Real z = ob ? c(std::abs(ib), tjb) * std::abs(ib) : 0;
                    e_t += ha * (a + z) / Real{2};
                }
            };
            // the first line's start: the step's right limit, else the last end
            Real ta = t0, va = d.v, ia = d.i, tja = tj0;
            bool oa = d.on;
            if (L.start) {
                oa = L.m_start.get(static_cast<Size>(d.law.bit));
                va = v_of_(d, L.x_start);
                ia = i_of_(d, L.x_start, oa);
                tja = tj_at(L.x_start);
            }
            for (Size j = 0; j < L.n; ++j) {
                const bool ol = L.m_left[j].get(static_cast<Size>(d.law.bit));
                const Real vl = v_of_(d, L.left[j]);
                const Real il = i_of_(d, L.left[j], ol);
                line(L.t[j] - ta, va, ia, oa, tja, vl, il, ol, tj_at(L.left[j]));
                ta = L.t[j];
                const Vector& xr = L.right_ok[j] ? L.right[j] : L.left[j];
                oa = (L.right_ok[j] ? L.m_right[j] : L.m_left[j])
                         .get(static_cast<Size>(d.law.bit));
                va = v_of_(d, xr);
                ia = i_of_(d, xr, oa);
                tja = tj_at(xr);
            }
            const Real tj1 = closed ? stamping::read_node_voltage(x1, d.node) : tj0;
            line(t1 - ta, va, ia, oa, tja, v1, i1, on1, tj1);
            d.e_el.add(e_el);
            Real heat = 0;
            if (d.spec.v_on) {
                d.e_tab_last_ = e_t;
                d.e_tab.add(e_t);
                if (d.spec.conduction == thermal::ConductionHeat::Table) heat = e_t;
            }
            if (d.spec.conduction == thermal::ConductionHeat::Electrical) heat = e_el;
            d.last = {h, e_el, d.spec.v_on ? d.e_tab_last_ : Real{0}, false};
            if (on1 != d.on) {
                const auto pt = thermal::commutation_point(on1, d.v, d.i, v1, i1);
                const auto& tab = on1 ? d.spec.e_on : d.spec.e_off;
                const Real e = tab ? (*tab)(pt.v, pt.i, tj0) : Real{0};
                d.rec.events.push_back({t0, on1, pt.v, pt.i, tj0, e});
                (on1 ? d.e_on : d.e_off).add(e);
                ++(on1 ? d.rec.n_on : d.rec.n_off);
                heat += e;
            }
            if (d.node >= 0) add_(queue_, d.node, heat);
            d.on = on1;
            d.v = v1;
            d.i = i1;
            d.x_tj = closed ? stamping::read_node_voltage(x1, d.node) : d.x_tj;
        }
    }

public:
    /// Variable-step accept of [t0, t1] ending in x1 under `mask`. The
    /// quadrature of the conduction power is w0·p(start) + wg·p(xg) +
    /// w1·p(x1) (the integrator's stage weights, Σw = 1; xg may be null
    /// when wg = 0). The start is the last accepted end, or the right
    /// limit of an event() whose mask this step carries.
    /// `e_exact` (one per device, in set_device_losses order): the step's
    /// exact ∫v·i, which replaces the quadrature (the exact engine, whose
    /// impulses are at event instants: they never replace a step's energy).
    void accept_step(Real t0, Real t1, const Vector& x1,
                     const topology::SwitchStateMask& mask, const Vector* xg,
                     Real w0, Real wg, Real w1, const Real* e_exact = nullptr) {
        accept_step_<1>(t0, t1, x1, mask, {xg}, {wg}, w0, w1, e_exact);
    }
    /// The same with N interior points (ESDIRK4(3): the stages Y1..Y4 at
    /// weights b2..b5; w0 = b1 on the start, w1 = b6 on x1 = Y5).
    template <std::size_t N>
    void accept_step(Real t0, Real t1, const Vector& x1,
                     const topology::SwitchStateMask& mask,
                     const std::array<const Vector*, N>& xs,
                     const std::array<Real, N>& ws, Real w0, Real w1) {
        accept_step_<N>(t0, t1, x1, mask, xs, ws, w0, w1);
    }

private:
    template <std::size_t N>
    void accept_step_(Real t0, Real t1, const Vector& x1,
                      const topology::SwitchStateMask& mask,
                      const std::array<const Vector*, N>& xs,
                      const std::array<Real, N>& ws, Real w0, Real w1,
                      const Real* e_exact = nullptr) {
        const Real h = t1 - t0;
        // the history just committed holds this power (DECISIONS 17)
        slots_rewritten();
        // the predicted conduction power reached the network over h; the
        // step's own energy (queued below) owes only the difference
        for (const auto& q : pred_) {
            injected_.add(q.e * h);
            add_(queue_, q.node, -q.e * h);
        }
        // the power E/h_now_ reached the network over h of this step: a
        // committed step shorter than the attempt (a landing) keeps the rest,
        // at the same power until the next advance (no jump for the solves
        // between: the edge's probe)
        const Real frac = h >= h_now_ * (Real{1} - Real{1e-12}) ? Real{1} : h / h_now_;
        for (auto& n : now_) {
            const Real e = frac == Real{1} ? n.e : n.e * frac;
            injected_.add(e);
            n.e -= e;
        }
        if (frac == Real{1}) {
            now_.clear();
        } else if (!shares_.empty()) {
            h_now_ -= h;
        }
        for (std::size_t di = 0; di < devs_.size(); ++di) {
            auto& d = devs_[di];
            const bool on1 = mask.get(static_cast<Size>(d.law.bit));
            if (d.ev_pending && t0 >= d.ev_t) {
                // the first step after an event: under its right side it
                // starts from the right limit; otherwise the event is stale
                if (on1 == d.ev_on) {
                    d.on = d.ev_on;
                    d.v = d.ev_v;
                    d.i = d.ev_i;
                } else {
                    // the engine never committed the event's right side (a
                    // cascade that undid it): take the booking back
                    unbook_(d);
                }
                d.ev_pending = false;
            }
            const Real v1 = v_of_(d, x1);
            const Real i1 = i_of_(d, x1, on1);
            const Real tj0 = tj_of_(d);
            const Real tj1 = d.node >= 0 && d.spec.closed_loop()
                ? stamping::read_node_voltage(x1, d.node) : tj0;
            Real e_el, e_t = 0;
            const bool same = on1 == d.on;
            // interior point k: (v, i, T_j) when it is read
            auto read_k = [&](std::size_t k, Real& vk, Real& ik, Real& tjk) {
                vk = 0, ik = 0, tjk = tj0;
                if (same && xs[k] != nullptr && ws[k] != Real{0}) {
                    vk = v_of_(d, *xs[k]);
                    ik = i_of_(d, *xs[k], on1);
                    if (d.node >= 0 && d.spec.closed_loop())
                        tjk = stamping::read_node_voltage(*xs[k], d.node);
                }
            };
            std::array<Real, N> vs{}, is{}, tjs{};
            for (std::size_t k = 0; k < N; ++k) read_k(k, vs[k], is[k], tjs[k]);
            if (e_exact != nullptr) {
                e_el = e_exact[di];
            } else if (same) {
                if constexpr (N == 1) {   // TR-BDF2 / SDIRK: one expression, as before
                    e_el = h * (w0 * d.v * d.i + ws[0] * vs[0] * is[0] + w1 * v1 * i1);
                } else {
                    Real s = w0 * d.v * d.i;
                    for (std::size_t k = 0; k < N; ++k) s += ws[k] * vs[k] * is[k];
                    e_el = h * (s + w1 * v1 * i1);
                }
            } else {
                e_el = h * (d.v * d.i + v1 * i1) / Real{2};
            }
            d.e_el.add(e_el);
            Real heat = 0;
            if (d.spec.v_on) {
                const auto& c = *d.spec.v_on;
                const Real a = d.on ? c(std::abs(d.i), tj0) * std::abs(d.i) : 0;
                const Real z = on1 ? c(std::abs(i1), tj1) * std::abs(i1) : 0;
                if (same) {
                    auto g_k = [&](std::size_t k) {
                        return on1 && ws[k] != Real{0}
                            ? c(std::abs(is[k]), tjs[k]) * std::abs(is[k]) : Real{0};
                    };
                    if constexpr (N == 1) {
                        const Real g = g_k(0);
                        e_t = h * (w0 * a + ws[0] * g + w1 * z);
                    } else {
                        Real s = w0 * a;
                        for (std::size_t k = 0; k < N; ++k) s += ws[k] * g_k(k);
                        e_t = h * (s + w1 * z);
                    }
                } else {
                    e_t = h * (a + z) / Real{2};
                }
                d.e_tab.add(e_t);
                if (d.spec.conduction == thermal::ConductionHeat::Table) heat = e_t;
            }
            if (d.spec.conduction == thermal::ConductionHeat::Electrical) heat = e_el;
            d.last = {h, e_el, e_t, e_exact != nullptr};
            // a flip no event() reported: the reading rule on the step's ends
            if (!same) heat += book_(d, t0, on1, d.v, d.i, v1, i1, tj0);
            if (d.node >= 0) add_(queue_, d.node, heat);
            d.on = on1;
            d.v = v1;
            d.i = i1;
            d.x_tj = tj1;
        }
    }

public:
    /// A commutation at t between the engine's left limit (x_l under m_l)
    /// and right limit (x_r under m_r): every tracked device whose bit
    /// differs is booked by the reading rule on the two limits.
    void event(Real t, const Vector& x_l, const topology::SwitchStateMask& m_l,
               const Vector& x_r, const topology::SwitchStateMask& m_r) {
        for (auto& d : devs_) {
            const auto bit = static_cast<Size>(d.law.bit);
            const bool on_l = m_l.get(bit), on_r = m_r.get(bit);
            if (on_l == on_r) continue;
            // the same flip already booked at t (a stale left mask after a
            // landing, re-reported by the settle at a coincident gate edge)
            if (d.ev_pending && d.ev_t == t && d.ev_on == on_r) continue;
            const Real v_l = v_of_(d, x_l), i_l = i_of_(d, x_l, on_l);
            const Real v_r = v_of_(d, x_r), i_r = i_of_(d, x_r, on_r);
            const Real e = book_(d, t, on_r, v_l, i_l, v_r, i_r, tj_of_(d));
            if (d.node >= 0) add_(queue_, d.node, e);
            d.ev_e = e;
            d.ev_pending = true;
            d.ev_t = t;
            d.ev_on = on_r;
            d.ev_v = v_r;
            d.ev_i = i_r;
        }
    }

    /// Book the energy of the re-initialisation records appended since the
    /// last call to the device each one names (`cause`), if it is tracked.
    template <typename Records>
    void impulses(const Records& recs) {
        for (; n_reinit_ < recs.size(); ++n_reinit_) {
            const auto& r = recs[n_reinit_];
            if (!(r.dW > 0)) continue;
            for (auto& d : devs_) {
                if (d.spec.device != r.cause) continue;
                d.e_imp.add(r.dW);
                Real heat = r.dW;
                if (!d.last.replaced) {
                    // The step that carried the impulse conducts its end
                    // power only (device_loss_summary's pulse step): its
                    // v·i quadrature read the impulse's mean current
                    // against a voltage it collapsed, which is not a
                    // conduction loss (the capacitor loop read -38.6 µJ).
                    const Real el = d.last.h * d.v * d.i;
                    Real tab = 0;
                    if (d.spec.v_on && d.on) {
                        tab = d.last.h * (*d.spec.v_on)(std::abs(d.i), tj_of_(d)) * std::abs(d.i);
                    }
                    d.e_el.add(el - d.last.el);
                    if (d.spec.v_on) d.e_tab.add(tab - d.last.tab);
                    if (d.spec.conduction == thermal::ConductionHeat::Electrical) heat += el - d.last.el;
                    if (d.spec.conduction == thermal::ConductionHeat::Table) heat += tab - d.last.tab;
                    d.last.replaced = true;
                }
                if (d.node >= 0) add_(queue_, d.node, heat);
                break;
            }
        }
    }

    /// The heat queued since the last call, per deposit node, counted as
    /// injected now: an engine that delivers it itself (the exact engine: the
    /// conduction heat in closed form along the step, an event's energy as an
    /// impulse at its instant) takes it in place of advance() / inject().
    [[nodiscard]] std::vector<std::pair<Index, Real>> take_heat() {
        std::vector<std::pair<Index, Real>> out;
        // A tracker restored from another engine's snapshot (field 17) may
        // still owe the rest of a step's energy (now_) and hold a variable-step
        // engine's predicted conduction power (pred_). Neither is delivered
        // here by inject(): the owed energy goes out now, once, and the
        // prediction is dropped (accept_step would otherwise book it each step
        // and take it back out of the queue: heat lost). Empty on a run that
        // started on this engine.
        pred_.clear();
        for (const auto& n : now_) add_(queue_, n.node, n.e);   // one entry per node
        now_.clear();
        for (const auto& q : queue_) {
            if (q.e == Real{0}) continue;
            injected_.add(q.e);
            out.emplace_back(q.node, q.e);
        }
        queue_.clear();
        return out;
    }

    /// The queued energy becomes the power of the next step, of length
    /// h_next. Called once per outer step; a second call before accept()
    /// (no step was accepted) keeps the power already owed.
    void advance(Real h_next) {
        h_now_ = h_next;
        // a resumed trbdf2 prediction is never booked by pwl's accept();
        // advance(h, hist) rebuilds it for the variable-step engines
        pred_.clear();
        for (const auto& q : queue_) {
            if (q.e == Real{0}) continue;
            add_(now_, q.node, q.e);
        }
        queue_.clear();
    }

    /// advance() for the variable-step integrators (trbdf2's TR-BDF2,
    /// SDIRK restart and ESDIRK steps). The conduction power at the step
    /// start is delivered over the step, so a constant loss has no lag; and
    /// with `right_limit` (a step whose stages and estimate read the
    /// capacitors' stored current; NOT a restart, whose backward-Euler stages
    /// read none) the power's change at the step START, from the one the
    /// history was committed under (left_), goes into the capacitors' slots:
    /// each gets its share, the capacitors' response to a current step with
    /// every state held (DECISIONS 17). `hist` holds the last commit plus
    /// what an earlier attempt of this step added (a restart's snapshot /
    /// restore keeps it).
    void advance(Real h_next, pwl::HistoryState& hist, bool right_limit) {
        advance(h_next);
        if (!bound_) bind_(hist);
        if (shares_.empty()) return;   // no network it can correct: the old rule
        // the energy queue then owes only the step's departure from it
        pred_.clear();
        for (const auto& d : devs_) {
            // a network bind_ could not correct keeps the old rule
            if (d.node < 0 || !corrected_(d.node)) continue;
            const bool on = d.ev_pending ? d.ev_on : d.on;
            const Real v = d.ev_pending ? d.ev_v : d.v;
            const Real i = d.ev_pending ? d.ev_i : d.i;
            Real pw = 0;
            if (d.spec.conduction == thermal::ConductionHeat::Electrical) {
                pw = v * i;
            } else if (d.spec.v_on && on) {
                pw = (*d.spec.v_on)(std::abs(i), tj_of_(d)) * std::abs(i);
            }
            if (pw != Real{0}) add_(pred_, d.node, pw);
        }
        if (!right_limit || !left_known_) return;
        for (auto& s : shares_) {
            const Real dj = power_(pred_, s.node) + power_(now_, s.node) / h_now_
                - power_(left_, s.node);
            for (auto& m : s.members) {
                const Real d = m.share * dj;
                hist.add_to_i_prev(m.entry, d - m.applied);
                m.applied = d;
            }
        }
    }

    /// The engine replaced the capacitors' stored currents without a commit
    /// (a slots-only refresh from a probe solved under the power owed now).
    void slots_rewritten() {
        left_.clear();
        left_known_ = true;
        if (bound_ && shares_.empty()) return;   // nothing to correct
        for (const auto& n : now_) add_(left_, n.node, n.e / h_now_);
        for (const auto& q : pred_) add_(left_, q.node, q.e);
        for (auto& s : shares_) {
            for (auto& m : s.members) m.applied = 0;
        }
    }

    /// Take this attempt's share back out of the history (the run ends on
    /// an attempt that was not committed, a final sliver: the snapshot must
    /// hold the committed slots, which a resume corrects again).
    void retract(pwl::HistoryState& hist) {
        for (auto& s : shares_) {
            for (auto& m : s.members) {
                hist.add_to_i_prev(m.entry, -m.applied);
                m.applied = 0;
            }
        }
    }

private:
    // ---- DECISIONS 17: the capacitors' share of a step of heat power ----
    [[nodiscard]] bool corrected_(Index node) const noexcept {
        for (const auto& s : shares_) {
            if (s.node == node) return true;
        }
        return false;
    }
    struct Member {
        Size entry;          // index in HistoryState::entries()
        Real share;          // its current per ampere into the deposit node
        Real applied = 0;    // what advance() added to its i_prev since the commit
    };
    struct Share {
        Index node;
        std::vector<Member> members;
    };
    template <typename Heats>
    static Real power_(const Heats& v, Index node) {
        for (const auto& q : v) {
            if (q.node == node) return q.e;
        }
        return 0;
    }
    /// A node is FIXED when an independent voltage source ties it to ground
    /// or to a fixed node (its voltage does not answer a current step); it is
    /// PLAIN when every element on it is a resistor, capacitor, inductor, PWL
    /// switch / diode, current source or such a source.
    void classify_nodes_(const topology::Graph& g, const pwl::DevicePool& pool) {
        using K = pwl::DevicePool::StoredKind;
        const auto nn = static_cast<Size>(g.num_nodes());
        fixed_.assign(nn, 0);
        plain_.assign(nn, 1);
        auto vsrc = [](K k) {
            return k == K::VoltageSource || k == K::PWMVoltageSource ||
                   k == K::SineVoltageSource || k == K::PulseVoltageSource;
        };
        auto is_fixed = [&](Index n) { return n < 0 || fixed_[static_cast<Size>(n)] != 0; };
        for (bool grew = true; grew;) {
            grew = false;
            for (Index b = 0; b < g.num_branches(); ++b) {
                if (!pool.is_registered(b) || !vsrc(pool.kind_of(b))) continue;
                const auto& br = g.branch(b);
                for (const auto [n, o] : {std::pair{br.from, br.to}, std::pair{br.to, br.from}}) {
                    if (n >= 0 && !is_fixed(n) && is_fixed(o)) {
                        fixed_[static_cast<Size>(n)] = 1;
                        grew = true;
                    }
                }
            }
        }
        for (Index n = 0; n < g.num_nodes(); ++n) {
            for (const Index b : g.branches_of(n)) {
                // shortcut: a branch outside the pool counts as plain; mark it
                // not plain if a controlled source ever sits on a thermal node
                if (!pool.is_registered(b)) continue;
                const auto k = pool.kind_of(b);
                const bool ok = k == K::Resistor || k == K::Capacitor ||
                    k == K::Inductor || k == K::Switch || k == K::Diode ||
                    k == K::CurrentSource || k == K::SineCurrentSource ||
                    k == K::PulseCurrentSource || k == K::PwlCurrentSource ||
                    (vsrc(k) && is_fixed(n));
                if (!ok) plain_[static_cast<Size>(n)] = 0;
            }
        }
    }
    /// For each deposit node d: the capacitors reached from d through
    /// capacitors (fixed nodes end the walk) and Kc·y = e_d, Kc the
    /// capacitance matrix of those nodes; capacitor k's share is
    /// C_k·(y_from − y_to). Skipped (no correction, the old left limit) when
    /// a node of the walk is not plain, the walk never reaches a fixed node
    /// (Kc singular) or it spans more than 256 nodes.
    void bind_(const pwl::HistoryState& hist) {
        bound_ = true;
        if (fixed_.empty()) return;
        using K = pwl::DevicePool::StoredKind;
        const auto& es = hist.entries();
        auto fixed = [&](Index n) { return n < 0 || fixed_[static_cast<Size>(n)] != 0; };
        std::vector<Index> done;
        for (const auto& dv : devs_) {
            if (dv.node < 0 || std::find(done.begin(), done.end(), dv.node) != done.end()) continue;
            done.push_back(dv.node);
            if (fixed(dv.node)) continue;
            std::vector<Index> nodes{dv.node};
            bool ok = true;
            for (std::size_t q = 0; q < nodes.size() && ok; ++q) {
                ok = plain_[static_cast<Size>(nodes[q])] != 0 && nodes.size() <= 256;
                for (const auto& e : es) {
                    if (e.kind != K::Capacitor) continue;
                    const Index o = e.from == nodes[q] ? e.to : e.to == nodes[q] ? e.from : -2;
                    if (o == -2 || fixed(o)) continue;
                    if (std::find(nodes.begin(), nodes.end(), o) == nodes.end()) nodes.push_back(o);
                }
            }
            if (!ok) continue;
            const auto pos = [&](Index n) -> Index {
                if (fixed(n)) return -1;
                return static_cast<Index>(std::find(nodes.begin(), nodes.end(), n) - nodes.begin());
            };
            const auto m = static_cast<Index>(nodes.size());
            DenseMatrix kc = DenseMatrix::Zero(m, m);
            std::vector<Size> caps;
            for (Size k = 0; k < es.size(); ++k) {
                const auto& e = es[k];
                if (e.kind != K::Capacitor) continue;
                const Index a = pos(e.from), b = pos(e.to);
                if ((a < 0 && b < 0) || a == m || b == m) continue;   // outside the walk
                caps.push_back(k);
                if (a >= 0) kc(a, a) += e.C_or_L;
                if (b >= 0) kc(b, b) += e.C_or_L;
                if (a >= 0 && b >= 0) {
                    kc(a, b) -= e.C_or_L;
                    kc(b, a) -= e.C_or_L;
                }
            }
            const Eigen::FullPivLU<DenseMatrix> lu(kc);
            if (!lu.isInvertible()) continue;
            Vector rhs = Vector::Zero(m);
            rhs[0] = 1;   // nodes[0] is the deposit node
            const Vector y = lu.solve(rhs);
            Share s{dv.node, {}};
            for (const Size k : caps) {
                const Index a = pos(es[k].from), b = pos(es[k].to);
                const Real share = es[k].C_or_L * ((a >= 0 ? y[a] : 0) - (b >= 0 ? y[b] : 0));
                if (share != Real{0}) s.members.push_back({k, share, 0});
            }
            shares_.push_back(std::move(s));
        }
    }

    struct Dev {
        thermal::DeviceLossSpec spec;   // a copy: the pool may grow mid-run
        Index branch = -1, node = -1;
        pwl::PwlBranchLaw law;   // v and i under the stamped mode (one law)
        bool on = false;     // state at the last accepted step
        Real v = 0, i = 0;   // device voltage / current there
        Real x_tj = 0;       // junction temperature there (closed loop)
        thermal::CompensatedSum e_on, e_off, e_el, e_tab, e_imp;
        thermal::DeviceLossRecord rec;
        // the last accepted step's length and conduction energies (an
        // impulse booked to the device replaces them by its end power)
        struct Last {
            Real h = 0, el = 0, tab = 0;
            bool replaced = true;
        } last;
        Real e_tab_last_ = 0;
        // an event() whose right limit the next accept_step() starts from
        bool ev_pending = false, ev_on = false;
        Real ev_t = 0, ev_v = 0, ev_i = 0, ev_e = 0;
    };

    /// Undo the pending event's booking (its record, ledger and heat).
    void unbook_(Dev& d) {
        if (!d.rec.events.empty()) d.rec.events.pop_back();
        (d.ev_on ? d.e_on : d.e_off).add(-d.ev_e);
        --(d.ev_on ? d.rec.n_on : d.rec.n_off);
        if (d.node >= 0 && d.ev_e != Real{0}) add_(queue_, d.node, -d.ev_e);
    }

    /// One commutation of `d` by the reading rule: record, ledger, energy.
    Real book_(Dev& d, Real t, bool on1, Real v0, Real i0, Real v1, Real i1,
               Real tj) {
        const auto pt = thermal::commutation_point(on1, v0, i0, v1, i1);
        const auto& tab = on1 ? d.spec.e_on : d.spec.e_off;
        const Real e = tab ? (*tab)(pt.v, pt.i, tj) : Real{0};
        d.rec.events.push_back({t, on1, pt.v, pt.i, tj, e});
        (on1 ? d.e_on : d.e_off).add(e);
        ++(on1 ? d.rec.n_on : d.rec.n_off);
        return e;
    }
    struct Heat {
        Index node;
        Real e;   // J
    };

    [[nodiscard]] static Real v_of_(const Dev& d, const Vector& x) { return d.law.v(x); }
    [[nodiscard]] static Real i_of_(const Dev& d, const Vector& x, bool on) {
        return d.law.i(x, on);
    }
    [[nodiscard]] static Real tj_of_(const Dev& d) {
        return d.spec.closed_loop() ? d.x_tj : d.spec.tj;
    }
    static void add_(std::vector<Heat>& v, Index node, Real e) {
        for (auto& q : v) {
            if (q.node == node) {
                q.e += e;
                return;
            }
        }
        v.push_back({node, e});
    }

    static Index find_branch_(const topology::Graph& g, std::string_view name) {
        for (Index b = 0; b < g.num_branches(); ++b) {
            if (g.branch_name(b) == name) return b;
        }
        return -1;
    }
    static Index find_node_(const topology::Graph& g, std::string_view name) {
        for (Index n = 0; n < g.num_nodes(); ++n) {
            if (g.node(n).name == name) return n;
        }
        return -1;
    }
    /// The deposit node must absorb an impulse: a capacitor on it, and
    /// no voltage source holding it (the heat would leave through it).
    static void require_thermal_node_(const topology::Graph& g,
                                      const pwl::DevicePool& pool,
                                      std::string_view dev,
                                      std::string_view node_name, Index node) {
        using K = pwl::DevicePool::StoredKind;
        bool cap = false;
        for (const Index b : g.branches_of(node)) {
            if (!pool.is_registered(b)) continue;
            const auto k = pool.kind_of(b);
            if (k == K::VoltageSource || k == K::PWMVoltageSource ||
                k == K::SineVoltageSource || k == K::PulseVoltageSource ||
                k == K::VCVS || k == K::CCVS) {
                throw std::invalid_argument(std::format(
                    "device losses: '{}' deposits its heat into node '{}', "
                    "which voltage source '{}' holds: the heat would leave "
                    "through the source and the temperature would not move. "
                    "Deposit into the junction node of the thermal network.",
                    dev, node_name, g.branch_name(b)));
            }
            cap = cap || k == K::Capacitor;
        }
        if (cap) return;
        throw std::invalid_argument(std::format(
            "device losses: '{}' deposits its heat into node '{}', which has "
            "no thermal capacitance (no capacitor on it): its temperature rise "
            "would depend on the step size. Give the junction a capacitor "
            "(a Cauer or Foster stage).", dev, node_name));
    }

    std::vector<Dev> devs_;
    std::vector<Heat> queue_, now_;   // energy owed next step / this step
    std::vector<Heat> left_;          // power (W) the last commit was solved under
    std::vector<Heat> pred_;          // conduction power (W) delivered over this step
    std::vector<Share> shares_;       // per deposit node (bound on the first advance(h, hist))
    std::vector<char> fixed_, plain_; // per node (empty: no deposit node)
    bool bound_ = false;
    bool left_known_ = true;          // false after a version-1 resume, until a commit
    Real h_now_ = 1;
    thermal::CompensatedSum injected_;
    std::size_t n_reinit_ = 0;   // reinit records already booked

public:
    /// The run state as a flat vector (SolverSnapshot::loss_state): the
    /// pending heat, the ledgers, each device's last accepted reading and
    /// its events. A resume restores it in place of begin(), so the
    /// resumed run's records and junction heat continue the first run's.
    [[nodiscard]] std::vector<Real> save() const {
        // version 2 adds left_ and pred_ (two more lists), written only when
        // they hold something (pwl never fills them: its state stays version 1)
        const bool v2 = !left_.empty() || !pred_.empty();
        std::vector<Real> o{v2 ? Real{2} : Real{1}, static_cast<Real>(devs_.size()), h_now_,
                            injected_.s, injected_.c};
        for (const auto* v : {&queue_, &now_, &left_, &pred_}) {
            if (v == &left_ && !v2) break;
            o.push_back(static_cast<Real>(v->size()));
            for (const auto& q : *v) {
                o.push_back(static_cast<Real>(q.node));
                o.push_back(q.e);
            }
        }
        for (const auto& d : devs_) {
            o.insert(o.end(), {Real(d.on), d.v, d.i, d.x_tj, Real(d.ev_pending),
                               Real(d.ev_on), d.ev_t, d.ev_v, d.ev_i, d.ev_e});
            for (const auto* c : {&d.e_on, &d.e_off, &d.e_el, &d.e_tab, &d.e_imp}) {
                o.push_back(c->s);
                o.push_back(c->c);
            }
            o.push_back(static_cast<Real>(d.rec.n_on));
            o.push_back(static_cast<Real>(d.rec.n_off));
            o.push_back(static_cast<Real>(d.rec.events.size()));
            for (const auto& e : d.rec.events) {
                o.insert(o.end(), {e.t, Real(e.turn_on), e.v, e.i, e.tj, e.e});
            }
        }
        return o;
    }

    void restore(const std::vector<Real>& s) {
        std::size_t k = 0;
        auto next = [&]() {
            if (k >= s.size()) {
                throw std::invalid_argument(
                    "device losses: the snapshot's loss state is truncated");
            }
            return s[k++];
        };
        auto count = [&]() { return static_cast<std::size_t>(next()); };
        const Real version = next();
        if ((version != Real{1} && version != Real{2}) || count() != devs_.size()) {
            throw std::invalid_argument(
                "device losses: the snapshot's loss state is for another set "
                "of loss ports (set_device_losses) than this circuit's");
        }
        h_now_ = next();
        injected_.s = next();
        injected_.c = next();
        left_.clear();
        pred_.clear();
        // a version-1 state does not say which power its slots hold
        left_known_ = version == Real{2};
        for (auto* v : {&queue_, &now_, &left_, &pred_}) {
            if (v == &left_ && version == Real{1}) break;
            v->clear();
            for (std::size_t n = count(); n > 0; --n) {
                const auto node = static_cast<Index>(next());
                v->push_back({node, next()});
            }
        }
        for (auto& d : devs_) {
            d.on = next() != 0;
            d.v = next();
            d.i = next();
            d.x_tj = next();
            d.ev_pending = next() != 0;
            d.ev_on = next() != 0;
            d.ev_t = next();
            d.ev_v = next();
            d.ev_i = next();
            d.ev_e = next();
            for (auto* c : {&d.e_on, &d.e_off, &d.e_el, &d.e_tab, &d.e_imp}) {
                c->s = next();
                c->c = next();
            }
            d.rec.n_on = count();
            d.rec.n_off = count();
            d.rec.events.clear();
            for (std::size_t n = count(); n > 0; --n) {
                thermal::LossEvent e;
                e.t = next();
                e.turn_on = next() != 0;
                e.v = next();
                e.i = next();
                e.tj = next();
                e.e = next();
                d.rec.events.push_back(e);
            }
        }
        if (k != s.size()) {
            throw std::invalid_argument(
                "device losses: the snapshot's loss state has trailing values");
        }
        n_reinit_ = 0;   // a resumed result's reinit_records start empty
    }

    /// Move the records into a result.
    template <typename Result>
    void write(Result& out) const {
        out.device_losses.clear();
        // queued, or owed to a step that was not accepted (an abort)
        Real pending = 0;
        for (const auto& q : queue_) pending += q.e;
        for (const auto& n : now_) pending += n.e;
        for (const auto& d : devs_) {
            auto r = d.rec;
            r.E_on = d.e_on.value();
            r.E_off = d.e_off.value();
            r.E_cond_electrical = d.e_el.value();
            r.E_cond_table = d.e_tab.value();
            r.E_impulse = d.e_imp.value();
            out.device_losses.push_back(std::move(r));
        }
        out.heat_injected = injected_.value();
        out.heat_pending = pending;
    }
};

}  // namespace pulsim::solver
