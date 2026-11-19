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
//
// impulses(result.reinit_records): the energy of an impulsive
// re-initialisation (a capacitor loop closed, a cut-set opened, by an
// ideal element) is dissipated by the element the record is booked to
// (`cause`): it joins that device's heat as E_impulse, in the step that
// carried it. The engine calls it after its reinit.record().
//
// Engines without these calls refuse a circuit with loss ports by name
// (`require_supported`, feature "device_losses").

#include "pulsim/numeric/types.hpp"
#include "pulsim/pwl/device_pool.hpp"
#include "pulsim/stamping/branch_coord.hpp"
#include "pulsim/thermal/device_losses.hpp"
#include "pulsim/topology/graph.hpp"
#include "pulsim/topology/switch_state.hpp"

#include <format>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace pulsim::solver {

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
            d.diode = k == K::Diode;
            if (k != K::Switch && !d.diode) {
                throw std::invalid_argument(std::format(
                    "device losses: '{}' is a {}, not a PWL switch or diode",
                    s.device, pwl::stored_kind_name(k)));
            }
            const auto& br = graph.branch(d.branch);
            d.from = br.from;
            d.to = br.to;
            d.var = pool.branch_var_id_for_switch(d.branch, graph);
            if (d.diode) {
                const auto& p = pool.diode_params(d.branch);
                d.g_on = p.g_on;
                d.g_off = p.g_off;
                d.v_f = p.V_f;
            } else {
                d.g_on = pool.switch_g_on(d.branch);
                d.g_off = pool.switch_g_off(d.branch);
            }
            d.bit = switch_bit_(graph, d.branch);
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
    }

    [[nodiscard]] bool active() const noexcept { return !devs_.empty(); }

    /// The state the run starts from and the mask it was stamped under.
    void begin(const Vector& x0, const topology::SwitchStateMask& m0) {
        for (auto& d : devs_) {
            d.on = m0.get(static_cast<Size>(d.bit));
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
    }

    /// One accepted step from (t0, x0) to (t1, x1) under `mask`.
    void accept(Real t0, Real t1, const Vector& x1,
                const topology::SwitchStateMask& mask) {
        const Real h = t1 - t0;
        // the power of this step has reached the network
        for (const auto& n : now_) injected_.add(n.e);
        now_.clear();
        for (auto& d : devs_) {
            const bool on1 = mask.get(static_cast<Size>(d.bit));
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

    /// Variable-step accept of [t0, t1] ending in x1 under `mask`. The
    /// quadrature of the conduction power is w0·p(start) + wg·p(xg) +
    /// w1·p(x1) (the integrator's stage weights, Σw = 1; xg may be null
    /// when wg = 0). The start is the last accepted end, or the right
    /// limit of an event() whose mask this step carries.
    void accept_step(Real t0, Real t1, const Vector& x1,
                     const topology::SwitchStateMask& mask, const Vector* xg,
                     Real w0, Real wg, Real w1) {
        const Real h = t1 - t0;
        // the power E/h_now_ reached the network over h of this step: a
        // committed step shorter than the attempt (a landing) keeps the rest
        const Real frac = h >= h_now_ * (Real{1} - Real{1e-12}) ? Real{1} : h / h_now_;
        for (auto& n : now_) {
            const Real e = frac == Real{1} ? n.e : n.e * frac;
            injected_.add(e);
            n.e -= e;
        }
        if (frac == Real{1}) now_.clear();
        for (auto& d : devs_) {
            const bool on1 = mask.get(static_cast<Size>(d.bit));
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
            Real vg = 0, ig = 0, tjg = tj0;
            const bool same = on1 == d.on;
            if (same && xg != nullptr && wg != Real{0}) {
                vg = v_of_(d, *xg);
                ig = i_of_(d, *xg, on1);
                if (d.node >= 0 && d.spec.closed_loop())
                    tjg = stamping::read_node_voltage(*xg, d.node);
            }
            if (same) {
                e_el = h * (w0 * d.v * d.i + wg * vg * ig + w1 * v1 * i1);
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
                    const Real g = on1 && wg != Real{0} ? c(std::abs(ig), tjg) * std::abs(ig) : 0;
                    e_t = h * (w0 * a + wg * g + w1 * z);
                } else {
                    e_t = h * (a + z) / Real{2};
                }
                d.e_tab.add(e_t);
                if (d.spec.conduction == thermal::ConductionHeat::Table) heat = e_t;
            }
            if (d.spec.conduction == thermal::ConductionHeat::Electrical) heat = e_el;
            d.last = {h, e_el, e_t, false};
            // a flip no event() reported: the reading rule on the step's ends
            if (!same) heat += book_(d, t0, on1, d.v, d.i, v1, i1, tj0);
            if (d.node >= 0) add_(queue_, d.node, heat);
            d.on = on1;
            d.v = v1;
            d.i = i1;
            d.x_tj = tj1;
        }
    }

    /// A commutation at t between the engine's left limit (x_l under m_l)
    /// and right limit (x_r under m_r): every tracked device whose bit
    /// differs is booked by the reading rule on the two limits.
    void event(Real t, const Vector& x_l, const topology::SwitchStateMask& m_l,
               const Vector& x_r, const topology::SwitchStateMask& m_r) {
        for (auto& d : devs_) {
            const auto bit = static_cast<Size>(d.bit);
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

    /// The queued energy becomes the power of the next step, of length
    /// h_next. Called once per outer step; a second call before accept()
    /// (no step was accepted) keeps the power already owed.
    void advance(Real h_next) {
        h_now_ = h_next;
        for (const auto& q : queue_) {
            if (q.e == Real{0}) continue;
            add_(now_, q.node, q.e);
        }
        queue_.clear();
    }

private:
    struct Dev {
        thermal::DeviceLossSpec spec;   // a copy: the pool may grow mid-run
        Index branch = -1, from = -1, to = -1, var = -1, node = -1;
        long bit = -1;
        bool diode = false;
        Real g_on = 1, g_off = 0, v_f = 0;
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

    [[nodiscard]] static Real v_of_(const Dev& d, const Vector& x) {
        return stamping::read_node_voltage(x, d.from) -
               stamping::read_node_voltage(x, d.to);
    }
    [[nodiscard]] static Real i_of_(const Dev& d, const Vector& x, bool on) {
        if (d.var >= 0) return x[d.var];
        const Real v = v_of_(d, x);
        if (d.diode) return on ? d.g_on * (v - d.v_f) : d.g_off * v;
        return v * (on ? d.g_on : d.g_off);
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
    static long switch_bit_(const topology::Graph& g, Index branch) {
        long bit = 0;
        for (Index b = 0; b < branch; ++b) {
            if (g.branch(b).kind == topology::BranchKind::Switch) ++bit;
        }
        return bit;
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
    Real h_now_ = 1;
    thermal::CompensatedSum injected_;
    std::size_t n_reinit_ = 0;   // reinit records already booked

public:
    /// The run state as a flat vector (SolverSnapshot::loss_state): the
    /// pending heat, the ledgers, each device's last accepted reading and
    /// its events. A resume restores it in place of begin(), so the
    /// resumed run's records and junction heat continue the first run's.
    [[nodiscard]] std::vector<Real> save() const {
        std::vector<Real> o{Real{1}, static_cast<Real>(devs_.size()), h_now_,
                            injected_.s, injected_.c};
        for (const auto* v : {&queue_, &now_}) {
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
        if (next() != Real{1} || count() != devs_.size()) {
            throw std::invalid_argument(
                "device losses: the snapshot's loss state is for another set "
                "of loss ports (set_device_losses) than this circuit's");
        }
        h_now_ = next();
        injected_.s = next();
        injected_.c = next();
        for (auto* v : {&queue_, &now_}) {
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
