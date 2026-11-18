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
// Engines without this hook refuse a circuit with loss ports by name
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
                d.e_tab.add(e_t);
                if (d.spec.conduction == thermal::ConductionHeat::Table) heat = e_t;
            }
            if (d.spec.conduction == thermal::ConductionHeat::Electrical) heat = e_el;
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
        thermal::CompensatedSum e_on, e_off, e_el, e_tab;
        thermal::DeviceLossRecord rec;
    };
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

public:
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
            out.device_losses.push_back(std::move(r));
        }
        out.heat_injected = injected_.value();
        out.heat_pending = pending;
    }
};

}  // namespace pulsim::solver
