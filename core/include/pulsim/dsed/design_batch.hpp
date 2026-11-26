#pragma once

// =============================================================================
// Pulsim -- the kernel batch evaluator of `pulsim.design.Evaluator`
// =============================================================================
//
// add-design-optimization-workflow DECISIONS 26. One design of the
// circuit form on the exact steady-state route is, in one call and with
// no Python: `with_params` / `with_value` of the shared base `Circuit`,
// the periodic orbit (`periodic_orbit_newton`, the kernel shooting), the
// period integrals, the losses, the KPIs, the energy balance, the
// constraints and the status. A batch runs a table of designs on
// `std::thread`s; every design owns its builder, cache, adapter and
// registers, and reads the base circuit, the plan and the loss tables,
// which are immutable (the base's symbolic plan is mutex-guarded).
//
// What a record holds is a short register program the Python plan
// writes (`pulsim/_design_kernel.py`): it mirrors, operation for
// operation, the arithmetic of the Python path (`design._kpis`,
// `_period_kpis.loss_rows`, `_energy_balance`), so the records are the
// Python path's bit for bit. A design the kernel cannot finish -- an
// orbit that does not converge, a refusal, an energy-balance or a
// non-finite value -- is returned as `fallback` with the reason; the
// caller re-runs it on the Python path, which owns every non-`ok`
// outcome's wording. Nothing here throws out of a worker.

#include "pulsim/analysis/cancellation.hpp"
#include "pulsim/dsed/builder_adapter.hpp"
#include "pulsim/dsed/period_integrals.hpp"
#include "pulsim/dsed/period_thermal.hpp"
#include "pulsim/dsed/periodic_orbit.hpp"
#include "pulsim/ir/circuit.hpp"
#include "pulsim/numeric/types.hpp"
#include "pulsim/pwl/cache.hpp"
#include "pulsim/solver/voltage_sanity.hpp"
#include "pulsim/thermal/device_losses.hpp"
#include "pulsim/thermal/loss_table.hpp"
#include "pulsim/topology/switch_state.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <exception>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <span>
#include <system_error>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace pulsim::dsed::design {

using SSM = topology::SwitchStateMask;

/// Register-program operations. Same numbering as `_design_kernel.OPS`.
enum class Op : std::int32_t {
    Const = 0,   // push c
    Var = 1,     // push design variable a
    Slot = 2,    // push value slot a (get_value of the design's circuit)
    Stat = 3,    // push stats[a] column b (int1, int2, peak, t_peak, valley, t_valley)
    Cross = 4,   // push cross[a]
    YStart = 5,  // push y_start[a]
    YEnd = 6,    // push y_end[a]
    Sw = 7,      // push switching entry a field b (E_on, E_off, n_on, n_off)
    Period = 8,  // push T
    Load = 9,    // push register a
    Store = 10,  // pop into register a
    Add = 11, Sub = 12, Mul = 13, Div = 14, Neg = 15, Pow = 16,
    Max2 = 17,   // Python max(a, b): b if b > a else a
    Min2 = 18,   // Python min(a, b): b if b < a else a
    Abs = 19, Sqrt = 20, Exp = 21, Log = 22,
    DivPos = 23, // a / b if b > 0 else NaN (the efficiency guard)
    SumN = 24,   // pop a values; CPython >= 3.12 sum() (Neumaier) of them
};

struct Instr {
    Op op = Op::Const;
    std::int32_t a = 0;
    std::int32_t b = 0;
    Real c = 0;
};

/// A switch's switching energy from its loss tables at its edges.
struct SwitchingEntry {
    int bit = -1;
    int obs_v = -1;
    int obs_i = -1;
    std::shared_ptr<const thermal::LossTable3D> on, off;
    Real tj = 0;
    thermal::EdgeLaw edge{true, 0.0, 0.0};   // commutation_point's soft edges
};

/// Everything a design needs besides its row, period and schedule.
struct Plan {
    // the design table's columns: a named parameter, else "dev.field"
    std::vector<std::string> names;
    std::vector<bool> is_param;
    // observables of the period integrals
    std::vector<PeriodObservable> obs;
    std::vector<std::pair<int, int>> pairs;
    std::vector<std::string> slots;        // "R1.R": read per design
    std::vector<SwitchingEntry> switching;
    std::vector<Instr> code;
    int n_regs = 0;
    std::vector<int> kpi_regs, loss_regs, viol_regs;
    // energy balance: applied when energy_static and p_in > 0
    bool energy_static = false;
    int p_in_reg = -1, res_reg = -1, p_table_reg = -1;
    Real energy_tol = Real{1e-8};
    // implausible-voltage scan: obs index of node 0 (nodes in order)
    int node_obs0 = -1;
    int n_nodes = 0;
    int n_full = 0;
    // orbit
    OrbitOptions orbit;
    bool mode_guard = false;
    Real h_e_rel = Real{1e-7};
};

enum class Status : std::int32_t { Ok = 0, Infeasible = 1, Fallback = 2 };

struct DesignOut {
    Status status = Status::Fallback;
    std::string reason;                  // fallback only
    std::vector<Real> kpis, losses, viol;
    bool energy_applied = false;
    Real energy_res = 0, p_table = 0;
    int n_runs = 0, n_iterations = 0;
    std::size_t n_steps = 0;
    std::uint64_t symbolic_analyses = 0;
    std::vector<std::pair<std::string, Real>> pattern;   // (name, t/T)
    Vector x0;
    double wall_s = 0;
};

namespace detail {

using period_thermal::np_pairwise;
using period_thermal::python_total;

struct Fallback {
    std::string why;
};

/// `_period_kpis.Period.edges` + `_table_switching_loss` for one
/// switch: (E_on, E_off, n_on, n_off), each edge read by
/// thermal::commutation_point (a soft edge adds no entry).
inline std::array<Real, 4> switching_energy(const PeriodMapResult<SSM>& p,
                                            const SwitchingEntry& s, Real T) {
    struct Pt { bool closed; Real v, i; };
    std::vector<Pt> seq;
    const auto bit = static_cast<Size>(s.bit);
    const bool lb = p.last_mask.get(bit), fb = p.first_mask.get(bit);
    if (lb != fb) {
        seq.push_back({lb, p.y_end[s.obs_v], p.y_end[s.obs_i]});
        seq.push_back({fb, p.y_start[s.obs_v], p.y_start[s.obs_i]});
    }
    const Real tol = Real{1e-12} * T;
    for (std::size_t k = 0; k < p.events.size(); ++k) {
        const auto& e = p.events[k];
        const bool o = e.old_mask.get(bit), n = e.new_mask.get(bit);
        if (o != n && tol < e.t && e.t < T - tol) {
            const auto r = static_cast<Index>(k);
            seq.push_back({o, p.y_left(r, s.obs_v), p.y_left(r, s.obs_i)});
            seq.push_back({n, p.y_right(r, s.obs_v), p.y_right(r, s.obs_i)});
        }
    }
    for (std::size_t q = 1; q + 1 < seq.size(); q += 2) {
        if (seq[q].closed != seq[q + 1].closed) {
            throw Fallback{"the period map's edges do not alternate"};
        }
    }
    if (seq.empty()) seq.push_back({fb, 0.0, 0.0});
    std::vector<Real> on, off;
    for (std::size_t q = 0; q + 1 < seq.size(); ++q) {
        const auto& a = seq[q];
        const auto& b = seq[q + 1];
        if (a.closed == b.closed) continue;
        const auto pt = thermal::commutation_point(b.closed, a.v, a.i, b.v, b.i, s.edge, false);
        const Real v = pt.v, i = pt.i;
        if (!std::isfinite(v) || !std::isfinite(i)) {
            throw Fallback{"a non-finite switching point"};
        }
        if (pt.soft) {
            // a soft edge costs nothing (thermal::commutation_point)
        } else if (b.closed) {
            if (s.on) on.push_back((*s.on)(v, i, s.tj));
        } else if (s.off) {
            off.push_back((*s.off)(v, i, s.tj));
        }
    }
    // counts are of the edges, whether or not the table is given
    Real n_on = 0, n_off = 0;
    for (std::size_t q = 0; q + 1 < seq.size(); ++q) {
        if (seq[q].closed != seq[q + 1].closed) (seq[q + 1].closed ? n_on : n_off) += 1;
    }
    return {s.on ? python_total(on) : Real{0}, s.off ? python_total(off) : Real{0},
            n_on, n_off};
}

inline Real py_max(Real a, Real b) { return b > a ? b : a; }
inline Real py_min(Real a, Real b) { return b < a ? b : a; }

/// The register program on one design's period.
inline void run_program(const Plan& plan, const Real* var,
                        const std::vector<Real>& slots,
                        const PeriodMapResult<SSM>& p,
                        const std::vector<std::array<Real, 4>>& sw, Real T,
                        std::vector<Real>& regs) {
    std::vector<Real> st;
    st.reserve(16);
    auto pop = [&] {
        const Real v = st.back();
        st.pop_back();
        return v;
    };
    regs.assign(static_cast<std::size_t>(plan.n_regs),
                std::numeric_limits<Real>::quiet_NaN());
    for (const auto& in : plan.code) {
        switch (in.op) {
            case Op::Const: st.push_back(in.c); break;
            case Op::Var: st.push_back(var[in.a]); break;
            case Op::Slot: st.push_back(slots[static_cast<std::size_t>(in.a)]); break;
            case Op::Stat: {
                const auto& s = p.stats[static_cast<std::size_t>(in.a)];
                const Real col[6] = {s.int1, s.int2, s.peak, s.t_peak, s.valley, s.t_valley};
                st.push_back(col[in.b]);
                break;
            }
            case Op::Cross: st.push_back(p.cross[static_cast<std::size_t>(in.a)]); break;
            case Op::YStart: st.push_back(p.y_start[in.a]); break;
            case Op::YEnd: st.push_back(p.y_end[in.a]); break;
            case Op::Sw: st.push_back(sw[static_cast<std::size_t>(in.a)][static_cast<std::size_t>(in.b)]); break;
            case Op::Period: st.push_back(T); break;
            case Op::Load: st.push_back(regs[static_cast<std::size_t>(in.a)]); break;
            case Op::Store: regs[static_cast<std::size_t>(in.a)] = pop(); break;
            case Op::Neg: st.back() = -st.back(); break;
            case Op::Abs: st.back() = std::abs(st.back()); break;
            case Op::Sqrt: st.back() = std::sqrt(st.back()); break;
            case Op::Exp: st.back() = std::exp(st.back()); break;
            case Op::Log: st.back() = std::log(st.back()); break;
            case Op::SumN: {
                // pulsim.design._psum: CPython's float sum, compensated
                const auto k = static_cast<std::size_t>(in.a);
                Real f = 0, c = 0;
                for (std::size_t q = st.size() - k; q < st.size(); ++q) {
                    const Real x = st[q];
                    const Real t = f + x;
                    if (std::abs(f) >= std::abs(x)) c += (f - t) + x;
                    else c += (x - t) + f;
                    f = t;
                }
                st.resize(st.size() - k);
                if (c != 0 && std::isfinite(c)) f += c;
                st.push_back(f);
                break;
            }
            default: {
                const Real b = pop();
                const Real a = pop();
                Real r = 0;
                switch (in.op) {
                    case Op::Add: r = a + b; break;
                    case Op::Sub: r = a - b; break;
                    case Op::Mul: r = a * b; break;
                    case Op::Div: r = a / b; break;
                    case Op::Pow: r = std::pow(a, b); break;
                    case Op::Max2: r = py_max(a, b); break;
                    case Op::Min2: r = py_min(a, b); break;
                    case Op::DivPos:
                        r = b > 0 ? a / b : std::numeric_limits<Real>::quiet_NaN();
                        break;
                    default: throw Fallback{"design program: unknown operation"};
                }
                st.push_back(r);
            }
        }
    }
}

/// `periodic_steady_state._check_schedule`'s periodicity test.
template <class Sched>
bool schedule_is_periodic(const Sched& sf, Real T) {
    auto edges = [&](Real lo, Real hi) {
        std::vector<Real> out;
        Real t = lo;
        for (;;) {
            const Real e = sf.next_edge_after(t);
            if (!(e <= hi) || !std::isfinite(e)) return out;
            if (e <= t) throw Fallback{"the schedule's edges cannot be located"};
            out.push_back(e);
            if (out.size() > 100000) throw Fallback{"more than 1e5 edges in one period"};
            t = e;
        }
    };
    const auto e1 = edges(0.0, T);
    const auto e2 = edges(T, 2 * T);
    const Real tol = Real{1e-12} * T * 4;
    if (e1.size() != e2.size()) return false;
    for (std::size_t k = 0; k < e1.size(); ++k) {
        if (!(std::abs((e2[k] - T) - e1[k]) <= tol)) return false;
    }
    std::vector<Real> grid{0.0};
    grid.insert(grid.end(), e1.begin(), e1.end());
    grid.push_back(T);
    for (std::size_t k = 0; k + 1 < grid.size(); ++k) {
        const Real tm = 0.5 * (grid[k] + grid[k + 1]);
        if (!(sf(tm) == sf(tm + T))) return false;
    }
    return true;
}

/// Shortest period of the sine / PWM sources (`_min_source_period`).
inline Real min_source_period(const topology::Graph& g, const pwl::DevicePool& pool) {
    using K = pwl::DevicePool::StoredKind;
    Real T = std::numeric_limits<Real>::infinity();
    for (Index b = 0; b < g.num_branches(); ++b) {
        K k;
        try { k = pool.kind_of(b); } catch (const std::out_of_range&) { continue; }
        Real f = 0;
        if (k == K::SineVoltageSource) f = pool.sine_voltage_source_params(b).frequency;
        else if (k == K::PWMVoltageSource) f = pool.pwm_voltage_source_params(b).frequency;
        if (f > 0) T = std::min(T, Real{1} / f);
    }
    return T;
}

inline ir::Circuit derive(const ir::Circuit& base, const Plan& plan, const Real* row) {
    std::vector<std::pair<std::string_view, Real>> ps;
    for (std::size_t j = 0; j < plan.names.size(); ++j) {
        if (plan.is_param[j]) ps.emplace_back(plan.names[j], row[j]);
    }
    ir::Circuit c = ps.empty() ? base : base.with_params(ps);
    for (std::size_t j = 0; j < plan.names.size(); ++j) {
        if (!plan.is_param[j]) c = c.with_value(plan.names[j], row[j]);
    }
    return c;
}

}  // namespace detail

/// One design. `run_period(adapter, sched, x, T, obs, pairs, h_e)`
/// returns a `PeriodMapResult<SSM>` (throwing `PeriodRunFailed` on an
/// engine non-convergence).
template <class Sched, class RunPeriod>
DesignOut evaluate_design(const ir::Circuit& base, const Plan& plan,
                          const Real* row, Real T, const Sched& sf,
                          RunPeriod& run_period) {
    using detail::Fallback;
    DesignOut out;
    const auto t0 = std::chrono::steady_clock::now();
    try {
        ir::Circuit c = detail::derive(base, plan, row);
        auto b = c.to_builder();
        pwl::PwlStateSpaceCache cache(b.graph(), b.pool());
        cache.share_symbolic(c.symbolic_plan());
        cache.build_lazy(T);
        NativeCircuitBuilderAdapter<SSM> ad(b.graph(), b.pool(), cache);
        if (ad.has_dynamic_sources()) throw Fallback{"time-varying sources"};
        if (!detail::schedule_is_periodic(sf, T)) {
            throw Fallback{"the schedule is not periodic in T"};
        }
        const auto m0 = sf(0.0);
        if (m0.size() != b.graph().num_switches()) {
            // the Python path refuses it by name (require_mask_width)
            throw Fallback{"a switch mask of another width than the circuit's"};
        }
        ad.set_mask(m0);
        const int n = ad.n_state();
        if (n == 0) throw Fallback{"no state"};
        Real h_e = 0;
        if (plan.mode_guard) {
            Real m = std::min(T, detail::min_source_period(b.graph(), b.pool()));
            const Real e1 = sf.next_edge_after(0.0);
            const Real e3 = sf.next_edge_after(sf.next_edge_after(e1));
            if (std::isfinite(e3) && e3 > e1) m = std::min(m, e3 - e1);
            h_e = plan.h_e_rel * m;
        }
        const std::vector<PeriodObservable> no_obs;
        const std::vector<std::pair<int, int>> no_pairs;
        auto run = [&](const Vector& x, bool with_obs) {
            return run_period(ad, sf, x, T, with_obs ? plan.obs : no_obs,
                              with_obs ? plan.pairs : no_pairs, h_e);
        };
        auto r = periodic_orbit_newton<SSM>(run, Vector::Zero(n), T, plan.orbit);
        out.n_runs = r.n_runs;
        out.n_steps = r.n_steps;
        out.n_iterations = static_cast<int>(r.history.size());
        out.symbolic_analyses = cache.metrics().symbolic_analyses;
        if (r.status != OrbitStatus::Converged) {
            throw Fallback{std::string("orbit: ") + to_string(r.status)};
        }
        for (Index k = 0; k < r.multipliers.size(); ++k) {
            if (std::abs(r.multipliers[k]) >= 1.0) throw Fallback{"unstable orbit"};
        }
        const auto& p = r.period;
        if (!r.has_period) throw Fallback{"no verification period"};
        // the Python path checks periodicity here, before the integrals
        for (const auto& e : p.events) out.pattern.emplace_back(e.name, e.t / T);
        out.x0 = r.x0;
        if (!p.integrals_ok) throw Fallback{"no closed-form period integrals"};
        // a cut-set record holds the spike the node extrema do not (an
        // instantaneous cut destroys L i^2 / 2): the Python path warns by it
        for (const auto& rec : ad.reinit_log()) {
            if (rec.kind == "cut_set" && rec.dW > 0) {
                throw Fallback{"an impulsive re-initialisation (a cut-set record)"};
            }
        }
        // the implausible-voltage scan of a kernel period (node extrema)
        if (plan.node_obs0 >= 0) {
            solver::SimulationResult sr;
            sr.times = {0.0, T};
            Vector hi = Vector::Zero(plan.n_full), lo = Vector::Zero(plan.n_full);
            for (int k = 0; k < plan.n_nodes; ++k) {
                const auto& s = p.stats[static_cast<std::size_t>(plan.node_obs0 + k)];
                hi[k] = s.peak;
                lo[k] = s.valley;
            }
            sr.states = {hi, lo};
            const auto v = solver::find_implausible_voltage(b.graph(), b.pool(), sr, 100.0);
            if (v.node >= 0) throw Fallback{"implausible voltage"};
        }
        std::vector<Real> slots;
        slots.reserve(plan.slots.size());
        for (const auto& s : plan.slots) slots.push_back(c.lowered().get_value(s));
        std::vector<std::array<Real, 4>> sw;
        sw.reserve(plan.switching.size());
        for (const auto& s : plan.switching) sw.push_back(detail::switching_energy(p, s, T));
        std::vector<Real> regs;
        detail::run_program(plan, row, slots, p, sw, T, regs);
        auto take = [&](const std::vector<int>& idx) {
            std::vector<Real> v;
            v.reserve(idx.size());
            for (int k : idx) v.push_back(regs[static_cast<std::size_t>(k)]);
            return v;
        };
        out.kpis = take(plan.kpi_regs);
        out.losses = take(plan.loss_regs);
        out.viol = take(plan.viol_regs);
        if (plan.energy_static && regs[static_cast<std::size_t>(plan.p_in_reg)] > 0) {
            out.energy_applied = true;
            out.energy_res = regs[static_cast<std::size_t>(plan.res_reg)];
            out.p_table = regs[static_cast<std::size_t>(plan.p_table_reg)];
            if (!(std::abs(out.energy_res) <= plan.energy_tol)) throw Fallback{"energy balance"};
        }
        for (Real v : out.kpis) {
            if (!std::isfinite(v)) throw Fallback{"a non-finite KPI"};
        }
        out.status = Status::Ok;
        for (Real v : out.viol) {
            if (v > 0) out.status = Status::Infeasible;
        }
    } catch (const Fallback& f) {
        out.status = Status::Fallback;
        out.reason = f.why;
    } catch (const pulsim::analysis::Cancelled&) {
        throw;   // the batch stops; not a design outcome (evaluate_batch)
    } catch (const std::exception& e) {
        out.status = Status::Fallback;
        out.reason = std::string("kernel error: ") + e.what();
    } catch (...) {
        out.status = Status::Fallback;
        out.reason = "kernel error: unknown exception";
    }
    if (out.status == Status::Fallback) {
        out.kpis.clear();
        out.losses.clear();
        out.viol.clear();
    }
    out.wall_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return out;
}

/// A table of designs (`rows` row-major, one row per design) on
/// `n_threads` threads; outputs in input order. `scheds[i]` and
/// `periods[i]` belong to row i.
template <class Sched, class RunPeriod>
std::vector<DesignOut> evaluate_batch(const ir::Circuit& base, const Plan& plan,
                                      const Real* rows, std::size_t n_rows,
                                      std::span<const Real> periods,
                                      std::span<const Sched* const> scheds,
                                      RunPeriod run_period, int n_threads) {
    std::vector<DesignOut> out(n_rows);
    std::atomic<std::size_t> next{0};
    const std::size_t width = plan.names.size();
    // the first exception that escapes a design (a cancellation; every
    // design outcome is caught in evaluate_design) stops the other
    // workers and is re-thrown after the join
    std::mutex err_mu;
    std::exception_ptr err;
    std::atomic<bool> stop{false};
    auto work = [&] {
        RunPeriod rp = run_period;   // per-thread copy of the functor
        for (std::size_t i; !stop.load() && (i = next.fetch_add(1)) < n_rows;) {
            try {
                out[i] = evaluate_design(base, plan, rows + i * width, periods[i],
                                         *scheds[i], rp);
            } catch (...) {
                std::lock_guard<std::mutex> lk(err_mu);
                if (!err) err = std::current_exception();
                stop.store(true);
            }
        }
    };
    const auto nt = static_cast<std::size_t>(std::max(1, n_threads));
    std::vector<std::thread> pool;
    try {
        for (std::size_t t = 1; t < std::min(nt, n_rows); ++t) pool.emplace_back(work);
    } catch (const std::system_error&) {
        // no more threads on this machine: the started ones and this
        // thread drain the table (a destroyed joinable thread terminates)
    }
    work();
    for (auto& th : pool) th.join();
    if (err) std::rethrow_exception(err);
    return out;
}

}  // namespace pulsim::dsed::design
