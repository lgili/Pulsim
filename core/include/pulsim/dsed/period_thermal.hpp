#pragma once

// =============================================================================
// Pulsim -- event losses and the junction temperature on a periodic orbit
// =============================================================================
//
// add-in-kernel-event-losses, the exact-route part
// (add-design-optimization-workflow DECISIONS 28). The electrical orbit is
// fixed; what depends on the junction temperature is how the loss tables
// are read. Per device:
//
//   P(T_j) = P_fixed + (sum E_on(v, i, T_j) + sum E_off(v, i, T_j)) / T
//            [+ (a(T_j) int|i| + b(T_j) int i^2) / T   with a ConductionTable]
//
// where the (v, i) of every commutation follow the reading rule (turn-on:
// v before, i after; turn-off: v after, i before), read by the caller from
// the period map's left and right limits. `solve_junction` finds the
// smallest T_j >= T_amb with T_j = T_amb + R_th P(T_j): the temperature
// the physical iteration T <- T_amb + R_th P(T) climbs to from T_amb. It
// refuses a runaway (no such root) by name, with R_th dP/dT.
//
// Pure functions of immutable inputs: safe on any number of threads.

#include "pulsim/thermal/loss_table.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace pulsim::dsed::period_thermal {

using Real = double;

/// numpy's pairwise summation of `np.sum` over a contiguous float64
/// vector (checked against numpy 2.4 on 5840 random vectors, n 8..299).
inline Real np_pairwise(const Real* a, std::size_t n) {
    if (n < 8) {
        Real res = 0.;
        for (std::size_t i = 0; i < n; ++i) res += a[i];
        return res;
    }
    if (n <= 128) {
        Real r[8];
        for (int j = 0; j < 8; ++j) r[j] = a[j];
        std::size_t i = 8;
        for (; i < n - (n % 8); i += 8) {
            for (int j = 0; j < 8; ++j) r[j] += a[i + static_cast<std::size_t>(j)];
        }
        Real res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for (; i < n; ++i) res += a[i];
        return res;
    }
    std::size_t n2 = n / 2;
    n2 -= n2 % 8;
    return np_pairwise(a, n2) + np_pairwise(a + n2, n - n2);
}

/// `_period_kpis._table_switching_loss`'s sum: a plain loop below 8
/// terms, `np.sum` from 8 on.
inline Real python_total(const std::vector<Real>& e) {
    if (e.size() < 8) {
        Real acc = 0.0;
        for (Real x : e) acc += x;
        return acc;
    }
    return np_pairwise(e.data(), e.size());
}

/// Sum of a table over the commutation points at one T_j.
inline Real table_total(const thermal::LossTable3D* t,
                        const std::vector<std::array<Real, 2>>& pts, Real tj) {
    if (t == nullptr || pts.empty()) return 0.0;
    std::vector<Real> e;
    e.reserve(pts.size());
    for (const auto& p : pts) e.push_back((*t)(p[0], p[1], tj));
    return python_total(e);
}

/// `(1/T) int V_on(|i|, T_j) |i| dt` of a conduction table that is
/// linear in i (one or two current knots): `(a q1 + b q2) / T` with
/// q1 = int|i| dt, q2 = int i^2 dt over the period. Refused, by name,
/// when the table is not linear in i (its integral needs the waveform)
/// or when it clamps at 0 inside [0, i_max] (then it is not a + b i).
inline Real conduction_power(const thermal::ConductionTable& c, Real q1, Real q2,
                             Real i_max, Real tj, Real period) {
    const auto& ia = c.i_axis();
    if (ia.size() > 2) {
        throw std::invalid_argument(
            "a conduction table with more than two current knots is not linear in i: "
            "its period integral needs the waveform, which the period map does not keep");
    }
    const auto at = thermal::detail::axis_weight("ConductionTable", "tj", c.tj_axis(), tj);
    const std::size_t nt = c.tj_axis().size();
    auto knot = [&](std::size_t k) {   // unclamped V_on at current knot k
        return c.v_on()[k * nt + at.lo] * (1.0 - at.w) + c.v_on()[k * nt + at.hi] * at.w;
    };
    Real a = knot(0), b = 0.0;
    if (ia.size() == 2) {
        b = (knot(1) - knot(0)) / (ia[1] - ia[0]);
        a = knot(0) - b * ia[0];
    }
    if (a < 0.0 || a + b * i_max < 0.0) {
        throw std::invalid_argument(
            "the conduction table clamps V_on at 0 inside the device's current range");
    }
    return (a * q1 + b * q2) / period;
}

/// One device's loss on a fixed orbit as a function of T_j. The pointers
/// are borrowed and must outlive the call.
struct DevicePower {
    std::vector<std::array<Real, 2>> on_pts, off_pts;   // (|v|, |i|), reading rule applied
    const thermal::LossTable3D* on = nullptr;
    const thermal::LossTable3D* off = nullptr;
    const thermal::ConductionTable* cond = nullptr;
    Real q1 = 0, q2 = 0, i_max = 0;   // int|i| dt, int i^2 dt, max|i| (on-state current)
    Real p_fixed = 0;                 // W that does not depend on T_j
    Real period = 1;

    /// (P, E_on, E_off, P_cond_table) at T_j.
    [[nodiscard]] std::array<Real, 4> at(Real tj) const {
        const Real e_on = table_total(on, on_pts, tj);
        const Real e_off = table_total(off, off_pts, tj);
        const Real pc = cond ? conduction_power(*cond, q1, q2, i_max, tj, period) : 0.0;
        // record order: _psum([conduction, switching]) is a plain sum of two
        return {(p_fixed + pc) + (e_on + e_off) / period, e_on, e_off, pc};
    }

    /// The T_j knots of every table: P is linear in T_j between them
    /// (but for a table entry that extrapolates through 0 and clamps).
    [[nodiscard]] std::vector<Real> knots() const {
        std::vector<Real> k;
        for (const auto* t : {on, off}) {
            if (t) k.insert(k.end(), t->tj_axis().begin(), t->tj_axis().end());
        }
        if (cond) k.insert(k.end(), cond->tj_axis().begin(), cond->tj_axis().end());
        std::sort(k.begin(), k.end());
        k.erase(std::unique(k.begin(), k.end()), k.end());
        return k;
    }
};

struct Junction {
    bool runaway = false;
    Real tj = 0;          // the fixed point (runaway: where the search stopped)
    Real p = 0;           // P(T_j)
    Real residual = 0;    // T_j - T_amb - R_th P(T_j), K
    Real slope = 0;       // R_th dP/dT_j at T_j (secant of the last bracket)
    int evaluations = 0;  // evaluations of P
};

/// The smallest T_j >= t_amb with g(T_j) = T_j - t_amb - r_th P(T_j) = 0.
/// g(t_amb) = -r_th P <= 0. Segment by segment between the tables' T_j
/// knots: the first segment whose end has g >= 0 brackets the root,
/// which regula falsi (Illinois) finds -- in one step when g is linear
/// there. Above the last knot P is linear, so the search steps to twice
/// the linear root; it stops as a runaway when g does not rise
/// (r_th dP/dT >= 1) or the bracket passes 1e6 C.
/// shortcut: a segment is tested at its ends only, so a root inside a
/// segment whose ends both have g < 0 is missed; that needs a table
/// entry falling with T_j through 0 (clamped) inside one segment.
inline Junction solve_junction(const DevicePower& d, Real r_th, Real t_amb,
                               Real tol = 1e-9) {
    if (!(r_th >= 0.0) || !std::isfinite(r_th) || !std::isfinite(t_amb)) {
        throw std::invalid_argument("solve_junction: r_th must be finite and >= 0, "
                                    "t_amb finite");
    }
    Junction out;
    auto g = [&](Real t) {
        ++out.evaluations;
        const Real p = d.at(t)[0];
        if (!std::isfinite(p)) throw std::domain_error("a non-finite loss at T_j");
        return t - t_amb - r_th * p;
    };
    auto finish = [&](Real t, Real gt, Real slope_g) {
        out.tj = t;
        out.residual = gt;
        out.p = d.at(t)[0];
        out.slope = 1.0 - slope_g;
        return out;
    };
    // Illinois regula falsi on [lo, hi], g(lo) < 0 <= g(hi)
    auto bracket = [&](Real lo, Real glo, Real hi, Real ghi) {
        int side = 0;
        Real x = hi, gx = ghi;
        bool done = false;
        for (int it = 0; it < 200; ++it) {
            if (ghi == 0.0) return finish(hi, 0.0, (ghi - glo) / (hi - lo));
            x = hi - ghi * (hi - lo) / (ghi - glo);
            gx = g(x);
            if (std::abs(gx) <= tol || hi - lo <= 1e-13 * std::max(Real{1}, std::abs(x))) {
                done = true;
                break;
            }
            if (gx < 0) {
                lo = x;
                glo = gx;
                if (side == -1) ghi *= 0.5;
                side = -1;
            } else {
                hi = x;
                ghi = gx;
                if (side == 1) glo *= 0.5;
                side = 1;
            }
        }
        if (!done) {
            throw std::runtime_error("solve_junction: the root was not located in 200 "
                                     "regula falsi steps");
        }
        return finish(x, gx, (ghi - glo) / (hi - lo));
    };
    Real a = t_amb, ga = g(a);
    if (ga >= 0.0) return finish(a, ga, 1.0);
    for (Real b : d.knots()) {
        if (b <= a) continue;
        const Real gb = g(b);
        if (gb >= 0.0) return bracket(a, ga, b, gb);
        a = b;
        ga = gb;
    }
    Real hi = a + std::max(Real{1}, std::abs(a));
    for (int it = 0; it < 200; ++it) {
        const Real ghi = g(hi);
        if (ghi >= 0.0) return bracket(a, ga, hi, ghi);
        const Real s = (ghi - ga) / (hi - a);
        // runaway iff g stops rising (r_th dP/dT_j >= 1); beyond the last
        // knot g is linear, so a rising g is bracketed by the next step
        if (!(s > 0.0)) {
            out.runaway = true;
            out.tj = hi;
            out.residual = ghi;
            out.slope = 1.0 - s;
            out.p = d.at(hi)[0];
            return out;
        }
        a = hi;
        ga = ghi;
        hi = a + std::max(Real{1}, -2.0 * ga / s);
    }
    throw std::runtime_error("solve_junction: no root bracketed in 200 expansions");
}

}  // namespace pulsim::dsed::period_thermal
