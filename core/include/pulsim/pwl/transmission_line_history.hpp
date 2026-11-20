#pragma once

// =============================================================================
// Pulsim — the committed past of the run's Bergeron lines (D18)
// =============================================================================
//
// openspec add-realistic-passives section 5, DECISIONS D18. Each port of a
// lossless line is v − Z0·i = E(t) (models/transmission_line.hpp) with
// E_a(t) = w_b(t − tau), E_b(t) = w_a(t − tau) and w = v + Z0·i the wave a
// port sends. This object keeps, per line, the committed samples
// (t, w_a, w_b) and adds −E to the two port rows of a solve's b.
//
// Rules (D18 (1), (2)):
// - READ on every solve (`inject`): nominal steps, retries, restart stages,
//   event iterations, TR-BDF2 stages, rejected attempts, probes. WRITTEN
//   only by `commit`, which the engines call once per committed time; a
//   commit at a time not after the last sample is ignored (a hook that
//   repeats the step's commit), so the samples are strictly increasing.
// - Lookup by TIME (binary search), never by index: the committed grid is
//   not uniform (resume offsets, TR-BDF2). A query within 64 eps of a
//   sample time returns that sample exactly (tau = k·dt on the fixed grid:
//   no interpolation, the line is then exact to rounding); otherwise the
//   linear interpolant of the bracketing samples (second order, never
//   amplifying). Before the first sample: the first sample (the line's
//   past is constant: the DC point, or at rest).
// - Bounded: samples older than the last one at or before t − tau are
//   dropped (compacted in place, no allocation once the capacity holds).

#include "pulsim/numeric/types.hpp"
#include "pulsim/pwl/device_pool.hpp"
#include "pulsim/stamping/branch_coord.hpp"
#include "pulsim/topology/graph.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace pulsim::pwl {

class LineHistory {
public:
    LineHistory(const topology::Graph& graph, const DevicePool& pool) {
        for (const auto& lp : pool.line_pairs()) {
            Line l;
            const auto& p = pool.transmission_line_params(lp.a);
            l.Z0 = p.Z0;
            l.tau = p.delay;
            l.name = std::string(graph.branch_name(lp.a));
            const auto& ba = graph.branch(lp.a);
            const auto& bb = graph.branch(lp.b);
            l.af = ba.from; l.at = ba.to;
            l.bf = bb.from; l.bt = bb.to;
            l.ra = pool.branch_var_id_for_source(lp.a, graph);
            l.rb = pool.branch_var_id_for_source(lp.b, graph);
            lines_.push_back(std::move(l));
        }
    }

    [[nodiscard]] bool empty() const noexcept { return lines_.empty(); }
    [[nodiscard]] Size size() const noexcept { return lines_.size(); }

    /// The shortest delay of the run and the line that has it.
    [[nodiscard]] Real tau_min() const noexcept {
        Real m = std::numeric_limits<Real>::infinity();
        for (const auto& l : lines_) m = std::min(m, l.tau);
        return m;
    }
    [[nodiscard]] const std::string& tau_min_name() const {
        const auto it = std::min_element(lines_.begin(), lines_.end(),
            [](const Line& a, const Line& b) { return a.tau < b.tau; });
        return it->name;
    }

    /// The run's start: the past is constant. `consistent` (a DC operating
    /// point or another solved state): the waves of `x`; otherwise the line
    /// is at rest (zero waves).
    void start(Real t0, const Vector& x, bool consistent) {
        for (auto& l : lines_) {
            l.t.clear(); l.wa.clear(); l.wb.clear(); l.head = 0;
            l.t.push_back(t0);
            l.wa.push_back(consistent ? wave_(x, l.af, l.at, l.ra, l.Z0) : Real{0});
            l.wb.push_back(consistent ? wave_(x, l.bf, l.bt, l.rb, l.Z0) : Real{0});
        }
    }

    /// b += the port rows' sources at solve time `t` (J x + b = 0, so the
    /// row v − Z0·i − E = 0 holds −E).
    void inject(Vector& b, Real t) const {
        for (const auto& l : lines_) {
            Real wa_d, wb_d;
            l.at_time(t - l.tau, t, wa_d, wb_d);
            b[l.ra] -= wb_d;
            b[l.rb] -= wa_d;
        }
    }

    /// Append the committed state `x` at time `t` (ignored at or before
    /// the last sample), then drop what no later query can reach.
    void commit(Real t, const Vector& x) {
        for (auto& l : lines_) {
            if (!(t > l.t.back())) continue;
            l.t.push_back(t);
            l.wa.push_back(wave_(x, l.af, l.at, l.ra, l.Z0));
            l.wb.push_back(wave_(x, l.bf, l.bt, l.rb, l.Z0));
            l.prune(t - l.tau);
            if (l.t.size() - l.head > (Size{1} << 20)) {
                throw std::runtime_error(
                    "LineHistory: the transmission line '" + l.name + "' holds "
                    "more than 2^20 samples of its delay (steps far shorter "
                    "than the delay); raise the minimum step or shorten the line.");
            }
        }
    }

    /// Snapshot: per line Z0, tau, n, then n × (t, w_a, w_b).
    [[nodiscard]] std::vector<Real> save() const {
        std::vector<Real> s;
        s.push_back(static_cast<Real>(lines_.size()));
        for (const auto& l : lines_) {
            const Size n = l.t.size() - l.head;
            s.insert(s.end(), {l.Z0, l.tau, static_cast<Real>(n)});
            for (Size k = l.head; k < l.t.size(); ++k) {
                s.insert(s.end(), {l.t[k], l.wa[k], l.wb[k]});
            }
        }
        return s;
    }

    /// Resume: refuses, naming the line, a snapshot whose lines are not
    /// this circuit's (count, Z0, tau) or that carries no line state.
    void restore(const std::vector<Real>& s, const char* engine) {
        auto refuse = [&](const std::string& why) {
            throw std::invalid_argument(
                std::string(engine) + "(resume_from=...): " + why
                + " Resuming would start the travelling waves from a wrong past.");
        };
        if (s.empty()) {
            refuse("the snapshot carries no transmission-line state, but this "
                   "circuit has the line '" + lines_.front().name + "'.");
        }
        if (static_cast<Size>(s[0]) != lines_.size()) {
            refuse("the snapshot holds " + std::to_string(static_cast<Size>(s[0]))
                   + " transmission lines, this circuit "
                   + std::to_string(lines_.size()) + ".");
        }
        Size p = 1;
        for (auto& l : lines_) {
            if (p + 3 > s.size() || s[p] != l.Z0 || s[p + 1] != l.tau) {
                refuse("the line '" + l.name + "' differs from the snapshot's "
                       "(Z0 or delay).");
            }
            const auto n = static_cast<Size>(s[p + 2]);
            p += 3;
            if (n == 0 || p + 3 * n > s.size()) refuse("the line state is truncated.");
            l.t.clear(); l.wa.clear(); l.wb.clear(); l.head = 0;
            for (Size k = 0; k < n; ++k, p += 3) {
                l.t.push_back(s[p]);
                l.wa.push_back(s[p + 1]);
                l.wb.push_back(s[p + 2]);
            }
        }
        if (p != s.size()) refuse("the line state has trailing values.");
    }

    /// Samples held for line `k` (tests: the buffer is bounded).
    [[nodiscard]] Size samples(Size k) const { return lines_[k].t.size() - lines_[k].head; }

private:
    struct Line {
        Real Z0 = 0, tau = 0;
        Index af = 0, at = 0, bf = 0, bt = 0, ra = 0, rb = 0;
        std::string name;
        std::vector<Real> t, wa, wb;
        Size head = 0;   // first live sample

        /// The waves at `tq` (`t_solve` = the solve's time, for the
        /// rounding scale and the refusal).
        void at_time(Real tq, Real t_solve, Real& wa_o, Real& wb_o) const {
            const auto first = t.begin() + static_cast<std::ptrdiff_t>(head);
            const auto it = std::upper_bound(first, t.end(), tq);   // first sample after tq
            const Real snap = Real{64} * std::numeric_limits<Real>::epsilon()
                * (std::abs(t_solve) + tau);
            if (it == first) { pick(head, wa_o, wb_o); return; }   // the constant past
            const auto j = static_cast<Size>(it - t.begin());     // tq in [t[j-1], t[j])
            if (tq - t[j - 1] <= snap || j == t.size()) {
                if (j == t.size() && tq - t[j - 1] > snap) {
                    throw std::logic_error(
                        "LineHistory: the line '" + name + "' was read at "
                        + std::to_string(tq) + " s, after its last committed "
                        "sample (" + std::to_string(t[j - 1])
                        + " s): a step longer than the delay reached the engine");
                }
                pick(j - 1, wa_o, wb_o);
                return;
            }
            if (t[j] - tq <= snap) {
                Size k = j;   // the right limit of a repeated time
                while (k + 1 < t.size() && t[k + 1] == t[k]) ++k;
                pick(k, wa_o, wb_o);
                return;
            }
            const Real th = (tq - t[j - 1]) / (t[j] - t[j - 1]);
            wa_o = wa[j - 1] + th * (wa[j] - wa[j - 1]);
            wb_o = wb[j - 1] + th * (wb[j] - wb[j - 1]);
        }
        void pick(Size k, Real& wa_o, Real& wb_o) const { wa_o = wa[k]; wb_o = wb[k]; }

        /// Keep from the last sample at or before `t_old` (the earliest
        /// time a later solve reads, minus nothing: solves move forward).
        void prune(Real t_old) {
            Size k = head;
            while (k + 1 < t.size() && t[k + 1] <= t_old) ++k;
            head = k;
            if (head > 64 && 2 * head > t.size()) {
                const auto h = static_cast<std::ptrdiff_t>(head);
                t.erase(t.begin(), t.begin() + h);
                wa.erase(wa.begin(), wa.begin() + h);
                wb.erase(wb.begin(), wb.begin() + h);
                head = 0;
            }
        }
    };

    static Real wave_(const Vector& x, Index f, Index to, Index row, Real Z0) {
        return stamping::read_node_voltage(x, f) - stamping::read_node_voltage(x, to)
            + Z0 * x[row];
    }

    std::vector<Line> lines_;
};

}  // namespace pulsim::pwl
