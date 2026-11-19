#pragma once

// =============================================================================
// Pulsim — external devices as one stateful Newton kind
// =============================================================================
//
// add-external-device-abi. Every external device of a run, as one entry of
// `StatefulKinds`, so the time loops reach it through the same folds as the
// built-in kinds (no loop edit). Its storage IS the flat device-state block:
// the kind's `entries()` is a vector of reals (one "entry" per real), and
// each device keeps an offset into it — per row q_n, q̇_n, q_γ, then the
// model's n_state reals (DECISIONS #32's offset view, for this kind).
//
// The stamp is the device contract's `stamp_fq` at run-time arity: the same
// time-formula functions (`device::formula_a`, `device::history_term`) and
// the same row rules (pin row: committed q̇ from the formula; internal row:
// q̇_n ← −f). An external device has no hand stamp, so it runs through its
// law on BOTH device paths ('legacy' routes it through the law too; the
// rest of the circuit follows the path asked for).

#include "pulsim/device/device.hpp"
#include "pulsim/device/device_state.hpp"
#include "pulsim/numeric/types.hpp"
#include "pulsim/pwl/device_pool.hpp"
#include "pulsim/pwl/trbdf2_stage.hpp"
#include "pulsim/sparse/matrix.hpp"
#include "pulsim/stamping/branch_coord.hpp"
#include "pulsim/topology/graph.hpp"
#include "pulsim/xdev/host.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace pulsim::pwl {

class ExternalDeviceHistory {
public:
    using Entry = Real;
    static constexpr device::StateBlock kBlock = device::StateBlock::ExternalDevice;
    static constexpr std::size_t kFlatPerDevice = 1;
    static constexpr DevicePool::StoredKind kStoredKind = DevicePool::StoredKind::ExternalDevice;

    struct Device {
        std::shared_ptr<const xdev::Instance> inst;
        std::vector<Index> at;   ///< pins: node index; internal rows: unknown index
        std::size_t off = 0;     ///< into cells_: q_n[R], q̇_n[R], q_γ[R], state[n_state]
        mutable xdev::EvalOut o;
        mutable std::vector<Real> x, scratch;
        /// Each pattern entry's offset into J_nl's value array, validated
        /// before every use (-1: not resolved yet).
        mutable std::vector<std::ptrdiff_t> fslot, qslot;
    };

    void init(const topology::Graph& graph, const DevicePool& pool) {
        devices_.clear();
        cells_.clear();
        for (const Index b : pool.external_device_branches()) {
            const auto& p = pool.external_device_params(b);
            Device d;
            d.inst = p.inst;
            d.at = p.pins;
            const std::size_t ni = p.inst->model().d().n_internal;
            for (std::size_t k = 0; k < ni; ++k) {
                d.at.push_back(pool.branch_var_id_for_inductor(p.branches[k], graph));
            }
            const std::size_t R = d.inst->rows();
            d.off = cells_.size();
            p.inst->size(d.o);
            d.x.assign(R, Real{0});
            d.scratch.assign(d.inst->n_state() + 1, Real{0});
            d.fslot.assign(d.inst->model().d().nnz_f, -1);
            d.qslot.assign(d.inst->model().d().nnz_q, -1);
            cells_.resize(cells_.size() + 3 * R + d.inst->n_state(), Real{0});
            devices_.push_back(std::move(d));
        }
        // consistent start at x = 0: q_n = q(0), model state from init(0)
        for (auto& d : devices_) start_(d, nullptr);
    }

    [[nodiscard]] bool empty() const noexcept { return devices_.empty(); }
    [[nodiscard]] const std::vector<Real>& entries() const noexcept { return cells_; }
    [[nodiscard]] std::vector<Real>& mutable_entries() noexcept { return cells_; }
    [[nodiscard]] const std::vector<Device>& devices() const noexcept { return devices_; }
    template <class E, class F> static void for_each_state(E& e, F&& f) { f(e); }

    void seed_from_dc_op(const Vector& x) {
        for (auto& d : devices_) start_(d, &x);
    }

    /// The Newton contribution of every device under (h, stage).
    Real stamp(const Vector& x, sparse::Matrix& J_nl, Vector& f_nl, Real h,
               TrBdf2Stage st) const {
        const device::StepCtx c{Real{0}, formula_of_(st), h};
        const Real a = device::formula_a(c);
        Real worst = Real{0};
        for (const auto& d : devices_) {
            const std::size_t R = d.inst->rows(), NP = d.inst->model().d().n_pins;
            gather_(d, x);
            d.inst->eval(d.x.data(), state_(d), d.o);
            const auto& md = d.inst->model().d();
            const Real* qn = &cells_[d.off];
            const Real* qd = qn + R;
            const Real* qg = qd + R;
            auto on = [&](std::size_t r) {
                return r >= NP || stamping::node_is_active(d.at[r]);
            };
            for (std::size_t r = 0; r < R; ++r) {
                // F = f + a·q + r, the operation order of device::stamp_fq
                const Real F = d.o.f[r] + a * d.o.q[r]
                               + device::history_term(c, qn[r], qd[r], qg[r]);
                if (r < NP) worst = std::max(worst, std::abs(F));   // the device current, as the built-in stamps return
                if (on(r)) f_nl[d.at[r]] += F;
            }
            for (std::uint32_t k = 0; k < md.nnz_f; ++k) {
                const auto r = md.f_rows[k], col = md.f_cols[k];
                if (on(r) && on(col)) slot_(J_nl, d.fslot[k], d.at[r], d.at[col]) += d.o.df[k];
            }
            if (a != Real{0}) {
                for (std::uint32_t k = 0; k < md.nnz_q; ++k) {
                    const auto r = md.q_rows[k], col = md.q_cols[k];
                    if (on(r) && on(col)) slot_(J_nl, d.qslot[k], d.at[r], d.at[col]) += a * d.o.dq[k];
                }
            }
        }
        return worst;
    }

    /// J(r, c) through a cached value offset: valid only while the slot
    /// still holds (r, c) in the matrix's CURRENT pattern (checked: inside
    /// column c's stored range and row index r), else coeffRef resolves it
    /// again. Any insertion, reallocation or rebuild is therefore safe.
    static Real& slot_(sparse::Matrix& J, std::ptrdiff_t& k, Index r, Index c) {
        static_assert(!sparse::Matrix::IsRowMajor);
        if (k >= 0) {
            const auto beg = J.outerIndexPtr()[c];
            const auto end = J.innerNonZeroPtr() ? beg + J.innerNonZeroPtr()[c]
                                                 : J.outerIndexPtr()[c + 1];
            if (k >= beg && k < end && J.innerIndexPtr()[k] == r) return J.valuePtr()[k];
        }
        Real& v = J.coeffRef(r, c);
        k = &v - J.valuePtr();
        return v;
    }

    void commit(const Vector& x, Real h, TrBdf2Stage st) {
        const device::StepCtx c{Real{0}, formula_of_(st), h};
        const Real a = device::formula_a(c);
        for (auto& d : devices_) {
            const std::size_t R = d.inst->rows(), NP = d.inst->model().d().n_pins;
            gather_(d, x);
            d.inst->eval(d.x.data(), state_(d), d.o);
            Real* qn = &cells_[d.off];
            Real* qd = qn + R;
            const Real* qg = qd + R;
            for (std::size_t r = 0; r < R; ++r) {
                const Real dot = r >= NP ? -d.o.f[r]
                                         : a * d.o.q[r] + device::history_term(c, qn[r], qd[r], qg[r]);
                qn[r] = d.o.q[r];
                qd[r] = dot;
            }
            accept_(d, h);
        }
    }

    void capture_gamma(const Vector& xg) {
        for (auto& d : devices_) {
            const std::size_t R = d.inst->rows();
            gather_(d, xg);
            d.inst->eval(d.x.data(), state_(d), d.o);
            std::copy(d.o.q.begin(), d.o.q.end(), cells_.begin() + static_cast<std::ptrdiff_t>(d.off + 2 * R));
        }
    }

    /// SDIRK restart: q_n ← q_n + c·(q(x1) − q_n).
    void extrapolate_previous(const Vector& x1, Real c) {
        for (auto& d : devices_) {
            gather_(d, x1);
            d.inst->eval(d.x.data(), state_(d), d.o);
            for (std::size_t r = 0; r < d.inst->rows(); ++r) {
                cells_[d.off + r] += c * (d.o.q[r] - cells_[d.off + r]);
            }
        }
    }

    /// TR-BDF2 post-event refresh of the derivative slots from a
    /// trapezoidal probe of step h: internal row −f(x_p) (the saturable
    /// inductor's rule), pin row the trapezoidal derivative of the probe.
    void refresh_derivatives(const Vector& x_probe, Real h) {
        for (auto& d : devices_) {
            const std::size_t R = d.inst->rows(), NP = d.inst->model().d().n_pins;
            gather_(d, x_probe);
            d.inst->eval(d.x.data(), state_(d), d.o);
            for (std::size_t r = 0; r < R; ++r) {
                Real& dot = cells_[d.off + R + r];
                if (r >= NP) dot = -d.o.f[r];
                else if (h > Real{0}) dot = Real{2} / h * (d.o.q[r] - cells_[d.off + r]) - dot;
            }
        }
    }

    [[nodiscard]] std::vector<Real> to_flat() const { return cells_; }
    void from_flat(const std::vector<Real>& flat) {
        if (flat.size() != cells_.size()) {
            throw std::invalid_argument(
                "ExternalDeviceHistory::from_flat: expected " + std::to_string(cells_.size())
                + " reals for this circuit's external devices but got "
                + std::to_string(flat.size()) + " — the snapshot belongs to a different circuit");
        }
        cells_ = flat;
    }

private:
    [[nodiscard]] static device::Formula formula_of_(TrBdf2Stage s) noexcept {
        switch (s) {
        case TrBdf2Stage::BackwardEuler: return device::Formula::BackwardEuler;
        case TrBdf2Stage::Bdf2Stage2:    return device::Formula::Bdf2Stage2;
        default:                         return device::Formula::Trapezoidal;
        }
    }
    void gather_(const Device& d, const Vector& x) const {
        const std::size_t NP = d.inst->model().d().n_pins;
        for (std::size_t r = 0; r < d.at.size(); ++r) {
            d.x[r] = r < NP ? stamping::read_node_voltage(x, d.at[r]) : x[d.at[r]];
        }
    }
    [[nodiscard]] const Real* state_(const Device& d) const {
        return cells_.data() + d.off + 3 * d.inst->rows();
    }
    void start_(Device& d, const Vector* x) {
        const std::size_t R = d.inst->rows(), NP = d.inst->model().d().n_pins;
        if (x) gather_(d, *x);
        else std::fill(d.x.begin(), d.x.end(), Real{0});
        const auto& md = d.inst->model().d();
        if (md.init && md.n_state > 0) {
            guarded_(d, [&](const pulsim_xdev_ctx& c, Real* s) {
                return md.init(&c, d.x.data(), s);
            }, "init");
        }
        d.inst->eval(d.x.data(), state_(d), d.o);
        for (std::size_t r = 0; r < R; ++r) {
            cells_[d.off + r] = d.o.q[r];
            cells_[d.off + R + r] = r >= NP ? -d.o.f[r] : Real{0};
            cells_[d.off + 2 * R + r] = d.o.q[r];
        }
    }
    void accept_(Device& d, Real h) {
        const auto& md = d.inst->model().d();
        if (!md.accept || md.n_state == 0) return;
        guarded_(d, [&](const pulsim_xdev_ctx& c, Real* s) {
            return md.accept(&c, d.x.data(), h, s);
        }, "accept");
    }
    /// Run init / accept on a copy with a guard real past n_state; a write
    /// past the declared size or a non-finite state is refused by name, and
    /// nothing reaches the flat state unless it passed.
    template <class Fn>
    void guarded_(Device& d, Fn&& fn, const char* what) {
        const std::size_t NS = d.inst->n_state();
        Real* st = cells_.data() + d.off + 3 * d.inst->rows();
        std::copy(st, st + NS, d.scratch.begin());
        const Real guard = Real{-7.25e-301};
        d.scratch[NS] = guard;
        const pulsim_xdev_ctx c = d.inst->ctx_(st);
        if (fn(c, d.scratch.data()) != PULSIM_XDEV_OK) {
            d.inst->refuse_(std::string(what) + " returned an error status");
        }
        if (std::memcmp(&d.scratch[NS], &guard, sizeof guard) != 0) {
            d.inst->refuse_(std::string(what) + " wrote past the declared n_state = "
                            + std::to_string(NS));
        }
        for (std::size_t k = 0; k < NS; ++k) {
            if (!std::isfinite(d.scratch[k])) {
                d.inst->refuse_(std::string(what) + " left state[" + std::to_string(k)
                                + "] non-finite");
            }
        }
        std::copy(d.scratch.begin(), d.scratch.begin() + static_cast<std::ptrdiff_t>(NS), st);
    }

    std::vector<Device> devices_;
    std::vector<Real> cells_;
};

}  // namespace pulsim::pwl
