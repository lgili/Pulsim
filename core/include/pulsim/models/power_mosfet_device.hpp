#pragma once

// =============================================================================
// Pulsim — the power MOSFET core as ONE f/q device (add-power-mosfet-
// datasheet-device task 5.2)
// =============================================================================
//
// The nonlinear part of `add_power_mosfet` — the channel (Level-1 law or a
// `ChannelTable`), the C_gd and C_ds charge laws and the Shockley or
// Lauritzen body diode — is one device on the f/q contract (the external
// device ABI's row convention, run in-process: `xdev::Model` with no
// library). Pins {d', s', g'}; with a Lauritzen body diode one internal
// row, its stored charge q_M. Rows: each pin's KCL, f = current leaving the
// node into the device, q = charge; the integrator owns d/dt, so charge is
// conserved whatever C(v) does. The law of each element is the one its
// composite kind used (device/laws.hpp bodies), so the device's Newton
// matrix is the composite's entry by entry (up to the order of the sums).
// The Jacobian is forward AD of the same expressions the residual
// evaluates. R_d, R_g, L_s and C_gs stay the MNA's own linear stamps; a
// constant C_gd / C_ds is a linear charge INSIDE the device, so the
// device's drain-pin current (f + dq/dt) is the whole current at d'.

#include "pulsim/ad/ad_scalar.hpp"
#include "pulsim/device/laws.hpp"
#include "pulsim/models/channel_table.hpp"
#include "pulsim/models/lauritzen_diode.hpp"
#include "pulsim/models/mosfet_level1.hpp"
#include "pulsim/models/nonlinear_capacitor.hpp"
#include "pulsim/models/power_mosfet.hpp"
#include "pulsim/models/shockley_diode.hpp"
#include "pulsim/xdev/host.hpp"

#include <array>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <vector>

namespace pulsim::models::power_mosfet_device {

/// The structural pattern (row, col) of df and dq.
struct Pattern {
    std::vector<std::uint32_t> fr, fc, qr, qc;
};

/// Everything the law reads, at T_j (K_eff, V_T_eff).
struct Core {
    device::laws::MosfetL1Params<Real> ch{};
    std::shared_ptr<const ChannelTable> table;         //!< replaces the Level-1 law
    std::optional<NonlinearCapacitor::Params> gd, ds;  //!< charge laws (u = v_gd, v_ds)
    Real c_gd = 0, c_ds = 0;                           //!< constant capacitances [F]
    PowerMosfet::BodyDiode body = PowerMosfet::BodyDiode::None;
    device::laws::ShockleyParams<Real> sh{};
    device::laws::LauritzenParams<Real> la{};
    Pattern pat;   //!< set by make_model

    [[nodiscard]] std::size_t rows() const noexcept {
        return body == PowerMosfet::BodyDiode::Lauritzen ? 4 : 3;
    }
};

enum Row : std::uint32_t { D = 0, S = 1, G = 2, QM = 3 };
using T = ad::ADRealN<4>;

/// f, q of every row and their dense Jacobians at x (rows() entries used).
struct Dense {
    std::array<Real, 4> f{}, q{};
    std::array<std::array<Real, 4>, 4> df{}, dq{};
};

inline Dense evaluate(const Core& c, const Real* x) {
    const std::size_t R = c.rows();
    std::array<T, 4> v{};
    for (std::size_t k = 0; k < R; ++k) {
        std::array<Real, 4> s{};
        s[k] = Real{1};
        v[k] = T(x[k], s);
    }
    std::array<T, 4> f{}, q{};
    struct O2 { std::array<T, 3> f{}, q{}; };
    // channel d' → s'
    if (c.table) {
        const Real vgs = x[G] - x[S], vds = x[D] - x[S];
        const auto e = c.table->current(vgs, vds);
        const T i = e.i + e.di_dvgs * ((v[G] - v[S]) - vgs) + e.di_dvds * ((v[D] - v[S]) - vds);
        f[D] += i;
        f[S] -= i;
    } else {
        const T vv[3] = {v[D], v[S], v[G]};
        const T i = models::MosfetLevel1::current<T>(vv, c.ch);
        f[D] += i;
        f[S] -= i;
    }
    // charge laws: C_gd placed gate → drain (u = v_gd), C_ds drain → source
    auto charge = [&](const NonlinearCapacitor::Params& m, Row a, Row b) {
        O2 o;
        device::laws::NonlinearCapacitor::eval(device::laws::NonlinearCapacitor::aux_from(m),
                                               device::laws::NonlinearCapacitor::from(m),
                                               std::array<T, 2>{v[a], v[b]}, o);
        q[a] += o.q[0];
        q[b] += o.q[1];
    };
    if (c.gd) charge(*c.gd, G, D);
    if (c.ds) charge(*c.ds, D, S);
    const T qgd = c.c_gd * (v[G] - v[D]), qds = c.c_ds * (v[D] - v[S]);
    q[G] += qgd; q[D] += qds - qgd; q[S] -= qds;
    // body diode s' (anode) → d' (cathode)
    if (c.body == PowerMosfet::BodyDiode::Shockley) {
        O2 o;
        device::laws::ShockleyDiode::eval(c.sh, std::array<T, 2>{v[S], v[D]}, o);
        f[S] += o.f[0];
        f[D] += o.f[1];
    } else if (c.body == PowerMosfet::BodyDiode::Lauritzen) {
        O2 o;
        device::laws::LauritzenDiode::eval(c.la, std::array<T, 3>{v[S], v[D], v[QM]}, o);
        f[S] += o.f[0];
        f[D] += o.f[1];
        f[QM] += o.f[2];
        q[QM] += o.q[2];
    }
    Dense out;
    for (std::size_t r = 0; r < R; ++r) {
        out.f[r] = f[r].value();
        out.q[r] = q[r].value();
        for (std::size_t k = 0; k < R; ++k) {
            out.df[r][k] = f[r].deriv(k);
            out.dq[r][k] = q[r].deriv(k);
        }
    }
    return out;
}

inline Pattern pattern(const Core& c) {
    std::array<std::array<bool, 4>, 4> F{}, Q{};
    for (Row r : {D, S}) for (Row k : {D, S, G}) F[r][k] = true;    // channel (+ body diode)
    if (c.gd || c.c_gd != Real{0}) for (Row r : {G, D}) for (Row k : {G, D}) Q[r][k] = true;
    if (c.ds || c.c_ds != Real{0}) for (Row r : {D, S}) for (Row k : {D, S}) Q[r][k] = true;
    if (c.body == PowerMosfet::BodyDiode::Lauritzen) {
        for (Row r : {D, S, QM}) for (Row k : {D, S, QM}) F[r][k] = true;
        Q[QM][QM] = true;
    }
    Pattern p;
    for (std::uint32_t r = 0; r < c.rows(); ++r) {
        for (std::uint32_t k = 0; k < c.rows(); ++k) {
            if (F[r][k]) { p.fr.push_back(r); p.fc.push_back(k); }
            if (Q[r][k]) { p.qr.push_back(r); p.qc.push_back(k); }
        }
    }
    return p;
}

/// The Core of a power MOSFET record (at T_j): the builder and the YAML
/// reader build the device from this one function.
inline Core core_of(const PowerMosfet::Record& rec) {
    using PM = PowerMosfet;
    const auto& p = rec.params;
    Core c;
    c.ch = {rec.K_eff, rec.V_T_eff, p.lambda, p.kappa};
    c.table = p.channel;
    c.body = p.body;
    if (p.body == PM::BodyDiode::Shockley) {
        const auto& d = p.shockley;
        c.sh = device::laws::ShockleyDiode::from(ShockleyDiode::Params{
            .I_S = d.I_S, .n = d.n, .V_T = d.V_T, .G_min = d.G_min, .BV = d.BV});
    } else if (p.body == PM::BodyDiode::Lauritzen) {
        const auto& d = p.lauritzen;
        c.la = device::laws::LauritzenDiode::from(LauritzenDiode::Params{
            .I_S = d.I_S, .n = d.n, .V_T = d.V_T, .tau = d.tau, .T_M = d.T_M,
            .G_min = d.G_min});
    }
    auto put = [](const PM::Capacitance& x, std::optional<NonlinearCapacitor::Params>& law,
                  Real& C) {
        if (const Real* v = std::get_if<Real>(&x)) C = *v;
        else law = std::get<NonlinearCapacitor::Params>(x);
    };
    put(p.C_gd, c.gd, c.c_gd);
    put(p.C_ds, c.ds, c.c_ds);
    return c;
}

// ---- the in-process model: one descriptor per device, the Core behind create()
namespace detail {
inline std::mutex& mu() { static std::mutex m; return m; }
inline std::unordered_map<std::uint64_t, const Core*>& registry() {
    static std::unordered_map<std::uint64_t, const Core*> r;
    return r;
}
inline std::uint64_t& next_id() { static std::uint64_t n = 0; return n; }

inline int32_t PULSIM_XDEV_CALL create(const double* p, void** inst) {
    const std::lock_guard<std::mutex> lk(mu());
    const auto it = registry().find(static_cast<std::uint64_t>(p[0]));
    if (it == registry().end()) return 1;
    *inst = const_cast<Core*>(it->second);
    return PULSIM_XDEV_OK;
}

inline int32_t PULSIM_XDEV_CALL eval(const pulsim_xdev_ctx* ctx, const double* x, double* f,
                                     double* q, double* df, double* dq) {
    const Core& c = *static_cast<const Core*>(ctx->instance);
    const Dense d = evaluate(c, x);
    for (std::size_t r = 0; r < c.rows(); ++r) { f[r] = d.f[r]; q[r] = d.q[r]; }
    const Pattern& pt = c.pat;
    for (std::size_t k = 0; k < pt.fr.size(); ++k) df[k] = d.df[pt.fr[k]][pt.fc[k]];
    for (std::size_t k = 0; k < pt.qr.size(); ++k) dq[k] = d.dq[pt.qr[k]][pt.qc[k]];
    return PULSIM_XDEV_OK;
}
}  // namespace detail

/// Owns the Core for as long as any Instance of the model lives (the model
/// pointer handed out aliases this holder).
struct Holder {
    Core core;
    std::uint64_t id = 0;
    std::optional<xdev::Model> model;
    explicit Holder(Core c) : core(std::move(c)) {
        const std::lock_guard<std::mutex> lk(detail::mu());
        id = ++detail::next_id();
        detail::registry()[id] = &core;
    }
    Holder(const Holder&) = delete;
    Holder& operator=(const Holder&) = delete;
    ~Holder() {
        const std::lock_guard<std::mutex> lk(detail::mu());
        detail::registry().erase(id);
    }
};

/// The model of one device and its parameter vector ({handle}).
inline std::pair<std::shared_ptr<const xdev::Model>, std::vector<Real>> make_model(Core core) {
    core.pat = pattern(core);
    auto h = std::make_shared<Holder>(std::move(core));
    const Pattern& pt = h->core.pat;
    static const char* const kPins[3] = {"d", "s", "g"};
    static const pulsim_xdev_field kField{"handle", "", 0.0, 0.0, 9007199254740992.0};
    const std::size_t R = h->core.rows();
    std::vector<double> lo(R, -1e3), hi(R, 1e3);
    if (R == 4) { lo[QM] = -1e-6; hi[QM] = 1e-6; }
    pulsim_xdev_descriptor d{};
    d.abi_version = PULSIM_XDEV_ABI_VERSION;
    d.struct_size = sizeof d;
    d.name = "power_mosfet";
    d.n_pins = 3;
    d.n_internal = R == 4 ? 1 : 0;
    d.n_fields = 1;
    d.nnz_f = static_cast<std::uint32_t>(pt.fr.size());
    d.nnz_q = static_cast<std::uint32_t>(pt.qr.size());
    d.claims = PULSIM_XDEV_CLAIM_SMOOTH;
    d.pin_names = kPins;
    d.fields = &kField;
    d.f_rows = pt.fr.data();
    d.f_cols = pt.fc.data();
    d.q_rows = pt.qr.empty() ? nullptr : pt.qr.data();
    d.q_cols = pt.qc.empty() ? nullptr : pt.qc.data();
    d.probe_lo = lo.data();
    d.probe_hi = hi.data();
    d.eval = &detail::eval;
    d.create = &detail::create;
    h->model.emplace(nullptr, &d);   // the model copies the descriptor
    std::shared_ptr<const xdev::Model> m(h, &*h->model);
    return {std::move(m), {static_cast<Real>(h->id)}};
}

}  // namespace pulsim::models::power_mosfet_device
