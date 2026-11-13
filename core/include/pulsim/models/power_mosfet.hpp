#pragma once

// =============================================================================
// Pulsim — datasheet power MOSFET (composite parameters)
// =============================================================================
//
// add-power-mosfet-datasheet-device, steps 1–2. The device is built by
// `CircuitBuilder::add_power_mosfet` as a COMPOSITE of kinds that already
// pass their tests — no new kernel device, no new history class, no new
// wrapper in the Newton refresh chain:
//
//     D ─ R_d ─ d' ─┬─ channel (d' → s')      G ─ R_g ─ g'
//                   ├─ C_ds(v_d's')           C_gd between g' and d'
//                   └─ body diode (s' → d')   C_gs between g' and s'
//     s' ─ L_s ─ S
//
// An internal node exists only when its element is non-zero. Sub-element
// names: `{name}` (Level-1 channel), `{name}.rd`, `{name}.rg`,
// `{name}.ls`, `{name}.cgs`, `{name}.cgd`, `{name}.cds`, `{name}_body`.
//
// C_gd and C_ds are either a number (a linear capacitor, stamped exactly
// as `add_capacitor` stamps it) or a `NonlinearCapacitor` charge law. C_gd
// is placed gate → drain, so its law sees u = v_gd, the VDMOS convention.
//
// Run-constant junction temperature (ΔT = T_j − 25 °C):
//     r(T) = 1 + rds_tc1·ΔT + rds_tc2·ΔT²,  R_d → r·R_d,  K → K/r,
//     V_T → V_T + vt_tc·ΔT,
// so the small-signal R_DS(on) = R_d + 1/(2K·V_OV) scales by exactly r.

#include "pulsim/models/lauritzen_diode.hpp"
#include "pulsim/models/nonlinear_capacitor.hpp"
#include "pulsim/models/shockley_diode.hpp"
#include "pulsim/numeric/types.hpp"

#include <cmath>
#include <format>
#include <stdexcept>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace pulsim::models {

struct PowerMosfet {
    /// A capacitance: a number [F] (linear; 0 = absent) or a charge law.
    using Capacitance = std::variant<Real, NonlinearCapacitor::Params>;

    enum class BodyDiode : int { None = 0, Shockley = 1, Lauritzen = 2 };

    struct Params {
        Real K      = Real{1};       //!< [A/V²] Level-1 transconductance at 25 °C
        Real V_T    = Real{3};       //!< [V] threshold at 25 °C
        Real lambda = Real{0};       //!< [1/V] channel-length modulation
        Real kappa  = Real{15};      //!< [1/V] Level-1 sigmoid sharpness
        Real R_d    = Real{0};       //!< [Ω] drift resistance at 25 °C
        Real C_gs   = Real{0};       //!< [F] constant (0 = absent)
        Capacitance C_gd = Real{0};  //!< Miller capacitance (law of v_gd)
        Capacitance C_ds = Real{0};  //!< output capacitance (law of v_ds)
        Real R_g    = Real{0};       //!< [Ω] internal gate resistance
        Real L_s    = Real{0};       //!< [H] common source inductance
        BodyDiode body = BodyDiode::None;
        ShockleyDiode::Params shockley{};
        LauritzenDiode::Params lauritzen{};
        Real T_j     = Real{25};     //!< [°C] run-constant junction temperature
        Real rds_tc1 = Real{0};      //!< [1/K]
        Real rds_tc2 = Real{0};      //!< [1/K²]
        Real vt_tc   = Real{0};      //!< [V/K]
    };

    [[nodiscard]] static Real r_of_T(const Params& p) noexcept {
        const Real dT = p.T_j - Real{25};
        return Real{1} + p.rds_tc1 * dT + p.rds_tc2 * dT * dT;
    }
    [[nodiscard]] static Real V_T_of_T(const Params& p) noexcept {
        return p.V_T + p.vt_tc * (p.T_j - Real{25});
    }

    /// Throws std::invalid_argument naming `name` and the parameter.
    static void validate(const Params& p, std::string_view name) {
        auto bad = [&](std::string_view what, Real v, std::string_view need) {
            throw std::invalid_argument(std::format(
                "add_power_mosfet(\"{}\"): {} = {} — {}.", name, what, v, need));
        };
        if (!(p.K > Real{0}) || !std::isfinite(p.K)) bad("K", p.K, "must be positive and finite");
        if (!std::isfinite(p.V_T)) bad("V_T", p.V_T, "must be finite");
        if (!(p.kappa > Real{0})) bad("kappa", p.kappa, "must be positive");
        if (!(p.R_d >= Real{0}) || !std::isfinite(p.R_d)) bad("R_d", p.R_d, "must be >= 0");
        if (!(p.R_g >= Real{0}) || !std::isfinite(p.R_g)) bad("R_g", p.R_g, "must be >= 0");
        if (!(p.L_s >= Real{0}) || !std::isfinite(p.L_s)) bad("L_s", p.L_s, "must be >= 0");
        if (!(p.C_gs >= Real{0}) || !std::isfinite(p.C_gs)) {
            bad("C_gs", p.C_gs, "must be >= 0 (0 = absent)");
        }
        auto check_cap = [&](const Capacitance& c, std::string_view what) {
            if (const Real* x = std::get_if<Real>(&c)) {
                if (!(*x >= Real{0}) || !std::isfinite(*x)) bad(what, *x, "must be >= 0 (0 = absent)");
                return;
            }
            try {
                NonlinearCapacitor::validate(std::get<NonlinearCapacitor::Params>(c));
            } catch (const std::invalid_argument& e) {
                throw std::invalid_argument(std::format(
                    "add_power_mosfet(\"{}\"): {}: {}", name, what, e.what()));
            }
        };
        check_cap(p.C_gd, "C_gd");
        check_cap(p.C_ds, "C_ds");
        // A gate with no capacitance at all has no charge to move: R_g
        // would do nothing and there is no Miller plateau. (C_gs = 0 alone
        // is allowed: an imported VDMOS card may carry TT or CJO only.)
        const Real* gd = std::get_if<Real>(&p.C_gd);
        if (p.R_g > Real{0} && p.C_gs == Real{0} && gd != nullptr && *gd == Real{0}) {
            throw std::invalid_argument(std::format(
                "add_power_mosfet(\"{}\"): R_g = {} with C_gs = 0 and no C_gd — "
                "the gate has no charge, so the gate resistance would have no "
                "effect. Give C_gs (and C_gd) or drop R_g.", name, p.R_g));
        }
        if (!std::isfinite(p.T_j)) bad("T_j", p.T_j, "must be finite");
        const Real r = r_of_T(p);
        if (!(r > Real{0}) || !std::isfinite(r)) {
            throw std::invalid_argument(std::format(
                "add_power_mosfet(\"{}\"): the temperature factor r(T_j = {} °C) "
                "= 1 + rds_tc1·ΔT + rds_tc2·ΔT² = {} must be positive "
                "(rds_tc1 = {}, rds_tc2 = {}).",
                name, p.T_j, r, p.rds_tc1, p.rds_tc2));
        }
        try {
            if (p.body == BodyDiode::Shockley) ShockleyDiode::validate(p.shockley);
            if (p.body == BodyDiode::Lauritzen) LauritzenDiode::validate(p.lauritzen);
        } catch (const std::invalid_argument& e) {
            throw std::invalid_argument(std::format(
                "add_power_mosfet(\"{}\"): body_diode: {}", name, e.what()));
        }
    }

    /// What `CircuitBuilder` records for introspection (`components()`).
    struct Record {
        std::string name;
        std::string drain, source, gate;  //!< external pins
        Params params;                    //!< as given (25 °C values)
        Real K_eff = 0, V_T_eff = 0, R_d_eff = 0;  //!< at T_j
        std::vector<std::string> sub_elements;
    };
};

}  // namespace pulsim::models
