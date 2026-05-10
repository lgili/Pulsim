#pragma once

#include "pulsim/v1/templates/registry.hpp"
#include "pulsim/v1/runtime_circuit.hpp"

#include <cmath>
#include <numbers>
#include <string>

namespace pulsim::v1::templates {

// =============================================================================
// add-converter-templates — buck (Phase 2.1)
// =============================================================================
//
// Synchronous buck converter:
//
//      Vin ──┬── Q1 ── L1 ──┬── Vout
//            │              │
//            └── D1 ──┬── C1 ── Rload
//                     gnd
//
// Auto-design heuristics:
//   - L1 sized for ≤ 30 % current ripple at full load:
//       L = (Vin - Vout) · D / (ΔI · fsw)
//     where D = Vout/Vin and ΔI = ripple_pct · Iout.
//   - C1 sized for ≤ 1 % output-voltage ripple:
//       C = ΔI / (8 · fsw · ΔV)
//     where ΔV = vout_ripple_pct · Vout.
//   - Rload = Vout / Iout.
//
// Parameters (all in SI; ratios as 0–1 fractions):
//   Required: Vin, Vout, Iout, fsw
//   Optional: ripple_pct (default 0.30),
//              vout_ripple_pct (default 0.01),
//              q_g_on (default 1e3 S),
//              q_g_off (default 1e-9 S)

[[nodiscard]] inline ConverterExpansion expand_buck(
    const ConverterParameters& params) {
    ConverterExpansion out;
    out.topology = "buck";

    const Real Vin  = require_param(params, "Vin",  "buck");
    const Real Vout = require_param(params, "Vout", "buck");
    const Real Iout = require_param(params, "Iout", "buck");
    const Real fsw  = require_param(params, "fsw",  "buck");

    if (!(Vin > Vout) || !(Vout > Real{0}) || !(fsw > Real{0}) ||
        !(Iout > Real{0})) {
        throw std::invalid_argument(
            "buck: require Vin > Vout > 0, Iout > 0, fsw > 0");
    }

    const Real ripple_pct = param_or(params, "ripple_pct",      Real{0.30});
    const Real vout_ripple_pct = param_or(params, "vout_ripple_pct", Real{0.01});

    // Auto-design.
    const Real D    = Vout / Vin;
    const Real dI   = ripple_pct * Iout;
    const Real L    = (Vin - Vout) * D / (dI * fsw);
    const Real dV   = vout_ripple_pct * Vout;
    const Real C    = dI / (Real{8} * fsw * dV);
    const Real R_load = Vout / Iout;

    // User overrides take precedence.
    const Real L_actual    = param_or(params, "L",       L);
    const Real C_actual    = param_or(params, "C",       C);
    const Real R_actual    = param_or(params, "Rload",   R_load);
    const Real q_g_on      = param_or(params, "q_g_on",  Real{1e3});
    const Real q_g_off     = param_or(params, "q_g_off", Real{1e-9});

    // Build the circuit.
    Circuit ckt;
    auto in   = ckt.add_node("in");
    auto sw_n = ckt.add_node("sw");
    auto out_n = ckt.add_node("out");
    auto ctrl = ckt.add_node("ctrl");

    ckt.add_voltage_source("Vin", in, Circuit::ground(), Vin);

    // PWM control source — duty pre-set to D so the open-loop output
    // settles at the target. Frequency is fsw.
    PWMParams pwm;
    pwm.frequency = fsw;
    pwm.duty      = D;
    pwm.v_high    = 5.0;
    pwm.v_low     = 0.0;
    pwm.phase     = 0.0;
    ckt.add_pwm_voltage_source("Vctrl", ctrl, Circuit::ground(), pwm);

    // High-side switch — voltage-controlled, gate from `ctrl`.
    ckt.add_vcswitch("Q1", ctrl, in, sw_n,
                     /*v_threshold*/2.5, q_g_on, q_g_off);
    // Free-wheeling diode from gnd to sw_n (anode to ground in this
    // convention so current flows when sw_n drops below 0).
    ckt.add_diode("D1", Circuit::ground(), sw_n, q_g_on, q_g_off);
    ckt.add_inductor("L1", sw_n, out_n, L_actual, /*i_init*/0.0);
    ckt.add_capacitor("C1", out_n, Circuit::ground(), C_actual, /*v_init*/0.0);
    ckt.add_resistor("Rload", out_n, Circuit::ground(), R_actual);

    out.circuit = std::move(ckt);
    out.resolved_parameters = {
        {"Vin", Vin}, {"Vout", Vout}, {"Iout", Iout}, {"fsw", fsw},
        {"ripple_pct", ripple_pct},
        {"vout_ripple_pct", vout_ripple_pct},
        {"D", D},
        {"L", L_actual}, {"C", C_actual}, {"Rload", R_actual},
        {"q_g_on", q_g_on}, {"q_g_off", q_g_off},
    };
    if (!params.count("L")) {
        out.design_notes["L"] = "Auto-sized for " +
            std::to_string(ripple_pct * 100.0) + " % current ripple at full load";
    }
    if (!params.count("C")) {
        out.design_notes["C"] = "Auto-sized for " +
            std::to_string(vout_ripple_pct * 100.0) + " % output-voltage ripple";
    }
    if (!params.count("Rload")) {
        out.design_notes["Rload"] = "Auto-sized for nominal Iout = " +
            std::to_string(Iout) + " A";
    }
    return out;
}

/// Static registrar — pulls the expander into the global registry at
/// translation-unit load time. Use `register_buck_template()` from a
/// .cpp / test to ensure the registration runs even when the header is
/// only included indirectly.
inline void register_buck_template() {
    static const bool _ = [] {
        ConverterRegistry::instance().register_template("buck", expand_buck);
        return true;
    }();
    (void)_;
}

}  // namespace pulsim::v1::templates
