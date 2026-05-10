#pragma once

#include "pulsim/v1/templates/registry.hpp"
#include "pulsim/v1/runtime_circuit.hpp"

#include <cmath>
#include <string>

namespace pulsim::v1::templates {

// =============================================================================
// add-converter-templates — boost (Phase 2.2)
// =============================================================================
//
// Boost converter (Vout > Vin):
//
//      Vin ── L1 ──┬── D1 ──┬── Vout
//                  │        │
//                  Q1       C1 ── Rload
//                  │        │
//                  gnd      gnd
//
// Auto-design:
//   D = 1 - Vin/Vout
//   ΔI = ripple_pct · I_in_avg, where I_in_avg = Iout · (Vout/Vin)
//   L = Vin · D / (ΔI · fsw)
//   ΔV = vout_ripple_pct · Vout
//   C = Iout · D / (fsw · ΔV)
//   Rload = Vout / Iout

[[nodiscard]] inline ConverterExpansion expand_boost(
    const ConverterParameters& params) {
    ConverterExpansion out;
    out.topology = "boost";

    const Real Vin  = require_param(params, "Vin",  "boost");
    const Real Vout = require_param(params, "Vout", "boost");
    const Real Iout = require_param(params, "Iout", "boost");
    const Real fsw  = require_param(params, "fsw",  "boost");

    if (!(Vout > Vin) || !(Vin > Real{0}) || !(fsw > Real{0}) ||
        !(Iout > Real{0})) {
        throw std::invalid_argument(
            "boost: require Vout > Vin > 0, Iout > 0, fsw > 0");
    }

    const Real ripple_pct      = param_or(params, "ripple_pct",      Real{0.30});
    const Real vout_ripple_pct = param_or(params, "vout_ripple_pct", Real{0.01});

    const Real D       = Real{1} - Vin / Vout;
    const Real I_in    = Iout * (Vout / Vin);
    const Real dI      = ripple_pct * I_in;
    const Real L       = Vin * D / (dI * fsw);
    const Real dV      = vout_ripple_pct * Vout;
    const Real C       = Iout * D / (fsw * dV);
    const Real R_load  = Vout / Iout;

    const Real L_actual = param_or(params, "L",     L);
    const Real C_actual = param_or(params, "C",     C);
    const Real R_actual = param_or(params, "Rload", R_load);
    const Real q_g_on   = param_or(params, "q_g_on",  Real{1e3});
    const Real q_g_off  = param_or(params, "q_g_off", Real{1e-9});

    Circuit ckt;
    auto in    = ckt.add_node("in");
    auto sw_n  = ckt.add_node("sw");
    auto out_n = ckt.add_node("out");
    auto ctrl  = ckt.add_node("ctrl");

    ckt.add_voltage_source("Vin", in, Circuit::ground(), Vin);
    ckt.add_inductor("L1", in, sw_n, L_actual, 0.0);

    PWMParams pwm;
    pwm.frequency = fsw;
    pwm.duty      = D;
    pwm.v_high    = 5.0;
    pwm.v_low     = 0.0;
    ckt.add_pwm_voltage_source("Vctrl", ctrl, Circuit::ground(), pwm);

    // Low-side switch from sw_n to gnd.
    ckt.add_vcswitch("Q1", ctrl, sw_n, Circuit::ground(),
                     2.5, q_g_on, q_g_off);
    // Diode from sw_n (anode) to out_n (cathode).
    ckt.add_diode("D1", sw_n, out_n, q_g_on, q_g_off);
    ckt.add_capacitor("C1", out_n, Circuit::ground(), C_actual, 0.0);
    ckt.add_resistor("Rload", out_n, Circuit::ground(), R_actual);

    out.circuit = std::move(ckt);
    out.resolved_parameters = {
        {"Vin", Vin}, {"Vout", Vout}, {"Iout", Iout}, {"fsw", fsw},
        {"ripple_pct", ripple_pct}, {"vout_ripple_pct", vout_ripple_pct},
        {"D", D}, {"I_in", I_in},
        {"L", L_actual}, {"C", C_actual}, {"Rload", R_actual},
        {"q_g_on", q_g_on}, {"q_g_off", q_g_off},
    };
    if (!params.count("L")) out.design_notes["L"] = "Auto-sized for ripple_pct";
    if (!params.count("C")) out.design_notes["C"] = "Auto-sized for vout_ripple_pct";
    if (!params.count("Rload")) out.design_notes["Rload"] = "Auto-sized for Iout";
    return out;
}

inline void register_boost_template() {
    static const bool _ = [] {
        ConverterRegistry::instance().register_template("boost", expand_boost);
        return true;
    }();
    (void)_;
}

}  // namespace pulsim::v1::templates
