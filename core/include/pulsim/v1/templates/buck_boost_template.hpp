#pragma once

#include "pulsim/v1/templates/registry.hpp"
#include "pulsim/v1/runtime_circuit.hpp"

#include <cmath>
#include <string>

namespace pulsim::v1::templates {

// =============================================================================
// add-converter-templates — buck-boost (Phase 2.3, inverting topology)
// =============================================================================
//
// Inverting buck-boost (Vout opposite polarity to Vin):
//
//      Vin ── Q1 ──┬── D1 ── Vout (negative)
//                  │
//                  L1
//                  │
//                  gnd
//
// Switch closed: L1 charges from Vin via Q1, D1 reverse-blocks.
// Switch open:   L1 discharges into Vout (negative) via D1.
//
// Auto-design (CCM, |Vout| can be > or < Vin):
//   D = |Vout| / (Vin + |Vout|)
//   I_in_avg = Iout · |Vout| / Vin
//   ΔI = ripple_pct · I_in_avg   (or use Iout, depending on convention)
//   L = Vin · D / (ΔI · fsw)
//   C = Iout · D / (fsw · ΔV)

[[nodiscard]] inline ConverterExpansion expand_buck_boost(
    const ConverterParameters& params) {
    ConverterExpansion out;
    out.topology = "buck_boost";

    const Real Vin  = require_param(params, "Vin",  "buck_boost");
    const Real Vout_mag = std::abs(require_param(params, "Vout", "buck_boost"));
    const Real Iout = require_param(params, "Iout", "buck_boost");
    const Real fsw  = require_param(params, "fsw",  "buck_boost");

    if (!(Vin > Real{0}) || !(Vout_mag > Real{0}) || !(fsw > Real{0}) ||
        !(Iout > Real{0})) {
        throw std::invalid_argument(
            "buck_boost: require Vin > 0, |Vout| > 0, Iout > 0, fsw > 0");
    }

    const Real ripple_pct      = param_or(params, "ripple_pct",      Real{0.30});
    const Real vout_ripple_pct = param_or(params, "vout_ripple_pct", Real{0.01});

    const Real D    = Vout_mag / (Vin + Vout_mag);
    const Real I_in = Iout * (Vout_mag / Vin);
    const Real dI   = ripple_pct * I_in;
    const Real L    = Vin * D / (dI * fsw);
    const Real dV   = vout_ripple_pct * Vout_mag;
    const Real C    = Iout * D / (fsw * dV);
    const Real R_load = Vout_mag / Iout;

    const Real L_actual = param_or(params, "L",     L);
    const Real C_actual = param_or(params, "C",     C);
    const Real R_actual = param_or(params, "Rload", R_load);
    const Real q_g_on   = param_or(params, "q_g_on",  Real{1e3});
    const Real q_g_off  = param_or(params, "q_g_off", Real{1e-9});

    Circuit ckt;
    auto in    = ckt.add_node("in");
    auto sw_n  = ckt.add_node("sw");
    auto out_n = ckt.add_node("out");           // negative terminal
    auto ctrl  = ckt.add_node("ctrl");

    ckt.add_voltage_source("Vin", in, Circuit::ground(), Vin);

    PWMParams pwm;
    pwm.frequency = fsw;
    pwm.duty      = D;
    pwm.v_high    = 5.0;
    pwm.v_low     = 0.0;
    ckt.add_pwm_voltage_source("Vctrl", ctrl, Circuit::ground(), pwm);

    ckt.add_vcswitch("Q1", ctrl, in, sw_n, 2.5, q_g_on, q_g_off);
    // Inductor from sw_n to ground (charges when Q1 closes).
    ckt.add_inductor("L1", sw_n, Circuit::ground(), L_actual, 0.0);
    // Diode anode at sw_n, cathode at out_n. With sw_n above ground
    // during charge, the diode is reverse-biased; when Q1 opens and
    // L1 reverses sw_n below ground, the diode conducts and pumps
    // charge into out_n (which sits negative w.r.t. ground).
    ckt.add_diode("D1", out_n, sw_n, q_g_on, q_g_off);   // out_n→sw_n (cathode, anode)
    ckt.add_capacitor("C1", out_n, Circuit::ground(), C_actual, 0.0);
    ckt.add_resistor("Rload", out_n, Circuit::ground(), R_actual);

    out.circuit = std::move(ckt);
    out.resolved_parameters = {
        {"Vin", Vin}, {"Vout", -Vout_mag}, {"Iout", Iout}, {"fsw", fsw},
        {"ripple_pct", ripple_pct}, {"vout_ripple_pct", vout_ripple_pct},
        {"D", D}, {"I_in", I_in},
        {"L", L_actual}, {"C", C_actual}, {"Rload", R_actual},
        {"q_g_on", q_g_on}, {"q_g_off", q_g_off},
    };
    if (!params.count("L")) out.design_notes["L"] = "Auto-sized for ripple_pct";
    if (!params.count("C")) out.design_notes["C"] = "Auto-sized for vout_ripple_pct";
    return out;
}

inline void register_buck_boost_template() {
    static const bool _ = [] {
        ConverterRegistry::instance().register_template("buck_boost", expand_buck_boost);
        return true;
    }();
    (void)_;
}

}  // namespace pulsim::v1::templates
