#pragma once

#include "pulsim/circuit.hpp"

namespace pulsim {
namespace devices {

// =============================================================================
// DIODE LIBRARY
// =============================================================================

// General purpose rectifier diodes
inline DiodeParams diode_1N4007() {
    DiodeParams p;
    p.is = 2.55e-9;     // Saturation current
    p.n = 1.75;         // Ideality factor
    p.rs = 0.042;       // Series resistance
    p.vt = 0.026;       // Thermal voltage at 300K
    p.ideal = false;
    p.cj0 = 15e-12;     // Junction capacitance (15pF typical)
    p.vj = 0.7;         // Junction potential
    p.m = 0.5;          // Grading coefficient
    p.tt = 3e-6;        // Transit time (3us for rectifier)
    p.bv = 1000.0;      // Reverse breakdown (1000V for 1N4007)
    p.ibv = 5e-6;
    return p;
}

inline DiodeParams diode_1N4148() {
    DiodeParams p;
    p.is = 2.52e-9;
    p.n = 1.752;
    p.rs = 0.568;
    p.vt = 0.026;
    p.ideal = false;
    p.cj0 = 4e-12;      // 4pF typical
    p.vj = 0.65;
    p.m = 0.5;
    p.tt = 4e-9;        // Fast switching (4ns)
    p.bv = 100.0;
    p.ibv = 1e-7;
    return p;
}

// Schottky diodes (low forward voltage, fast switching)
inline DiodeParams diode_1N5819() {
    DiodeParams p;
    p.is = 1e-5;        // Higher saturation current for Schottky
    p.n = 1.1;          // Lower ideality factor
    p.rs = 0.04;
    p.vt = 0.026;
    p.ideal = false;
    p.cj0 = 110e-12;    // Higher capacitance
    p.vj = 0.35;        // Lower junction potential (Schottky)
    p.m = 0.5;
    p.tt = 0.0;         // No minority carrier storage
    p.bv = 40.0;        // Low reverse voltage for Schottky
    p.ibv = 1e-3;
    return p;
}

// Fast recovery diodes for power electronics
inline DiodeParams diode_MUR860() {
    DiodeParams p;
    p.is = 3.3e-9;
    p.n = 1.8;
    p.rs = 0.02;
    p.vt = 0.026;
    p.ideal = false;
    p.cj0 = 50e-12;
    p.vj = 0.7;
    p.m = 0.5;
    p.tt = 50e-9;       // 50ns recovery time
    p.bv = 600.0;       // High voltage
    p.ibv = 1e-5;
    return p;
}

// Silicon Carbide (SiC) Schottky diode
inline DiodeParams diode_C3D10065A() {
    DiodeParams p;
    p.is = 1e-15;       // Very low leakage
    p.n = 1.05;
    p.rs = 0.065;       // Low on-resistance
    p.vt = 0.026;
    p.ideal = false;
    p.cj0 = 35e-12;
    p.vj = 1.2;         // Higher for SiC
    p.m = 0.5;
    p.tt = 0.0;         // Zero reverse recovery
    p.bv = 650.0;
    p.ibv = 1e-6;
    return p;
}

// =============================================================================
// MOSFET LIBRARY
// =============================================================================

// Low-side driver N-channel MOSFETs
inline MOSFETParams mosfet_IRF540N() {
    MOSFETParams p;
    p.type = MOSFETType::NMOS;
    p.vth = 4.0;        // 2-4V threshold
    p.kp = 20e-6;       // Default transconductance parameter
    p.lambda = 0.01;
    p.w = 1.0;
    p.l = 1.0;
    p.body_diode = true;
    p.is_body = 1e-10;
    p.n_body = 1.3;
    p.cgs = 2500e-12;   // Input capacitance ~1700pF + Cgd
    p.cgd = 300e-12;    // Reverse transfer cap ~300pF
    p.cds = 500e-12;    // Output capacitance
    p.rds_on = 0.044;   // 44mOhm typical
    p.rds_off = 1e9;
    return p;
}

inline MOSFETParams mosfet_IRFZ44N() {
    MOSFETParams p;
    p.type = MOSFETType::NMOS;
    p.vth = 4.0;
    p.kp = 20e-6;
    p.lambda = 0.01;
    p.w = 1.0;
    p.l = 1.0;
    p.body_diode = true;
    p.is_body = 1e-10;
    p.n_body = 1.3;
    p.cgs = 1600e-12;
    p.cgd = 250e-12;
    p.cds = 400e-12;
    p.rds_on = 0.0175;  // 17.5mOhm
    p.rds_off = 1e9;
    return p;
}

// High-side P-channel MOSFET
inline MOSFETParams mosfet_IRF9540() {
    MOSFETParams p;
    p.type = MOSFETType::PMOS;
    p.vth = -4.0;       // Negative threshold for PMOS
    p.kp = 10e-6;       // Lower mobility for PMOS
    p.lambda = 0.01;
    p.w = 1.0;
    p.l = 1.0;
    p.body_diode = true;
    p.is_body = 1e-10;
    p.n_body = 1.3;
    p.cgs = 1200e-12;
    p.cgd = 200e-12;
    p.cds = 350e-12;
    p.rds_on = 0.117;   // Higher for PMOS
    p.rds_off = 1e9;
    return p;
}

// High-efficiency N-channel for DC-DC converters
inline MOSFETParams mosfet_BSC0902NS() {
    MOSFETParams p;
    p.type = MOSFETType::NMOS;
    p.vth = 1.8;        // Logic level gate
    p.kp = 20e-6;
    p.lambda = 0.01;
    p.w = 1.0;
    p.l = 1.0;
    p.body_diode = true;
    p.is_body = 1e-10;
    p.n_body = 1.3;
    p.cgs = 2800e-12;
    p.cgd = 80e-12;
    p.cds = 300e-12;
    p.rds_on = 0.0021;  // 2.1mOhm - very low
    p.rds_off = 1e9;
    return p;
}

// GaN FET (high-efficiency, fast switching)
inline MOSFETParams mosfet_EPC2001C() {
    MOSFETParams p;
    p.type = MOSFETType::NMOS;
    p.vth = 1.4;        // Low threshold for enhancement mode GaN
    p.kp = 50e-6;       // Higher transconductance
    p.lambda = 0.005;
    p.w = 1.0;
    p.l = 1.0;
    p.body_diode = false;  // GaN FETs have no body diode
    p.cgs = 150e-12;    // Very low capacitance
    p.cgd = 2e-12;      // Extremely low Miller cap
    p.cds = 70e-12;
    p.rds_on = 0.004;   // 4mOhm
    p.rds_off = 1e9;
    return p;
}

// =============================================================================
// IGBT LIBRARY
// =============================================================================

// General purpose IGBT for motor drives
inline IGBTParams igbt_IRG4PC40UD() {
    IGBTParams p;
    p.vth = 5.5;        // Gate threshold
    p.vce_sat = 1.5;    // Vce(sat) at 20A
    p.rce_on = 0.01;
    p.rce_off = 1e9;
    p.tf = 150e-9;      // Fall time
    p.tr = 50e-9;       // Rise time
    p.cies = 2600e-12;  // Input capacitance
    p.body_diode = true;
    p.is_diode = 1e-12;
    p.n_diode = 1.0;
    p.vf_diode = 1.5;   // Fast recovery diode
    return p;
}

// High-speed IGBT for resonant converters
inline IGBTParams igbt_IRG4BC30KD() {
    IGBTParams p;
    p.vth = 5.0;
    p.vce_sat = 1.65;
    p.rce_on = 0.02;
    p.rce_off = 1e9;
    p.tf = 60e-9;       // Faster switching
    p.tr = 25e-9;
    p.cies = 1400e-12;
    p.body_diode = true;
    p.is_diode = 1e-12;
    p.n_diode = 1.0;
    p.vf_diode = 1.2;
    return p;
}

// High-voltage IGBT for inverters (1200V class)
inline IGBTParams igbt_IKW40N120H3() {
    IGBTParams p;
    p.vth = 6.0;
    p.vce_sat = 1.95;
    p.rce_on = 0.025;
    p.rce_off = 1e9;
    p.tf = 130e-9;
    p.tr = 40e-9;
    p.cies = 3200e-12;
    p.body_diode = true;
    p.is_diode = 1e-12;
    p.n_diode = 1.0;
    p.vf_diode = 1.7;
    return p;
}

// =============================================================================
// SWITCH LIBRARY (for ideal switches in simulation)
// =============================================================================

// Ideal switch (very fast, low resistance)
inline SwitchParams switch_ideal() {
    SwitchParams p;
    p.ron = 1e-6;       // 1 micro-ohm
    p.roff = 1e12;      // 1 tera-ohm
    p.vth = 0.5;
    p.initial_state = false;
    return p;
}

// Mechanical relay model
inline SwitchParams switch_relay() {
    SwitchParams p;
    p.ron = 0.1;        // 100mOhm contact resistance
    p.roff = 1e9;
    p.vth = 3.0;        // Typical 5V relay pulls in around 3V
    p.initial_state = false;
    return p;
}

// Solid-state relay model
inline SwitchParams switch_ssr() {
    SwitchParams p;
    p.ron = 0.02;       // Lower than mechanical
    p.roff = 1e10;
    p.vth = 1.5;        // LED drive threshold
    p.initial_state = false;
    return p;
}

}  // namespace devices
}  // namespace pulsim
