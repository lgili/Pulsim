#pragma once

#include "pulsim/types.hpp"
#include <cmath>
#include <memory>
#include <string>
#include <unordered_map>

namespace pulsim::models {

// =============================================================================
// Advanced MOSFET Models
// Implements Level 1, 2, 3, and BSIM3v3 models
// =============================================================================

// MOSFET operating region
enum class MOSRegion {
    CUTOFF,
    LINEAR,      // Triode/Ohmic
    SATURATION,
    SUBTHRESHOLD
};

// Base MOSFET model parameters (common to all levels)
struct MOSFETModelParams {
    // Device type
    bool is_pmos = false;

    // Basic parameters (Level 1)
    Real vth0 = 0.7;      // Zero-bias threshold voltage (V)
    Real kp = 110e-6;     // Transconductance parameter (A/V²)
    Real lambda = 0.0;    // Channel-length modulation (1/V)
    Real phi = 0.6;       // Surface potential (V)
    Real gamma = 0.0;     // Body effect parameter (V^0.5)

    // Geometry
    Real tox = 10e-9;     // Gate oxide thickness (m)
    Real ld = 0.0;        // Lateral diffusion (m)
    Real wd = 0.0;        // Width narrowing (m)

    // Mobility
    Real u0 = 600.0;      // Low-field mobility (cm²/V·s)
    Real ucrit = 1e4;     // Critical field for mobility degradation (V/cm)
    Real uexp = 0.0;      // Mobility degradation exponent
    Real utra = 0.0;      // Transverse field mobility coefficient
    Real theta = 0.0;     // Mobility degradation coefficient (1/V)
    Real ua = 0.0;        // First-order mobility degradation coefficient
    Real ub = 0.0;        // Second-order mobility degradation coefficient

    // Velocity saturation (Level 2, 3)
    Real vmax = 0.0;      // Maximum carrier velocity (m/s), 0 = infinite
    Real neff = 1.0;      // Total channel charge coefficient

    // Subthreshold (Level 2, 3)
    Real nfs = 0.0;       // Fast surface state density (1/cm²)
    Real eta = 0.0;       // Static feedback on threshold voltage

    // Junction parameters
    Real js = 1e-14;      // Bulk junction saturation current density (A/m²)
    Real pb = 0.8;        // Bulk junction potential (V)
    Real cj = 0.0;        // Zero-bias bulk junction capacitance (F/m²)
    Real mj = 0.5;        // Bulk junction grading coefficient
    Real cjsw = 0.0;      // Zero-bias sidewall capacitance (F/m)
    Real mjsw = 0.33;     // Sidewall grading coefficient

    // Overlap capacitances
    Real cgso = 0.0;      // Gate-source overlap capacitance per width (F/m)
    Real cgdo = 0.0;      // Gate-drain overlap capacitance per width (F/m)
    Real cgbo = 0.0;      // Gate-bulk overlap capacitance per length (F/m)

    // Temperature parameters
    Real tnom = 300.15;   // Nominal temperature (K)

    // BSIM3v3 specific parameters
    Real k1 = 0.5;        // First-order body effect coefficient
    Real k2 = 0.0;        // Second-order body effect coefficient
    Real k3 = 80.0;       // Narrow width effect coefficient
    Real k3b = 0.0;       // Body effect coefficient of k3
    Real dvt0 = 2.2;      // Short channel effect coefficient 0
    Real dvt1 = 0.53;     // Short channel effect coefficient 1
    Real dvt2 = -0.032;   // Short channel effect coefficient 2
    Real nlx = 1.74e-7;   // Lateral non-uniform doping coefficient
    Real w0 = 0.0;        // Narrow width effect parameter
    Real vsat = 1.5e5;    // Saturation velocity (m/s)
    Real a0 = 1.0;        // Non-uniform depletion width effect coefficient
    Real ags = 0.2;       // Gate bias coefficient of Abulk
    Real a1 = 0.0;        // Non-saturation effect coefficient
    Real a2 = 1.0;        // Non-saturation effect coefficient
    Real b0 = 0.0;        // Abulk narrow width parameter
    Real b1 = 0.0;        // Abulk narrow width parameter
    Real voff = -0.11;    // Threshold voltage offset
    Real nfactor = 1.0;   // Subthreshold swing factor
    Real cit = 0.0;       // Interface state capacitance
    Real cdsc = 0.0;      // Drain/source and channel coupling capacitance
    Real cdscb = 0.0;     // Body-bias sensitivity of cdsc
    Real cdscd = 0.0;     // Drain-bias sensitivity of cdsc
    Real pclm = 1.3;      // Channel length modulation coefficient
    Real pdiblc1 = 0.39;  // DIBL coefficient 1
    Real pdiblc2 = 0.0086;// DIBL coefficient 2
    Real pdiblcb = 0.0;   // Body effect coefficient of DIBL
    Real drout = 0.56;    // DIBL coefficient in output resistance
    Real pscbe1 = 4.24e8; // Substrate current body-effect coefficient 1
    Real pscbe2 = 1e-5;   // Substrate current body-effect coefficient 2
    Real pvag = 0.0;      // Gate dependence of output resistance
    Real delta = 0.01;    // Effective Vds parameter
    Real alpha0 = 0.0;    // Substrate current model parameter
    Real beta0 = 30.0;    // Substrate current model parameter
};

// Instance parameters
struct MOSFETInstance {
    Real w = 10e-6;       // Channel width (m)
    Real l = 1e-6;        // Channel length (m)
    Real as = 0.0;        // Source area (m²)
    Real ad = 0.0;        // Drain area (m²)
    Real ps = 0.0;        // Source perimeter (m)
    Real pd = 0.0;        // Drain perimeter (m)
    Real nrd = 0.0;       // Drain squares
    Real nrs = 0.0;       // Source squares
    int m = 1;            // Multiplier
};

// MOSFET operating point
struct MOSFETOpPoint {
    MOSRegion region = MOSRegion::CUTOFF;
    Real ids = 0.0;       // Drain-source current (A)
    Real vgs = 0.0;       // Gate-source voltage (V)
    Real vds = 0.0;       // Drain-source voltage (V)
    Real vbs = 0.0;       // Bulk-source voltage (V)
    Real vth = 0.0;       // Threshold voltage (V)
    Real vdsat = 0.0;     // Saturation voltage (V)
    Real gm = 0.0;        // Transconductance (S)
    Real gds = 0.0;       // Output conductance (S)
    Real gmbs = 0.0;      // Body transconductance (S)
    Real cgs = 0.0;       // Gate-source capacitance (F)
    Real cgd = 0.0;       // Gate-drain capacitance (F)
    Real cgb = 0.0;       // Gate-bulk capacitance (F)
    Real cbs = 0.0;       // Bulk-source capacitance (F)
    Real cbd = 0.0;       // Bulk-drain capacitance (F)
};

// =============================================================================
// Level 1 MOSFET (Shichman-Hodges)
// =============================================================================

class MOSFETLevel1 {
public:
    explicit MOSFETLevel1(const MOSFETModelParams& params = {});

    void set_instance(const MOSFETInstance& inst) { inst_ = inst; }

    // Evaluate drain current and derivatives
    MOSFETOpPoint evaluate(Real vgs, Real vds, Real vbs, Real temp = 300.15) const;

    // Get model parameters
    const MOSFETModelParams& params() const { return params_; }

private:
    MOSFETModelParams params_;
    MOSFETInstance inst_;

    Real compute_vth(Real vbs, Real temp) const;
    Real compute_beta(Real temp) const;
};

// =============================================================================
// Level 2 MOSFET (Grove-Frohman)
// Includes velocity saturation and subthreshold conduction
// =============================================================================

class MOSFETLevel2 {
public:
    explicit MOSFETLevel2(const MOSFETModelParams& params = {});

    void set_instance(const MOSFETInstance& inst) { inst_ = inst; }

    MOSFETOpPoint evaluate(Real vgs, Real vds, Real vbs, Real temp = 300.15) const;

    const MOSFETModelParams& params() const { return params_; }

private:
    MOSFETModelParams params_;
    MOSFETInstance inst_;

    Real compute_vth(Real vbs, Real temp) const;
    Real compute_mobility(Real vgs, Real vds, Real temp) const;
    Real compute_vdsat(Real vgs, Real vth, Real temp) const;
};

// =============================================================================
// Level 3 MOSFET (Semi-empirical short-channel model)
// =============================================================================

class MOSFETLevel3 {
public:
    explicit MOSFETLevel3(const MOSFETModelParams& params = {});

    void set_instance(const MOSFETInstance& inst) { inst_ = inst; }

    MOSFETOpPoint evaluate(Real vgs, Real vds, Real vbs, Real temp = 300.15) const;

    const MOSFETModelParams& params() const { return params_; }

private:
    MOSFETModelParams params_;
    MOSFETInstance inst_;

    Real compute_vth(Real vbs, Real vds, Real temp) const;
    Real compute_mobility(Real vgs, Real vbs, Real temp) const;
};

// =============================================================================
// BSIM3v3 MOSFET Model
// Industry-standard model for deep submicron devices
// =============================================================================

class MOSFETBSIM3 {
public:
    explicit MOSFETBSIM3(const MOSFETModelParams& params = {});

    void set_instance(const MOSFETInstance& inst);

    MOSFETOpPoint evaluate(Real vgs, Real vds, Real vbs, Real temp = 300.15) const;

    const MOSFETModelParams& params() const { return params_; }

private:
    MOSFETModelParams params_;
    MOSFETInstance inst_;

    // Precomputed size-dependent parameters
    mutable bool size_params_valid_ = false;
    mutable Real leff_ = 0.0;
    mutable Real weff_ = 0.0;
    mutable Real vth0_eff_ = 0.0;
    mutable Real k1_eff_ = 0.0;
    mutable Real k2_eff_ = 0.0;

    void compute_size_params() const;
    Real compute_vth(Real vbs, Real vds, Real temp) const;
    Real compute_vdsat(Real vgs, Real vth, Real vbs, Real temp) const;
    Real compute_ids_linear(Real vgs, Real vds, Real vth, Real vdsat, Real temp) const;
    Real compute_ids_sat(Real vgs, Real vds, Real vth, Real vdsat, Real temp) const;
};

// =============================================================================
// EKV MOSFET Model (Enz-Krummenacher-Vittoz)
// Single-equation model valid in all regions
// =============================================================================

struct EKVParams {
    bool is_pmos = false;
    Real vto = 0.5;       // Long-channel threshold voltage (V)
    Real gamma = 1.0;     // Body effect parameter (V^0.5)
    Real phi = 0.7;       // Bulk Fermi potential (V)
    Real kp = 50e-6;      // Transconductance parameter (A/V²)
    Real theta = 0.0;     // Mobility reduction coefficient (1/V)
    Real ucrit = 4.0e6;   // Longitudinal critical field (V/m)
    Real lambda = 0.5;    // Depletion length coefficient
    Real weta = 0.0;      // Narrow-channel effect coefficient
    Real leta = 0.0;      // Short-channel effect coefficient
    Real q0 = 0.0;        // RSCE peak charge density
    Real lk = 0.0;        // RSCE characteristic length
    Real iba = 0.0;       // First impact ionization coefficient
    Real ibb = 0.0;       // Second impact ionization coefficient
    Real ibn = 1.0;       // Saturation voltage factor for impact ionization
    Real kf = 0.0;        // Flicker noise coefficient
    Real af = 1.0;        // Flicker noise exponent
};

class MOSFETEKV {
public:
    explicit MOSFETEKV(const EKVParams& params = {});

    void set_instance(const MOSFETInstance& inst) { inst_ = inst; }

    MOSFETOpPoint evaluate(Real vgs, Real vds, Real vbs, Real temp = 300.15) const;

    const EKVParams& params() const { return params_; }

private:
    EKVParams params_;
    MOSFETInstance inst_;

    // EKV specific functions
    Real compute_vp(Real vgs, Real vbs) const;
    Real compute_n(Real vp, Real vbs) const;
    Real compute_if(Real vp, Real vs) const;
    Real compute_ir(Real vp, Real vd) const;
};

// =============================================================================
// MOSFET Model Factory
// =============================================================================

enum class MOSFETModelLevel {
    LEVEL1,
    LEVEL2,
    LEVEL3,
    BSIM3,
    EKV
};

// Factory function to create MOSFET models
std::unique_ptr<class MOSFETModelBase> create_mosfet_model(
    MOSFETModelLevel level,
    const MOSFETModelParams& params = {});

// Abstract base for polymorphic usage
class MOSFETModelBase {
public:
    virtual ~MOSFETModelBase() = default;
    virtual void set_instance(const MOSFETInstance& inst) = 0;
    virtual MOSFETOpPoint evaluate(Real vgs, Real vds, Real vbs, Real temp = 300.15) const = 0;
    virtual MOSFETModelLevel level() const = 0;
};

}  // namespace pulsim::models
