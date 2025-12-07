#pragma once

#include "pulsim/types.hpp"
#include <cmath>
#include <memory>
#include <string>
#include <vector>

namespace pulsim::models {

// =============================================================================
// Magnetic Core Models with Saturation
// Implements Jiles-Atherton, Preisach, and simplified piecewise-linear models
// =============================================================================

// =============================================================================
// Magnetic Core Operating Point
// =============================================================================

struct MagneticCoreOpPoint {
    Real H = 0.0;         // Magnetic field intensity (A/m)
    Real B = 0.0;         // Magnetic flux density (T)
    Real M = 0.0;         // Magnetization (A/m)
    Real mu = 0.0;        // Differential permeability (H/m)
    Real dB_dH = 0.0;     // Incremental permeability
    Real flux = 0.0;      // Total magnetic flux (Wb)
    Real energy = 0.0;    // Stored energy (J)
    Real loss = 0.0;      // Core loss power (W)
};

// =============================================================================
// Core Geometry
// =============================================================================

struct CoreGeometry {
    Real Ae = 1e-4;       // Effective cross-sectional area (m²)
    Real le = 0.1;        // Effective magnetic path length (m)
    Real Ve = 1e-5;       // Effective volume (m³) - computed if 0
    Real lg = 0.0;        // Air gap length (m)
    int N = 1;            // Number of turns (for flux linkage)

    // Compute volume if not specified
    Real effective_volume() const {
        return Ve > 0 ? Ve : Ae * le;
    }
};

// =============================================================================
// Jiles-Atherton Model Parameters
// Standard hysteresis model for magnetic cores
// =============================================================================

struct JilesAthertonParams {
    // Saturation magnetization
    Real Ms = 1.6e6;      // Saturation magnetization (A/m)

    // Anhysteretic parameters
    Real a = 1000.0;      // Shape parameter for anhysteretic curve (A/m)
    Real alpha = 1e-3;    // Domain coupling parameter (dimensionless)

    // Hysteresis parameters
    Real k = 500.0;       // Pinning parameter (A/m)
    Real c = 0.1;         // Reversibility coefficient (dimensionless)

    // Initial state
    Real M0 = 0.0;        // Initial magnetization (A/m)

    // Integration parameters
    Real tolerance = 1e-6;
    int max_iterations = 100;
};

// =============================================================================
// Jiles-Atherton Hysteresis Model
// =============================================================================

class JilesAthertonModel {
public:
    explicit JilesAthertonModel(const JilesAthertonParams& params = {},
                                 const CoreGeometry& geometry = {});

    // Evaluate B given H and current state
    MagneticCoreOpPoint evaluate(Real H, Real dH_dt = 0.0);

    // Reset to initial state
    void reset();

    // Get/set current magnetization
    Real magnetization() const { return M_; }
    void set_magnetization(Real M) { M_ = M; }

    // Get parameters
    const JilesAthertonParams& params() const { return params_; }
    const CoreGeometry& geometry() const { return geometry_; }

    // Compute anhysteretic magnetization
    Real compute_Man(Real He) const;

private:
    JilesAthertonParams params_;
    CoreGeometry geometry_;

    // State variables
    Real M_ = 0.0;        // Current magnetization
    Real H_prev_ = 0.0;   // Previous H for direction detection

    // Internal computations
    Real compute_He(Real H, Real M) const;
    Real compute_dM_dH(Real H, Real M, Real Man, int delta) const;
};

// =============================================================================
// Piecewise-Linear (PWL) Core Model
// Simplified model using linear segments
// =============================================================================

struct PWLCorePoint {
    Real H;               // Field intensity (A/m)
    Real B;               // Flux density (T)
};

struct PWLCoreParams {
    std::vector<PWLCorePoint> curve;  // B-H curve points (first quadrant)
    bool symmetric = true;             // Assume symmetric B-H curve

    // Default curve (typical ferrite)
    static PWLCoreParams default_ferrite();
    static PWLCoreParams default_silicon_steel();
    static PWLCoreParams default_amorphous();
};

class PWLCoreModel {
public:
    explicit PWLCoreModel(const PWLCoreParams& params = PWLCoreParams::default_ferrite(),
                          const CoreGeometry& geometry = {});

    MagneticCoreOpPoint evaluate(Real H) const;

    const PWLCoreParams& params() const { return params_; }
    const CoreGeometry& geometry() const { return geometry_; }

private:
    PWLCoreParams params_;
    CoreGeometry geometry_;

    // Sorted curve for interpolation
    void sort_curve();
};

// =============================================================================
// Steinmetz Core Loss Model
// P = k * f^a * B^b
// =============================================================================

struct SteinmetzParams {
    Real k = 1.0;         // Steinmetz coefficient
    Real alpha = 1.3;     // Frequency exponent
    Real beta = 2.5;      // Flux density exponent

    // Improved Generalized Steinmetz (iGSE) parameters
    Real ki = 0.0;        // iGSE coefficient (computed from k, alpha, beta if 0)

    // Default parameters for common materials
    static SteinmetzParams ferrite_3C90();
    static SteinmetzParams ferrite_3F3();
    static SteinmetzParams silicon_steel();
    static SteinmetzParams amorphous_2605SA1();
};

class SteinmetzLossModel {
public:
    explicit SteinmetzLossModel(const SteinmetzParams& params = {});

    // Calculate loss density (W/m³) for sinusoidal excitation
    Real loss_density_sine(Real frequency, Real Bpk) const;

    // Calculate loss density using iGSE for arbitrary waveforms
    Real loss_density_igse(Real frequency, Real dB_dt, Real Bpk) const;

    // Calculate total loss (W) given volume
    Real total_loss_sine(Real frequency, Real Bpk, Real volume) const;

    const SteinmetzParams& params() const { return params_; }

private:
    SteinmetzParams params_;
    Real ki_;  // Computed iGSE coefficient

    void compute_ki();
};

// =============================================================================
// Combined Core Model (Saturation + Hysteresis + Losses)
// =============================================================================

struct CombinedCoreParams {
    JilesAthertonParams hysteresis;
    SteinmetzParams losses;
    CoreGeometry geometry;

    bool include_hysteresis = true;
    bool include_eddy_current = true;

    // Eddy current parameters
    Real sigma = 0.0;     // Conductivity (S/m), 0 = no eddy current
    Real d = 0.0;         // Lamination thickness (m)
};

class CombinedCoreModel {
public:
    explicit CombinedCoreModel(const CombinedCoreParams& params = {});

    // Evaluate complete operating point
    MagneticCoreOpPoint evaluate(Real H, Real dH_dt = 0.0, Real frequency = 0.0);

    // Reset state
    void reset();

    const CombinedCoreParams& params() const { return params_; }

    // Access internal models
    JilesAthertonModel& hysteresis_model() { return hysteresis_; }
    SteinmetzLossModel& loss_model() { return losses_; }

private:
    CombinedCoreParams params_;
    JilesAthertonModel hysteresis_;
    SteinmetzLossModel losses_;
};

// =============================================================================
// Inductor with Saturable Core
// For circuit integration
// =============================================================================

struct SaturableInductorParams {
    Real L0 = 1e-3;       // Unsaturated inductance (H)
    Real Isat = 10.0;     // Saturation current (A)
    Real Lsat = 0.0;      // Saturated inductance (H), 0 = 10% of L0

    // Optional: use detailed core model
    bool use_core_model = false;
    CombinedCoreParams core;
};

class SaturableInductor {
public:
    explicit SaturableInductor(const SaturableInductorParams& params = {});

    // Compute inductance at given current
    Real inductance(Real current) const;

    // Compute flux linkage at given current
    Real flux_linkage(Real current) const;

    // Compute voltage given current and di/dt
    Real voltage(Real current, Real di_dt) const;

    // For transient simulation - compute di/dt given voltage and current
    Real di_dt(Real voltage, Real current) const;

    const SaturableInductorParams& params() const { return params_; }

private:
    SaturableInductorParams params_;
    Real Lsat_;  // Computed saturated inductance

    // For detailed model
    mutable std::unique_ptr<CombinedCoreModel> core_model_;
};

// =============================================================================
// Transformer with Saturable Core
// =============================================================================

struct SaturableTransformerParams {
    // Winding parameters
    int N1 = 1;           // Primary turns
    int N2 = 1;           // Secondary turns
    Real R1 = 0.0;        // Primary resistance (Ω)
    Real R2 = 0.0;        // Secondary resistance (Ω)
    Real Llk1 = 0.0;      // Primary leakage inductance (H)
    Real Llk2 = 0.0;      // Secondary leakage inductance (H)

    // Core parameters
    CombinedCoreParams core;

    // Simplified model
    Real Lm = 1e-3;       // Magnetizing inductance (H)
    Real Isat = 10.0;     // Magnetizing saturation current (A)
};

struct TransformerOpPoint {
    Real v1 = 0.0;        // Primary voltage
    Real v2 = 0.0;        // Secondary voltage
    Real i1 = 0.0;        // Primary current
    Real i2 = 0.0;        // Secondary current
    Real im = 0.0;        // Magnetizing current
    Real phi = 0.0;       // Core flux (Wb)
    Real Lm_eff = 0.0;    // Effective magnetizing inductance
    Real Pcore = 0.0;     // Core loss power
};

class SaturableTransformer {
public:
    explicit SaturableTransformer(const SaturableTransformerParams& params = {});

    // Evaluate operating point
    TransformerOpPoint evaluate(Real v1, Real i1, Real i2, Real di1_dt, Real di2_dt);

    // Get turns ratio
    Real turns_ratio() const { return static_cast<Real>(params_.N1) / params_.N2; }

    const SaturableTransformerParams& params() const { return params_; }

private:
    SaturableTransformerParams params_;
    std::unique_ptr<CombinedCoreModel> core_model_;
};

}  // namespace pulsim::models
