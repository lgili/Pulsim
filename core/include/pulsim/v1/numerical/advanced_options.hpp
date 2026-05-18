#pragma once

// simplify-and-harden-numerical-surface — Phase 3 MVP.
//
// `AdvancedOptions` is a **reference view** over the existing
// flat-namespace fields on `SimulationOptions`. It provides users
// with a discoverable namespace (`opts.advanced().newton.*`) without
// breaking back-compat (the flat fields keep working).
//
// This is the MVP scaffolding for the full Phase 3 god-class refactor
// — when the dedicated refactor PR ships, the AdvancedOptions struct
// will OWN the fields (instead of just referencing them) and the
// top-level aliases will be deprecated.
//
// For now, this header just provides the view + the accessor. Users
// who prefer the namespaced form can use it; users who prefer the
// flat form continue to do so. Same data, two access paths.

#include "pulsim/v1/numerical/dc_strategy.hpp"
#include "pulsim/v1/numerical/linear_solver.hpp"
#include "pulsim/v1/numerical/newton.hpp"
#include "pulsim/v1/numerical/stiffness.hpp"
#include "pulsim/v1/numerical/timestep_control.hpp"
#include "pulsim/v1/numerical/formulation.hpp"

namespace pulsim::v1 {

// Forward decl — SimulationOptions is defined in simulation.hpp,
// which includes this header (so this header can't pull in
// simulation.hpp without creating a cycle).
struct SimulationOptions;
struct AdvancedTimestepConfig;
struct RichardsonLTEConfig;
struct BDFOrderConfig;
struct FallbackPolicyOptions;
struct StiffnessConfig;
// `FormulationMode` is an enum class defined in simulation.hpp;
// forward-declare it here so we can take a pointer/reference to it
// without including simulation.hpp.
enum class FormulationMode;

/// Mutable reference view over the advanced numerical knobs on a
/// `SimulationOptions` instance. Constructed by
/// `SimulationOptions::advanced()`.
///
/// Each field is a reference to the corresponding member of the
/// underlying `SimulationOptions` — mutating through the view mutates
/// the original. No data duplication.
struct AdvancedOptions {
    NewtonOptions&             newton;
    AdvancedTimestepConfig&    timestep;
    RichardsonLTEConfig&       lte;
    BDFOrderConfig&            bdf_order;
    DCConvergenceConfig&       dc;
    StiffnessConfig&           stiffness;
    FallbackPolicyOptions&     fallback;
    FormulationMode&           formulation;
    LinearSolverStackConfig&   linear_solver;
};

/// Const reference view (read-only).
struct AdvancedOptionsConst {
    const NewtonOptions&             newton;
    const AdvancedTimestepConfig&    timestep;
    const RichardsonLTEConfig&       lte;
    const BDFOrderConfig&            bdf_order;
    const DCConvergenceConfig&       dc;
    const StiffnessConfig&           stiffness;
    const FallbackPolicyOptions&     fallback;
    const FormulationMode&           formulation;
    const LinearSolverStackConfig&   linear_solver;
};

}  // namespace pulsim::v1
