#pragma once

// simplify-and-harden-numerical-surface — Phase 1 reorganization.
//
// New canonical location for `StiffnessConfig` (stiffness-detection +
// automatic integrator switch-over). The struct currently lives inside
// `simulation.hpp`; we re-export here so new code can use the
// reorganized include path.

#include "pulsim/v1/simulation.hpp"
