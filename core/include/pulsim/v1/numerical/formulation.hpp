#pragma once

// simplify-and-harden-numerical-surface — Phase 1 reorganization.
//
// New canonical location for `FormulationMode` (DAE assembly choice:
// `ProjectedWrapper` vs `Direct`). The enum currently lives inside
// `simulation.hpp`; we re-export here so new code can use the
// reorganized include path.

#include "pulsim/v1/simulation.hpp"
