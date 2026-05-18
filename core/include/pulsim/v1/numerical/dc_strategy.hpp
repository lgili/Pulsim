#pragma once

// simplify-and-harden-numerical-surface — Phase 1 reorganization.
//
// New canonical location for `DCStrategy`, `DCConvergenceConfig`,
// `GminConfig`, `SourceSteppingConfig`, `PseudoTransientConfig`, and
// `InitializationConfig`. Re-exports from the legacy `convergence_aids.hpp`
// path so this is purely additive in the first release.

#include "pulsim/v1/convergence_aids.hpp"
