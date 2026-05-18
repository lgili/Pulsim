#pragma once

// simplify-and-harden-numerical-surface — Phase 1 reorganization.
//
// New canonical location for `TransientStepMode`, `AdvancedTimestepConfig`,
// `RichardsonLTEConfig`, `BDFOrderConfig`, `TimestepConfig`, and the
// transient step controller. Re-exports from `transient_services.hpp`
// (the canonical home for these primitives today).

#include "pulsim/v1/transient_services.hpp"
