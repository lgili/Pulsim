#pragma once

// simplify-and-harden-numerical-surface — Phase 1 reorganization.
//
// New canonical location for the `Integrator` enum + integration helpers.
// This header re-exports the same symbols from the legacy `integration.hpp`
// path so the migration is purely additive in the first release.
//
// Migration policy:
//   - New code SHALL include this path.
//   - Legacy code that includes `pulsim/v1/integration.hpp` keeps
//     compiling — that header is now considered the legacy alias.
//   - In a follow-up release we will swap the source of truth (move the
//     definitions here, leave a forwarder + `#pragma message` on the old
//     path).

#include "pulsim/v1/integration.hpp"
