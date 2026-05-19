#pragma once

// simplify-and-harden-numerical-surface — Phase 5.1.
//
// Canonical location for the PWL event-detection + simultaneous-event
// coalescence machinery.
//
// The actual implementation lives on `Circuit` in `runtime_circuit.hpp`:
//
//   - `Circuit::scan_pwl_commutations(x, circuit_default)`
//     → returns the list of pending state transitions on PWL devices.
//
//   - `Circuit::bisect_pwl_event_alpha(...)`
//     → bisects the fractional step `α ∈ [0, 1]` at which the event(s)
//       cross. Re-scans at `α_hi + 16·tolerance` after convergence and
//       merges any newly-found events into the committed batch. That
//       re-scan is the simultaneous-event coalescence — multiple gates
//       crossing inside the bisection tolerance are committed as a
//       single atomic Newton solve instead of being processed serially
//       across N consecutive macro steps.
//
//   - `Circuit::commit_pwl_commutations(...)`
//     → applies the coalesced batch to each device's `pwl_state_`.
//
// Extracting these into a free function would require threading the
// circuit's device store, parasitic-helpers, and PWL admissibility
// callback through every call — net cost > net benefit. Extraction is
// deferred to a Phase 3-style operator-network refactor.
//
// Telemetry: `result.backend_telemetry.simultaneous_event_groups`
// counts steps where ≥ 2 PWL commutations coalesced.
//
// Reference test: `core/tests/test_simultaneous_events.cpp`
// (3 cases, 7 assertions).

#include "pulsim/v1/runtime_circuit.hpp"
