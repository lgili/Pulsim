#pragma once

// =============================================================================
// Pulsim — Layer 2 V15: VCVS (voltage-controlled voltage source)
// =============================================================================
//
// 4-terminal linear-controlled source:
//
//   V_out = Gain · (V_in_pos − V_in_neg)
//
// where:
//   in_pos, in_neg:  sense nodes (no current flow — ideal
//                    high-impedance inputs)
//   out_pos, out_neg: branch terminals, branch-current
//                     unknown like a regular VoltageSource
//   Gain:            transfer-function gain [V/V]
//
// MNA stamping pattern:
//   * Two KCL contributions (out_pos: +I_vcvs, out_neg: −I_vcvs)
//   * One constraint row (the branch-var row):
//       V_out_pos − V_out_neg − Gain·V_in_pos + Gain·V_in_neg = 0
//
// LINEAR device (kind = Source). Stamps entirely from the
// PWL cache — no Newton refresh needed.
//
// Use cases:
//   * Building block for OP-AMP IDEAL (set Gain very high,
//     e.g. 10⁵–10⁷; the feedback network forces V_in_pos ≈
//     V_in_neg via the closed-loop "virtual short").
//   * Sensor amplifiers (sample voltage divider, output to
//     load).
//   * Error amplifiers in SMPS control loops.
//   * Comparators (with limit clipping in user code).

#include "pulsim/numeric/types.hpp"
#include "pulsim/topology/graph.hpp"

namespace pulsim::models {

struct VCVS {
    struct Params {
        Real gain = Real{1};   // [V/V] open-loop gain
    };

    static constexpr topology::BranchKind kind =
        topology::BranchKind::Source;
    // 4 terminals (in_pos, in_neg, out_pos, out_neg) but only
    // out_pos→out_neg is the BRANCH. The two input terminals
    // are sense nodes stored separately in DevicePool, not
    // part of the branch geometry. The Graph only sees the
    // out branch.
    static constexpr Size num_terminals = 2;
    static constexpr bool is_linear  = true;
    static constexpr bool is_dynamic = false;
    static constexpr bool needs_branch_unknown = true;
};

}  // namespace pulsim::models
