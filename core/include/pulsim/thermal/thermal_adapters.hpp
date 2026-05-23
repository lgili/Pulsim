// SPDX-License-Identifier: MIT
//
// Pulsim — Phase C.1 / C++ port: thermal observer adapters.
//
// The Foster network itself is built from primitive R/C devices in
// the builder, so no kernel additions are needed for the topology.
// What moves to C++ here is the per-step bridge that converts an
// electrical-side quantity (current through a switch, etc.) into a
// thermal-side current injection at the junction node.
//
// Two block factories:
//
//   * `add_thermal_power_injection_to_chain(...)` — generic:
//     given an InputRef `P_ref` resolving to power (W) and a
//     `junction_node_idx` (state-vector index of the junction
//     temperature node), each step writes
//     b_extra[junction_node_idx] = −P (the "thermal current
//     source" convention).
//
//   * `add_resistive_power_injection_to_chain(...)` — convenience
//     for the common case of P = R · I² (a MOSFET conduction loss):
//     reads i from one InputRef, computes R·i², injects.

#pragma once

#include "pulsim/blockchain/chain.hpp"
#include "pulsim/numeric/types.hpp"

#include <string>
#include <utility>

namespace pulsim::thermal {

using blockchain::BlockChain;
using blockchain::ChainContext;
using blockchain::InputRef;

/// Generic thermal injection. The closure resolves `P_ref` (could be
/// a Channel reading P from another block, or a Const for a fixed
/// dissipation, etc.) and pushes the value into
/// `ctx.b_extra[junction_node_idx]` with a flipped sign (current
/// flowing INTO the node from outside the network ↔ heat source).
inline void add_thermal_power_injection_to_chain(
    BlockChain& chain,
    InputRef P_ref,
    Index junction_node_idx,
    std::string power_channel = std::string{}) {

    auto step = [P_ref = std::move(P_ref),
                    junction_node_idx,
                    power_channel = std::move(power_channel)]
                   (ChainContext& ctx) {
        const Real P = blockchain::resolve(P_ref, ctx);
        if (ctx.b_extra) {
            (*ctx.b_extra)[junction_node_idx] = -P;
        }
        if (!power_channel.empty()) {
            ctx.channels[power_channel] = P;
        }
    };
    chain.add(std::move(step));
}


/// Convenience: P = R · I² for a single resistive element. Reads
/// the current via `i_ref` (typically a Node InputRef pointing at
/// the inductor branch current state index, or a Channel reading
/// from a prior current-measurement block), squares it, multiplies
/// by R_ohm, and injects.
///
/// Optionally writes the computed power to `power_channel` so other
/// blocks can read it (e.g. a moving-average block to smooth over
/// PWM ripple before logging).
inline void add_resistive_power_injection_to_chain(
    BlockChain& chain,
    Real R_ohm,
    InputRef i_ref,
    Index junction_node_idx,
    std::string power_channel = std::string{}) {

    auto step = [R_ohm, i_ref = std::move(i_ref),
                    junction_node_idx,
                    power_channel = std::move(power_channel)]
                   (ChainContext& ctx) {
        const Real i = blockchain::resolve(i_ref, ctx);
        const Real P = R_ohm * i * i;
        if (ctx.b_extra) {
            (*ctx.b_extra)[junction_node_idx] = -P;
        }
        if (!power_channel.empty()) {
            ctx.channels[power_channel] = P;
        }
    };
    chain.add(std::move(step));
}

}  // namespace pulsim::thermal
