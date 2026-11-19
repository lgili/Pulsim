#pragma once

// =============================================================================
// Pulsim — the run's flat device state and its layout
// =============================================================================
//
// add-unified-device-interface step 3. Every piece of device state a run
// carries from step to step — the linear companion history and the five
// stateful-device histories — as ONE vector of reals in blocks of a fixed
// order. A snapshot holds this vector, the instance count of every block
// and a signature of the layout; a resume compares the layout with the
// circuit's and refuses a mismatch naming the first block that differs.
//
// Each block is the owning history's own flat layout (`to_flat`), so the
// histories' arithmetic is untouched; the storage moves into the vector
// when each kind is ported onto the device interface (steps 5–7).

#include "pulsim/numeric/types.hpp"

#include <array>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace pulsim::device {

/// The blocks of the device state, in layout order.
enum class StateBlock : std::uint8_t {
    LinearCompanion = 0,   ///< (v_prev, i_prev) per dynamic L / C branch
    SaturableInductor,     ///< 7 per saturable / gapped / JA inductor
    NonlinearCapacitor,    ///< (v, q, i) per charge-based capacitor
    LauritzenDiode,        ///< (q_M, f) per Lauritzen diode
    IgbtTail,              ///< (q, f) per IGBT turn-off tail
    PmsmMna,               ///< (λ, i, v) × 3 per MNA PMSM
    /// add-external-device-abi: ONE real per "instance" — the count is the
    /// number of reals all external devices carry (per device: q_n, q̇_n,
    /// q_γ per row, then the model's n_state), since the size varies per model.
    ExternalDevice,
};
inline constexpr std::size_t kNumStateBlocks = 7;

struct StateBlockInfo {
    std::string_view name;
    std::uint32_t reals_per_instance;
};

/// Name and reals per instance of each block, in layout order. The
/// names are the ones a refusal prints.
inline constexpr std::array<StateBlockInfo, kNumStateBlocks> kStateBlocks{{
    {"linear_companion", 2},
    {"saturable_inductor", 7},
    {"nonlinear_capacitor", 3},
    {"lauritzen_diode", 2},
    {"igbt_tail", 2},
    {"pmsm_mna", 9},
    {"external_device", 1},
}};

/// Instance count per block, in layout order.
struct DeviceStateLayout {
    std::array<std::uint32_t, kNumStateBlocks> count{};

    [[nodiscard]] std::size_t size() const noexcept {
        std::size_t n = 0;
        for (std::size_t b = 0; b < kNumStateBlocks; ++b) {
            n += std::size_t{count[b]} * kStateBlocks[b].reals_per_instance;
        }
        return n;
    }
    [[nodiscard]] std::size_t offset(StateBlock blk) const noexcept {
        std::size_t n = 0;
        for (std::size_t b = 0; b < static_cast<std::size_t>(blk); ++b) {
            n += std::size_t{count[b]} * kStateBlocks[b].reals_per_instance;
        }
        return n;
    }
    [[nodiscard]] std::size_t block_size(StateBlock blk) const noexcept {
        const auto b = static_cast<std::size_t>(blk);
        return std::size_t{count[b]} * kStateBlocks[b].reals_per_instance;
    }
    /// FNV-1a over (block name, count, reals per instance) of the first
    /// `n_blocks` blocks. A snapshot is signed over the blocks it has, so
    /// a build with a block appended still verifies (and zero-pads) an
    /// older snapshot.
    [[nodiscard]] std::uint64_t signature(
        std::size_t n_blocks = kNumStateBlocks) const noexcept {
        std::uint64_t h = 0xcbf29ce484222325ULL;
        auto mix = [&h](std::uint64_t byte) {
            h ^= byte;
            h *= 0x100000001b3ULL;
        };
        for (std::size_t b = 0; b < kNumStateBlocks && b < n_blocks; ++b) {
            for (char c : kStateBlocks[b].name) mix(static_cast<unsigned char>(c));
            for (int s = 0; s < 32; s += 8) mix((count[b] >> s) & 0xffU);
            mix(kStateBlocks[b].reals_per_instance);
        }
        return h;
    }
    /// Empty when equal; else a sentence naming the first differing block.
    [[nodiscard]] std::string first_difference(
        const DeviceStateLayout& other) const {
        for (std::size_t b = 0; b < kNumStateBlocks; ++b) {
            if (count[b] != other.count[b]) {
                return "device-state block '"
                       + std::string(kStateBlocks[b].name) + "' has "
                       + std::to_string(count[b]) + " instance(s) here but "
                       + std::to_string(other.count[b]) + " in the other";
            }
        }
        return {};
    }
};

}  // namespace pulsim::device
