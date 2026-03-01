#pragma once

#include "pulsim/v1/components/base.hpp"

namespace pulsim::v1 {

// =============================================================================
// Transformer Device (CRTP - Linear, 4-terminal ideal transformer)
// =============================================================================

/// Ideal transformer with turns ratio N:1
/// Terminals: Primary+ (0), Primary- (1), Secondary+ (2), Secondary- (3)
class Transformer : public LinearDeviceBase<Transformer> {
public:
    using Base = LinearDeviceBase<Transformer>;
    static constexpr std::size_t num_pins = 4;
    static constexpr int device_type = static_cast<int>(DeviceType::Transformer);

    struct Params {
        Scalar turns_ratio = 1.0;   // N:1 (primary:secondary)
        Scalar magnetizing_inductance = 1e-3;  // Lm (H), large for ideal
    };

    explicit Transformer(Scalar turns_ratio, std::string name = "")
        : Base(std::move(name))
        , turns_ratio_(turns_ratio)
        , branch_index_p_(-1)
        , branch_index_s_(-1) {}

    /// Set branch indices for MNA (primary and secondary currents)
    void set_branch_indices(Index primary, Index secondary) {
        branch_index_p_ = primary;
        branch_index_s_ = secondary;
    }

    /// Stamp implementation using coupled inductors formulation
    /// V1 = N * V2, I2 = -N * I1
    template<typename Matrix, typename Vec>
    void stamp_impl(Matrix& G, Vec& b, std::span<const NodeIndex> nodes) {
        if (nodes.size() < 4 || branch_index_p_ < 0 || branch_index_s_ < 0) return;

        const NodeIndex np_plus = nodes[0];   // Primary +
        const NodeIndex np_minus = nodes[1];  // Primary -
        const NodeIndex ns_plus = nodes[2];   // Secondary +
        const NodeIndex ns_minus = nodes[3];  // Secondary -
        const NodeIndex br_p = branch_index_p_;
        const NodeIndex br_s = branch_index_s_;

        Scalar n = turns_ratio_;

        // Ideal transformer equations:
        // V1 - N*V2 = 0  (voltage relation)
        // I1 + N*I2 = 0  (current relation, power conservation)

        // Primary winding: branch equation V_p+ - V_p- = L*di_p/dt
        // For ideal: V_p+ - V_p- = N * (V_s+ - V_s-)
        if (np_plus >= 0) {
            G.coeffRef(np_plus, br_p) += 1.0;
            G.coeffRef(br_p, np_plus) += 1.0;
        }
        if (np_minus >= 0) {
            G.coeffRef(np_minus, br_p) -= 1.0;
            G.coeffRef(br_p, np_minus) -= 1.0;
        }

        // Secondary winding coupling
        if (ns_plus >= 0) {
            G.coeffRef(ns_plus, br_s) += 1.0;
            G.coeffRef(br_p, ns_plus) -= n;  // Coupling: V_p = n * V_s
        }
        if (ns_minus >= 0) {
            G.coeffRef(ns_minus, br_s) -= 1.0;
            G.coeffRef(br_p, ns_minus) += n;
        }

        // Secondary branch equation
        if (ns_plus >= 0) G.coeffRef(br_s, ns_plus) += 1.0;
        if (ns_minus >= 0) G.coeffRef(br_s, ns_minus) -= 1.0;

        // Current relationship: I_p + n * I_s = 0
        G.coeffRef(br_s, br_p) += n;
        G.coeffRef(br_s, br_s) += 0.0;  // Placeholder

        // No RHS contribution for ideal transformer
        (void)b;
    }

    static constexpr auto jacobian_pattern_impl() {
        // 6x6 contributions for 4 nodes + 2 branches
        return StaticSparsityPattern<16>{{
            JacobianEntry{0, 4}, JacobianEntry{4, 0},  // np+ <-> br_p
            JacobianEntry{1, 4}, JacobianEntry{4, 1},  // np- <-> br_p
            JacobianEntry{2, 5}, JacobianEntry{4, 2},  // ns+ <-> br_s, coupling
            JacobianEntry{3, 5}, JacobianEntry{4, 3},  // ns- <-> br_s, coupling
            JacobianEntry{5, 2}, JacobianEntry{5, 3},  // br_s <-> secondary nodes
            JacobianEntry{5, 4}, JacobianEntry{5, 5},  // br_s <-> currents
            JacobianEntry{0, 0}, JacobianEntry{1, 1},  // Diagonal placeholders
            JacobianEntry{2, 2}, JacobianEntry{3, 3}
        }};
    }

    [[nodiscard]] Scalar turns_ratio() const { return turns_ratio_; }

private:
    Scalar turns_ratio_;
    Index branch_index_p_;
    Index branch_index_s_;
};

template<>
struct device_traits<Transformer> {
    static constexpr DeviceType type = DeviceType::Transformer;
    static constexpr std::size_t num_pins = 4;
    static constexpr std::size_t num_internal_nodes = 2;  // Two branch currents
    static constexpr bool is_linear = true;
    static constexpr bool is_dynamic = false;  // Ideal transformer (no Lm dynamics here)
    static constexpr bool has_loss_model = false;
    static constexpr bool has_thermal_model = false;
    static constexpr std::size_t jacobian_size = 16;
};

}  // namespace pulsim::v1
