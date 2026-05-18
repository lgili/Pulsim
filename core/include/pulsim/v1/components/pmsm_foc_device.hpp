#pragma once

#include "pulsim/v1/components/base.hpp"
#include "pulsim/v1/motors/pmsm_foc.hpp"
#include "pulsim/v1/motors/pmsm.hpp"

#include <array>
#include <string>
#include <utility>

namespace pulsim::v1 {

// =============================================================================
// PMSM-FOC current loop — runtime device (consolidate-motors-and-three-phase,
//                                         Phase B.2b)
// =============================================================================
//
// Signal-domain controller device that wraps `motors::PmsmFocCurrentLoop`
// (two PI compensators tuned via pole-zero cancellation against the PMSM's
// L_d / L_q / R_s). The device has **no electrical pins**: it is a pure
// signal block that the user wires explicitly between a PMSM's measured
// (i_d, i_q) currents and an inverter's (V_d, V_q) references.
//
// Per-step usage from Python / mixed-domain blocks:
//   1. ckt.set_pmsm_foc_references("Ctrl1", id_ref, iq_ref)
//   2. ckt.set_pmsm_foc_measurements("Ctrl1",
//                                    ckt.pmsm_i_d("M1"),
//                                    ckt.pmsm_i_q("M1"))
//   3. The variant walker calls the device's `update_history()` on each
//      accepted step, which runs `loop.step(...)` and caches
//      (V_d_ref, V_q_ref) for the next inverter command.
//   4. ckt.pmsm_foc_vd_ref("Ctrl1") / pmsm_foc_vq_ref("Ctrl1") returns the
//      result for inverter wiring.
//
// Retuning is online via `set_pmsm_foc_params` (no rebuild required).
// A standalone helper `motors::PmsmFocCurrentLoop` is also exposed
// through pybind11 for users who prefer to drive the loop manually
// without a Circuit attachment.

class PmsmFocDevice : public DynamicDeviceBase<PmsmFocDevice> {
public:
    using Base = DynamicDeviceBase<PmsmFocDevice>;
    using ScalarType = Scalar;

    struct Params {
        motors::PmsmParams motor{};
        motors::PmsmFocCurrentLoopParams foc{};
    };

    // 0 electrical pins — signal-domain controller.
    static constexpr std::size_t num_pins = 0;
    static constexpr int device_type = static_cast<int>(DeviceType::Inductor);

    PmsmFocDevice() = default;

    explicit PmsmFocDevice(Params params, std::string name = "")
        : Base(std::move(name))
        , params_(std::move(params))
        , loop_(params_.motor, params_.foc)
    {}

    // ---- DynamicDeviceBase interface (CRTP impls) --------------------------

    template<SparseMatrixLike Matrix, VectorLike Vec>
    void stamp_impl(Matrix& /*G*/, Vec& /*b*/,
                    std::span<const NodeIndex> /*nodes*/) {
        // No electrical contribution — controller-only device.
    }

    static constexpr auto jacobian_pattern_impl() {
        return std::array<std::array<int, 2>, 0>{};
    }

    void update_history_impl() {
        const auto dt = Base::timestep();
        if (dt <= Scalar{0}) return;
        const auto [vd, vq] = loop_.step(id_ref_, iq_ref_,
                                         id_meas_, iq_meas_, dt);
        vd_ref_last_ = vd;
        vq_ref_last_ = vq;
    }

    // ---- External coupling -------------------------------------------------

    void set_references(Real id_ref, Real iq_ref) noexcept {
        id_ref_ = id_ref;
        iq_ref_ = iq_ref;
    }
    void set_measurements(Real id_meas, Real iq_meas) noexcept {
        id_meas_ = id_meas;
        iq_meas_ = iq_meas;
    }

    /// Online retune (e.g., adaptive control updates L_d / L_q estimates).
    void retune(const motors::PmsmParams& motor,
                const motors::PmsmFocCurrentLoopParams& foc) {
        params_.motor = motor;
        params_.foc   = foc;
        loop_.retune(motor, foc);
    }

    void reset() {
        loop_.reset();
        vd_ref_last_ = 0.0;
        vq_ref_last_ = 0.0;
    }

    [[nodiscard]] Real vd_ref() const noexcept { return vd_ref_last_; }
    [[nodiscard]] Real vq_ref() const noexcept { return vq_ref_last_; }
    [[nodiscard]] Real id_ref() const noexcept { return id_ref_; }
    [[nodiscard]] Real iq_ref() const noexcept { return iq_ref_; }
    [[nodiscard]] Real id_meas() const noexcept { return id_meas_; }
    [[nodiscard]] Real iq_meas() const noexcept { return iq_meas_; }
    [[nodiscard]] const Params& params() const noexcept { return params_; }
    [[nodiscard]] const motors::PmsmFocCurrentLoop& loop() const noexcept {
        return loop_;
    }

private:
    Params params_{};
    motors::PmsmFocCurrentLoop loop_{};
    Real id_ref_  = 0.0;
    Real iq_ref_  = 0.0;
    Real id_meas_ = 0.0;
    Real iq_meas_ = 0.0;
    Real vd_ref_last_ = 0.0;
    Real vq_ref_last_ = 0.0;
};

}  // namespace pulsim::v1
