#pragma once

// =============================================================================
// Pulsim — the per-step trace of external devices' readings (3.4)
// =============================================================================
//
// add-external-device-abi 3.4. For every external device added with
// `record=True`, one row per accepted step of a pwl / trbdf2 run: the pin
// currents (f + the committed q̇, current INTO the device), the absorbed
// power Σ v·i and the model's named outputs (`outputs_fn` with the
// committed state). Fed by `StepHooks::device_state` (after the commit), so
// no loop edit and nothing on a run without a recorded device. The row at
// the run's start is computed on request from a copy of the histories as
// they were at the first accepted step (their start cells) and the stored
// initial state.

#include "pulsim/events/step_hooks.hpp"
#include "pulsim/pwl/external_device_history.hpp"
#include "pulsim/pwl/stateful_histories.hpp"

#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace pulsim::xdev {

class OutputTrace {
public:
    struct Dev {
        std::string name;
        std::vector<std::string> columns;
        std::size_t k = 0;          ///< index in ExternalDeviceHistory::devices()
        std::vector<Real> rows;     ///< row-major, columns.size() per accepted step
    };

    void attach(events::StepHooks& h) {
        h.on_device_state([this](Real t, const Vector& x, const pwl::StatefulHistories& hs) {
            on_state_(t, x, hs);
        });
    }
    /// Hooks of its own, for a run without a recorder feed.
    [[nodiscard]] events::StepHooks& hooks() noexcept { return hooks_; }

    [[nodiscard]] const std::vector<Real>& times() const noexcept { return t_; }
    [[nodiscard]] const std::vector<Dev>& devices() const noexcept { return devs_; }
    [[nodiscard]] const Dev& device(const std::string& name) const {
        for (const auto& d : devs_) if (d.name == name) return d;
        if (!start_) {
            throw std::invalid_argument("device_output: '" + name
                                        + "': the run committed no step, so nothing was recorded");
        }
        std::string have;
        for (const auto& d : devs_) have += (have.empty() ? "" : ", ") + d.name;
        throw std::invalid_argument("device_output: '" + name + "' was not recorded (recorded: "
                                    + (have.empty() ? "none" : have)
                                    + "); add it with add_external_device(..., record=True)");
    }
    /// The readings at the run's start: x0 the stored initial state, t0 its time.
    [[nodiscard]] std::vector<Real> start_row(const std::string& name, const Vector& x0,
                                              Real t0) const {
        const Dev& d = device(name);
        std::vector<Real> row(d.columns.size());
        if (start_) start_->reading(d.k, x0, t0, /*at_start=*/true, row.data());
        return row;
    }

private:
    void on_state_(Real t, const Vector& x, const pwl::StatefulHistories& hs) {
        const auto& ext = hs.get<pwl::ExternalDeviceHistory>();
        if (!start_) {
            start_ = ext;   // the start cells are still the run's start here
            const auto& ds = ext.devices();
            for (std::size_t k = 0; k < ds.size(); ++k) {
                if (!ds[k].record) continue;
                const auto& md = ds[k].inst->model().d();
                Dev d;
                d.name = ds[k].inst->device();
                d.k = k;
                for (std::uint32_t r = 0; r < md.n_pins; ++r) {
                    d.columns.push_back(std::string("i(") + md.pin_names[r] + ")");
                }
                d.columns.emplace_back("power");
                for (std::uint32_t r = 0; r < md.n_outputs; ++r) d.columns.emplace_back(md.outputs[r].name);
                devs_.push_back(std::move(d));
            }
        }
        // a row the engine stored and then dropped (a post-edge sample a
        // later commit did not lie after, release D3): superseded
        while (!t_.empty() && t_.back() >= t) {
            t_.pop_back();
            for (auto& d : devs_) d.rows.resize(d.rows.size() - d.columns.size());
        }
        t_.push_back(t);
        for (auto& d : devs_) {
            const std::size_t w = d.columns.size();
            d.rows.resize(d.rows.size() + w);
            ext.reading(d.k, x, t, /*at_start=*/false, d.rows.data() + d.rows.size() - w);
        }
    }

    events::StepHooks hooks_;
    std::optional<pwl::ExternalDeviceHistory> start_;
    std::vector<Real> t_;
    std::vector<Dev> devs_;
};

}  // namespace pulsim::xdev
