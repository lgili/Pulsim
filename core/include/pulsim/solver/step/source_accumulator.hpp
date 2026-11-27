#pragma once

// refactor-unified-transient-stepper, task group 1: the built-in sources'
// right-hand-side terms, one implementation for the fixed engine, the
// TR-BDF2 engine and the DSED adapter.
//
// Only the SOURCE terms are shared. Each caller keeps its own association
// of them with the history terms and the user's b_extra_fn (DECISIONS 7,
// 21): the accumulator adds into the vector it is given, so the caller
// decides what that vector already holds.
//
// `b[row] += 0 + (−v)` per source row is the dense helpers' `out[row] +=
// −v` on a zeroed buffer followed by `b += out`, bit for bit; the rows a
// source does not own skip `b + (+0.0)`, which only differs from `b` for a
// −0.0 entry, and none of the three callers holds one (a zeroed buffer
// plus anything is never −0.0; the fixed engine's history rows were
// already accumulated this way since update-fixed-step-loop-bookkeeping).

#include "pulsim/models/pulse_voltage_source.hpp"
#include "pulsim/models/pwm_voltage_source.hpp"
#include "pulsim/models/sine_voltage_source.hpp"
#include "pulsim/numeric/types.hpp"
#include "pulsim/pwl/device_pool.hpp"
#include "pulsim/sources/current_waveform_b_extra.hpp"
#include "pulsim/sources/source_jumps.hpp"
#include "pulsim/topology/graph.hpp"

#include <utility>
#include <vector>

namespace pulsim::step {

class SourceAccumulator {
public:
    SourceAccumulator(const pwl::DevicePool& pool, const topology::Graph& graph)
        : has_iwave_{sources::has_current_waveform_sources(pool, graph)} {
        // the (branch, row) lists, in the branch order of the dense passes
        for (Index b_id = 0; b_id < graph.num_branches(); ++b_id) {
            const auto& br = graph.branch(b_id);
            if (br.kind != topology::BranchKind::Source) {
                continue;
            }
            switch (pool.kind_of(br.id)) {
            case pwl::DevicePool::StoredKind::PWMVoltageSource:
                pwm_.emplace_back(br.id, pool.branch_var_id_for_source(br.id, graph));
                break;
            case pwl::DevicePool::StoredKind::SineVoltageSource:
                sine_.emplace_back(br.id, pool.branch_var_id_for_source(br.id, graph));
                break;
            case pwl::DevicePool::StoredKind::PulseVoltageSource:
                pulse_.emplace_back(br.id, pool.branch_var_id_for_source(br.id, graph));
                break;
            default:
                break;
            }
        }
    }

    [[nodiscard]] bool has_current_waveforms() const noexcept { return has_iwave_; }
    [[nodiscard]] bool has_pwm() const noexcept { return !pwm_.empty(); }
    [[nodiscard]] bool has_pulse() const noexcept { return !pulse_.empty(); }

    /// The PWM, sine and pulse voltage-source rows, in that order. `src_t`:
    /// the PWM / pulse sample time; `sine_t`: the sine's; `lb` > 0: a source
    /// that jumps at `src_t` reads value_at(src_t − lb), its left limit.
    /// src_t != sine_t: a jump snapped to the step end; only the source that
    /// jumps there is re-timed (the others read `sine_t`).
    void add_voltage_rows(const pwl::DevicePool& pool, Vector& b, Real src_t,
                          Real sine_t, Real lb) const {
        const auto read_t = [&](Index id) {
            const bool jumps = lb > Real{0} && sources::source_jumps_at(pool, id, src_t);
            if (src_t != sine_t && !jumps) {
                return sine_t;
            }
            return jumps ? src_t - lb : src_t;
        };
        for (const auto& [id, row] : pwm_) {
            b[row] += Real{0} + -models::PWMVoltageSource::value_at(
                pool.pwm_voltage_source_params(id), read_t(id));
        }
        for (const auto& [id, row] : sine_) {
            b[row] += Real{0} + -models::SineVoltageSource::value_at(
                pool.sine_voltage_source_params(id), sine_t);
        }
        for (const auto& [id, row] : pulse_) {
            b[row] += Real{0} + -models::PulseVoltageSource::value_at(
                pool.pulse_voltage_source_params(id), read_t(id));
        }
    }

    /// The waveform current sources (node rows, two per source: the dense
    /// helper), `b += value(t)`; `lb` as above for a jumping pulse.
    void add_current_waveforms(const pwl::DevicePool& pool,
                               const topology::Graph& graph, Vector& b, Real t,
                               Real lb) {
        if (!has_iwave_) return;
        sources::compute_current_waveform_b_extra(pool, graph, t, be_iwave_, lb);
        b += be_iwave_;
    }

    /// Both, at one time: what the TR-BDF2 engine and the DSED adapter add.
    /// Inline with the emptiness tests in front: most circuits have neither,
    /// and this runs on every solve (the instructions-per-step gate).
#if defined(__GNUC__) || defined(__clang__)
    [[gnu::always_inline]]
#endif
    void add(const pwl::DevicePool& pool, const topology::Graph& graph,
             Vector& b, Real t, Real lb) {
        if (!pwm_.empty() || !sine_.empty() || !pulse_.empty()) {
            add_voltage_rows(pool, b, t, t, lb);
        }
        if (has_iwave_) add_current_waveforms(pool, graph, b, t, lb);
    }

private:
    using Rows = std::vector<std::pair<Index, Index>>;
    Rows pwm_, sine_, pulse_;
    bool has_iwave_ = false;
    Vector be_iwave_;
};

}  // namespace pulsim::step
