#pragma once

// =============================================================================
// Pulsim — linear controlled sources and waveform current sources
// =============================================================================
//
// Siblings of `models/vcvs.hpp` (`openspec/changes/add-controlled-source-set`):
//
//   VCCS     i = g_m·(v_cp − v_cn)      [S]   no extra unknown
//   CCCS     i = β·i_c                  [–]   no extra unknown
//   CCVS     v_op − v_on = r_m·i_c      [Ω]   one branch unknown
//   Gyrator  i_1 = g·v_2, i_2 = −g·v_1  [S]   no extra unknown, lossless
//
// and three time-varying current sources whose value is overlaid on the
// node rows each step (`sources/current_waveform_b_extra.hpp`), the way
// the sine / pulse voltage sources are overlaid on their branch rows.
//
// Terminal convention of every current OUTPUT (VCCS, CCCS, the waveform
// sources): that of `add_current_source` — a positive value leaves the
// source at `from` into the external circuit and returns at `to`.
// `i_c` is the controlling branch's current unknown, positive from that
// branch's `from` to its `to`. Gyrator currents `i_1`, `i_2` ENTER the
// device at `p1_pos`, `p2_pos`.
//
// All of them are linear and independent of the switch mask: the
// controlled ones go into the cached `J` once, the waveform ones only
// into the right-hand side.

#include "pulsim/models/pulse_voltage_source.hpp"
#include "pulsim/models/pwl_waveform.hpp"
#include "pulsim/models/sine_voltage_source.hpp"
#include "pulsim/numeric/types.hpp"
#include "pulsim/topology/graph.hpp"

#include <algorithm>
#include <cmath>
#include <format>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

namespace pulsim::models {

struct VCCS {
    struct Params {
        Real gm = Real{0};   //!< transconductance [S]
    };
};

struct CCCS {
    struct Params {
        Real gain = Real{0};   //!< current gain β [A/A]
    };
};

struct CCVS {
    struct Params {
        Real rm = Real{0};   //!< transresistance [Ω]
    };
};

struct Gyrator {
    struct Params {
        Real g = Real{0};   //!< gyration conductance [S]
    };
};

/// i(t) = i_dc + i_amplitude·sin(2π·f·t + φ) — the sine voltage
/// source's evaluator with the value read as a current.
struct SineCurrentSource {
    struct Params {
        SineVoltageSource::Params w;
    };
    [[nodiscard]] static Real value_at(const Params& p, Real t) noexcept {
        return SineVoltageSource::value_at(p.w, t);
    }
};

/// The pulse voltage source's trapezoid (same parameters and edge
/// rules) read as a current.
struct PulseCurrentSource {
    struct Params {
        PulseVoltageSource::Params w;
    };
    [[nodiscard]] static Real value_at(const Params& p, Real t) noexcept {
        return PulseVoltageSource::value_at(p.w, t);
    }
    /// First corner of the trapezoid strictly after `t`; +inf when
    /// there is none (a single shot that has finished).
    [[nodiscard]] static Real next_breakpoint_after(const Params& p,
                                                    Real t) noexcept {
        const auto& w = p.w;
        const Real c[4] = {Real{0}, w.rise_time, w.rise_time + w.pulse_width,
                           w.rise_time + w.pulse_width + w.fall_time};
        const Real inf = std::numeric_limits<Real>::infinity();
        Real k0 = Real{0};
        if (w.period > Real{0} && t > w.t_start) {
            k0 = std::floor((t - w.t_start) / w.period);
        }
        for (int rep = 0; rep < 2; ++rep) {
            const Real base = w.t_start + (k0 + rep) * w.period;
            for (const Real ci : c) {
                if (base + ci > t) return base + ci;
            }
            if (!(w.period > Real{0})) break;
        }
        return inf;
    }
};

struct PwlCurrentSource {
    struct Params {
        PwlWaveform w;
    };
    [[nodiscard]] static Real value_at(const Params& p, Real t) noexcept {
        return p.w.value_at(t);
    }
};

}  // namespace pulsim::models
