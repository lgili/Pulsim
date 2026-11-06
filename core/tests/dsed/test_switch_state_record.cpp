// =============================================================================
// SimulationResult::switch_state_bits — the per-sample STAMPED mask
// =============================================================================
//
// A PWL diode is g_on or g_off with no voltage offset, so a diode
// carrying current sits at i/g_on, inside 0 < v_D < V_th, where BOTH
// states are self-consistent: the stored voltage alone cannot say which
// one the solver stamped. The kernel therefore records, with every
// stored sample, the mask of the solve that produced it. These tests pin
// the record on every engine that writes a SimulationResult:
//
//   * one entry per sample, parallel to `times`;
//   * the recorded bit IS the stamp: g(bit)·v_D equals the series
//     inductor's current — a STATE variable, not a reconstruction — at
//     every sample, to solver precision, with V_th = 0.7 (where the
//     voltage rule v_D > V_th would call every conducting sample OFF);
//   * known instants: ON at the positive crest, OFF at the negative one;
//   * a controlled switch's bit is what switch_fn returned for the step
//     (fixed engine: switch_fn(t_k); TR-BDF2: the step midpoint);
//   * a seeded initial state is flagged unsolved, a zero IC is not.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "pulsim/builder/circuit_builder.hpp"
#include "pulsim/solver/bdf1.hpp"
#include "pulsim/solver/run_transient.hpp"
#include "pulsim/solver/trbdf2_transient.hpp"

#include <cmath>
#include <numbers>

using namespace pulsim;
using Catch::Approx;

namespace {

constexpr Real kF = 1e3;          // source frequency [Hz]
constexpr Real kGon = 1e3;
constexpr Real kGoff = 1e-9;
constexpr Real kVth = 0.7;
constexpr Real kFsw = 4e3;        // S1 toggle frequency [Hz]

// Vac (10 V, 1 kHz) -> D1 (V_th = 0.7) -> k -> L (1 mH) -> m -> R 10 Ω
// -> gnd, with S1 from m to gnd through 20 Ω, toggled at 4 kHz. D1 is
// added AFTER S1, so the diode's mask bit (1) follows the switch's (0):
// the record must follow SwitchStateMask indices, not "diodes first".
struct Rig {
    builder::CircuitBuilder b;
    Index d_bit = -1, s_bit = -1;
    Index i_l_var = -1;
    Index n_a = -1, n_k = -1, n_m = -1, n_x = -1;

    Rig() {
        b.add_sine_voltage_source("Vac", "a", "gnd", 0.0, 10.0, kF);
        b.add_switch("S1", "m", "x", kGon, kGoff);
        b.add_resistor("Rx", "x", "gnd", 20.0);
        b.add_diode("D1", "a", "k", kGon, kGoff, kVth);
        b.add_inductor("L", "k", "m", 1e-3);
        b.add_resistor("R", "m", "gnd", 10.0);
        d_bit = b.switch_index_of("D1");
        s_bit = b.switch_index_of("S1");
        i_l_var = b.pool().branch_var_id_for_inductor(
            b.branch_id_of("L"), b.graph());
        n_a = b.node_id_of("a");
        n_k = b.node_id_of("k");
        n_m = b.node_id_of("m");
        n_x = b.node_id_of("x");
    }

    [[nodiscard]] topology::SwitchStateMask gate(Real t) const {
        topology::SwitchStateMask m(b.graph().num_switches());
        m.set(static_cast<Size>(s_bit),
              std::fmod(t * kFsw, 1.0) < 0.5);
        return m;
    }
};

// Every recorded diode bit, applied to the stored v_D, reproduces the
// inductor's (state) current; every switch bit, applied to the stored
// v_S, closes KCL at node m against the same state current.
void check_record(const Rig& rig, const solver::SimulationResult& r) {
    REQUIRE(r.has_switch_states);
    REQUIRE(r.switch_states_complete());
    REQUIRE(r.num_switch_bits == 2);
    REQUIRE(r.switch_state_bits.size() == r.times.size());  // 1 byte
    Real i_peak = 0.0;
    for (Size k = 0; k < r.times.size(); ++k) {
        i_peak = std::max(i_peak, std::abs(r.states[k][rig.i_l_var]));
    }
    REQUIRE(i_peak > 0.5);
    Size n_on_in_band = 0;
    for (Size k = 0; k < r.times.size(); ++k) {
        const auto x = r.states[k];
        const Real v_d = x[rig.n_a] - x[rig.n_k];
        const bool on = r.switch_state(k, static_cast<Size>(rig.d_bit));
        const Real i_d = (on ? kGon : kGoff) * v_d;
        const Real i_l = x[rig.i_l_var];
        INFO("k=" << k << " t=" << r.times[k] << " v_D=" << v_d
                  << " on=" << on << " i_D=" << i_d << " i_L=" << i_l);
        REQUIRE(std::abs(i_d - i_l) <= 1e-7 * i_peak + 1e-9);
        const bool s_on = r.switch_state(k, static_cast<Size>(rig.s_bit));
        const Real v_s = x[rig.n_m] - x[rig.n_x];
        const Real i_s = (s_on ? kGon : kGoff) * v_s;
        const Real kcl_m = i_l - x[rig.n_m] / 10.0 - i_s;
        REQUIRE(std::abs(kcl_m) <= 1e-7 * i_peak + 1e-9);
        if (on && v_d > 0.0 && v_d < kVth && i_l > 0.1) {
            ++n_on_in_band;
        }
    }
    // The case the voltage rule cannot decide is actually exercised.
    REQUIRE(n_on_in_band > 5);
}

// Index of the stored sample nearest to `t`.
Size nearest(const solver::SimulationResult& r, Real t) {
    Size best = 0;
    for (Size k = 1; k < r.times.size(); ++k) {
        if (std::abs(r.times[k] - t) < std::abs(r.times[best] - t)) {
            best = k;
        }
    }
    return best;
}

}  // namespace

TEST_CASE("Stamped-mask record: fixed engine, diode in its V_th band",
          "[v2][record][run_transient]") {
    Rig rig;
    constexpr Real dt = 1e-6;
    pwl::PwlStateSpaceCache cache{rig.b.graph(), rig.b.pool()};
    cache.build(dt);
    solver::SimulationOptions opts{.t_start = 0.0, .t_end = 3e-3, .dt = dt};
    const auto r = solver::run_transient(
        cache, rig.b.graph(), rig.b.pool(), opts,
        [&](Real t) { return rig.gate(t); });
    REQUIRE(r.times.size() == opts.expected_sample_count());
    check_record(rig, r);
    // Zero IC: consistent with any mask, so nothing is flagged.
    REQUIRE(r.switch_state_unsolved.empty());
    // Known instants (third period): ON at the positive crest (with
    // v_D far below V_th — the band), OFF at the negative crest.
    const Size k_pos = nearest(r, 2.0e-3 + 0.25e-3);
    const Size k_neg = nearest(r, 2.0e-3 + 0.75e-3);
    REQUIRE(r.switch_state(k_pos, static_cast<Size>(rig.d_bit)));
    REQUIRE_FALSE(r.switch_state(k_neg, static_cast<Size>(rig.d_bit)));
    // Fixed engine: the switch bit is switch_fn at the step's END.
    for (Size k = 0; k < r.times.size(); ++k) {
        REQUIRE(r.switch_state(k, static_cast<Size>(rig.s_bit)) ==
                rig.gate(r.times[k]).get(static_cast<Size>(rig.s_bit)));
    }
}

TEST_CASE("Stamped-mask record: fixed engine with store_every and a "
          "seeded initial state",
          "[v2][record][run_transient]") {
    Rig rig;
    constexpr Real dt = 1e-6;
    pwl::PwlStateSpaceCache cache{rig.b.graph(), rig.b.pool()};
    cache.build(dt);
    solver::SimulationOptions opts{
        .t_start = 0.0, .t_end = 3e-3, .dt = dt, .store_every = 7};
    const Vector x0 = Vector::Zero(static_cast<Index>(
        rig.b.pool().state_size(rig.b.graph())));
    const auto r = solver::run_transient(
        cache, rig.b.graph(), rig.b.pool(), opts,
        [&](Real t) { return rig.gate(t); }, {}, false, {}, {}, &x0);
    REQUIRE(r.times.size() == opts.expected_sample_count());
    check_record(rig, r);
    // initial_state was supplied: sample 0 was solved by no mask.
    REQUIRE(r.switch_state_unsolved.size() == 1);
    REQUIRE(r.switch_state_unsolved[0] == 0);
}

TEST_CASE("Stamped-mask record: TR-BDF2 accepted steps and landings",
          "[v2][record][trbdf2]") {
    Rig rig;
    pwl::PwlStateSpaceCache cache{rig.b.graph(), rig.b.pool()};
    cache.build_lazy(1e-6);
    solver::TrBdf2Options o;
    o.t_start = 0.0;
    o.t_end = 3e-3;
    solver::TrBdf2Stats st;
    const auto r = solver::run_transient_trbdf2(
        cache, rig.b.graph(), rig.b.pool(), o,
        [&](Real t) { return rig.gate(t); }, {}, std::nullopt, &st);
    REQUIRE(st.n_diode_events > 2);   // landings were stored
    check_record(rig, r);
    REQUIRE(r.switch_state_unsolved.empty());
    const Size k_pos = nearest(r, 2.0e-3 + 0.25e-3);
    const Size k_neg = nearest(r, 2.0e-3 + 0.75e-3);
    REQUIRE(r.switch_state(k_pos, static_cast<Size>(rig.d_bit)));
    REQUIRE_FALSE(r.switch_state(k_neg, static_cast<Size>(rig.d_bit)));
    // TR-BDF2 stamps switch_fn at the step MIDPOINT; gate edges are
    // landed exactly, so the midpoint is inside one gate interval.
    for (Size k = 1; k < r.times.size(); ++k) {
        const Real mid = 0.5 * (r.times[k - 1] + r.times[k]);
        REQUIRE(r.switch_state(k, static_cast<Size>(rig.s_bit)) ==
                rig.gate(mid).get(static_cast<Size>(rig.s_bit)));
    }
}

TEST_CASE("Stamped-mask record: BDF1 records switch_fn's mask",
          "[v2][record][bdf1]") {
    builder::CircuitBuilder b;
    b.add_voltage_source("V", "in", "gnd", 5.0);
    b.add_switch("S", "in", "n", 1e3, 1e-9);
    b.add_resistor("R", "n", "gnd", 10.0);
    b.add_capacitor("C", "n", "gnd", 1e-6);
    const Size n_sw = b.graph().num_switches();
    auto gate = [n_sw](Real t) {
        topology::SwitchStateMask m(n_sw);
        m.set(0, std::fmod(t * 1e4, 1.0) < 0.3);
        return m;
    };
    solver::SimulationOptions opts{.t_start = 0.0, .t_end = 4e-4,
                                   .dt = 1e-6, .store_every = 3};
    const auto r = solver::run_transient_bdf1(b, opts, gate);
    REQUIRE(r.switch_states_complete());
    REQUIRE(r.switch_state_bits.size() == r.times.size());
    for (Size k = 0; k < r.times.size(); ++k) {
        REQUIRE(r.switch_state(k, 0) == gate(r.times[k]).get(0));
    }
}

TEST_CASE("Stamped-mask record: wide masks pack LSB-first",
          "[v2][record]") {
    solver::SimulationResult r;
    r.begin_switch_state_record(70);
    REQUIRE(r.switch_state_stride == 9);
    topology::SwitchStateMask m(70);
    m.set(0, true);
    m.set(9, true);
    m.set(64, true);
    m.set(69, true);
    r.times.push_back(0.0);
    r.record_switch_state(m, /*solved=*/false);
    REQUIRE(r.switch_states_complete());
    for (Size i = 0; i < 70; ++i) {
        REQUIRE(r.switch_state(0, i) == m.get(i));
    }
    REQUIRE(r.switch_state_unsolved == std::vector<Size>{0});
    REQUIRE(r.approx_bytes() >= 9);
    // A result no engine recorded says so.
    solver::SimulationResult blank;
    blank.times.push_back(0.0);
    REQUIRE_FALSE(blank.switch_states_complete());
}

// A resume's sample 0 is the previous run's last sample: same x, and
// the mask that x was solved under (SolverSnapshot::stamped_mask),
// solved. Before the field it was labelled with the carried diode bits
// + switch_fn(t_start): a TR-BDF2 run that ends exactly on a gate edge
// stores the pre-edge solve while its diode bits are already
// post-edge, so the label disagreed with x. check_record closes KCL on
// every row, row 0 included. A snapshot without the field (an older
// pickle) resumes with row 0 unsolved.
TEST_CASE("Stamped-mask record: a resume's row 0 is the snapshot's "
          "stamped mask",
          "[v2][record][resume]") {
    // 4 kHz gate: t = 1e-3 s is a gate edge (S1 turns ON).
    constexpr Real t_edge = 1e-3;
    constexpr Real t_end = 3e-3;
    constexpr Real dt = 1e-6;
    auto check_resume = [](const Rig& rig,
                           const solver::SimulationResult& r0,
                           const solver::SimulationResult& r1) {
        const auto& snap = r0.final_snapshot;
        REQUIRE(snap.valid);
        REQUIRE(snap.stamped_mask.size() == 2);
        const Size last = r0.times.size() - 1;
        for (Size i = 0; i < 2; ++i) {
            REQUIRE(snap.stamped_mask[i] == r0.switch_state(last, i));
            REQUIRE(r1.switch_state(0, i) == r0.switch_state(last, i));
        }
        REQUIRE(r1.switch_state_unsolved.empty());
        REQUIRE(r1.states[0].isApprox(r0.states[last]));
        check_record(rig, r1);
    };

    SECTION("fixed engine") {
        Rig rig;
        pwl::PwlStateSpaceCache cache{rig.b.graph(), rig.b.pool()};
        cache.build(dt);
        auto gate = [&](Real t) { return rig.gate(t); };
        solver::SimulationOptions o0{.t_start = 0.0, .t_end = t_edge,
                                     .dt = dt};
        const auto r0 = solver::run_transient(
            cache, rig.b.graph(), rig.b.pool(), o0, gate);
        solver::SimulationOptions o1{
            .t_start = r0.final_snapshot.t, .t_end = t_end, .dt = dt};
        const auto r1 = solver::run_transient(
            cache, rig.b.graph(), rig.b.pool(), o1, gate, {}, false, {},
            {}, nullptr, {}, &r0.final_snapshot);
        check_resume(rig, r0, r1);

        // No stamped_mask: row 0 is flagged unsolved, nothing else.
        solver::SolverSnapshot old = r0.final_snapshot;
        old.stamped_mask.clear();
        const auto r2 = solver::run_transient(
            cache, rig.b.graph(), rig.b.pool(), o1, gate, {}, false, {},
            {}, nullptr, {}, &old);
        REQUIRE(r2.switch_state_unsolved == std::vector<Size>{0});
    }

    SECTION("TR-BDF2, ending exactly on a gate edge") {
        Rig rig;
        pwl::PwlStateSpaceCache cache{rig.b.graph(), rig.b.pool()};
        cache.build_lazy(dt);
        auto gate = [&](Real t) { return rig.gate(t); };
        solver::TrBdf2Options o0;
        o0.t_start = 0.0;
        o0.t_end = t_edge;
        const auto r0 = solver::run_transient_trbdf2(
            cache, rig.b.graph(), rig.b.pool(), o0, gate);
        REQUIRE(r0.times.back() == Approx(t_edge));
        solver::TrBdf2Options o1;
        o1.t_start = r0.final_snapshot.t;
        o1.t_end = t_end;
        const auto r1 = solver::run_transient_trbdf2(
            cache, rig.b.graph(), rig.b.pool(), o1, gate, {},
            std::nullopt, nullptr, {}, {}, {}, {}, &r0.final_snapshot);
        check_resume(rig, r0, r1);
        // The stored sample is the pre-edge solve: S1 still OFF in the
        // record, while switch_fn(t_start) already has it ON.
        REQUIRE_FALSE(r1.switch_state(0, static_cast<Size>(rig.s_bit)));
        REQUIRE(rig.gate(o1.t_start).get(static_cast<Size>(rig.s_bit)));
    }
}
