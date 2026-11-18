// =============================================================================
// DSED — the kernel batch evaluator of pulsim.design
// =============================================================================
//
// add-design-optimization-workflow DECISIONS 26
// (pulsim/dsed/design_batch.hpp): numpy's pairwise sum and CPython's
// compensated sum as the register program runs them, and a table of buck
// designs on 1 and 4 threads — equal outputs, input order kept, a bad
// design returned as a fallback with its reason, never thrown.

#include <catch2/catch_test_macros.hpp>

#include "pulsim/builder/circuit_builder.hpp"
#include "pulsim/dsed/design_batch.hpp"
#include "pulsim/dsed/scheduler.hpp"
#include "pulsim/ir/circuit.hpp"

#include <cmath>
#include <vector>

using namespace pulsim;
namespace dz = pulsim::dsed::design;

namespace {

using SSM = topology::SwitchStateMask;

struct Gate {
    Real T, D;
    SSM operator()(Real t) const {
        SSM m(1);
        const Real ph = t - T * std::floor(t / T);
        m.set(0, ph < D * T);
        return m;
    }
    Real next_edge_after(Real t) const {
        const Real k = std::floor(t / T);
        for (Real c : {k * T + D * T, (k + 1) * T, (k + 1) * T + D * T}) {
            if (c > t * (1 + 1e-15) + 1e-300) return c;
        }
        return (k + 2) * T;
    }
};

struct Run {
    template <class Ad>
    dsed::PeriodMapResult<SSM> operator()(
        Ad& ad, const Gate& g, const Vector& x0, Real T,
        const std::vector<dsed::PeriodObservable>& obs,
        const std::vector<std::pair<int, int>>& pairs, Real) const {
        dsed::EventPredictor pred;
        auto diodes = dsed::add_diode_predicates(ad, pred);
        dsed::PEDSimulator<Ad, Gate> sim(ad, g, dsed::PIController(1e-11, 1e-14),
                                         std::move(pred), T * 1e-6, T, 1);
        if (!diodes.list.empty()) {
            sim.enable_diode_commutation(diodes.owned, SSM(1), diodes.list);
        }
        sim.set_event_projection(false);
        const auto res = sim.simulate(x0, T);
        return dsed::period_map_from_run(ad, res, diodes, obs, pairs, true);
    }
};

dz::Instr op(dz::Op o, int a = 0, int b = 0, Real c = 0) { return {o, a, b, c}; }

}  // namespace

TEST_CASE("Design batch: numpy pairwise and CPython compensated sums",
          "[v2][dsed][design_batch]") {
    // below 8 terms: a plain loop from 0
    REQUIRE(dz::detail::python_total({1.0, 2.0, 3.0}) == 6.0);
    // 9 terms: eight accumulators, then the ninth added last
    std::vector<Real> e{1e16, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    Real r[8];
    for (int j = 0; j < 8; ++j) r[j] = e[static_cast<std::size_t>(j)];
    const Real want = (((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]))) + e[8];
    REQUIRE(dz::detail::python_total(e) == want);
    // SumN: 1e16 + 1 + 1 compensated is exactly 1e16 + 2
    dz::Plan plan;
    plan.code = {op(dz::Op::Const, 0, 0, 1e16), op(dz::Op::Const, 0, 0, 1.0),
                 op(dz::Op::Const, 0, 0, 1.0), op(dz::Op::SumN, 3),
                 op(dz::Op::Store, 0)};
    plan.n_regs = 1;
    std::vector<Real> regs;
    dsed::PeriodMapResult<SSM> p;
    dz::detail::run_program(plan, nullptr, {}, p, {}, 1.0, regs);
    REQUIRE(regs[0] == 1e16 + 2.0);
    REQUIRE(1e16 + 1.0 + 1.0 == 1e16);   // what a plain loop gives
}

TEST_CASE("Design batch: a table on 1 and 4 threads, in order, fallbacks named",
          "[v2][dsed][design_batch]") {
    builder::CircuitBuilder b;
    b.add_voltage_source("Vin", "vin", "gnd", 24.0);
    b.add_switch("S", "vin", "sw", 1e3, 1e-9);
    b.add_diode("D1", "gnd", "sw", 1e3, 1e-9);
    b.add_inductor("L", "sw", "vout", 100e-6);
    b.add_capacitor("C", "vout", "gnd", 47e-6);
    b.add_resistor("RL", "vout", "gnd", 5.0);
    const auto base = ir::Circuit::from_builder(b);
    const Index vout = b.node_id_of("vout");

    dz::Plan plan;
    plan.names = {"L.L"};
    plan.is_param = {false};
    dsed::PeriodObservable o;
    o.w = {{vout, 1.0}};
    o.extrema = true;
    plan.obs = {o};
    // v_out = int1 / T
    plan.code = {op(dz::Op::Stat, 0, 0), op(dz::Op::Period), op(dz::Op::Div),
                 op(dz::Op::Store, 0)};
    plan.n_regs = 1;
    plan.kpi_regs = {0};
    plan.mode_guard = true;
    const Real T = 1e-5;
    const Gate g{T, 0.5};
    std::vector<Real> rows{60e-6, 100e-6, -1.0, 220e-6, 330e-6, 470e-6};
    std::vector<Real> periods(rows.size(), T);
    std::vector<const Gate*> scheds(rows.size(), &g);
    auto one = dz::evaluate_batch<Gate>(base, plan, rows.data(), rows.size(),
                                        periods, scheds, Run{}, 1);
    auto four = dz::evaluate_batch<Gate>(base, plan, rows.data(), rows.size(),
                                         periods, scheds, Run{}, 4);
    for (std::size_t i = 0; i < rows.size(); ++i) {
        REQUIRE(one[i].status == four[i].status);
        REQUIRE(one[i].kpis == four[i].kpis);
        REQUIRE(one[i].x0 == four[i].x0);
        if (i == 2) {        // L = -1: with_value's bound, not a throw
            REQUIRE(one[i].status == dz::Status::Fallback);
            REQUIRE(one[i].reason.find("kernel error") == 0);
            continue;
        }
        REQUIRE(one[i].status == dz::Status::Ok);
        // CCM at D = 0.5 into 5 ohm: v_out within 2 % of D Vin
        REQUIRE(std::abs(one[i].kpis[0] - 12.0) < 0.24);
        REQUIRE(one[i].n_runs >= 2);
    }
}

TEST_CASE("Design batch: a cancellation stops the batch and is re-thrown",
          "[v2][dsed][design_batch]") {
    // it was caught as a std::runtime_error and turned into a fallback,
    // so a cancelled batch finished every design and re-ran it in Python
    builder::CircuitBuilder b;
    b.add_voltage_source("Vin", "vin", "gnd", 24.0);
    b.add_switch("S", "vin", "sw", 1e3, 1e-9);
    b.add_diode("D1", "gnd", "sw", 1e3, 1e-9);
    b.add_inductor("L", "sw", "vout", 100e-6);
    b.add_capacitor("C", "vout", "gnd", 47e-6);
    b.add_resistor("RL", "vout", "gnd", 5.0);
    const auto base = ir::Circuit::from_builder(b);
    dz::Plan plan;
    plan.names = {"L.L"};
    plan.is_param = {false};
    plan.n_regs = 1;
    const Real T = 1e-5;
    const Gate g{T, 0.5};
    std::vector<Real> rows(64, 100e-6);
    std::vector<Real> periods(rows.size(), T);
    std::vector<const Gate*> scheds(rows.size(), &g);
    auto cancel = [](auto&, const Gate&, const Vector&, Real,
                     const std::vector<dsed::PeriodObservable>&,
                     const std::vector<std::pair<int, int>>&,
                     Real) -> dsed::PeriodMapResult<SSM> {
        throw analysis::Cancelled("design batch");
    };
    for (int nt : {1, 4}) {
        REQUIRE_THROWS_AS(dz::evaluate_batch<Gate>(base, plan, rows.data(),
                                                   rows.size(), periods, scheds,
                                                   cancel, nt),
                          analysis::Cancelled);
    }
}
