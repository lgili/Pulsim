// add-external-device-abi: loading, descriptor validation and the
// registration checks — every lying test model is refused by name, the
// honest ones and the twins pass. The libraries are built by this test
// build (CMake passes their paths); engine runs are in
// python/tests/test_external_device.py.
#include "pulsim/xdev/checks.hpp"
#include "pulsim/xdev/host.hpp"

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <string>
#include <vector>

using namespace pulsim;

namespace {
std::string refusal(const std::string& lib, const std::string& model,
                    std::vector<Real> p = {1e-3}) {
    try {
        auto m = xdev::load_model(lib, model);
        xdev::Instance in(model + "_1", m, std::move(p));
        xdev::check_contract(in);
    } catch (const xdev::ContractError& e) {
        return e.what();
    }
    return {};
}
bool has(const std::string& s, const std::string& sub) {
    return s.find(sub) != std::string::npos;
}
}  // namespace

TEST_CASE("xdev: honest models and both twins pass registration", "[xdev]") {
    CHECK(refusal(PULSIM_XDEV_CONTRACT, "good_resistor") == std::string{});
    CHECK(refusal(PULSIM_XDEV_CONTRACT, "energy_meter") == std::string{});
    CHECK(refusal(PULSIM_XDEV_CONTRACT, "nan_outside") == std::string{});   // lie outside the box
    CHECK(refusal(PULSIM_XDEV_CONTRACT, "liar_outside") == std::string{});  // lie outside the box
    CHECK(refusal(PULSIM_XDEV_CONTRACT, "tiny_cap_outside") == std::string{});  // lie outside the box
    CHECK(refusal(PULSIM_XDEV_TWINS, "satind_twin", {1e-3, 1.0, 1e-4}) == std::string{});
    CHECK(refusal(PULSIM_XDEV_TWINS, "shockley_twin", {1e-12, 1.0, 0.025852, 1e-12, 1e6, 0.0})
               == std::string{});
}

TEST_CASE("xdev: every lie inside the probe box is refused by name", "[xdev]") {
    struct Case {
        const char* model;
        const char* says;
    };
    const Case cases[] = {
        {"liar_jacobian", "is not the derivative"},
        {"liar_pattern", "missing from the declared pattern"},
        {"liar_kink", "claims SMOOTH"},
        {"liar_nonmonotone", "non-decreasing"},
        {"liar_active", "claims PASSIVE"},
        {"liar_reciprocal", "claims RECIPROCAL"},
        {"liar_impure", "not pure"},
        {"liar_kcl", "KCL"},
        {"liar_overrun", "wrote past the end of df"},
        {"liar_state", "wrote past the declared n_state"},
        {"liar_abi", "ABI version 99"},
        {"liar_writes_x", "wrote through its const x"},
        {"liar_writes_p", "wrote through its const ctx->p"},
        {"liar_offaxis", "off the axes"},
        {"liar_state_via_ctx", "wrote through a const input"},
    };
    for (const auto& c : cases) {
        INFO(c.model);
        const auto r = refusal(PULSIM_XDEV_CONTRACT, c.model);
        INFO(r);
        CHECK(has(r, c.says));
        CHECK(has(r, c.model));   // the model (and, past the descriptor, the device) by name
    }
}

TEST_CASE("xdev: loader and parameter refusals", "[xdev]") {
    CHECK(has(refusal(PULSIM_XDEV_CONTRACT, "no_such_model"), "has no model 'no_such_model'"));
    CHECK(has(refusal("/nonexistent/libnothing.so", "x"), "/nonexistent/libnothing.so"));
    CHECK(has(refusal(PULSIM_XDEV_CONTRACT, "good_resistor", {-1.0}),
              "outside its declared bounds"));
    CHECK(has(refusal(PULSIM_XDEV_TWINS, "satind_twin", {1e-3, 1.0, 2e-3}), "L_residual"));
    auto m = xdev::load_model(PULSIM_XDEV_TWINS, "satind_twin");
    try {
        (void)xdev::resolve_params(*m, "L1", {{"L_0", 1e-3, "mH", true}});
        FAIL("a unit mismatch must be refused");
    } catch (const xdev::ContractError& e) {
        CHECK(has(e.what(), "declares 'H'"));
    }
    try {
        (void)xdev::resolve_params(*m, "L1", {{"L0", 1e-3, "", false}});
        FAIL("an unknown name must be refused");
    } catch (const xdev::ContractError& e) {
        CHECK(has(e.what(), "no parameter 'L0'"));
    }
}

TEST_CASE("xdev: run-time finiteness and the debug Jacobian check", "[xdev]") {
    auto nan = xdev::load_model(PULSIM_XDEV_CONTRACT, "nan_outside");
    xdev::Instance a("Xn", nan, {1e-3});
    xdev::EvalOut o;
    a.size(o);
    const Real x[2] = {10.0, 0.0};
    try {
        a.eval(x, nullptr, o);
        FAIL("a NaN out of the model must be refused");
    } catch (const xdev::ContractError& e) {
        CHECK(has(e.what(), "'Xn'"));
        CHECK(has(e.what(), "non-finite f[0]"));
    }
    auto lie = xdev::load_model(PULSIM_XDEV_CONTRACT, "liar_outside");
    xdev::Instance b("Xo", lie, {1e-3}, /*debug_jacobian=*/true);
    b.size(o);
    const Real y[2] = {4.0, 0.0};
    try {
        b.eval(y, nullptr, o);
        FAIL("the debug check must catch the Jacobian outside the box");
    } catch (const xdev::ContractError& e) {
        CHECK(has(e.what(), "run-time debug check"));
    }
    // a 0.1 pF capacitance: dq/dx = 1e-13, below the old absolute 1e-12 floor
    auto cap = xdev::load_model(PULSIM_XDEV_CONTRACT, "tiny_cap_outside");
    xdev::Instance c("Xc", cap, {1e-3}, /*debug_jacobian=*/true);
    c.size(o);
    const Real in_box[2] = {0.5, 0.0};
    CHECK_NOTHROW(c.eval(in_box, nullptr, o));
    try {
        c.eval(y, nullptr, o);
        FAIL("the debug check must catch a doubled sub-pF capacitance");
    } catch (const xdev::ContractError& e) {
        CHECK(has(e.what(), "declared dq[0]/dx[0]"));
    }
}

TEST_CASE("xdev: the reviewer's open list (#16) and the solve's time", "[xdev]") {
    xdev::EvalOut o;
    // a finite output whose SUM overflows is not a refusal
    auto huge = xdev::load_model(PULSIM_XDEV_CONTRACT, "huge_outside");
    xdev::Instance h("Xh", huge, {1e-3});
    xdev::check_contract(h);
    h.size(o);
    const Real out_of_box[2] = {5.0, 0.0};
    CHECK_NOTHROW(h.eval(out_of_box, nullptr, o));
    CHECK(o.df[0] == 1e308);
    // an honest kink (no SMOOTH claim) at the probe centre: one-sided slopes
    CHECK(refusal(PULSIM_XDEV_CONTRACT, "honest_kink").empty());
    auto kink = xdev::load_model(PULSIM_XDEV_CONTRACT, "honest_kink");
    xdev::Instance k("Xk", kink, {1e-3}, /*debug_jacobian=*/true);
    k.size(o);
    const Real at_kink[2] = {0.0, 0.0};
    CHECK_NOTHROW(k.eval(at_kink, nullptr, o));
    // a kink CLAIMED smooth stays refused
    CHECK(has(refusal(PULSIM_XDEV_CONTRACT, "liar_kink"), "claims SMOOTH"));
    // the time reaches ctx->t, with host_features saying so
    auto ramp = xdev::load_model(PULSIM_XDEV_CONTRACT, "ramp_conductance");
    xdev::Instance r("Xr", ramp, {1e-3});
    r.size(o);
    const Real v1[2] = {1.0, 0.0};
    r.eval(v1, nullptr, o, /*t=*/2e-3);
    CHECK(std::abs(o.f[0] - 3e-3) <= 1e-15);   // G (1 + t / 1 ms) at t = 2 ms
    CHECK((r.ctx_(nullptr, 1.0).host_features & PULSIM_XDEV_HOST_TIME) != 0u);
}
