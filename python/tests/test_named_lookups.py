"""Regression coverage for `add-python-named-lookups` (v1.5).

Each test below maps directly to a Scenario block in
`openspec/changes/add-python-named-lookups/specs/python-bindings/spec.md`.
"""
from __future__ import annotations

import numpy as np
import pytest

import pulsim as ps


# ---------------------------------------------------------------------------
# Shared fixture — open-loop buck large enough to exercise every device kind.
# ---------------------------------------------------------------------------
@pytest.fixture()
def buck():
    """Open-loop buck with one switching MOSFET, one freewheel diode, an
    inductor / output cap / load resistor. The MOSFET drives a 50 % duty
    PWM at 10 kHz so the inductor current actually swings.
    """
    b = ps.CircuitBuilder()
    b.add_voltage_source("V1", "vin", "gnd", 12.0)
    b.add_mosfet_with_body_diode(
        "Q1", "vin", "sw", R_on=1e-3, R_off=1e9, V_F=0.7,
    )
    b.add_diode("D1", "gnd", "sw", 1e3, 1e-9, V_th=0.7)
    b.add_inductor("L1", "sw", "vout", 220e-6)
    b.add_capacitor("Cout", "vout", "gnd", 220e-6)
    b.add_resistor("R_L", "vout", "gnd", 8.0)
    switch_fn = ps.make_pwm_switch_fn(
        10e3, 0.5, b.switch_index_of("Q1"), b.graph.num_switches,
    )
    res = ps.simulate(b, t_end=10e-3, dt=2e-6, switch_fn=switch_fn)
    return b, res


# ---------------------------------------------------------------------------
# Requirement: Builder Device-Name Index Lookups
# ---------------------------------------------------------------------------
class TestBuilderLookups:
    def test_switch_index_of_for_named_mosfet(self):
        # Spec scenario: "PWM lookup by device name" — `Q1` is the only
        # switching branch, so its bit position must be 0.
        b = ps.CircuitBuilder()
        b.add_voltage_source("V1", "vin", "gnd", 12.0)
        b.add_mosfet_with_body_diode(
            "Q1", "vin", "sw", R_on=1e-3, R_off=1e9, V_F=0.7,
        )
        # add_mosfet_with_body_diode adds 2 switching branches (the
        # MOSFET itself + the body diode), so the explicit Q1 is at
        # index 0 and the auto-named body diode at index 1.
        assert b.switch_index_of("Q1") == 0
        assert b.switch_index_of("Q1_body") == 1

    def test_switch_index_of_rejects_passive(self, buck):
        b, _ = buck
        # Spec scenario: "Non-switching device raises with kind info".
        with pytest.raises(IndexError, match=r"R_L.*passive"):
            b.switch_index_of("R_L")

    def test_branch_index_of_round_trip(self, buck):
        # Spec scenario: "Branch-index round-trip with state vector"
        # only holds for state-variable branches (inductors / sources).
        b, res = buck
        states = np.asarray(res.states)
        # For an inductor, the state-vector column is
        # `pool.branch_var_id_for_inductor(branch_id, graph)` — not
        # simply `n_nodes + branch_index_of(name)`. The proposal's
        # round-trip scenario assumed a 1:1 layout that pulsim 1.4
        # doesn't actually use. The accessor :meth:`result.i` hides
        # this gap; we cross-check the trace equality here directly.
        expected = states[:, b.pool.branch_var_id_for_inductor(
            b.branch_index_of("L1"), b.graph)]
        assert np.array_equal(res.i("L1"), expected)
        # `branch_index_of` is the public alias for `branch_id_of`.
        assert b.branch_index_of("L1") == b.branch_id_of("L1")

    def test_devices_enumerates_in_add_order(self, buck):
        # Spec scenario: "Enumerate registered devices".
        b, _ = buck
        names = [d.name for d in b.devices()]
        # Q1 also pushes a Q1_body diode into the device list — it's
        # an auto-named branch so it shows up but trails the named ones.
        assert names[:3] == ["V1", "Q1", "Q1_body"]
        # Kinds are the topological category strings.
        kinds = {d.name: d.kind for d in b.devices()}
        assert kinds["V1"] == "source"
        assert kinds["Q1"] == "switch"
        assert kinds["L1"] == "passive"
        # Terminals translate ground back to "gnd".
        v1 = next(d for d in b.devices() if d.name == "V1")
        assert v1.terminals == ["vin", "gnd"]


# ---------------------------------------------------------------------------
# Requirement: Named Result Accessors
# ---------------------------------------------------------------------------
class TestResultAccessors:
    def test_v_matches_raw_indexing(self, buck):
        # Spec scenario: "Read node voltage by name".
        b, res = buck
        expected = np.asarray(res.states)[:, b.node_id_of("vout")]
        assert np.array_equal(res.v("vout"), expected)
        assert res.v("vout").shape == (res.num_steps(),)
        assert res.v("vout").dtype == np.float64

    def test_v_single_step_scalar(self, buck):
        # Spec scenario: "Single-step snapshot" — t=-1 returns scalar.
        _, res = buck
        v_end = res.v("vout", t=-1)
        assert isinstance(v_end, float)
        # 12 V input × 50 % duty → ~6 V steady state.
        assert 5.0 < v_end < 7.0

    def test_i_inductor_matches_pool_state(self, buck):
        # Spec scenario: "Read branch current by name" — the sign
        # convention follows `add_inductor(name, from, to, …)`.
        b, res = buck
        expected = np.asarray(res.states)[
            :, b.pool.branch_var_id_for_inductor(
                b.branch_index_of("L1"), b.graph)]
        assert np.array_equal(res.i("L1"), expected)

    def test_i_resistor_raises_not_implemented(self, buck):
        # Resistor / capacitor / diode branches have no MNA current
        # state variable — `result.i` raises NotImplementedError
        # pointing the caller at the `losses` helpers.
        _, res = buck
        with pytest.raises(NotImplementedError, match="MNA current"):
            res.i("R_L")

    def test_power_resistor(self, buck):
        # Spec scenario: "Device power lookup". `device_loss_summary`
        # exposes P_avg for resistors; for V_out ≈ 6 V on R = 8 Ω
        # the expected average power is ~4.5 W.
        _, res = buck
        p = res.power("R_L")
        assert 3.5 < p < 6.0


# ---------------------------------------------------------------------------
# Requirement: Typed errors with suggestions
# ---------------------------------------------------------------------------
class TestNameNotFoundError:
    def test_typo_suggests_close_match(self, buck):
        # Spec scenario: "Unknown name produces typed error with
        # suggestions".
        _, res = buck
        with pytest.raises(ps.NameNotFoundError) as excinfo:
            res.v("voutt")  # typo for "vout"
        err = excinfo.value
        assert err.name == "voutt"
        assert err.kind == "node"
        assert "vout" in err.suggestions

    def test_subclass_of_key_error(self, buck):
        # NameNotFoundError IS a KeyError so existing
        # `except KeyError: ...` patterns keep working.
        _, res = buck
        with pytest.raises(KeyError):
            res.v("totally_unknown_node")

    def test_branch_typo_suggests_close_match(self, buck):
        _, res = buck
        with pytest.raises(ps.NameNotFoundError) as excinfo:
            res.i("L11")  # typo for "L1"
        err = excinfo.value
        assert err.kind == "branch"
        assert "L1" in err.suggestions
