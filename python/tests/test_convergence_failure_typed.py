"""The exact engine's give-ups reach Python typed (integration batch 9).

C++ ``pulsim::ConvergenceFailure`` (eight sites in dsed/scheduler.hpp and
scheduler_auto.hpp, five kinds) is translated to ``pulsim.ConvergenceError``
with ``kind`` / ``t_failed`` / ``device``; a design evaluation turns each
into a ``failed`` record without reading the message.
"""

import pytest

import pulsim as p
from pulsim.design import SteadySpec

KINDS = ("settle", "chatter", "no_progress", "max_steps", "nan")


def _chattering_rectifier(_design=None):
    # core/tests/dsed/test_convergence_failure.cpp, chatter case: tau = 1 ps
    # at 500 MHz, the diode commutes twice per 2 ns period (deterministic).
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("V", "s", "gnd", 0.0, 10.0, 5e8, 0.0)
    b.add_diode("D1", "s", "o", 1e3, 1e-9, 0.0)
    b.add_resistor("R", "o", "gnd", 1e3)
    b.add_capacitor("C", "o", "gnd", 1e-15)
    return b


@pytest.mark.parametrize("engine", ["dsed", "exact"])
def test_a_chattering_diode_is_a_typed_convergence_error(engine):
    with pytest.raises(p.ConvergenceError) as ei:
        p.simulate(_chattering_rectifier(), t_end=1e-6, engine=engine)
    e = ei.value
    assert type(e) is p.ConvergenceError            # not SimulationAborted
    assert isinstance(e, RuntimeError)              # old `except RuntimeError` still works
    assert e.kind == "chatter"
    assert e.device.startswith("D1 ")
    assert 0.0 <= e.t_failed < 2e-7
    assert f"{e.device} is chattering" in str(e)


def test_a_real_kernel_give_up_is_a_failed_design_record(monkeypatch):
    # The evaluator's own transient of a design lands on other checks first, so its
    # simulate() is replaced by the chattering run: the exception object is the
    # translator's, raised from C++.
    real = p.simulate

    def chatter(*a, **k):
        return real(_chattering_rectifier(), t_end=1e-6, engine="dsed")
    monkeypatch.setattr(p, "simulate", chatter)
    from . import test_design_evaluation as tde
    rec = tde._ev(engine="dsed", steady=SteadySpec(method="transient")).evaluate(tde.CCM)
    assert rec.status == "failed"
    assert rec.reason.startswith("ConvergenceError: ")
    assert "chattering" in rec.reason


@pytest.mark.parametrize("kind", KINDS)
def test_every_kind_is_failed_by_type_not_by_message(monkeypatch, kind):
    def give_up(*a, **k):
        e = p.ConvergenceError("an opaque message")   # no keyword to match on
        e.kind, e.t_failed, e.device = kind, 1e-3, ""
        raise e
    monkeypatch.setattr(p, "simulate", give_up)
    monkeypatch.setattr(p._pulsim, "dsed_period_map", give_up)
    monkeypatch.setattr(p._pulsim, "dsed_periodic_orbit", give_up)
    from . import test_design_evaluation as tde
    rec = tde._ev().evaluate(tde.CCM)
    assert rec.status == "failed"
    assert rec.reason.endswith("an opaque message")


def test_a_give_up_on_the_native_adapter_does_not_fall_through_to_the_next_bridge(monkeypatch):
    # A diode-free circuit: the native path's `except RuntimeError` fell through to the Python
    # adapter (Bridge.10), re-running the circuit without the implicit-segment refusal.
    calls = []

    def give_up(*a, **k):
        e = p.ConvergenceError("PEDSimulator: exceeded max-step cap")
        e.kind, e.t_failed, e.device = "max_steps", 1e-3, ""
        raise e

    def bridge10(*a, **k):
        calls.append(1)
        raise AssertionError("fell through to Bridge.10")
    for name in ("run_ped_native_builder", "run_bdf2_native_builder", "run_auto_native_builder"):
        monkeypatch.setattr(p._pulsim, name, give_up)
    for name in ("run_ped_native", "run_bdf2_native", "run_auto_native"):
        monkeypatch.setattr(p._pulsim, name, bridge10)
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "in", "gnd", 10.0)
    b.add_resistor("R", "in", "o", 1e3)
    b.add_capacitor("C", "o", "gnd", 1e-6)
    with pytest.raises(p.ConvergenceError, match="max-step cap"):
        p.simulate(b, t_end=1e-3, engine="exact")
    assert calls == []
