"""Regression guards for the adversarial review of PRs #100-#105.

30 findings survived refutation, clustering into ten defects. These
pin the ones with user-visible behaviour; the C++-only ones (the
saturable-inductor dt mismatch, the sub-step breach accounting, the
undefined shift) are pinned by their own kernel guards and by the
comments at those sites.
"""

import warnings

import numpy as np
import pytest

import pulsim as p


def _rect(vpk=400.0, kappa=100.0):
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("Vac", "ac", "gnd", 0.0, vpk, 60.0)
    q = p.IdealDiodeParams()
    q.kappa = kappa
    b.add_nonlinear_diode("D", "ac", "vout", q)
    b.add_resistor("R", "vout", "gnd", 50.0)
    b.add_capacitor("C", "vout", "gnd", 100e-6)
    return b


# --- A: a Python callback's exception is not a step-size problem ---

def test_a_callback_error_is_not_retried_or_relabelled():
    """The retry caught every std::exception, which includes
    pybind11::error_already_set from the user's switch_fn — and that
    type's constructor CLEARS the Python error indicator, so a
    KeyboardInterrupt was swallowed and the run continued. A
    deterministic error was re-invoked up to 126 times and surfaced
    stripped of its type."""
    calls = {"n": 0}

    class Boom(RuntimeError):
        pass

    def exploding_switch(t):
        calls["n"] += 1
        if calls["n"] > 3:
            raise Boom("from the user's switch_fn")
        return p.SwitchStateMask(0)

    b = p.CircuitBuilder()
    b.add_voltage_source("V", "vin", "gnd", 12.0)
    b.add_resistor("R", "vin", "vout", 1.0)
    b.add_capacitor("C", "vout", "gnd", 1e-6)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(Boom):        # the ORIGINAL type, intact
            p.simulate(b, t_end=1e-4, dt=1e-6,
                        switch_fn=exploding_switch)
    # And it was not re-invoked a hundred more times looking for a
    # smaller step that could never help.
    assert calls["n"] < 20, calls["n"]


# --- F/G/L: the detectors have to be able to speak -----------------

def test_the_retry_says_it_happened():
    """The one new diagnostic that was recorded and never surfaced.
    Its own documentation says the user is entitled to know."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = p.simulate(_rect(), t_end=1.7e-2, dt=2e-4,
                          enable_nonlinear_refresh=True)
    assert len(res.dt_retries) >= 1
    hits = [w for w in caught if "re-taken at a smaller step"
            in str(w.message)]
    assert len(hits) == 1
    assert "dt_retries" in str(hits[0].message)


def test_the_voltage_check_survives_warnings_as_errors():
    """The detector sat inside `except Exception: pass`, so under
    -W error its own warning became an exception and was swallowed —
    no warning AND no attribute. The user who asked for warnings to
    be fatal is the last one who should be spared this."""
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", 48.0)
    b.add_inductor("L", "vin", "sw", 1e-3)
    b.add_switch("S", "sw", "gnd", 1e3, 1e-12)

    def sw(t, T=1e-4):
        m = p.SwitchStateMask(1)
        m.set(0, (t % T) < 0.5 * T)
        return m

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(UserWarning, match="largest voltage"):
            p.simulate(b, t_end=1e-3, dt=1e-8, switch_fn=sw)


def test_the_voltage_factor_is_reachable():
    """The bound is a judgement, so it has to be tunable from the
    entry point users actually call — not only from the binding."""
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", 48.0)
    b.add_inductor("L", "vin", "sw", 1e-3)
    b.add_switch("S", "sw", "gnd", 1e3, 1e-12)

    def sw(t, T=1e-4):
        m = p.SwitchStateMask(1)
        m.set(0, (t % T) < 0.5 * T)
        return m

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        p.simulate(b, t_end=1e-3, dt=1e-8, switch_fn=sw,
                    voltage_sanity_factor=1e9)
    assert [w for w in caught
            if "largest voltage" in str(w.message)] == []


# --- D: a dependent source defeats the bound -----------------------

def test_an_op_amp_circuit_is_not_falsely_accused():
    """`add_op_amp_ideal` is a VCVS at gain 1e5. Its output is a
    function of the circuit, so the bound built from independent
    sources alone would be ~1e5 times too small and every op-amp
    circuit would trip. The check declines to have an opinion
    instead."""
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", 1.0)
    b.add_resistor("R1", "vin", "vm", 1e3)
    b.add_resistor("Rf", "vm", "vout", 100e3)
    b.add_vcvs("A1", "vout", "gnd", "gnd", "vm", 1e5)
    b.add_resistor("Rl", "vout", "gnd", 10e3)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = p.simulate(b, t_end=1e-4, dt=1e-7)
    assert [w for w in caught
            if "largest voltage" in str(w.message)] == []
    assert getattr(res, "_implausible_voltage", None) is None
    assert np.isfinite(np.asarray(res.states)).all()


# --- J: the wreckage gets the same handles as the survivors --------

def test_the_partial_trace_can_resolve_names():
    """`e.partial` is a fresh pybind object with none of simulate()'s
    side-table attributes, so every name-based accessor failed with a
    message telling the user to do what they had already done."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(p.SimulationAborted) as exc:
            p.simulate(_rect(), t_end=1.7e-2, dt=2e-4,
                        enable_nonlinear_refresh=True,
                        max_dt_halvings=0)
    partial = exc.value.partial
    v = np.asarray(partial.v("vout"))
    assert len(v) == partial.num_steps()
    assert np.isfinite(v).all()
