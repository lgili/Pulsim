"""Every device the builder can add must report its real kind.

`DevicePool::kind_of` is what Python dispatches on — the engine
router's TR-BDF2 blocker list, `result.i()`'s device-family
refusal, and the schematic exporter all read it. A device missing
from the pybind enum comes back as the string `StoredKind.???`,
and every one of those consumers then silently treats it as
something it is not.

That is not hypothetical: `NonlinearCapacitor`, `ShockleyDiode`
and `LauritzenDiode` were all added to the C++ enum and to the
pool without being added to the binding, so all three reported
`???`. It surfaced when a router blocker written as
`kind.endswith("LauritzenDiode")` quietly never fired, which
would have routed a stored-charge diode onto an engine that does
not carry its history — converging, and reporting recovery that
never happened.

The C++ side already has a compile-time guard keeping `StoredKind`
and the variant in lock-step. This is the same guard for the
Python side, written as a property rather than a count so it needs
no bookkeeping when a device is added.
"""

import pulsim as p


def _one_of_everything():
    """One instance of every `add_*` that maps to a StoredKind."""
    b = p.CircuitBuilder()
    b.add_resistor("R1", "a", "gnd", 1.0)
    b.add_voltage_source("V1", "a", "gnd", 1.0)
    b.add_switch("S1", "a", "b", 1e3, 1e-9)
    b.add_capacitor("C1", "b", "gnd", 1e-6)
    b.add_inductor("L1", "b", "c", 1e-3)
    b.add_diode("D1", "c", "gnd", 1e3, 1e-9, 0.7)
    b.add_nonlinear_diode("D2", "c", "d",
                           p.IdealDiodeParams())
    b.add_current_source("I1", "d", "gnd", 1.0)
    b.add_pwm_voltage_source("VP", "e", "gnd", 10.0, 0.0,
                              1e5, 0.5)
    b.add_sine_voltage_source("VS", "f", "gnd", 0.0, 10.0,
                               50.0)
    b.add_pulse_voltage_source("VU", "g2", "gnd", 0.0, 5.0,
                                1e-6, 5e-7, 1e-6)
    b.add_mosfet_level1("M1", "h", "gnd", "g", 1e-3, 2.0)
    b.add_igbt_level1("Q1", "i", "gnd", "g")
    b.add_vcvs("E1", "a", "gnd", "j", "gnd", 2.0)
    b.add_saturable_inductor("L2", "k", "gnd", 1e-3, 1.0, 2.0)
    b.add_nonlinear_capacitor("C2", "m", "gnd", 2e-9, 25.0, 0.5)
    b.add_shockley_diode("D3", "n", "gnd")
    b.add_lauritzen_diode("D4", "o", "gnd")
    b.add_pmsm_mna("M2", "p1", "p2", "p3", "nn2", "w2", "th2",
                   0.5, 1e-3, 3e-3, 0.05, 4, 1e-3)
    return b


def test_no_device_reports_an_unknown_kind():
    b = _one_of_everything()
    unknown = [
        (bid, str(b.pool.kind_of(bid)))
        for bid in (br["id"] for br in b.graph.branches)
        if "?" in str(b.pool.kind_of(bid))
    ]
    assert not unknown, unknown


def test_the_new_level_two_devices_are_named():
    """The three that were missing, pinned by name."""
    b = _one_of_everything()
    def kind(name):
        return str(b.pool.kind_of(b.branch_id_of(name)))

    assert kind("C2") == "StoredKind.NonlinearCapacitor"
    assert kind("D3") == "StoredKind.ShockleyDiode"
    assert kind("D4") == "StoredKind.LauritzenDiode"
    assert kind("M2_a") == "StoredKind.PmsmMna"
