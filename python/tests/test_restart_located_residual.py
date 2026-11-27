"""A located event's residual is not a jump (restart line, item 2).

The release's findings F-Y-2 / F-Y-4 and the DCM buck of
test_sensitivity_events: at a diode turn-off located by event splitting the
state left by the locator (its residual current, 4.5e-7 A of 15 A, 3.1e-5 A
of 20 A, 6e-5 A of 51.5 A) was recorded as a cut-set "jump" (2e-15 J) or
refused as one (a saturable core). The rule: a jump is recorded only above the
event-time uncertainty times the state's rate plus the solve's tolerance;
below it the state is continuous — no record, no refusal, no restart from a
moved state.

Every run passes the release defaults EXPLICITLY (benchmarks/tools/
next_defaults.py --list: restart rule, event splitting, mode_selection='auto',
control_tick='state', device_path='interface', branch-form switches and
diodes), so these hold on this line before the defaults flip.
"""

import warnings

import numpy as np
import pytest

import pulsim as p

RELEASE = dict(engine="pwl", discontinuity_restart=True, enable_substep_state_correction=True,
               mode_selection="auto", control_tick="state", device_path="interface")


def _sim(b, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return p.simulate(b, **{**RELEASE, **kw})


def _freewheel_into(i0, L=1e-6, V=100.0, dt=1e-7):
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "vo", "gnd", V)
    b.add_diode("D", "gnd", "a", 1e3, 1e-9, form="branch")
    b.add_inductor("L", "a", "vo", L, i0=i0)
    return _sim(b, t_end=2e-6, dt=dt)


@pytest.mark.parametrize("i0", [12.0, 22.0, 51.5, 55.0])
def test_a_located_zero_current_stop_writes_no_record(i0):
    """F-Y-4 (6e-5 A of 51.5 A, 2.0e-15 J on the release line). Truth (closed form, R_on =
    1 mOhm of the ON diode): i(t) = (i0 + V/R) e^(-Rt/L) - V/R until 0, then 0."""
    L, V, R = 1e-6, 100.0, 1e-3
    r = _freewheel_into(i0)
    t, i = np.asarray(r.times), np.asarray(r.i("L"))
    ic = np.maximum((i0 + V / R) * np.exp(-R * t / L) - V / R, 0.0)
    t_c = -L / R * np.log((V / R) / (i0 + V / R))
    far = np.abs(t - t_c) > 1e-7
    assert np.max(np.abs(i - ic)[far]) < 1e-6 * i0
    assert list(r.reinit_records) == []


@pytest.mark.parametrize("saturable", [False, True])
@pytest.mark.parametrize("edge_at", [None, 0.8e-6])
def test_a_dcm_freewheel_stop_is_neither_recorded_nor_refused(saturable, edge_at):
    """F-Y-2 (3.09e-5 A of 20.4 A refused on a saturable core) and F-Y-4 (4.8e-16 J). Truth:
    i rises at (100 - 49.08)/L to 20.368 A at 0.4 us, falls at 49.08/L to 0 at 0.815 us."""
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", 100.0)
    b.add_voltage_source("Vo", "vo", "gnd", 49.08)
    b.add_switch("S", "vin", "a", 1e3, 1e-9, form="branch")
    b.add_diode("D", "gnd", "a", 1e3, 1e-9, form="branch")
    if saturable:
        b.add_saturable_inductor("L", "a", "vo", 1e-6, 1e3, 1e-6)
    else:
        b.add_inductor("L", "a", "vo", 1e-6)
    b.add_voltage_source("V2", "v2", "gnd", 10.0)
    b.add_switch("S2", "v2", "q", 1e3, 1e-9, form="branch")
    b.add_resistor("R2", "q", "gnd", 10.0)
    n, k, k2 = b.graph.num_switches, b.switch_index_of("S"), b.switch_index_of("S2")

    def fn(t):
        m = p.SwitchStateMask(n)
        m.set(k, t < 0.4e-6 - 1e-15)
        m.set(k2, edge_at is not None and t >= edge_at - 1e-15)
        return m
    r = _sim(b, t_end=1.2e-6, dt=1e-7, switch_fn=fn)
    assert list(r.reinit_records) == []
    t, i = np.asarray(r.times), np.asarray(r.i("L"))
    ip = 50.92e6 * 0.4e-6
    ic = np.where(t <= 0.4e-6, 50.92e6 * t, np.maximum(ip - 49.08e6 * (t - 0.4e-6), 0.0))
    far = np.abs(t - (0.4e-6 + ip / 49.08e6)) > 1e-7
    np.testing.assert_allclose(i[far], ic[far], rtol=0, atol=2e-3 * ip)


@pytest.mark.parametrize("kind", ["atan", "gapped"])
def test_a_flux_device_behind_a_located_diode_turn_off_runs(kind):
    """F-Y-2 (refused on the release line: 'the event moves saturable_inductor Lc by
    -4.52e-07 A in an instant', a located turn-off of 15 A). Truth: a turn-off at zero
    current moves no flux — no record; the DC output against TR-BDF2 at rtol 1e-6 (2 %, as
    the release's own rectifier test)."""
    def build():
        b = p.CircuitBuilder()
        b.add_sine_voltage_source("V", "a", "gnd", v_dc=0.0, v_amplitude=20.0,
                                  frequency=1e3, phase=0.0)
        if kind == "gapped":
            b.add_gapped_core_inductor("Lc", "a", "m", N=25, Ae=76e-6, le=72e-3,
                                       lg=0.5e-3, B_sat=0.35)
        else:
            b.add_saturable_inductor("Lc", "a", "m", 111e-6, 6.0, 1e-6)
        b.add_diode("D", "m", "o", 1e3, 1e-9, 0.7, form="branch")
        b.add_resistor("R", "o", "gnd", 2.0)
        b.add_capacitor("C", "o", "gnd", 10e-6)
        return b
    r = _sim(build(), t_end=3e-3, dt=2e-7)
    assert list(r.reinit_records) == []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = p.simulate(build(), t_end=3e-3, engine="trbdf2", dt=4e-6, rtol=1e-6, atol=1e-9)

    def dc(res):
        return p.time_average(np.asarray(res.times), np.asarray(res.v("o")), t_from=2e-3)
    assert dc(ref) > 5.0
    assert dc(r) == pytest.approx(dc(ref), rel=2e-2)
