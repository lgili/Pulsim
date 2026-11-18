"""add-branch-form-ideal-switch task 2: the conductance form is validated at build time.

``g_on`` finite and > 0, ``g_off`` finite and >= 0 (BREAKING for invalid values: they used to
build and fail mid-run, or run on a negative resistance). The error names the device and the
field the caller passed.
"""
import math

import pytest

import pulsim as p

BAD_ON = [0.0, -1.0, math.inf, math.nan]
BAD_OFF = [-1e-9, math.inf, math.nan]


def _b():
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "a", "gnd", 1.0)
    return b


@pytest.mark.parametrize("g_on", BAD_ON)
def test_add_switch_refuses_g_on(g_on):
    with pytest.raises(ValueError, match=r'add_switch\("S1"\): g_on = '):
        _b().add_switch("S1", "a", "b", g_on, 1e-9)


@pytest.mark.parametrize("g_off", BAD_OFF)
def test_add_switch_refuses_g_off(g_off):
    with pytest.raises(ValueError, match=r'add_switch\("S1"\): g_off = '):
        _b().add_switch("S1", "a", "b", 1e3, g_off)


@pytest.mark.parametrize("g_on", BAD_ON)
def test_add_diode_refuses_g_on(g_on):
    with pytest.raises(ValueError, match=r'add_diode\("D1"\): g_on = '):
        _b().add_diode("D1", "a", "b", g_on, 1e-9)


@pytest.mark.parametrize("g_off", BAD_OFF)
def test_add_diode_refuses_g_off(g_off):
    with pytest.raises(ValueError, match=r'add_diode\("D1"\): g_off = '):
        _b().add_diode("D1", "a", "b", 1e3, g_off)


@pytest.mark.parametrize("R_on", [0.0, -1e-3])
def test_mosfet_and_igbt_name_R_on(R_on):
    with pytest.raises(ValueError, match=r'add_mosfet\("M1"\): R_on = '):
        _b().add_mosfet("M1", "a", "b", R_on=R_on)
    with pytest.raises(ValueError, match=r'add_igbt\("Q1"\): R_on = '):
        _b().add_igbt("Q1", "a", "b", R_on=R_on)


def test_mosfet_names_R_off():
    with pytest.raises(ValueError, match=r'add_mosfet\("M1"\): R_off = -1'):
        _b().add_mosfet("M1", "a", "b", R_off=-1.0)


def test_valid_limits_still_build():
    b = _b()
    b.add_switch("S1", "a", "b", 1e3, 0.0)          # g_off = 0 is legal (no leak)
    b.add_diode("D1", "b", "gnd", 1e12, 0.0)
    b.add_mosfet("M1", "a", "c", R_off=math.inf)     # R_off = inf -> g_off = 0
    assert b.graph.num_switches == 4


def test_yaml_refuses_and_names_the_device():
    text = (
        "circuit:\n  devices:\n"
        "    - {type: voltage_source, name: V1, from: a, to: gnd, V: 1.0}\n"
        "    - {type: switch, name: S9, from: a, to: b, g_on: -5.0, g_off: 1.0e-9}\n")
    with pytest.raises(Exception, match=r'S9.*g_on = -5'):
        p.load_yaml_string(text)
