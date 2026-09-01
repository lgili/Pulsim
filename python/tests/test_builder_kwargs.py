"""`CircuitBuilder.add_*` kwarg authoring — Round 3 ergonomics polish.

`from`/`to` are reserved/awkward in Python (you can't write
`b.add_resistor(name="R", from="vin", ...)` because `from` is a hard
reserved word — only the `**kwargs` dict workaround was possible).
The pybind11 args were renamed to `n_pos`/`n_neg`, which keeps the
positional call sites working unchanged and unlocks the natural
kwarg form.

These tests pin:

1. The kwarg form `n_pos=...`, `n_neg=...` is accepted on every
   two-terminal `add_*` (sources, R/L/C, switches).
2. The positional form still works — purely additive change.
3. Two-terminal semantic-named kwargs (`anode`/`cathode`,
   `drain`/`source`, `collector`/`emitter`) keep working as well
   so we don't regress diode/MOSFET/IGBT ergonomics.
4. The signature documented by pybind11 mentions `n_pos`/`n_neg`
   so users see them in `help()` / IDE autocomplete.
"""
from __future__ import annotations

import math

import numpy as np

import pulsim as p


def test_signature_documents_n_pos_n_neg() -> None:
    """`help(b.add_resistor)` must mention `n_pos`/`n_neg` — that's
    what the user sees in IDE autocomplete + docstring."""
    sig = p._pulsim.CircuitBuilder.add_resistor.__doc__.split("\n", 1)[0]
    assert "n_pos" in sig, f"signature missing n_pos: {sig}"
    assert "n_neg" in sig, f"signature missing n_neg: {sig}"


def test_resistor_divider_via_kwargs() -> None:
    """Classic 2-resistor divider authored purely with kwargs.
    The numerical answer pins that node naming flows through."""
    b = p.CircuitBuilder()
    b.add_voltage_source(name="Vin", n_pos="vin", n_neg="gnd", V=5.0)
    b.add_resistor(name="R_top", n_pos="vin", n_neg="mid", R_ohms=2.0)
    b.add_resistor(name="R_bot", n_pos="mid", n_neg="gnd", R_ohms=3.0)
    res = p.simulate(b, t_end=1e-3, dt=1e-5)
    v_mid = float(np.asarray(res.v("mid"))[-1])
    # R_bot/(R_top+R_bot) · V_in = 3/5 · 5 = 3 V.
    assert math.isclose(v_mid, 3.0, abs_tol=1e-6)


def test_positional_form_still_works() -> None:
    """The rename is purely additive — every existing positional
    call site must keep functioning bit-for-bit."""
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", 5.0)
    b.add_resistor("R_top", "vin", "mid", 2.0)
    b.add_resistor("R_bot", "mid", "gnd", 3.0)
    res = p.simulate(b, t_end=1e-3, dt=1e-5)
    v_mid = float(np.asarray(res.v("mid"))[-1])
    assert math.isclose(v_mid, 3.0, abs_tol=1e-6)


def test_all_two_terminal_add_methods_accept_n_pos_n_neg() -> None:
    """Survey: every two-terminal `add_*` accepts the kwarg form.
    If any binding gets out of sync with the rename, this raises
    `TypeError` immediately."""
    b = p.CircuitBuilder()
    b.add_voltage_source(name="V", n_pos="a", n_neg="gnd", V=1.0)
    b.add_current_source(name="I", n_pos="a", n_neg="b", I=0.0)
    b.add_resistor(name="R", n_pos="a", n_neg="b", R_ohms=1.0)
    b.add_capacitor(name="C", n_pos="a", n_neg="b", C_farads=1e-6)
    b.add_inductor(name="L", n_pos="a", n_neg="b", L_henries=1e-3)
    b.add_switch(name="SW", n_pos="a", n_neg="b", g_on=1e3, g_off=1e-9)
    # 6 add_* calls = 6 branches.
    assert b.graph.num_branches == 6


def test_semantic_named_terminals_unchanged() -> None:
    """The rename only touched the generic `from`/`to` pair.
    Devices with semantic-named terminals (anode/cathode,
    drain/source, collector/emitter, p_from/p_to/s_from/s_to)
    must keep using those names so domain-correct authoring still
    works."""
    b = p.CircuitBuilder()
    b.add_diode(name="D", anode="a", cathode="b",
                g_on=1e3, g_off=1e-9, V_th=0.7)
    b.add_mosfet(name="M", drain="d", source="s",
                 R_on=1e-3, R_off=1e9)
    b.add_igbt(name="Q", collector="c", emitter="e",
               R_on=10e-3, R_off=1e9)
    b.add_transformer(name="T", p_from="p1", p_to="p2",
                      s_from="s1", s_to="s2",
                      L_p=1e-3, L_s=1e-3, k=0.99)
    # diode + mosfet + igbt + 2 transformer windings = 5.
    # 6, not 5, since v2.0: the MOSFET carries its intrinsic
    # body diode (audit C.1).
    assert b.graph.num_branches == 6
