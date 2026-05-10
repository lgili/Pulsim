"""Converter-template Python builder API.

Phase 7 of `add-converter-templates`. Wraps the C++-side template
registry behind ergonomic Python factories so the user goes from a
3-line "I want a 24V→5V buck at 100kHz, 2A load" to a fully-wired
`Circuit` ready for `Simulator.run_transient` / `run_ac_sweep`.

The C++-side registry isn't bound to Python today (deferred until the
template-variant integration follow-up) — this module wraps the
registry by building the equivalent `Circuit` directly via the existing
`pulsim.Circuit` Python API. Each `pulsim.templates.<topology>(...)`
returns:

  - `circuit` — the built `pulsim.Circuit`
  - `parameters` — dict of resolved (auto-designed) parameters
  - `notes` — dict of design-decision notes per parameter

Usage::

    import pulsim
    expansion = pulsim.templates.buck(Vin=24, Vout=5, Iout=2, fsw=100e3)
    sim = pulsim.Simulator(expansion.circuit, options)
    print(expansion.parameters)        # {'L': 6.6e-5, 'C': 1.5e-5, ...}
    print(expansion.notes['L'])        # 'Auto-sized for 30 % current ripple ...'
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from . import Circuit  # type: ignore


__all__ = ["TemplateExpansion", "buck", "boost", "buck_boost"]


@dataclass
class TemplateExpansion:
    """Result of expanding a converter template.

    Mirrors the C++ `ConverterExpansion` struct:
      - `circuit` is the assembled `pulsim.Circuit`
      - `parameters` is a dict of resolved (user + auto-designed) values
      - `notes` is a dict of human-readable notes per auto-designed knob
      - `topology` is the template name ("buck", "boost", ...)
    """
    circuit: "Circuit"
    parameters: dict = field(default_factory=dict)
    notes: dict = field(default_factory=dict)
    topology: str = ""


def _import_pulsim_runtime():
    """Lazy import so the templates module loads cleanly even when the
    user only wants the Python pre-processing surface."""
    from . import _pulsim
    return _pulsim


def buck(
    *,
    Vin: float,
    Vout: float,
    Iout: float,
    fsw: float,
    ripple_pct: float = 0.30,
    vout_ripple_pct: float = 0.01,
    L: float | None = None,
    C: float | None = None,
    Rload: float | None = None,
    q_g_on: float = 1e3,
    q_g_off: float = 1e-9,
) -> TemplateExpansion:
    """Build a synchronous buck converter circuit.

    Auto-designs `L` (≤ ripple_pct current ripple) and `C` (≤
    vout_ripple_pct output ripple) from the design intent. User-
    supplied `L`, `C`, `Rload` override the auto-designed values.
    """
    if not (Vin > Vout > 0.0):
        raise ValueError("buck: require Vin > Vout > 0")
    if not (fsw > 0 and Iout > 0):
        raise ValueError("buck: require Iout > 0 and fsw > 0")

    pl = _import_pulsim_runtime()
    D = Vout / Vin
    dI = ripple_pct * Iout
    L_d = (Vin - Vout) * D / (dI * fsw)
    dV = vout_ripple_pct * Vout
    C_d = dI / (8.0 * fsw * dV)
    R_d = Vout / Iout

    L_use = L_d if L is None else L
    C_use = C_d if C is None else C
    R_use = R_d if Rload is None else Rload

    ckt = pl.Circuit()
    in_  = ckt.add_node("in")
    sw_n = ckt.add_node("sw")
    out_n = ckt.add_node("out")
    ctrl = ckt.add_node("ctrl")
    ckt.add_voltage_source("Vin", in_, ckt.ground(), Vin)

    pwm = pl.PWMParams()
    pwm.frequency = fsw
    pwm.duty = D
    pwm.v_high = 5.0
    pwm.v_low = 0.0
    pwm.phase = 0.0
    ckt.add_pwm_voltage_source("Vctrl", ctrl, ckt.ground(), pwm)

    ckt.add_vcswitch("Q1", ctrl, in_, sw_n, 2.5, q_g_on, q_g_off)
    ckt.add_diode("D1", ckt.ground(), sw_n, q_g_on, q_g_off)
    ckt.add_inductor("L1", sw_n, out_n, L_use, 0.0)
    ckt.add_capacitor("C1", out_n, ckt.ground(), C_use, 0.0)
    ckt.add_resistor("Rload", out_n, ckt.ground(), R_use)

    notes: dict = {}
    if L is None:
        notes["L"] = f"Auto-sized for {ripple_pct*100:.1f} % current ripple"
    if C is None:
        notes["C"] = f"Auto-sized for {vout_ripple_pct*100:.2f} % output-voltage ripple"
    if Rload is None:
        notes["Rload"] = f"Auto-sized for nominal Iout = {Iout} A"

    return TemplateExpansion(
        circuit=ckt,
        parameters={
            "Vin": Vin, "Vout": Vout, "Iout": Iout, "fsw": fsw,
            "ripple_pct": ripple_pct, "vout_ripple_pct": vout_ripple_pct,
            "D": D, "L": L_use, "C": C_use, "Rload": R_use,
            "q_g_on": q_g_on, "q_g_off": q_g_off,
        },
        notes=notes,
        topology="buck",
    )


def boost(
    *,
    Vin: float,
    Vout: float,
    Iout: float,
    fsw: float,
    ripple_pct: float = 0.30,
    vout_ripple_pct: float = 0.01,
    L: float | None = None,
    C: float | None = None,
    Rload: float | None = None,
    q_g_on: float = 1e3,
    q_g_off: float = 1e-9,
) -> TemplateExpansion:
    """Build a boost converter circuit (Vout > Vin)."""
    if not (Vout > Vin > 0.0):
        raise ValueError("boost: require Vout > Vin > 0")
    if not (fsw > 0 and Iout > 0):
        raise ValueError("boost: require Iout > 0 and fsw > 0")

    pl = _import_pulsim_runtime()
    D = 1.0 - Vin / Vout
    I_in = Iout * (Vout / Vin)
    dI = ripple_pct * I_in
    L_d = Vin * D / (dI * fsw)
    dV = vout_ripple_pct * Vout
    C_d = Iout * D / (fsw * dV)
    R_d = Vout / Iout

    L_use = L_d if L is None else L
    C_use = C_d if C is None else C
    R_use = R_d if Rload is None else Rload

    ckt = pl.Circuit()
    in_   = ckt.add_node("in")
    sw_n  = ckt.add_node("sw")
    out_n = ckt.add_node("out")
    ctrl  = ckt.add_node("ctrl")
    ckt.add_voltage_source("Vin", in_, ckt.ground(), Vin)
    ckt.add_inductor("L1", in_, sw_n, L_use, 0.0)

    pwm = pl.PWMParams()
    pwm.frequency = fsw
    pwm.duty = D
    pwm.v_high = 5.0
    pwm.v_low = 0.0
    ckt.add_pwm_voltage_source("Vctrl", ctrl, ckt.ground(), pwm)

    ckt.add_vcswitch("Q1", ctrl, sw_n, ckt.ground(), 2.5, q_g_on, q_g_off)
    ckt.add_diode("D1", sw_n, out_n, q_g_on, q_g_off)
    ckt.add_capacitor("C1", out_n, ckt.ground(), C_use, 0.0)
    ckt.add_resistor("Rload", out_n, ckt.ground(), R_use)

    notes: dict = {}
    if L is None:
        notes["L"] = f"Auto-sized for {ripple_pct*100:.1f} % input-current ripple"
    if C is None:
        notes["C"] = f"Auto-sized for {vout_ripple_pct*100:.2f} % output-voltage ripple"
    if Rload is None:
        notes["Rload"] = f"Auto-sized for nominal Iout = {Iout} A"

    return TemplateExpansion(
        circuit=ckt,
        parameters={
            "Vin": Vin, "Vout": Vout, "Iout": Iout, "fsw": fsw,
            "ripple_pct": ripple_pct, "vout_ripple_pct": vout_ripple_pct,
            "D": D, "I_in": I_in,
            "L": L_use, "C": C_use, "Rload": R_use,
            "q_g_on": q_g_on, "q_g_off": q_g_off,
        },
        notes=notes,
        topology="boost",
    )


def buck_boost(
    *,
    Vin: float,
    Vout: float,
    Iout: float,
    fsw: float,
    ripple_pct: float = 0.30,
    vout_ripple_pct: float = 0.01,
    L: float | None = None,
    C: float | None = None,
    Rload: float | None = None,
    q_g_on: float = 1e3,
    q_g_off: float = 1e-9,
) -> TemplateExpansion:
    """Build an inverting buck-boost. `Vout` is conventionally negative
    (or pass its magnitude — the function takes |Vout|)."""
    Vout_mag = abs(Vout)
    if not (Vin > 0.0 and Vout_mag > 0.0 and fsw > 0 and Iout > 0):
        raise ValueError(
            "buck_boost: require Vin > 0, |Vout| > 0, Iout > 0, fsw > 0")

    pl = _import_pulsim_runtime()
    D = Vout_mag / (Vin + Vout_mag)
    I_in = Iout * (Vout_mag / Vin)
    dI = ripple_pct * I_in
    L_d = Vin * D / (dI * fsw)
    dV = vout_ripple_pct * Vout_mag
    C_d = Iout * D / (fsw * dV)
    R_d = Vout_mag / Iout

    L_use = L_d if L is None else L
    C_use = C_d if C is None else C
    R_use = R_d if Rload is None else Rload

    ckt = pl.Circuit()
    in_   = ckt.add_node("in")
    sw_n  = ckt.add_node("sw")
    out_n = ckt.add_node("out")
    ctrl  = ckt.add_node("ctrl")
    ckt.add_voltage_source("Vin", in_, ckt.ground(), Vin)

    pwm = pl.PWMParams()
    pwm.frequency = fsw
    pwm.duty = D
    pwm.v_high = 5.0
    pwm.v_low = 0.0
    ckt.add_pwm_voltage_source("Vctrl", ctrl, ckt.ground(), pwm)

    ckt.add_vcswitch("Q1", ctrl, in_, sw_n, 2.5, q_g_on, q_g_off)
    ckt.add_inductor("L1", sw_n, ckt.ground(), L_use, 0.0)
    ckt.add_diode("D1", out_n, sw_n, q_g_on, q_g_off)
    ckt.add_capacitor("C1", out_n, ckt.ground(), C_use, 0.0)
    ckt.add_resistor("Rload", out_n, ckt.ground(), R_use)

    notes: dict = {}
    if L is None:
        notes["L"] = f"Auto-sized for {ripple_pct*100:.1f} % current ripple"
    if C is None:
        notes["C"] = f"Auto-sized for {vout_ripple_pct*100:.2f} % output-voltage ripple"

    return TemplateExpansion(
        circuit=ckt,
        parameters={
            "Vin": Vin, "Vout": -Vout_mag, "Iout": Iout, "fsw": fsw,
            "ripple_pct": ripple_pct, "vout_ripple_pct": vout_ripple_pct,
            "D": D, "I_in": I_in,
            "L": L_use, "C": C_use, "Rload": R_use,
            "q_g_on": q_g_on, "q_g_off": q_g_off,
        },
        notes=notes,
        topology="buck_boost",
    )
