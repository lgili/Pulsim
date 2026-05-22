"""Pulsim — power-electronics circuit simulator.

This top-level package is now a thin alias over the **v2 kernel**.
Every symbol that used to be re-exported here from the legacy v1
binding (``Resistor``, ``Circuit``, ``Simulator``, ``Preset``,
``Compressor``, ``codegen``, ``fmu``, …) was removed when v1 was
retired. The v2 surface is the only supported API.

If you have legacy code:

* Run ``python scripts/migrate_v1_to_v2.py <file>`` for a
  mechanical conversion of v1 patterns to v2 idiom.
* Read ``docs/v2/migration-from-v1.md`` for the device / analysis
  mapping table.
* For features that didn't migrate (codegen, FMU export,
  schematic auto-layout, converter templates, ``Preset``,
  ``compressor_load``, MMC inverter templates, …) you can either
  pin a v0.10.x release of pulsim or open an issue requesting
  the feature on v2.

Usage:

    import pulsim as p          # alias of pulsim.v2

    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "n0", "gnd", 5.0)
    b.add_resistor("R1",   "n0", "n1", 100.0)
    b.add_capacitor("C1",  "n1", "gnd", 1e-6)
    res = p.simulate(b, t_end=1e-3, dt=1e-5)
"""

from __future__ import annotations

__version__ = "1.0.0"

# Re-export everything ``pulsim.v2`` exposes so plain
# ``import pulsim`` and the historical idiomatic
# ``import pulsim.v2 as p`` both resolve to the same surface.
from .v2 import *  # noqa: F401,F403
from .v2 import __all__ as _v2_all

# Expose the v2 submodule under its canonical name too —
# ``pulsim.v2.foo`` keeps working for any code that uses the
# fully-qualified path.
from . import v2  # noqa: F401

__all__ = list(_v2_all) + ["v2", "__version__"]


# ---------------------------------------------------------------------------
# Graceful failure for legacy v1 symbols
# ---------------------------------------------------------------------------
# v1 used to export ~150 symbols at the top level: Resistor, Circuit,
# Simulator, Preset, AdvancedOptions, codegen, fmu, parse_netlist, ...
# Code that referenced any of them now hits a NameError at attribute
# resolution. We override ``__getattr__`` to give a useful redirect
# instead of the raw Python error.

# Subset of v1 names users are most likely to reach for — keeps the
# lookup table small and gives a TARGETED migration hint per symbol.
_V1_SYMBOL_HINTS = {
    "Circuit":            "p.CircuitBuilder()",
    "RuntimeCircuit":     "p.CircuitBuilder()",
    "Simulator":          "p.simulate(b, t_end=..., dt=...)",
    "SimulationResult":   "p.SimulationResult (re-exported on v2)",
    "Resistor":           "b.add_resistor(name, from_node, to_node, R)",
    "Capacitor":          "b.add_capacitor(...)",
    "Inductor":           "b.add_inductor(...)",
    "VoltageSource":      "b.add_voltage_source(...)",
    "CurrentSource":      "b.add_current_source(...)",
    "Diode":              "b.add_diode(name, anode, cathode, g_on, g_off, V_th)",
    "MOSFET":             "b.add_mosfet_with_body_diode(...)",
    "IGBT":               "b.add_igbt(...)",
    "Preset":             "no v2 equivalent — pass explicit SimulationOptions",
    "AdvancedOptions":    "no v2 equivalent — pass explicit SimulationOptions",
    "PulseParams":        "p.make_pwm_switch_fn(...) for PWM gates",
    "PIController":       "p.PIController (under MixedDomainBlockChain)",
    "PIDController":      "p.PIDController",
    "RobustnessProfile":  "no v2 equivalent",
    "LossAccumulator":    "p.LossAccumulator",
    "EfficiencyCalculator":"p.EfficiencyCalculator",
    "ThermalSimulator":   "p.add_foster_network + p.make_thermal_observer",
    "codegen":            "no v2 equivalent",
    "fmu":                "no v2 equivalent",
    "templates":          "no v2 equivalent",
    "schematic":          "no v2 equivalent",
    "parse_netlist":      "p.parse_spice_netlist (SPICE subset)",
}


def __getattr__(name: str):
    """Module-level ``__getattr__`` (PEP 562) — fires only when a
    name isn't found in the normal namespace. Translates legacy v1
    attribute accesses into actionable migration hints."""
    if name in _V1_SYMBOL_HINTS:
        hint = _V1_SYMBOL_HINTS[name]
        raise AttributeError(
            f"pulsim.{name} was a v1 symbol and is no longer "
            f"available (pulsim 1.0.0 removed the legacy kernel). "
            f"Migration hint: {hint}. See "
            f"docs/v2/migration-from-v1.md for the full mapping, "
            f"or run scripts/migrate_v1_to_v2.py on your sources.")
    raise AttributeError(
        f"module 'pulsim' has no attribute {name!r}. "
        f"pulsim 1.0.0 ships only the v2 surface — try "
        f"``import pulsim.v2`` or check "
        f"docs/v2/migration-from-v1.md if you're porting v1 code.")
