"""Erickson & Maksimovic averaged small-signal buck model.

Open-loop CCM buck V_out(s) / duty(s):

    G_vd(s) = V_in / [ 1 + sL/(R_load) + s²·L·C ]

Reference: Erickson & Maksimovic, *Fundamentals of Power Electronics* (3rd ed),
§ 7.2 "Averaged Switch Modeling and Spice Simulation" eq. 7.39.

This is the textbook duty-to-output transfer function used to design
voltage-mode buck compensators.
"""
from __future__ import annotations

import math
from typing import Callable


def erickson_buck_plant(V_in: float, L: float, C: float, R_load: float) -> Callable[[float], complex]:
    """Return a callable that maps f_hz → G_vd(jω) (complex).

    Parameters:
        V_in   : DC input voltage (the modulator gain)
        L      : output filter inductance (H)
        C      : output filter capacitance (F)
        R_load : load resistance (Ω)
    """
    def G(f_hz: float) -> complex:
        s = 2j * math.pi * f_hz
        # Erickson eq. 7.39:
        # G_vd(s) = V_in / (1 + s·L/R + s²·L·C)
        den = 1.0 + s * L / R_load + s * s * L * C
        return V_in / den

    return G
