"""Analytical Bode of a series RLC driving an R load:

           R_s        L
    in ── R_s ── L ──┬── out
                     C
                     │
                    R_load

    H(s) = V_out / V_in
         = (R_load || 1/sC) / (R_s + sL + R_load || 1/sC)

with R_load || 1/sC = R_load / (1 + sR_load·C).
"""
from __future__ import annotations

import math
from typing import Callable


def rlc_low_pass(R_s: float, L: float, C: float, R_load: float) -> Callable[[float], complex]:
    """Return a callable that maps f_hz → H(jω) (complex).

    The complete transfer function:

      H(s) = R_load / [ (R_s + sL)(1 + sR_loadC) + R_load ]

    Sanity at DC: H(0) = R_load / (R_s + R_load).
    """
    def H(f_hz: float) -> complex:
        s = 2j * math.pi * f_hz
        num = R_load
        den = (R_s + s * L) * (1.0 + s * R_load * C) + R_load
        return num / den

    return H
