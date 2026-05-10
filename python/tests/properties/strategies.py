"""Hypothesis strategies for generating randomized Pulsim circuits.

Three generator families:
  - `gen_passive_rc`: 1-port passive RC with random R, C in physical
    ranges (1 Ω – 1 MΩ, 1 pF – 100 µF).
  - `gen_passive_rlc`: passive RLC with random R, L, C.
  - `gen_two_node_resistor_network`: a small star-shaped resistor
    network around a central node, used for KCL / Tellegen checks.

Every circuit factory returns `(circuit, sim_options, metadata)` so
the calling property test can construct a Simulator + record the
seed-derived parameters in the failure report.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from hypothesis import strategies as st

import pulsim


# Reasonable physical ranges so the random circuits don't pathologize
# the simulator (extremely small / extremely large values trigger
# numerical edge cases the property tests aren't meant to catch).
RESISTANCE = st.floats(min_value=1.0, max_value=1e6,
                        allow_nan=False, allow_infinity=False)
CAPACITANCE = st.floats(min_value=1e-12, max_value=1e-3,
                         allow_nan=False, allow_infinity=False)
INDUCTANCE = st.floats(min_value=1e-9, max_value=1e-1,
                        allow_nan=False, allow_infinity=False)
SOURCE_VOLTAGE = st.floats(min_value=0.1, max_value=100.0,
                            allow_nan=False, allow_infinity=False)


@dataclass
class GeneratedCircuit:
    """A circuit + the parameters that made it. Useful for failure
    reporting — when an invariant fails, the property test can dump
    the parameter dict alongside the auto-shrunken Hypothesis report."""
    circuit: Any
    parameters: dict[str, float] = field(default_factory=dict)
    description: str = ""


@st.composite
def gen_passive_rc(draw) -> GeneratedCircuit:
    R = draw(RESISTANCE)
    C = draw(CAPACITANCE)
    V = draw(SOURCE_VOLTAGE)
    ckt = pulsim.Circuit()
    in_ = ckt.add_node("in")
    out = ckt.add_node("out")
    ckt.add_voltage_source("V1", in_, ckt.ground(), V)
    ckt.add_resistor("R1", in_, out, R)
    ckt.add_capacitor("C1", out, ckt.ground(), C, 0.0)
    return GeneratedCircuit(
        circuit=ckt,
        parameters={"R": R, "C": C, "V": V},
        description="passive_rc",
    )


@st.composite
def gen_passive_rlc(draw) -> GeneratedCircuit:
    R = draw(RESISTANCE)
    L = draw(INDUCTANCE)
    C = draw(CAPACITANCE)
    V = draw(SOURCE_VOLTAGE)
    ckt = pulsim.Circuit()
    in_  = ckt.add_node("in")
    mid  = ckt.add_node("mid")
    out  = ckt.add_node("out")
    ckt.add_voltage_source("V1", in_, ckt.ground(), V)
    ckt.add_resistor("R1", in_, mid, R)
    ckt.add_inductor("L1", mid, out, L, 0.0)
    ckt.add_capacitor("C1", out, ckt.ground(), C, 0.0)
    return GeneratedCircuit(
        circuit=ckt,
        parameters={"R": R, "L": L, "C": C, "V": V},
        description="passive_rlc",
    )


@st.composite
def gen_resistor_divider(draw) -> GeneratedCircuit:
    """Voltage divider with two resistors. The DC OP must satisfy KCL
    at the midpoint trivially; useful as the simplest KCL property."""
    R1 = draw(RESISTANCE)
    R2 = draw(RESISTANCE)
    V = draw(SOURCE_VOLTAGE)
    ckt = pulsim.Circuit()
    in_ = ckt.add_node("in")
    mid = ckt.add_node("mid")
    ckt.add_voltage_source("V1", in_, ckt.ground(), V)
    ckt.add_resistor("R1", in_, mid, R1)
    ckt.add_resistor("R2", mid, ckt.ground(), R2)
    return GeneratedCircuit(
        circuit=ckt,
        parameters={"R1": R1, "R2": R2, "V": V},
        description="resistor_divider",
    )


def make_quick_options(*, tstop: float = 1e-3, dt: float = 1e-5):
    """Common simulation options: short fixed-step transient, BDF1.
    Keeps the per-property runtime under ~10 ms so a Hypothesis run
    of ~50 examples completes in a few seconds (Phase 10 G.5 budget).
    """
    opts = pulsim.SimulationOptions()
    opts.tstart = 0.0
    opts.tstop = tstop
    opts.dt = dt
    opts.dt_min = dt * 1e-3
    opts.dt_max = dt
    opts.adaptive_timestep = False
    opts.integrator = pulsim.Integrator.BDF1
    return opts
