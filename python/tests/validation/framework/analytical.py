"""
Analytical solutions for common circuit types.

This module provides closed-form analytical solutions for RC, RL, and RLC circuits
that can be used as gold-standard references for validation.
"""

import numpy as np
from typing import Dict, Callable


class AnalyticalSolutions:
    """Collection of analytical solutions for common circuits."""

    # =========================================================================
    # RC Circuits
    # =========================================================================

    @staticmethod
    def rc_step_response(
        t: np.ndarray,
        node: str,
        params: Dict[str, float]
    ) -> np.ndarray:
        """
        RC circuit step response (capacitor charging).

        Circuit: V_source -- R -- C -- GND

        Analytical solution:
            V_C(t) = V0 * (1 - exp(-t / tau))
            where tau = R * C

        Required params:
            V0: Source voltage
            R: Resistance (Ohms)
            C: Capacitance (Farads)
            V_initial: Initial capacitor voltage (default 0)
        """
        V0 = params["V0"]
        R = params["R"]
        C = params["C"]
        V_initial = params.get("V_initial", 0.0)
        tau = R * C

        node_key = node.lower()
        if node_key in ("v_cap", "v_c", "output", "out", "v(out)"):
            # Capacitor voltage
            V_final = V0
            return V_final + (V_initial - V_final) * np.exp(-t / tau)
        elif node_key in ("i_cap", "i_c", "i_r", "i(v1)"):
            # Current through circuit (same for R and C in series)
            return (V0 - V_initial) / R * np.exp(-t / tau)
        else:
            raise ValueError(f"Unknown node '{node}' for RC step response")

    @staticmethod
    def rc_discharge(
        t: np.ndarray,
        node: str,
        params: Dict[str, float]
    ) -> np.ndarray:
        """
        RC circuit discharge (capacitor discharging through resistor).

        Circuit: C (charged) -- R -- GND

        Analytical solution:
            V_C(t) = V0 * exp(-t / tau)
            where tau = R * C

        Required params:
            V0: Initial capacitor voltage
            R: Resistance (Ohms)
            C: Capacitance (Farads)
        """
        V0 = params["V0"]
        R = params["R"]
        C = params["C"]
        tau = R * C

        node_key = node.lower()
        if node_key in ("v_cap", "v_c", "output", "out", "v(out)"):
            return V0 * np.exp(-t / tau)
        elif node_key in ("i_cap", "i_c", "i_r"):
            # Discharge current (negative, capacitor is source)
            return -V0 / R * np.exp(-t / tau)
        else:
            raise ValueError(f"Unknown node '{node}' for RC discharge")

    # =========================================================================
    # RL Circuits
    # =========================================================================

    @staticmethod
    def rl_step_response(
        t: np.ndarray,
        node: str,
        params: Dict[str, float]
    ) -> np.ndarray:
        """
        RL circuit step response.

        Circuit: V_source -- R -- L -- GND

        Analytical solution:
            I_L(t) = (V0/R) * (1 - exp(-t / tau))
            V_L(t) = V0 * exp(-t / tau)
            where tau = L / R

        Required params:
            V0: Source voltage
            R: Resistance (Ohms)
            L: Inductance (Henrys)
            I_initial: Initial inductor current (default 0)
        """
        V0 = params["V0"]
        R = params["R"]
        L = params["L"]
        I_initial = params.get("I_initial", 0.0)
        tau = L / R
        I_final = V0 / R

        node_key = node.lower()
        if node_key in ("i_ind", "i_l", "i_r", "current", "i(l1)"):
            # Inductor current
            return I_final + (I_initial - I_final) * np.exp(-t / tau)
        elif node_key in ("v_ind", "v_l"):
            # Voltage across inductor
            return (V0 - R * I_initial) * np.exp(-t / tau)
        elif node_key in ("v_r"):
            # Voltage across resistor
            i_current = I_final + (I_initial - I_final) * np.exp(-t / tau)
            return R * i_current
        else:
            raise ValueError(f"Unknown node '{node}' for RL step response")

    @staticmethod
    def rl_current_decay(
        t: np.ndarray,
        node: str,
        params: Dict[str, float]
    ) -> np.ndarray:
        """
        RL circuit current decay (inductor discharging through resistor).

        Analytical solution:
            I_L(t) = I0 * exp(-t / tau)
            where tau = L / R

        Required params:
            I0: Initial inductor current
            R: Resistance (Ohms)
            L: Inductance (Henrys)
        """
        I0 = params["I0"]
        R = params["R"]
        L = params["L"]
        tau = L / R

        node_key = node.lower()
        if node_key in ("i_ind", "i_l", "i_r", "current", "i(l1)"):
            return I0 * np.exp(-t / tau)
        elif node_key in ("v_ind", "v_l"):
            # Inductor voltage during decay (opposing current reduction)
            return -R * I0 * np.exp(-t / tau)
        else:
            raise ValueError(f"Unknown node '{node}' for RL current decay")

    # =========================================================================
    # RLC Circuits
    # =========================================================================

    @staticmethod
    def rlc_series_step_response(
        t: np.ndarray,
        node: str,
        params: Dict[str, float]
    ) -> np.ndarray:
        """
        Series RLC circuit step response.

        Circuit: V_source -- R -- L -- C -- GND

        Required params:
            V0: Source voltage (step)
            R: Resistance (Ohms)
            L: Inductance (Henrys)
            C: Capacitance (Farads)
            V_C_initial: Initial capacitor voltage (default 0)
            I_L_initial: Initial inductor current (default 0)

        The response depends on damping:
            - Underdamped: R < 2*sqrt(L/C) - oscillatory
            - Critically damped: R = 2*sqrt(L/C)
            - Overdamped: R > 2*sqrt(L/C)
        """
        V0 = params["V0"]
        R = params["R"]
        L = params["L"]
        C = params["C"]
        V_C0 = params.get("V_C_initial", 0.0)
        I_L0 = params.get("I_L_initial", 0.0)

        # Natural frequency and damping
        omega_0 = 1.0 / np.sqrt(L * C)  # Natural frequency
        alpha = R / (2.0 * L)            # Damping coefficient
        zeta = alpha / omega_0           # Damping ratio

        # Calculate capacitor voltage V_C(t)
        if zeta < 1.0:
            # Underdamped - oscillatory response
            omega_d = omega_0 * np.sqrt(1 - zeta**2)  # Damped frequency

            # Initial conditions effect
            A = V_C0 - V0
            B = (alpha * A + I_L0 / C) / omega_d

            V_C = V0 + np.exp(-alpha * t) * (A * np.cos(omega_d * t) +
                                              B * np.sin(omega_d * t))

            # Current through circuit
            I_L = C * np.exp(-alpha * t) * (
                (-alpha * A + omega_d * B) * np.cos(omega_d * t) +
                (-alpha * B - omega_d * A) * np.sin(omega_d * t)
            )

        elif np.isclose(zeta, 1.0, rtol=1e-3):
            # Critically damped
            A = V_C0 - V0
            B = alpha * A + I_L0 / C

            V_C = V0 + (A + B * t) * np.exp(-alpha * t)
            I_L = C * (B - alpha * (A + B * t)) * np.exp(-alpha * t)

        else:
            # Overdamped
            s1 = -alpha + np.sqrt(alpha**2 - omega_0**2)
            s2 = -alpha - np.sqrt(alpha**2 - omega_0**2)

            # Solve for A1 and A2 from initial conditions
            # V_C(0) = V0 + A1 + A2 = V_C0
            # I_L(0) = C*(s1*A1 + s2*A2) = I_L0
            A_sum = V_C0 - V0
            A_diff = (I_L0 / C - s2 * A_sum) / (s1 - s2)
            A1 = A_diff
            A2 = A_sum - A1

            V_C = V0 + A1 * np.exp(s1 * t) + A2 * np.exp(s2 * t)
            I_L = C * (s1 * A1 * np.exp(s1 * t) + s2 * A2 * np.exp(s2 * t))

        node_key = node.lower()
        if node_key in ("v_cap", "v_c", "output", "out", "v(out)"):
            return V_C
        elif node_key in ("i_ind", "i_l", "i_cap", "i_c", "current", "i(l1)"):
            return I_L
        elif node_key in ("v_ind", "v_l"):
            return L * np.gradient(I_L, t)
        elif node_key in ("v_r"):
            return R * I_L
        else:
            raise ValueError(f"Unknown node '{node}' for RLC step response")

    @staticmethod
    def rlc_get_damping_info(params: Dict[str, float]) -> Dict[str, float]:
        """
        Get damping characteristics of an RLC circuit.

        Returns dict with:
            - omega_0: Natural frequency (rad/s)
            - f_0: Natural frequency (Hz)
            - alpha: Damping coefficient
            - zeta: Damping ratio
            - damping_type: "underdamped", "critically_damped", or "overdamped"
            - omega_d: Damped frequency (for underdamped, else None)
            - Q: Quality factor
        """
        R = params["R"]
        L = params["L"]
        C = params["C"]

        omega_0 = 1.0 / np.sqrt(L * C)
        alpha = R / (2.0 * L)
        zeta = alpha / omega_0
        Q = 1.0 / (2.0 * zeta) if zeta > 0 else float('inf')

        if zeta < 1.0:
            damping_type = "underdamped"
            omega_d = omega_0 * np.sqrt(1 - zeta**2)
        elif np.isclose(zeta, 1.0, rtol=1e-3):
            damping_type = "critically_damped"
            omega_d = None
        else:
            damping_type = "overdamped"
            omega_d = None

        return {
            "omega_0": omega_0,
            "f_0": omega_0 / (2 * np.pi),
            "alpha": alpha,
            "zeta": zeta,
            "damping_type": damping_type,
            "omega_d": omega_d,
            "Q": Q,
        }

    # =========================================================================
    # Voltage Divider
    # =========================================================================

    @staticmethod
    def voltage_divider(
        t: np.ndarray,
        node: str,
        params: Dict[str, float]
    ) -> np.ndarray:
        """
        Resistive voltage divider.

        Circuit: V_source -- R1 -- node -- R2 -- GND

        Analytical solution:
            V_out = V_in * R2 / (R1 + R2)

        Required params:
            V_in: Input voltage
            R1: Upper resistance (Ohms)
            R2: Lower resistance (Ohms)
        """
        V_in = params["V_in"]
        R1 = params["R1"]
        R2 = params["R2"]

        V_out = V_in * R2 / (R1 + R2)

        node_key = node.lower()
        if node_key in ("v_out", "output", "out", "v_out"):
            return np.full_like(t, V_out)
        elif node_key in ("i", "current"):
            return np.full_like(t, V_in / (R1 + R2))
        else:
            raise ValueError(f"Unknown node '{node}' for voltage divider")

    # =========================================================================
    # Sinusoidal Steady State
    # =========================================================================

    @staticmethod
    def rc_lowpass_sine(
        t: np.ndarray,
        node: str,
        params: Dict[str, float]
    ) -> np.ndarray:
        """
        RC lowpass filter response to sinusoidal input.

        Circuit: V_source(sine) -- R -- C -- GND

        Required params:
            V_amp: Input amplitude
            freq: Input frequency (Hz)
            R: Resistance (Ohms)
            C: Capacitance (Farads)
            phase_in: Input phase (radians, default 0)
        """
        V_amp = params["V_amp"]
        freq = params["freq"]
        R = params["R"]
        C = params["C"]
        phase_in = params.get("phase_in", 0.0)

        omega = 2 * np.pi * freq
        tau = R * C

        # Transfer function: H(jw) = 1 / (1 + jwRC)
        # |H(jw)| = 1 / sqrt(1 + (wRC)^2)
        # angle(H) = -arctan(wRC)

        magnitude = 1.0 / np.sqrt(1 + (omega * tau)**2)
        phase_shift = -np.arctan(omega * tau)

        node_key = node.lower()
        if node_key in ("v_cap", "v_c", "output", "out"):
            return V_amp * magnitude * np.sin(omega * t + phase_in + phase_shift)
        elif node_key in ("v_in", "input"):
            return V_amp * np.sin(omega * t + phase_in)
        else:
            raise ValueError(f"Unknown node '{node}' for RC lowpass sine")

    @staticmethod
    def rc_highpass_sine(
        t: np.ndarray,
        node: str,
        params: Dict[str, float]
    ) -> np.ndarray:
        """
        RC highpass filter response to sinusoidal input.

        Circuit: V_source(sine) -- C -- R -- GND

        Required params:
            V_amp: Input amplitude
            freq: Input frequency (Hz)
            R: Resistance (Ohms)
            C: Capacitance (Farads)
            phase_in: Input phase (radians, default 0)
        """
        V_amp = params["V_amp"]
        freq = params["freq"]
        R = params["R"]
        C = params["C"]
        phase_in = params.get("phase_in", 0.0)

        omega = 2 * np.pi * freq
        tau = R * C

        # Transfer function: H(jw) = jwRC / (1 + jwRC)
        # |H(jw)| = wRC / sqrt(1 + (wRC)^2)
        # angle(H) = pi/2 - arctan(wRC)

        magnitude = (omega * tau) / np.sqrt(1 + (omega * tau)**2)
        phase_shift = np.pi / 2 - np.arctan(omega * tau)

        node_key = node.lower()
        if node_key in ("v_r", "output", "out"):
            return V_amp * magnitude * np.sin(omega * t + phase_in + phase_shift)
        elif node_key in ("v_in", "input"):
            return V_amp * np.sin(omega * t + phase_in)
        else:
            raise ValueError(f"Unknown node '{node}' for RC highpass sine")


# Convenient function registry
ANALYTICAL_FUNCTIONS: Dict[str, Callable] = {
    "rc_step": AnalyticalSolutions.rc_step_response,
    "rc_discharge": AnalyticalSolutions.rc_discharge,
    "rl_step": AnalyticalSolutions.rl_step_response,
    "rl_decay": AnalyticalSolutions.rl_current_decay,
    "rlc_step": AnalyticalSolutions.rlc_series_step_response,
    "voltage_divider": AnalyticalSolutions.voltage_divider,
    "rc_lowpass_sine": AnalyticalSolutions.rc_lowpass_sine,
    "rc_highpass_sine": AnalyticalSolutions.rc_highpass_sine,
}
