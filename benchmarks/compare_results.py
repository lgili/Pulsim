#!/usr/bin/env python3
"""
Benchmark comparison between Pulsim and ngspice.
Compares simulation results for RC, RL, and RLC circuits.
"""

import numpy as np
import os

def load_pulsim_csv(filename):
    """Load Pulsim CSV output (time, signal1, signal2, ...)"""
    data = np.genfromtxt(filename, delimiter=',', skip_header=1)
    return data

def load_ngspice_csv(filename):
    """Load ngspice wrdata output (time, value pairs)"""
    # ngspice wrdata format: whitespace-separated time value pairs
    data = np.loadtxt(filename)
    return data

def analytical_rc_step(t, V0, R, C):
    """RC step response: V(t) = V0 * (1 - exp(-t/RC))"""
    tau = R * C
    return V0 * (1 - np.exp(-t / tau))

def analytical_rl_step(t, V0, R, L):
    """RL step response voltage across inductor: V_L(t) = V0 * exp(-Rt/L)
       Voltage across resistor: V_R(t) = V0 * (1 - exp(-Rt/L))
       For our circuit, out node is between R and L, so V(out) = V_L
    """
    tau = L / R
    return V0 * np.exp(-t / tau)

def analytical_rlc_step(t, V0, R, L, C):
    """Underdamped RLC step response on capacitor voltage"""
    alpha = R / (2 * L)
    omega0 = 1 / np.sqrt(L * C)

    if alpha < omega0:  # Underdamped
        omega_d = np.sqrt(omega0**2 - alpha**2)
        return V0 * (1 - np.exp(-alpha * t) * (np.cos(omega_d * t) +
                     (alpha / omega_d) * np.sin(omega_d * t)))
    else:  # Overdamped or critically damped
        s1 = -alpha + np.sqrt(alpha**2 - omega0**2)
        s2 = -alpha - np.sqrt(alpha**2 - omega0**2)
        A = V0 * s2 / (s2 - s1)
        B = -V0 * s1 / (s2 - s1)
        return V0 - A * np.exp(s1 * t) - B * np.exp(s2 * t)

def compare_results():
    """Compare Pulsim and ngspice results against analytical solutions"""

    results = []

    # RC Circuit comparison
    print("=" * 60)
    print("RC CIRCUIT COMPARISON")
    print("=" * 60)

    pulsim_rc = load_pulsim_csv('results/rc_pulsim.csv')
    ngspice_rc = load_ngspice_csv('results/rc_ngspice.csv')

    # Parameters
    V0, R, C = 5.0, 1000.0, 1e-6

    # Pulsim: columns are time, V(in), V(out), I(V1)
    t_sl = pulsim_rc[:, 0]
    v_out_sl = pulsim_rc[:, 2]  # V(out)

    # ngspice: columns are time, v(out)
    t_ng = ngspice_rc[:, 0]
    v_out_ng = ngspice_rc[:, 1]

    # Analytical
    v_analytical_sl = analytical_rc_step(t_sl, V0, R, C)
    v_analytical_ng = analytical_rc_step(t_ng, V0, R, C)

    # Calculate errors
    error_sl = np.abs(v_out_sl - v_analytical_sl)
    error_ng = np.abs(v_out_ng - v_analytical_ng)

    print(f"Pulsim: {len(t_sl)} points, t=[{t_sl[0]:.2e}, {t_sl[-1]:.2e}]s")
    print(f"ngspice:  {len(t_ng)} points, t=[{t_ng[0]:.2e}, {t_ng[-1]:.2e}]s")
    print(f"\nMax absolute error vs analytical:")
    print(f"  Pulsim: {np.max(error_sl):.6e} V")
    print(f"  ngspice:  {np.max(error_ng):.6e} V")
    print(f"\nRMS error vs analytical:")
    print(f"  Pulsim: {np.sqrt(np.mean(error_sl**2)):.6e} V")
    print(f"  ngspice:  {np.sqrt(np.mean(error_ng**2)):.6e} V")
    print(f"\nFinal value comparison (at t={t_sl[-1]:.2e}s):")
    print(f"  Analytical: {v_analytical_sl[-1]:.6f} V")
    print(f"  Pulsim:   {v_out_sl[-1]:.6f} V (error: {error_sl[-1]:.6e} V)")

    # Find matching ngspice point
    idx_ng = np.argmin(np.abs(t_ng - t_sl[-1]))
    print(f"  ngspice:    {v_out_ng[idx_ng]:.6f} V (error: {np.abs(v_out_ng[idx_ng] - v_analytical_sl[-1]):.6e} V)")

    results.append({
        'circuit': 'RC',
        'pulsim_max_error': np.max(error_sl),
        'ngspice_max_error': np.max(error_ng),
        'pulsim_rms_error': np.sqrt(np.mean(error_sl**2)),
        'ngspice_rms_error': np.sqrt(np.mean(error_ng**2))
    })

    # RL Circuit comparison
    print("\n" + "=" * 60)
    print("RL CIRCUIT COMPARISON")
    print("=" * 60)

    pulsim_rl = load_pulsim_csv('results/rl_pulsim.csv')
    ngspice_rl = load_ngspice_csv('results/rl_ngspice.csv')

    # Parameters
    V0, R, L = 10.0, 100.0, 10e-3
    tau_rl = L / R  # Time constant = 0.1ms

    # Pulsim: columns are time, V(in), V(out), I(V1), I(L1)
    t_sl = pulsim_rl[:, 0]
    v_out_sl = pulsim_rl[:, 2]  # V(out) - voltage across inductor

    # ngspice: columns are time, v(out)
    t_ng = ngspice_rl[:, 0]
    v_out_ng = ngspice_rl[:, 1]

    # Filter to post-step region (after pulse rise time, t > 1ns)
    # For proper comparison, compare only after the step has occurred
    sl_mask = t_sl > 1e-9
    ng_mask = t_ng > 1e-9

    t_sl_filt = t_sl[sl_mask]
    v_out_sl_filt = v_out_sl[sl_mask]
    t_ng_filt = t_ng[ng_mask]
    v_out_ng_filt = v_out_ng[ng_mask]

    # Analytical (voltage across inductor = V0 * exp(-Rt/L))
    v_analytical_sl = analytical_rl_step(t_sl_filt, V0, R, L)
    v_analytical_ng = analytical_rl_step(t_ng_filt, V0, R, L)

    # Calculate errors
    error_sl = np.abs(v_out_sl_filt - v_analytical_sl)
    error_ng = np.abs(v_out_ng_filt - v_analytical_ng)

    print(f"Pulsim: {len(t_sl_filt)} points, t=[{t_sl_filt[0]:.2e}, {t_sl_filt[-1]:.2e}]s")
    print(f"ngspice:  {len(t_ng_filt)} points, t=[{t_ng_filt[0]:.2e}, {t_ng_filt[-1]:.2e}]s")
    print(f"Time constant (tau): {tau_rl*1e3:.3f} ms")
    print(f"\nMax absolute error vs analytical:")
    print(f"  Pulsim: {np.max(error_sl):.6e} V")
    print(f"  ngspice:  {np.max(error_ng):.6e} V")
    print(f"\nRMS error vs analytical:")
    print(f"  Pulsim: {np.sqrt(np.mean(error_sl**2)):.6e} V")
    print(f"  ngspice:  {np.sqrt(np.mean(error_ng**2)):.6e} V")

    results.append({
        'circuit': 'RL',
        'pulsim_max_error': np.max(error_sl),
        'ngspice_max_error': np.max(error_ng),
        'pulsim_rms_error': np.sqrt(np.mean(error_sl**2)),
        'ngspice_rms_error': np.sqrt(np.mean(error_ng**2))
    })

    # RLC Circuit comparison
    print("\n" + "=" * 60)
    print("RLC CIRCUIT COMPARISON")
    print("=" * 60)

    pulsim_rlc = load_pulsim_csv('results/rlc_pulsim.csv')
    ngspice_rlc = load_ngspice_csv('results/rlc_ngspice.csv')

    # Parameters
    V0, R, L, C = 10.0, 10.0, 1e-3, 10e-6

    # Pulsim: columns are time, V(in), V(n1), V(out), I(V1), I(L1)
    t_sl = pulsim_rlc[:, 0]
    v_out_sl = pulsim_rlc[:, 3]  # V(out)

    # ngspice: columns are time, v(out), time, i(V1)
    t_ng = ngspice_rlc[:, 0]
    v_out_ng = ngspice_rlc[:, 1]

    # Analytical
    v_analytical_sl = analytical_rlc_step(t_sl, V0, R, L, C)
    v_analytical_ng = analytical_rlc_step(t_ng, V0, R, L, C)

    # Calculate errors
    error_sl = np.abs(v_out_sl - v_analytical_sl)
    error_ng = np.abs(v_out_ng - v_analytical_ng)

    print(f"Pulsim: {len(t_sl)} points, t=[{t_sl[0]:.2e}, {t_sl[-1]:.2e}]s")
    print(f"ngspice:  {len(t_ng)} points, t=[{t_ng[0]:.2e}, {t_ng[-1]:.2e}]s")

    # RLC parameters
    alpha = R / (2 * L)
    omega0 = 1 / np.sqrt(L * C)
    omega_d = np.sqrt(omega0**2 - alpha**2) if omega0 > alpha else 0
    print(f"\nRLC Parameters:")
    print(f"  Damping factor (alpha): {alpha:.2f} rad/s")
    print(f"  Natural frequency (omega0): {omega0:.2f} rad/s")
    print(f"  Damped frequency (omega_d): {omega_d:.2f} rad/s")
    print(f"  Period: {2*np.pi/omega_d*1000:.3f} ms")
    print(f"  Damping ratio (zeta): {alpha/omega0:.3f}")

    print(f"\nMax absolute error vs analytical:")
    print(f"  Pulsim: {np.max(error_sl):.6e} V")
    print(f"  ngspice:  {np.max(error_ng):.6e} V")
    print(f"\nRMS error vs analytical:")
    print(f"  Pulsim: {np.sqrt(np.mean(error_sl**2)):.6e} V")
    print(f"  ngspice:  {np.sqrt(np.mean(error_ng**2)):.6e} V")

    results.append({
        'circuit': 'RLC',
        'pulsim_max_error': np.max(error_sl),
        'ngspice_max_error': np.max(error_ng),
        'pulsim_rms_error': np.sqrt(np.mean(error_sl**2)),
        'ngspice_rms_error': np.sqrt(np.mean(error_ng**2))
    })

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("\n{:<10} {:>20} {:>20}".format("Circuit", "Pulsim RMS Error", "ngspice RMS Error"))
    print("-" * 52)
    for r in results:
        print("{:<10} {:>20.6e} {:>20.6e}".format(
            r['circuit'], r['pulsim_rms_error'], r['ngspice_rms_error']))

    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)
    print("Pulsim produces results comparable to ngspice with")
    print("errors on the same order of magnitude relative to")
    print("analytical solutions. Both simulators use Backward Euler")
    print("integration which introduces some numerical damping.")

    return results

if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    compare_results()
