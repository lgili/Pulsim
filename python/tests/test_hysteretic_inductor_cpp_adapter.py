"""Verify the C++ Jiles-Atherton HystereticInductor BlockChain adapter
matches the Python observer (`make_hysteretic_inductor_observer`) on a
representative 50 V / 50 Hz sinusoidal excitation through a ferrite-N87
core.

The C++ adapter at `core/include/pulsim/magnetics/hysteretic_inductor.hpp`
implements the same per-step JA integrator as
`python/pulsim/hysteresis.py::JilesAthertonModel`. Both paths should
produce identical (within tight numerical tolerance) M / B / H
trajectories and inductor current waveforms.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

pulsim = pytest.importorskip("pulsim")


_HAS_CXX_HYST = (
    hasattr(pulsim, "_pulsim") and
    hasattr(pulsim._pulsim, "CxxBlockChain") and
    hasattr(pulsim._pulsim.CxxBlockChain(),
              "add_hysteretic_inductor")
)


def _build_50hz_core(name):
    """50 V peak / 50 Hz mains source + 1 Ω series + HystereticInductor
    (ferrite_n87, 100 turns, 50 mm path, 1 cm² cross-section)."""
    b = pulsim.CircuitBuilder()
    V_amp = 50.0
    b.add_sine_voltage_source("Vs", "a", "gnd",
                                 0.0, V_amp, 50.0, 0.0)
    b.add_resistor("Rs", "a", "n1", 1.0)
    params = pulsim.reference_material("ferrite_n87")
    hyst = pulsim.add_hysteretic_inductor(
        b, name=name,
        from_node="n1", to_node="gnd",
        params=params, N_turns=100, l_m=0.05, A_core=1e-4,
    )
    return b, hyst, params


@pytest.mark.skipif(not _HAS_CXX_HYST,
                       reason="C++ HystereticInductor adapter not in this build")
def test_cpp_adapter_matches_python_observer():
    DT = 100e-6
    T_END = 0.05  # 50 ms = 2.5 mains cycles

    # --- Python-observer run -------------------------------------------------
    b_py, hyst_py, params = _build_50hz_core("L_core_py")
    obs_py, b_extra_py = pulsim.make_hysteretic_inductor_observer(
        b_py, hyst_py, dt=DT)

    M_log_py = []
    B_log_py = []
    H_log_py = []
    log_t = []

    def step_py(t, x):
        obs_py(t, x)
        if not log_t or t - log_t[-1] >= 0.5e-3:
            log_t.append(t)
            M_log_py.append(hyst_py.M)
            B_log_py.append(hyst_py.B)
            H_log_py.append(hyst_py.H)

    res_py = pulsim.simulate(
        b_py, t_end=T_END, dt=DT,
        switch_fn=lambda t: pulsim.SwitchStateMask(0),
        step_observer=step_py,
        b_extra_fn=b_extra_py,
    )
    assert res_py.num_steps() > 0
    i_idx_py = b_py.pool.branch_var_id_for_inductor(
        hyst_py.inductor_branch_id, b_py.graph)
    states_py = np.asarray(res_py.states)
    i_L_py = states_py[:, i_idx_py]

    # --- C++-adapter run ----------------------------------------------------
    b_cxx, hyst_cxx, _ = _build_50hz_core("L_core_cxx")

    # Resolve branch-var indices explicitly for the C++ adapter.
    inductor_idx = b_cxx.pool.branch_var_id_for_inductor(
        hyst_cxx.inductor_branch_id, b_cxx.graph)
    bemf_idx = b_cxx.pool.branch_var_id_for_source(
        hyst_cxx.bemf_source_id, b_cxx.graph)

    cxx_chain = pulsim._pulsim.CxxBlockChain()
    cxx_chain.add_hysteretic_inductor(
        Ms=params.Ms, a=params.a, alpha=params.alpha,
        c=params.c, k=params.k,
        N_turns=100, l_m=0.05, A_core=1e-4,
        inductor_branch_var_idx=inductor_idx,
        bemf_source_idx=bemf_idx,
        M_channel="M",
        B_channel="B",
        H_channel="H",
    )

    state_size = b_cxx.pool.state_size(b_cxx.graph)
    cxx_observer = cxx_chain.make_step_observer(DT)
    cxx_b_extra = cxx_chain.make_b_extra_fn(state_size)

    M_log_cxx = []
    B_log_cxx = []
    H_log_cxx = []
    log_t_cxx = []

    def step_cxx(t, x):
        cxx_observer(t, x)
        if not log_t_cxx or t - log_t_cxx[-1] >= 0.5e-3:
            log_t_cxx.append(t)
            M_log_cxx.append(cxx_chain.get_channel("M"))
            B_log_cxx.append(cxx_chain.get_channel("B"))
            H_log_cxx.append(cxx_chain.get_channel("H"))

    res_cxx = pulsim.simulate(
        b_cxx, t_end=T_END, dt=DT,
        switch_fn=lambda t: pulsim.SwitchStateMask(0),
        step_observer=step_cxx,
        b_extra_fn=cxx_b_extra,
    )
    assert res_cxx.num_steps() > 0
    states_cxx = np.asarray(res_cxx.states)
    i_L_cxx = states_cxx[:, inductor_idx]

    # --- Compare -------------------------------------------------------------
    n = min(len(M_log_py), len(M_log_cxx))
    assert n > 50, f"too few samples to compare: n={n}"

    def rms_rel_err(a, b, denom_floor=1e-6):
        sq = 0.0
        for ai, bi in zip(a, b):
            denom = max(abs(bi), denom_floor)
            sq += ((ai - bi) / denom) ** 2
        return math.sqrt(sq / len(a))

    M_err = rms_rel_err(M_log_cxx[:n], M_log_py[:n], denom_floor=1.0)
    B_err = rms_rel_err(B_log_cxx[:n], B_log_py[:n], denom_floor=0.01)
    H_err = rms_rel_err(H_log_cxx[:n], H_log_py[:n], denom_floor=0.1)

    # The two implementations port the same forward-Euler + sub-step
    # algorithm. Expect tight agreement.
    assert M_err < 0.05, f"M RMS rel err {M_err:.4f} > 5%"
    assert B_err < 0.05, f"B RMS rel err {B_err:.4f} > 5%"
    assert H_err < 0.05, f"H RMS rel err {H_err:.4f} > 5%"

    # The C++ adapter does NOT write back to the Python `HystereticInductor`
    # dataclass — it only emits channel values. Validate the channel
    # output instead.
    assert abs(cxx_chain.get_channel("M")) > 0.1 * params.Ms, (
        f"C++ M channel did not build up: M={cxx_chain.get_channel('M'):.3e}")
    # Both paths should reach near-saturation under 50 V mains.
    assert abs(M_log_py[-1]) > 0.1 * params.Ms, (
        f"Python observer M did not build up: M={M_log_py[-1]:.3e}")
    assert abs(M_log_cxx[-1]) > 0.1 * params.Ms, (
        f"C++ adapter M did not build up: M={M_log_cxx[-1]:.3e}")

    # Inductor-current waveforms should track too.
    n_state = min(len(i_L_py), len(i_L_cxx))
    iL_peak_py = float(np.max(np.abs(i_L_py[:n_state])))
    iL_peak_cxx = float(np.max(np.abs(i_L_cxx[:n_state])))
    assert abs(iL_peak_py - iL_peak_cxx) / max(iL_peak_py, 1.0) < 0.1, (
        f"i_L peaks mismatch: py={iL_peak_py:.3f} A, "
        f"cxx={iL_peak_cxx:.3f} A")
