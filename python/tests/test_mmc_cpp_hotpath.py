"""C++/Python equivalence tests for the MMC L1 hot path.

Phase 20.11 ports the PS-PWM quantizer and the L1 forward-Euler
step into the kernel (``core/include/pulsim/mmc/arm.hpp``). The
Python module dispatches to those via ``_HAS_CPP_MMC``. These tests
exercise both code paths over a representative parameter sweep and
assert bit-for-bit (PS-PWM) and numerical (L1 step) equivalence —
the C++ port must not change behaviour for users.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import pulsim as p
import pulsim.mmc as mmc_mod


PS_PWM_AVAILABLE = mmc_mod._HAS_CPP_MMC

pytestmark = pytest.mark.skipif(
    not PS_PWM_AVAILABLE,
    reason="C++ kernel not built — skipping C++ hot-path equivalence tests",
)


# ---------------------------------------------------------------------------
# Helpers — flip the dispatch flag inside the module under test.
# ---------------------------------------------------------------------------

def _with_cpp_dispatch(enabled: bool):
    """Context-manager-style toggle of ``_HAS_CPP_MMC``."""
    class _Toggle:
        def __enter__(self):
            self._saved = mmc_mod._HAS_CPP_MMC
            mmc_mod._HAS_CPP_MMC = enabled
            return self

        def __exit__(self, *exc):
            mmc_mod._HAS_CPP_MMC = self._saved
            return False
    return _Toggle()


# ---------------------------------------------------------------------------
# ps_pwm_switching_function — exact integer equivalence.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_sm", [1, 4, 8, 30])
@pytest.mark.parametrize("sm_type", ["half_bridge", "full_bridge"])
def test_ps_pwm_equivalence(n_sm: int, sm_type: str):
    """Same s_b for every (m_ref, t) sample across the two paths."""
    f_carrier = 1000.0
    # Sweep m_ref over both ranges + sample over one period at fine
    # resolution.
    if sm_type == "half_bridge":
        m_refs = np.linspace(-0.2, 1.2, 13)  # include out-of-range clamps
    else:
        m_refs = np.linspace(-1.2, 1.2, 13)
    ts = np.linspace(0.0, 1.0 / f_carrier, 41, endpoint=False)

    for m in m_refs:
        for t in ts:
            with _with_cpp_dispatch(True):
                cpp = mmc_mod.ps_pwm_switching_function(
                    float(m), float(t), n_sm, f_carrier,
                    sm_type=sm_type,  # type: ignore[arg-type]
                )
            with _with_cpp_dispatch(False):
                py = mmc_mod.ps_pwm_switching_function(
                    float(m), float(t), n_sm, f_carrier,
                    sm_type=sm_type,  # type: ignore[arg-type]
                )
            assert cpp == py, (
                f"PS-PWM mismatch (n_sm={n_sm}, sm={sm_type}, "
                f"m={m:.4f}, t={t:.6e}): cpp={cpp} py={py}"
            )


# ---------------------------------------------------------------------------
# mmc_arm_multilevel_step — bit-equivalent v_C / v_b / s_b.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sm_type", ["half_bridge", "full_bridge"])
def test_l1_step_equivalence(sm_type: str):
    """C++ and Python L1 steps produce the same (v_C, v_b, s_b)."""
    params = p.MmcArmMultilevelParams(
        n_sm=10, c_sm=1e-3, v_c0=500.0,
        sm_type=sm_type,  # type: ignore[arg-type]
        f_carrier=2000.0,
    )
    dt = 1e-6
    n_steps = 1000

    def trajectory(use_cpp: bool):
        v_C = params.v_c0
        v_b_arr = np.zeros(n_steps)
        s_b_arr = np.zeros(n_steps, dtype=np.int64)
        v_C_arr = np.zeros(n_steps)
        with _with_cpp_dispatch(use_cpp):
            for k in range(n_steps):
                t = k * dt
                # Use a time-varying m_ref so both branches of the
                # quantizer (rising/falling carriers) are exercised.
                if sm_type == "half_bridge":
                    m_ref = 0.5 + 0.4 * math.sin(2 * math.pi * 50 * t)
                else:
                    m_ref = 0.8 * math.sin(2 * math.pi * 50 * t)
                i_b = 5.0 * math.cos(2 * math.pi * 50 * t)
                v_C, v_b, s_b = mmc_mod.mmc_arm_multilevel_step(
                    v_C, m_ref, i_b, dt, t, params,
                )
                v_C_arr[k] = v_C
                v_b_arr[k] = v_b
                s_b_arr[k] = s_b
        return v_C_arr, v_b_arr, s_b_arr

    v_C_cpp, v_b_cpp, s_b_cpp = trajectory(True)
    v_C_py,  v_b_py,  s_b_py  = trajectory(False)

    # Switching counts must be identical.
    np.testing.assert_array_equal(s_b_cpp, s_b_py)
    # Forward-Euler accumulates the same float operations both ways;
    # the floats are bit-equivalent up to ULP-level rounding.
    np.testing.assert_allclose(v_C_cpp, v_C_py, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(v_b_cpp, v_b_py, rtol=1e-12, atol=1e-12)


# ---------------------------------------------------------------------------
# simulate_mmc_arm_multilevel — full end-to-end equivalence under
# the higher-level driver (covers the result-dataclass path).
# ---------------------------------------------------------------------------

def test_simulate_multilevel_cpp_vs_python():
    params = p.MmcArmMultilevelParams(
        n_sm=8, c_sm=1e-3, v_c0=400.0, f_carrier=1000.0,
    )

    def run(use_cpp: bool):
        with _with_cpp_dispatch(use_cpp):
            return p.simulate_mmc_arm_multilevel(
                duration=2e-3, dt=5e-6,
                m_ref=lambda t: 0.5 + 0.3 * math.sin(2 * math.pi * 50 * t),
                i_b=lambda t: 4.0 * math.cos(2 * math.pi * 50 * t),
                params=params,
            )

    cpp = run(True)
    py  = run(False)
    np.testing.assert_array_equal(cpp.s_b, py.s_b)
    np.testing.assert_allclose(cpp.v_C, py.v_C, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(cpp.v_b, py.v_b, rtol=1e-12, atol=1e-12)


# ---------------------------------------------------------------------------
# Direct-binding sanity: the underscored helpers exist + work.
# ---------------------------------------------------------------------------

def test_cpp_bindings_callable():
    s_b = mmc_mod._cpp_ps_pwm(0.5, 0.0, 4, 1000.0, "half_bridge")
    assert isinstance(s_b, int)
    assert 0 <= s_b <= 4

    v_C_next, v_b, s_b2 = mmc_mod._cpp_ml_step(
        100.0, 0.5, 2.0, 1e-6, 0.0, 4, 0.25e-3, 1000.0, "half_bridge", 0.0,
    )
    assert isinstance(s_b2, int)
    assert v_C_next > 0.0
    assert v_b == (s_b2 / 4) * 100.0
