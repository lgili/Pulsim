"""Regression: `simulate(engine='dsed', ...)` must use the C++
native path without scipy installed.

scipy lives only under `[project.optional-dependencies] dev` in
`pyproject.toml` — it's NOT a runtime dependency. Prior to v1.6.2,
`pulsim.dsed.bdf2_integrator` did a top-level
`from scipy.linalg import lu_factor, lu_solve`, which cascaded
through `pulsim.dsed.__init__` and crashed the whole
`pulsim.dsed` subpackage when scipy was missing — masking the
C++ native DSED path that the `_dsed_dispatch` would have taken
otherwise. End-users hitting `pip install pulsim` got a
``RuntimeError`` and silently lost the documented 24× speedup.

We pin two contracts:

1. **`pulsim.dsed` subpackage loads without scipy.**
2. **`simulate(engine='dsed', ...)` on a real buck CCM circuit
   produces results AND runs in well under 1s — confirming the
   native C++ path was selected, not a Python fallback.**

If either invariant breaks, the wheel is silently broken in the
same way v1.6.1 was.
"""
from __future__ import annotations

import builtins
import importlib
import subprocess
import sys
import textwrap
import time

import numpy as np
import pytest

import pulsim as p


def test_pulsim_dsed_imports_without_scipy_in_subprocess() -> None:
    """Spawn a fresh Python and block `import scipy`. Importing
    `pulsim.dsed` (and going down the dispatcher path) must still
    succeed — this is the exact wheel-install scenario.

    Pre-v1.7.0 the public ``pulsim.dsed`` re-exported the pure-Python
    scheduler classes whose ``bdf2_integrator`` did a top-level
    ``import scipy.linalg``. The fix lazified that import, but
    the bigger structural cleanup in v1.7.0 moved the whole
    Python scheduler to ``tests/_dsed_reference/`` — so today the
    public ``pulsim.dsed`` doesn't touch scipy at all. This test
    keeps the scipy-blocked subprocess assertion as a guard
    against any future regression that pulls an optional dep
    back into the public package.
    """
    script = textwrap.dedent("""
        import builtins, sys
        _real = builtins.__import__
        def _no_scipy(name, *a, **kw):
            if name == 'scipy' or name.startswith('scipy.'):
                raise ModuleNotFoundError("No module named 'scipy'")
            return _real(name, *a, **kw)
        builtins.__import__ = _no_scipy

        import pulsim  # must not crash
        import pulsim.dsed as _dsed  # the cascade that used to crash
        # Touch the public surface to confirm it loaded — these are
        # the only symbols the slim v1.7.0 dsed package exports.
        _ = _dsed.run_user_lti
        _ = _dsed.CircuitBuilderAdapter
        sys.stdout.write(f'OK {pulsim.__version__}')
    """)
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, (
        f"`import pulsim.dsed` failed without scipy:\n"
        f"stdout={result.stdout}\nstderr={result.stderr}"
    )
    assert "OK " in result.stdout, result.stdout


def _synchronous_buck():
    """Two controlled switches, complementary drive, no diode — the
    converter the dsed engine can genuinely simulate. (Its PWL-diode
    sibling is now refused: the diode bit froze at whatever switch_fn
    returned, and v_out came out 0.59 V where 12 V is correct.)"""
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", 24.0)
    b.add_switch("S_hi", "vin", "sw", 1e3, 1e-9)
    b.add_switch("S_lo", "sw", "gnd", 1e3, 1e-9)
    b.add_inductor("L1", "sw", "out", 100e-6)
    b.add_capacitor("C1", "out", "gnd", 100e-6)
    b.add_resistor("R1", "out", "gnd", 2.0)
    return b


def test_simulate_dsed_runs_on_buck_ccm_without_scipy() -> None:
    """The canonical buck CCM benchmark must succeed on
    engine='dsed' even when scipy is missing — the C++ native
    Bridge.11 path doesn't need scipy.

    We block scipy in-process via an import hook, force a fresh
    import of `pulsim.dsed`, then run a 5 ms buck simulation. The
    result must contain finite voltages around the expected
    steady-state v_out ≈ D · V_in = 12 V."""
    real_import = builtins.__import__

    def _no_scipy(name, *a, **kw):
        if name == "scipy" or name.startswith("scipy."):
            raise ModuleNotFoundError("No module named 'scipy'")
        return real_import(name, *a, **kw)

    builtins.__import__ = _no_scipy
    try:
        # Force a fresh import of the dsed subpackage with scipy
        # blocked — this is what the wheel install sees.
        sys.modules.pop("scipy", None)
        for mod in list(sys.modules):
            if mod.startswith("scipy"):
                sys.modules.pop(mod, None)
        if "pulsim.dsed" in sys.modules:
            importlib.reload(sys.modules["pulsim.dsed"])
        if "pulsim.dsed.bdf2_integrator" in sys.modules:
            importlib.reload(sys.modules["pulsim.dsed.bdf2_integrator"])

        # SYNCHRONOUS buck: two controlled switches, complementary
        # drive, NO diode. The original fixture used p.add_buck,
        # whose freewheel diode the dsed engine cannot commutate —
        # its bit stayed frozen OFF and v_out came out 0.59 V where
        # 12 V is correct, a 20x error these tests never caught
        # because they only asserted finiteness and speed (the
        # docstring promised "v_out ~ 12 V"; no assertion checked
        # it). engine='dsed' now REFUSES diode circuits; this
        # fixture is what it can genuinely simulate.
        b = _synchronous_buck()
        F_SW = 100e3
        T_SW = 1.0 / F_SW
        DUTY = 0.5

        def pwm(t):
            hi = (t % T_SW) < (DUTY * T_SW)
            m = p.SwitchStateMask(b.graph.num_switches)
            m.set(0, hi)
            m.set(1, not hi)
            return m

        t0 = time.perf_counter()
        res = p.simulate(b, t_end=5e-3, engine="dsed", switch_fn=pwm)
        elapsed = time.perf_counter() - t0
    finally:
        builtins.__import__ = real_import

    # Sanity: produced a trajectory.
    assert res.num_steps() > 0
    final_state = np.asarray(res.states[-1])
    assert final_state.size > 0
    assert np.all(np.isfinite(final_state)), final_state
    # THE NUMBER, not just finiteness. The reduced state is
    # [v_C, i_L]; at D = 0.5 from 24 V the output must settle near
    # 12 V and the load current near 6 A. The previous fixture's
    # 0.59 V passed the old finiteness-only checks for months.
    states = np.asarray(res.states)
    tail = states[int(0.9 * len(states)):]
    v_out = float(np.mean(tail[:, 0]))
    i_l = float(np.mean(tail[:, 1]))
    assert v_out == pytest.approx(12.0, rel=0.05), v_out
    assert i_l == pytest.approx(6.0, rel=0.10), i_l
    # Sanity: well under 1s on the native C++ path. The Python
    # fallback (had it triggered) would take 30+ seconds because
    # `pulsim.dsed.PEDSimulatorAuto` walks per-event in pure Python.
    # We use a generous 5s budget for slow CI runners — still
    # comfortably below the Python-fallback regime.
    assert elapsed < 5.0, (
        f"DSED 5ms buck took {elapsed*1000:.0f} ms — suggests the "
        f"C++ native path was not selected. The native path "
        f"completes in ~1 ms.")


def test_simulate_dsed_faster_than_pwl_on_buck_ccm() -> None:
    """End-to-end speedup check: on the canonical buck CCM benchmark
    with PWL at dt=100ns, DSED auto must be at least 5× faster.

    This pins the headline claim — if a regression makes the
    dispatcher take the Python fallback (e.g. another top-level
    optional-dep import sneaks in), this test fails loudly with
    measurable evidence."""
    pytest.importorskip("scipy", reason="speedup measurement requires "
                       "scipy installed so PWL has no warm-cache bias")

    # Synchronous buck — see _synchronous_buck: the old add_buck
    # fixture's diode froze OFF under dsed and this benchmark spent
    # months comparing the speed of a 20x-wrong answer.
    b = _synchronous_buck()
    F_SW = 100e3
    T_SW = 1.0 / F_SW
    DUTY = 0.5

    def pwm(t):
        hi = (t % T_SW) < (DUTY * T_SW)
        m = p.SwitchStateMask(b.graph.num_switches)
        m.set(0, hi)
        m.set(1, not hi)
        return m

    # Warm-up — PWL cache build cost.
    _ = p.simulate(b, t_end=1e-5, dt=1e-7, switch_fn=pwm)
    _ = p.simulate(b, t_end=1e-5, engine="dsed", switch_fn=pwm)

    n_trials = 3
    pwl_times = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        p.simulate(b, t_end=5e-3, dt=1e-7, switch_fn=pwm)
        pwl_times.append(time.perf_counter() - t0)
    dsed_times = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        p.simulate(b, t_end=5e-3, engine="dsed", switch_fn=pwm)
        dsed_times.append(time.perf_counter() - t0)

    speedup = min(pwl_times) / min(dsed_times)
    assert speedup > 5.0, (
        f"DSED/PWL speedup on buck CCM was only {speedup:.1f}× — "
        f"expected >>5×. PWL best={min(pwl_times)*1000:.1f} ms, "
        f"DSED best={min(dsed_times)*1000:.1f} ms. The dispatcher "
        f"may have silently fallen into a Python fallback.")
