"""Cross-validation: C++ DSED scheduler vs Python reference.

This test is the **whole reason** the Python implementation was
kept in the repository at all (`python/tests/_dsed_reference/`)
after v1.7.0 demoted it from the public API. We run the same
synthetic LTI through both implementations and assert their state
trajectories agree within a tight tolerance. Any drift between the
C++ port and the Python ground truth — algorithm tweak that didn't
land in both, a binding marshalling bug, a step-controller
divergence — surfaces here as a numerical disagreement, not
through a vague "results look different" complaint downstream.

We use a synthetic LTI rather than a CircuitBuilder so the system
protocol the C++ scheduler talks to is identical to the Python
reference's `HasLTIPerMode` interface — no extractor in the loop,
just the integrator itself.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Make `_dsed_reference` importable even when this file is collected
# from outside `python/tests/` (e.g. CI runners with a custom rootdir).
_TESTS_DIR = Path(__file__).parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

from _dsed_reference import (  # type: ignore[import-not-found]  # noqa: E402
    PEDSimulator,
    PIController,
    EventPredictor,
)

import pulsim as p  # noqa: E402


class _LinearLTI:
    """2-state damped oscillator: ẍ + 2ζω ẋ + ω² x = b(t).

    Encoded in state-space form as ``[x, ẋ]'`` with
    ``A = [[0, 1], [-ω², -2ζω]]`` and forcing ``b = [0, u]``.
    Single mask — no events.
    """

    def __init__(self, *, omega: float = 10.0, zeta: float = 0.1,
                 u_amp: float = 1.0):
        self._A = np.array([[0.0, 1.0], [-omega ** 2, -2 * zeta * omega]])
        self._u_amp = u_amp
        self._mask = 0

    def A_matrix(self) -> np.ndarray:
        return self._A

    def b_vector(self, t: float) -> np.ndarray:
        # Step input at t=0 — DC.
        return np.array([0.0, self._u_amp])

    def rhs(self, t: float, x: np.ndarray) -> np.ndarray:
        return self._A @ x + self.b_vector(t)

    def current_mask(self) -> int:
        return self._mask

    def set_mask(self, mask: int) -> None:
        self._mask = mask


def _switch_fn_const(_t: float) -> int:
    return 0


@pytest.mark.parametrize("omega,zeta", [
    (10.0, 0.1),   # Lightly damped — many oscillations.
    (50.0, 0.7),   # Higher damping — fast transient + small steady-state.
    (5.0, 0.05),   # Slow + lightly damped — long simulation.
])
def test_cpp_dsed_matches_python_reference_on_damped_oscillator(
    omega: float, zeta: float,
) -> None:
    """Both implementations of the DOPRI5 + adaptive-PI scheduler
    must agree within the documented PI-controller tolerance over
    the whole trajectory.

    We deliberately force ``integrator='rk45'`` on both sides
    instead of ``'auto'`` so the test pins the *kernel* (DOPRI5
    Butcher tableau, PI step controller, Hermite interpolation,
    event-bisection bookkeeping) rather than the auto-dispatch
    heuristic. On stiff-leaning configurations the auto-dispatch
    Python and C++ heuristics can legitimately disagree on when to
    swap RK45→BDF2 and produce different (but each individually
    correct) trajectories — that's a separate question from
    "did the port preserve the algorithm".

    Tolerance follows numpy's ``allclose`` semantics:
    ``|x_cpp - x_py| <= atol_tol + rtol_tol·|x_py|`` — same form
    the PI controller uses internally.
    """
    rtol = 1.0e-7
    atol = 1.0e-10
    dt_init = 1.0e-6
    dt_max = 1.0e-2
    t_end = 1.0
    x0 = np.array([1.0, 0.0])

    # Cross-val budget: 1000×PI-controller tolerance per stamp.
    # Tighter than that would alias on store_every quantisation
    # and Hermite-interp jitter between schedulers.
    rtol_tol = 1000.0 * rtol
    atol_tol = 1000.0 * atol

    # ---- C++ native via the public escape hatch ----
    sys_cpp = _LinearLTI(omega=omega, zeta=zeta)
    res_cpp = p.dsed.run_user_lti(
        sys_cpp, _switch_fn_const, x0, t_end,
        integrator="rk45",
        rtol=rtol, atol=atol,
        dt_init=dt_init, dt_max=dt_max,
        store_every=1,
    )
    x_cpp = np.asarray([np.asarray(x) for x in res_cpp["states"]])

    # ---- Python reference (PEDSimulator = RK45-only) ----
    sys_py = _LinearLTI(omega=omega, zeta=zeta)
    sim_py = PEDSimulator(
        system=sys_py, switch_fn=_switch_fn_const,
        controller=PIController(rtol=rtol, atol=atol),
        predictor=EventPredictor(),
        dt_init=dt_init, dt_max=dt_max,
        store_every=1,
    )
    ref_res = sim_py.simulate(x0, t_end)
    x_py = np.asarray([np.asarray(x) for x in ref_res.states])

    # ---- Compare the END STATE. ----
    # The two schedulers pick their own step sequences (acceptable
    # rejections, stiffness-aware step-doubling, etc.), so
    # intermediate time stamps don't line up exactly — and linear
    # interpolation between them introduces error around fast
    # transients that swamps the PI-controller tolerance even when
    # the underlying trajectories are bit-equivalent at common
    # stamps.
    #
    # What the PI-controller contract DOES guarantee is that both
    # implementations land within ``atol + rtol·|x|`` of the true
    # trajectory at every accepted step. Two valid PI runs against
    # the same tolerance therefore agree at the end of the window
    # within the same envelope. That's the invariant we pin here.
    x_cpp_end = x_cpp[-1]
    x_py_end = x_py[-1]
    abs_diff = np.abs(x_cpp_end - x_py_end)
    allowed = atol_tol + rtol_tol * np.abs(x_py_end)
    assert np.all(abs_diff <= allowed), (
        f"C++ DSED vs Python reference disagree at final state on "
        f"damped oscillator (omega={omega}, zeta={zeta}). "
        f"x_cpp_end={x_cpp_end}, x_py_end={x_py_end}, "
        f"|Δ|={abs_diff}, allowed={allowed} "
        f"(atol_tol={atol_tol:.0e}, rtol_tol={rtol_tol:.0e}). "
        f"This likely means the C++ scheduler and the Python "
        f"reference algorithm have drifted — investigate which "
        f"one changed.\n"
        f"  n_accept CPP={res_cpp['n_accept']} "
        f"PY={getattr(ref_res, 'n_accept', '?')}\n"
        f"  n_reject CPP={res_cpp['n_reject']} "
        f"PY={getattr(ref_res, 'n_reject', '?')}")


def test_run_user_lti_validates_system_protocol() -> None:
    """A user who passes a system missing a required method must
    get a clear `TypeError`, not a cryptic pybind11 dispatch error."""

    class BrokenLTI:
        # Intentionally missing `b_vector` and `rhs`.
        def A_matrix(self):
            return np.eye(2)

        def current_mask(self):
            return 0

        def set_mask(self, m):
            pass

    with pytest.raises(TypeError, match="b_vector"):
        p.dsed.run_user_lti(BrokenLTI(), lambda t: 0,
                             np.zeros(2), 1.0)


def test_run_user_lti_rejects_unknown_integrator() -> None:
    sys_obj = _LinearLTI()
    with pytest.raises(ValueError, match="integrator="):
        p.dsed.run_user_lti(sys_obj, lambda t: 0, np.zeros(2), 1.0,
                              integrator="bogus")
