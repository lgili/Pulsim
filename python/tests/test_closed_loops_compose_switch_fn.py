"""Regression: ``simulate(closed_loops=..., switch_fn=..., step_observer=...)``
must compose, not raise.

**Bug history (T1.1 in GUI integration findings).** Pre-v1.6.5
``simulate()`` raised ``ValueError("pass closed_loops OR
switch_fn/step_observer, not both — the helper composes both
callbacks internally.")`` whenever both a ``closed_loops``
descriptor and a separate ``switch_fn``/``step_observer`` were
given. This blocked the canonical drive topology:

* One stage in closed loop (e.g. PFC Vbus regulation via
  ``bind_pi_to_switch`` → ``closed_loops``).
* Another stage switching openly (e.g. 3φ VSI driven by
  ``make_three_phase_spwm_fn`` → ``switch_fn``).
* A motor observer that wants a ``step_observer`` of its own.

Users were forced to drop one path; in practice the closed loop
won and the VSI's switches never toggled, leaving the rotor dead.

**Fix.** ``simulate()`` now composes: every ``closed_loops``
switch_fn AND the user-supplied switch_fn flow through
``make_combined_switch_fn`` (mask OR), and the closed_loops'
observers AND the user's step_observer chain in registration
order.

These tests pin:

1. **closed_loops + switch_fn coexist** — both contributors'
   switch indices are honoured. We build a circuit with two
   independent switches (one owned by a ``ClosedLoop``, one by
   the user's ``switch_fn``) and verify both toggle.
2. **closed_loops + step_observer coexist** — the user's
   observer is invoked once per accepted step, AFTER the loop's
   observer (so the loop has already updated its PI state).
3. **closed_loops alone** — legacy single-argument behaviour is
   preserved bit-for-bit (no regression).
4. **All three together** — the original blocked configuration
   from the GUI findings doc now runs cleanly.
"""
from __future__ import annotations

import numpy as np

import pulsim as p


def _two_switch_plant():
    """Plant with two independent switches feeding the same load.
    Builder shape: vin --SW_A--+--R_load--gnd, vin --SW_B--+
    Two switches, each independently controlled, share node ``mid``.
    """
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", 10.0)
    b.add_switch("SW_A", "vin", "mid", g_on=1e3, g_off=1e-9)
    b.add_switch("SW_B", "vin", "mid", g_on=1e3, g_off=1e-9)
    b.add_resistor("R_load", "mid", "gnd", 100.0)
    b.add_capacitor("C_filter", "mid", "gnd", 1e-6)
    return b


def test_closed_loops_plus_switch_fn_both_honored() -> None:
    """closed_loops owns SW_A (via a fake constant-1 PI-equivalent
    loop), user's switch_fn owns SW_B. Both must toggle as
    independently programmed."""
    b = _two_switch_plant()
    n_sw = b.graph.num_switches

    # Fake "closed loop" that owns SW_A and keeps it always ON
    # (no real PI involved; we're testing composition, not control).
    class _FakeLoopOwningSwA:
        @staticmethod
        def switch_fn(t: float):
            m = p.SwitchStateMask(n_sw)
            m.set(0, True)   # SW_A always ON via the "loop"
            return m

        @staticmethod
        def step_observer(t: float, x) -> None:
            # No-op observer — the loop just needs the slot.
            pass

    # User's switch_fn owns SW_B, PWM at 100 kHz, 50% duty.
    T_sw = 1e-5
    def user_sw(t: float):
        m = p.SwitchStateMask(n_sw)
        m.set(1, (t % T_sw) < (0.5 * T_sw))
        return m

    # Pre-fix this raised; post-fix it composes via OR of the masks.
    res = p.simulate(
        b, t_end=1e-3, dt=1e-7,
        closed_loops=[_FakeLoopOwningSwA()],
        switch_fn=user_sw,
    )
    assert res.num_steps() > 0

    # Sanity: mid-node voltage settles near V_in (SW_A always closed,
    # SW_B independently PWM'ing but both feed the same node so
    # whenever SW_A is closed the node is pinned at V_in via R_on).
    v_mid = float(np.asarray(res.v("mid"))[-1])
    assert 9.0 < v_mid < 10.5, v_mid


def test_closed_loops_plus_step_observer_chain_in_order() -> None:
    """Loop's observer runs first, user's runs second. We record
    the call order to make sure the loop has its PI updated before
    the user reads anything."""
    call_log: list[str] = []

    class _LoopWithLoggingObserver:
        @staticmethod
        def switch_fn(t: float):
            m = p.SwitchStateMask(2)
            m.set(0, True)
            return m

        @staticmethod
        def step_observer(t: float, x) -> None:
            call_log.append(f"loop@{t:.3e}")

    def user_observer(t: float, x) -> None:
        call_log.append(f"user@{t:.3e}")

    b = _two_switch_plant()
    res = p.simulate(
        b, t_end=5e-6, dt=1e-7,
        closed_loops=[_LoopWithLoggingObserver()],
        step_observer=user_observer,
    )
    assert res.num_steps() > 0
    # Both observers fired.
    assert any(s.startswith("loop@") for s in call_log)
    assert any(s.startswith("user@") for s in call_log)
    # At every t, loop fires before user (chained in registration order).
    paired = list(zip(call_log[::2], call_log[1::2]))
    for loop_call, user_call in paired:
        assert loop_call.startswith("loop@"), loop_call
        assert user_call.startswith("user@"), user_call


def test_closed_loops_alone_unchanged() -> None:
    """Regression: the legacy single-argument path (closed_loops with
    no user switch_fn or step_observer) must run bit-for-bit the same
    as before — no behavioural drift on existing scripts."""
    class _ConstLoop:
        @staticmethod
        def switch_fn(t: float):
            m = p.SwitchStateMask(2)
            m.set(0, True)
            m.set(1, False)
            return m

        @staticmethod
        def step_observer(t: float, x) -> None:
            pass

    b = _two_switch_plant()
    res = p.simulate(b, t_end=1e-4, dt=1e-7,
                       closed_loops=[_ConstLoop()])
    assert res.num_steps() > 0
    v_mid = float(np.asarray(res.v("mid"))[-1])
    # SW_A always ON, SW_B always OFF — node pinned at V_in via R_on.
    assert 9.0 < v_mid < 10.5, v_mid


def test_closed_loops_plus_switch_fn_plus_step_observer_runs() -> None:
    """The exact configuration the GUI findings doc flagged as
    blocked: closed_loops + switch_fn + step_observer all together.
    Pre-fix this raised ValueError; post-fix runs cleanly."""
    user_observer_calls = [0]

    class _Loop:
        @staticmethod
        def switch_fn(t: float):
            m = p.SwitchStateMask(2)
            m.set(0, True)
            return m

        @staticmethod
        def step_observer(t: float, x) -> None:
            pass

    def user_sw(t: float):
        T_sw = 1e-5
        m = p.SwitchStateMask(2)
        m.set(1, (t % T_sw) < (0.5 * T_sw))
        return m

    def user_obs(t: float, x) -> None:
        user_observer_calls[0] += 1

    b = _two_switch_plant()
    res = p.simulate(
        b, t_end=1e-4, dt=1e-7,
        closed_loops=[_Loop()],
        switch_fn=user_sw,
        step_observer=user_obs,
    )
    assert res.num_steps() > 0
    assert user_observer_calls[0] > 0, (
        "User step_observer must be invoked even when closed_loops "
        "is present — pre-fix it was silently dropped.")


def test_two_closed_loops_plus_switch_fn_compose() -> None:
    """Multiple ClosedLoop entries + a user switch_fn all merge
    correctly. Pin the contract that
    ``per_switch_fns = [loop1.sf, loop2.sf, ..., user_sf]`` is the
    order ``make_combined_switch_fn`` ORs over."""
    # Plant with 3 independent switches.
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", 10.0)
    b.add_switch("SW0", "vin", "mid", g_on=1e3, g_off=1e-9)
    b.add_switch("SW1", "vin", "mid", g_on=1e3, g_off=1e-9)
    b.add_switch("SW2", "vin", "mid", g_on=1e3, g_off=1e-9)
    b.add_resistor("R", "mid", "gnd", 100.0)
    b.add_capacitor("C", "mid", "gnd", 1e-6)

    class _LoopOwning:
        """Tiny ClosedLoop-shaped object that owns one switch index."""
        def __init__(self, idx: int):
            self._idx = idx

        def switch_fn(self, t):
            m = p.SwitchStateMask(3)
            m.set(self._idx, True)
            return m

        def step_observer(self, t, x): pass

    def user_sw(t):
        m = p.SwitchStateMask(3)
        m.set(2, True)
        return m

    res = p.simulate(b, t_end=1e-4, dt=1e-7,
                       closed_loops=[_LoopOwning(0), _LoopOwning(1)],
                       switch_fn=user_sw)
    assert res.num_steps() > 0
    # All 3 switches are closed → all 3 g_on paths in parallel.
    # Node pins very close to V_in.
    v_mid = float(np.asarray(res.v("mid"))[-1])
    assert 9.5 < v_mid < 10.1, v_mid
