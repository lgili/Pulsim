"""MMC arms as exact Thevenin equivalents (GGJ aggregation).

v2.0 Phase 3, the audit's "obra №2". The L3 arm path couples through
a ``b_extra`` source read one step late — a delayed co-simulation
its own comment admits. This module couples the same physics
EXACTLY, using three pieces the kernel already has:

* the per-step aggregation lives in C++
  (:class:`pulsim.ThevArm`, ``core/include/pulsim/mmc/
  thevenin_arm.hpp``): trapezoidal companion per submodule,
  analytic series elimination, sort-and-select balancing,
  back-solve;
* the arm enters the network as ONE resistor branch — zero mask
  bits for any number of submodules — whose value change on a
  gating event is absorbed by ``cache.refactor_parametric`` (the
  etree path refactor) in O(path);
* the coupling is driven from ``simulate()``'s own step hooks,
  which run BEFORE each step's solve: the observer finalises the
  previous step's capacitors from the just-solved state, stamps
  this step's (R_eq, V_eq), and the ``b_extra`` hook injects the
  Norton current at the same instant. Nothing is read late — the
  only "previous step" quantity is the trapezoidal companion's own
  history term, same as every capacitor in the engine.

Usage::

    arm = p.add_mmc_thevenin_arm(
        b, "arm_u", "dc_p", "ac_a",
        n_sm=100, c_sm=8e-3, r_on=1e-3, v_c_init=16.0,
        n_on=lambda t: level_of(t))          # 0..n_sm
    res = p.simulate(b, t_end=..., dt=..., mmc_arms=[arm])
    arm.v_c                                   # per-SM voltages
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np


@dataclass
class MmcThevArm:
    """One registered arm: builder-side identity + deferred kernel
    state (the kernel arm needs ``dt``, which only ``simulate()``
    knows)."""

    name: str
    branch_id: int
    node_from: int          # -1 = ground
    node_to: int
    n_sm: int
    c_sm: float
    r_on: float
    v_c_init: float
    n_on: Callable[[float], int]
    v_c_preset: Optional[list] = None
    # The builder this arm's resistor branch lives in — the driver
    # refuses to mutate any OTHER builder's devices through this
    # handle's branch id.
    _builder: object = field(default=None, repr=False)
    # Filled by the driver:
    _kernel: object = field(default=None, repr=False)

    @property
    def v_c(self):
        if self._kernel is None:
            raise RuntimeError(
                "MmcThevArm.v_c: the arm has not been simulated yet "
                "— pass it to simulate(..., mmc_arms=[arm]) first.")
        return list(self._kernel.v_c)

    @property
    def total_stored_voltage(self):
        if self._kernel is None:
            raise RuntimeError("MmcThevArm: not simulated yet.")
        return float(self._kernel.total_stored_voltage)


def add_mmc_thevenin_arm(builder, name, n_from, n_to, *, n_sm,
                           c_sm, r_on=1e-3, v_c_init=0.0,
                           n_on, v_c_preset=None):
    """Register a GGJ Thevenin arm between two nodes.

    Adds a single resistor branch (placeholder value ``n_sm·r_on``,
    the all-bypassed arm; the driver stamps the true R_eq before the
    first solve) and returns the handle to pass to
    ``simulate(mmc_arms=[...])``.

    ``n_on(t) -> int`` is the insertion count for the coming step;
    it is sampled at each step's END time ``t_k`` — the same
    convention the engine uses for ``switch_fn``. Which submodules
    realise the count is chosen internally by sort-and-select
    capacitor balancing.

    State semantics (matches the rest of the engine): every
    ``simulate()`` call restarts the arm from ``v_c_init`` /
    ``v_c_preset``, exactly as explicit capacitors restart from
    their declared initial conditions. To CONTINUE a run, pass
    ``initial_state=res.states[-1]`` for the network AND a fresh
    handle with ``v_c_preset=old_arm.v_c`` for the arm —
    ``simulate()`` refuses ``initial_state`` with an arm that has no
    explicit preset, because silently resetting the capacitors mid-
    continuation is an unphysical energy jump. After a
    ``SimulationAborted``, ``arm.v_c`` reflects the observer call of
    the FAILING step (its back-solve ran; its solve did not), so it
    can sit O(dt) ahead of ``partial.states[-1]``.
    """
    if v_c_preset is not None and len(v_c_preset) != int(n_sm):
        raise ValueError(
            f"add_mmc_thevenin_arm({name!r}): v_c_preset has "
            f"{len(v_c_preset)} entries for n_sm={n_sm}")
    branch_id = int(builder.num_branches)
    builder.add_resistor(name, n_from, n_to,
                          float(n_sm) * float(r_on))
    # node_id_of resolves ground aliases to -1 itself, and the
    # add_resistor above has just registered both names.
    return MmcThevArm(
        name=name,
        branch_id=branch_id,
        node_from=int(builder.node_id_of(n_from)),
        node_to=int(builder.node_id_of(n_to)),
        n_sm=int(n_sm),
        c_sm=float(c_sm),
        r_on=float(r_on),
        v_c_init=float(v_c_init),
        n_on=n_on,
        v_c_preset=(list(v_c_preset)
                     if v_c_preset is not None else None),
        _builder=builder,
    )


def make_mmc_thevenin_driver(arms, builder, cache, dt, state_size):
    """Build the exact-coupling ``(step_observer, b_extra_fn)`` pair.

    Called by ``simulate()`` after it constructs the cache — the
    driver mutates THAT cache's factors in place through
    ``refactor_parametric`` whenever an arm's insertion count
    changes, which is what keeps the stamped R_eq exact instead of
    one step late.
    """
    from . import _pulsim as _k

    seen_branches = set()
    for a in arms:
        if a.branch_id in seen_branches:
            raise ValueError(
                f"simulate(mmc_arms=...): arm {a.name!r} "
                f"(branch {a.branch_id}) appears more than once — "
                "a duplicate handle would double its Norton "
                "injection on the same physical branch.")
        seen_branches.add(a.branch_id)
        if a._builder is not None and a._builder is not builder:
            raise ValueError(
                f"simulate(mmc_arms=...): arm {a.name!r} was "
                "registered on a DIFFERENT builder than the one "
                "being simulated — its branch id would mutate "
                "whatever device happens to hold that id here.")

    kernels = []
    for a in arms:
        params = _k.ThevArmParams(
            n_sm=a.n_sm, c_sm=a.c_sm, r_on=a.r_on, dt=float(dt),
            v_c_init=a.v_c_init)
        arm = _k.ThevArm(params)
        if a.v_c_preset is not None:
            for i, v in enumerate(a.v_c_preset):
                arm.set_v_c(i, float(v))
        a._kernel = arm
        kernels.append(arm)

    n_arms = len(arms)
    # Per-arm stash of the stamp the CURRENT step was solved with —
    # what lets the next observer recover i_arm from x_prev exactly:
    # i = G·(v_from − v_to) − I_N.
    g_eq = np.zeros(n_arms)
    i_n = np.zeros(n_arms)

    # run_transient's observer contract (read from the kernel, not
    # assumed): one PRIMING call at (t_start, x0) before the loop,
    # then one call per step at (t_k, x_{k-1}) BEFORE that step's
    # solve. So the first TWO calls both see x0 with no solve in
    # between. Call #1 does nothing; call #2 (the pre-solve-1 call,
    # t = solve 1's END time) arms the insertion set with i_prev = 0
    # — which makes EVERY solve's gating sample n_on at its step's
    # end, the same convention switch_fn gets; real finalisation
    # starts at call #3. Finalising at call #2 would back-solve a
    # phantom −I_N of discharge against x0.
    call_no = [0]

    def _stamp(a, arm, k, i_prev, t):
        n_on = int(a.n_on(t))
        if not 0 <= n_on <= a.n_sm:
            raise ValueError(
                f"mmc arm {a.name!r}: n_on(t={t:.9g}) returned "
                f"{n_on}, outside [0, n_sm={a.n_sm}]")
        r_eq, v_eq, changed = arm.pre_step(i_prev, n_on)
        if changed:
            builder.update_resistor_R(a.name, r_eq)
            # Path refactor of the resident factors — O(etree
            # path), not a rebuild, and the reason a gating event
            # costs microseconds instead of a refactorisation.
            cache.refactor_parametric(a.branch_id, r_eq)
        g_eq[k] = 1.0 / r_eq
        i_n[k] = v_eq / r_eq

    def step_observer(t, x_prev):
        call_no[0] += 1
        if call_no[0] == 1:
            return
        if call_no[0] == 2:
            for k, (a, arm) in enumerate(zip(arms, kernels)):
                _stamp(a, arm, k, 0.0, t)
            return
        x = np.asarray(x_prev)
        for k, (a, arm) in enumerate(zip(arms, kernels)):
            va = 0.0 if a.node_from < 0 else float(x[a.node_from])
            vb = 0.0 if a.node_to < 0 else float(x[a.node_to])
            _stamp(a, arm, k, g_eq[k] * (va - vb) - i_n[k], t)

    def b_extra_fn(_t):
        v = np.zeros(state_size)
        for k, a in enumerate(arms):
            # Calibrated convention: b_extra[node] = −I injects I
            # INTO the node. The Norton companion injects I_N into
            # `from` and out of `to`.
            if a.node_from >= 0:
                v[a.node_from] -= i_n[k]
            if a.node_to >= 0:
                v[a.node_to] += i_n[k]
        return v

    def finalize(t_end, x_final):
        # The observer only fires BEFORE solves, so the last step's
        # charge transfer is still pending when run_transient
        # returns. Fold it in with finalize_step — back-solve ONLY,
        # never pre_step: pre_step would re-run sort-and-select for
        # a step that never executes, and each capacitor "leaving"
        # that phantom selection would take a trailing half-step
        # r_c·i it never earned (~mV per cap, caught by the crown
        # parity test).
        del t_end
        if call_no[0] < 2:
            # Fewer than two observer calls means NO solve ran
            # (should_continue cancelled at k=1, or t_end − t_start
            # < dt gave a single-sample run) — "finalising" would
            # back-solve a phantom step with i = −V_eq/R_eq and
            # discharge every inserted capacitor by kiloamp-scale
            # nonsense. Caught by the adversarial review.
            return
        x = np.asarray(x_final)
        for k, (a, arm) in enumerate(zip(arms, kernels)):
            va = 0.0 if a.node_from < 0 else float(x[a.node_from])
            vb = 0.0 if a.node_to < 0 else float(x[a.node_to])
            arm.finalize_step(g_eq[k] * (va - vb) - i_n[k])

    return step_observer, b_extra_fn, finalize


__all__ = [
    "MmcThevArm",
    "add_mmc_thevenin_arm",
    "make_mmc_thevenin_driver",
]
