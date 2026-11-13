"""Periodic steady state on the fixed (``pwl``) and variable (``trbdf2``)
engines: add-periodic-steady-state-shooting group 5.

The unknown is the TRUE state at the section ``t0 = 0`` -- capacitor
voltages and inductor currents (design D1). A period run starts from the
MNA vector ``X(x)`` that carries exactly that state (minimum norm, every
other entry zero) and reads the state back at ``T``:

* ``pwl`` runs with ``discontinuity_restart=True`` and
  ``enable_substep_state_correction=True``: the first steps after ``t0``
  are the restart rule's SDIRK steps, which read no derivative history,
  so ``P`` is a function of the true state alone (measured: bit for bit
  the map from the engine's own full vector at ``kT``). That is the
  transient's own map only when the transient restarts at ``t0`` too,
  i.e. when a gate edge falls at ``t0``; without one the fixed engine is
  refused (feature ``'section'``), never answered with an orbit its
  transient does not reach.
* ``trbdf2`` runs at ``rtol = 1e-10``, ``atol = 1e-13`` unless the
  caller's ``options`` say otherwise, with the step ceiling ``dt``
  (default ``T/100``): with ``dt`` unset ``simulate`` caps the step at
  ``t_end/1000``, so a one-period run and a long transient would be two
  different integrations with two different orbits.

``M`` is a forward-difference Jacobian of ``P`` (``n`` period runs),
updated by Broyden rank-one steps between refreshes; it is refreshed when
the scaled residual falls by less than a factor 10 in an accepted step, when no
backtracking step is accepted, and always at the candidate orbit before
the stopping rule reads it, so the reported multipliers and the step
test are those of a fresh difference Jacobian (design D3). The Newton
iteration, its globalisation and its stopping rule are those of the
exact path (D4).
"""

from __future__ import annotations

import numpy as np

# Forward-difference step, of (scale_j + |x_j|), and the relative noise
# of one period run it divides: the fixed engine is deterministic to
# rounding; a variable-step run moves by its tolerance when x moves (its
# step sequence changes), so it takes a larger step (DECISIONS 20).
_FD_REL = {"pwl": 1e-6, "trbdf2": 1e-4}
_RUN_NOISE_PWL = 1e-14
_SLOW = 0.1             # residual ratio above which M is refreshed


def _unsup(feature, msg):
    from .periodic_steady_state import _unsup as u
    return u(feature, msg)


class TransientMap:
    """One period of ``engine`` from the true state (the map ``P``)."""

    loop = "fd-broyden"

    def __init__(self, builder, T, switch_fn, engine, dt=None, options=None):
        from .periodic_steady_state import (_check_circuit, _check_schedule,
                                            _err)
        self.builder, self.T, self.sf, self.engine = builder, float(T), \
            switch_fn, engine
        if engine not in ("pwl", "trbdf2"):
            raise _err(f"engine must be 'exact', 'dsed', 'pwl' or "
                       f"'trbdf2', got {engine!r}")
        comps = list(builder.components())
        _check_circuit(builder, self.T, comps)
        opts = dict(options or {})
        for k in ("initial_state", "t_end", "engine", "switch_fn",
                  "resume_from", "closed_loops", "step_observer"):
            if k in opts:
                raise _unsup(k, f"options[{k!r}]: the period runs set the "
                             "start, the end and the schedule themselves; "
                             "a controller with hidden state has no place "
                             "in the true state of the section.")
        if engine == "pwl":
            if dt is None or not (dt > 0):
                raise _err("engine='pwl' needs dt > 0.")
            N = self.T / dt
            if abs(N - round(N)) > 1e-9 * N:
                raise _err(f"engine='pwl': period/dt = {N!r} is not an "
                           "integer, so the section does not fall on the "
                           "grid.")
            self.n_grid = int(round(N))
            opts.update(dt=float(dt), discontinuity_restart=True,
                        enable_substep_state_correction=True)
        else:
            opts.setdefault("rtol", 1e-10)
            opts.setdefault("atol", 1e-13)
            opts["dt"] = float(dt) if dt is not None else self.T / 100.0
        self.kw = opts
        # |mu| within `marginal` of 1 cannot be told from 1 by a difference
        # Jacobian: 100 x (run noise / step) (measured noise 0.4 rtol/h)
        self.fd_rel = _FD_REL[engine]
        noise = (_RUN_NOISE_PWL if engine == "pwl"
                 else float(opts.get("rtol", 1e-10)))
        self.marginal = max(1e-10, 100.0 * noise / self.fd_rel)
        from . import _pulsim
        controlled = _pulsim._switch_census(builder.graph, builder.pool)[2]
        if switch_fn is None and len(controlled):
            raise _err(f"the circuit has {len(controlled)} controlled "
                       "switch(es) and no switch_fn: there is no gate "
                       "schedule to be periodic in.")
        self._check_gate(switch_fn, _check_schedule, _err)

        # the true state: capacitor voltages and inductor currents
        n_full = len(builder.state_var_names())
        rows, labels, devs = [], [], []
        graph = builder.graph
        for c in comps:
            if c["kind"] == "capacitor":
                a, k = (int(v) for v in c["nodes"])
                r = np.zeros(n_full)
                if a >= 0:
                    r[a] += 1.0
                if k >= 0:
                    r[k] -= 1.0
                rows.append(r)
                labels.append(f"v({graph.node_name(a)})" if k < 0 and a >= 0
                              else f"v({c['name']})")
                devs.append([c["name"]])
            elif c["kind"] == "inductor":
                r = np.zeros(n_full)
                r[int(builder.pool.branch_var_id_for_inductor(
                    c["branch_id"], graph))] = 1.0
                rows.append(r)
                labels.append(f"i({c['name']})")
                devs.append([c["name"]])
        if not rows:
            raise _err("the circuit has no capacitors or inductors, so it "
                       "has no periodic state: its solution is algebraic "
                       "at every instant.")
        self.Cm = np.array(rows)
        if np.linalg.matrix_rank(self.Cm) < len(rows):
            raise _unsup("state", "the capacitor voltages and inductor "
                         "currents are not independent (a capacitor loop or "
                         "an inductor cut-set), so they are not a state.")
        self.pinv = np.linalg.pinv(self.Cm)
        self.labels, self.label_devices = labels, devs
        self.n = len(rows)
        self.n_runs = self.n_steps = 0
        self.n_jacobians = self.n_broyden = 0

    def _check_gate(self, sf, check_schedule, err):
        T = self.T
        if sf is None:
            if self.engine == "pwl":
                raise _unsup("section", self._no_edge())
            return
        if hasattr(sf, "next_edge_after"):
            check_schedule(sf, T)
            edge0 = not (sf(0.0) == sf(T * (1.0 - 1e-9)))
        elif self.engine == "pwl":
            # an opaque gate is sampled at the grid points (the restart
            # rule's step-start sampling): periodic on the grid or refused
            dt = T / self.n_grid
            if any(not (sf(k * dt) == sf(T + k * dt))
                   for k in range(self.n_grid)):
                raise err(f"the gate schedule is not periodic with "
                          f"period={T!r} on the grid dt={dt!r}.")
            edge0 = not (sf(0.0) == sf(T - dt))
        else:
            raise _unsup("switch_fn", "switch_fn is an opaque Python "
                         "callable (no next_edge_after(t)): the variable-step "
                         "engine finds its edges by sampling, which is not "
                         "reproducible between map evaluations. Give it a "
                         "next_edge_after(t) method or use a built-in "
                         "schedule.")
        if self.engine == "pwl" and not edge0:
            raise _unsup("section", self._no_edge())

    def _no_edge(self):
        return ("no gate edge at the section t = 0. The fixed engine's map "
                "restarts its trapezoidal history at t = 0, which its "
                "transient does only at a discontinuity, so the orbit would "
                "not be the one the transient reaches. Put a gate edge at "
                "t = 0 (the built-in PWM helpers' rising edge is there) or "
                "use engine='exact' or 'trbdf2'.")

    # -- the map -------------------------------------------------------
    def full(self, x):
        """The MNA vector that carries the true state ``x``."""
        return self.pinv @ np.asarray(x, dtype=float)

    def simulate(self, x, periods=1):
        from . import simulate
        return simulate(self.builder, t_end=periods * self.T,
                        engine=self.engine, switch_fn=self.sf,
                        initial_state=self.full(x), **self.kw)

    def run(self, x):
        """``(x(T), max |x_i(t)|, the run)``; a failed run raises."""
        from ._result_views import states_as_array
        from .periodic_steady_state import _conv
        self.n_runs += 1
        try:
            res = self.simulate(x)
        except RuntimeError as e:
            if isinstance(e, (NotImplementedError, RecursionError)) or \
                    type(e).__name__ == "_CxxCancelled":
                raise
            raise _conv(f"a period run failed: {e}") from e
        t = np.asarray(res.times, dtype=float)
        if not abs(t[-1] - self.T) <= 1e-9 * self.T:
            raise _conv(f"a period run ended at t = {t[-1]!r}, not at the "
                        f"period {self.T!r}.")
        Y = states_as_array(res) @ self.Cm.T
        self.n_steps += max(len(t) - 1, 0)
        return Y[-1].copy(), np.abs(Y).max(axis=0), res

    # -- diagnostics shared with the exact path's messages --------------
    def name_states(self, v, scale=None):
        from .periodic_steady_state import _Solver
        return _Solver.name_states(self, v, scale)


def solve(m, x, *, rtol, max_period_runs, n_relax):
    """Newton shooting on ``m`` from ``x``: ``(x0, M, mu, history,
    full_state, residual, last run)`` or a ``ConvergenceError``."""
    from .periodic_steady_state import _ARMIJO, _N_LAMBDA, _conv
    n = m.n
    scale = np.zeros(n)
    history: list = []

    def scaled(v, x0):
        den = rtol * (scale + np.abs(x0))
        with np.errstate(divide="ignore", invalid="ignore"):
            q = np.where(v == 0.0, 0.0, np.abs(v) / np.where(
                den > 0, den, 1e-300))
        return float(q.max())

    def norm(xe, sc, x0):
        nonlocal scale
        scale = np.maximum(scale, sc)
        return scaled(xe - x0, x0)

    def budget(extra=""):
        if m.n_runs >= max_period_runs:
            raise _conv(
                f"no periodic orbit found within max_period_runs="
                f"{max_period_runs}{extra} on engine={m.engine!r}. Residual "
                f"history max|P(x)-x|: {[f'{h:.3e}' for h in history]}.")

    def jac(x0, xe):
        """Forward differences of P at x0 (n runs)."""
        M = np.empty((n, n))
        for j in range(n):
            h = m.fd_rel * max(scale[j] + abs(x0[j]), 1e-12)
            xj = x0.copy()
            xj[j] += h
            budget(" (computing M by finite differences)")
            M[:, j] = (m.run(xj)[0] - xe) / h
        m.n_jacobians += 1
        return M

    x = np.asarray(x, dtype=float).copy()
    xe, sc, last = m.run(x)
    r = norm(xe, sc, x)
    M = jac(x, xe)
    fresh = True
    relax_rounds = 0
    polished = False
    I_n = np.eye(n)
    while True:
        r = norm(xe, sc, x)
        G = xe - x
        history.append(float(np.abs(G).max()))
        if r <= 1.0 and not fresh:
            # the stopping rule reads M: never a Broyden estimate
            M, fresh = jac(x, xe), True
        IM = I_n - M
        sig = np.linalg.svd(IM, compute_uv=False)
        singular = sig[-1] <= m.marginal * max(1.0, sig[0])
        if r <= 1.0:
            if singular or polished or scaled(np.linalg.solve(IM, G), x,
                                               ) <= 1.0:
                break
        if singular:
            if not fresh:
                M, fresh = jac(x, xe), True
                continue
            _, _, Vt = np.linalg.svd(IM)
            raise _conv(
                "I - M is singular: a Floquet multiplier equals 1, so the "
                "orbit is not isolated (a floating capacitor, an inductor "
                "loop with no resistance, an integrator with no path to its "
                f"input). States of the marginal direction: "
                f"{m.name_states(Vt[-1], scale)}.")
        delta = np.linalg.solve(IM, G)
        accepted = False
        lam = 1.0
        for _ in range(_N_LAMBDA):
            budget()
            xt = x + lam * delta
            xte, tsc, tlast = m.run(xt)
            r_new = norm(xte, tsc, xt)
            if r_new <= (1.0 - _ARMIJO * lam) * norm(xe, sc, x) or (
                    r <= 1.0 and r_new <= 1.0):
                polished = r <= 1.0
                # Broyden on G = P - x: J += (dG - J s) s^T / s^T s
                s_ = xt - x
                J = M - I_n
                dG = (xte - xt) - G
                J = J + np.outer(dG - J @ s_, s_) / float(s_ @ s_)
                M, fresh = J + I_n, False
                m.n_broyden += 1
                x, xe, sc, last = xt, xte, tsc, tlast
                accepted = True
                if r_new > _SLOW * r and r_new > 1.0:
                    M, fresh = jac(x, xe), True
                break
            lam *= 0.5
        if accepted:
            continue
        if not fresh:
            M, fresh = jac(x, xe), True
            continue
        relax_rounds += 1
        if relax_rounds > 3:
            raise _conv(
                "Newton made no progress after three rounds of "
                f"{n_relax} relaxation periods on engine={m.engine!r}. "
                f"Residual history max|P(x)-x|: "
                f"{[f'{h:.3e}' for h in history]}.")
        for _ in range(n_relax):
            budget()
            x = xe
            xe, sc, last = m.run(x)
        M, fresh = jac(x, xe), True

    mu = np.linalg.eigvals(M)
    lam_, vec = np.linalg.eig(M)
    for i, m_ in enumerate(lam_):
        if abs(abs(m_) - 1.0) < m.marginal:
            raise _conv(
                f"the orbit is not isolated: Floquet multiplier {m_:.9g} "
                "lies on the unit circle within the accuracy of the "
                f"difference Jacobian ({m.marginal:.0e} on engine="
                f"{m.engine!r}), so neighbouring solutions need not decay "
                "to it (a lossless loop or an undamped integrator). "
                f"States of the marginal direction: "
                f"{m.name_states(vec[:, i], scale)}.")
    # No separate verification period: a period run is a fresh, fully
    # deterministic simulate() from X(x), so re-running x reproduces the
    # accepted run bit for bit (DECISIONS 17); that run is the orbit.
    residual = float(np.abs(xe - x).max())
    from ._result_views import states_as_array
    full = np.array(states_as_array(last)[-1], dtype=float)
    vrun = last
    return x, M, mu, history, full, residual, vrun


def periodic_steady_state_on(builder, *, period, switch_fn, engine, dt,
                             options, start_from, max_period_runs, n_relax,
                             rtol, sensitivities):
    """:func:`pulsim.periodic_steady_state` on ``engine`` (pwl, trbdf2)."""
    from .periodic_steady_state import (PeriodicSteadyStateResult, _classify,
                                        _err)
    if sensitivities is not None:
        raise _unsup("sensitivities", "orbit sensitivities exist on "
                     f"engine='exact' only (engine={engine!r}: refused, never "
                     "a finite difference).")
    m = TransientMap(builder, period, switch_fn, engine, dt, options)
    n = m.n
    if start_from is None:
        x = np.zeros(n)
    elif isinstance(start_from, PeriodicSteadyStateResult):
        x = np.asarray(start_from.x0, dtype=float).copy()
    else:
        x = np.asarray(start_from, dtype=float).reshape(-1).copy()
    if x.shape != (n,):
        raise _err(f"start_from has {x.size} entries; the state has {n} "
                   f"({m.labels}).")
    x0, M, mu, history, full, residual, vrun = solve(
        m, x, rtol=rtol, max_period_runs=max_period_runs, n_relax=n_relax)
    unstable = [(i, complex(v), _classify(v)) for i, v in enumerate(mu)
                if abs(v) >= 1.0]
    res = PeriodicSteadyStateResult(
        x0=x0, state_labels=list(m.labels), _orbit=vrun, monodromy=M,
        multipliers=mu, stable=not unstable, unstable_multipliers=unstable,
        pattern=[], residual=residual, residual_history=history,
        pattern_history=[], n_period_runs=m.n_runs,
        n_engine_steps=m.n_steps, period=m.T, _ctx=m, full_state=full)
    return res
