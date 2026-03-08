"""Example 02 — Stateful integrator via PythonCBlock.

The block uses a persistent context dict to accumulate the integral of the
input signal: y(t) = ∫ u(τ) dτ.

Run::

    PYTHONPATH=python python examples/cblock/02_python_integrator.py
"""

from pulsim.cblock import PythonCBlock


def integrator(ctx, t, dt, inputs):
    """Numerical integrator: y += u * dt."""
    ctx.setdefault("sum", 0.0)
    ctx["sum"] += inputs[0] * dt
    return [ctx["sum"]]


blk = PythonCBlock(integrator, n_inputs=1, n_outputs=1, name="integrator")

# Integrate a constant signal of 1.0 over 10 × 100 µs steps
# → expected final value: 0.001 (= 10 * 100e-6 * 1.0)
dt = 100e-6
for i in range(10):
    t = i * dt
    step_dt = dt if i > 0 else 0.0
    out = blk.step(t, step_dt, [1.0])
    print(f"t={t*1e3:.3f} ms  integral={out[0]:.6f}")

print(f"\nFinal integral value: {out[0]:.6f}  (expected ~{9*dt:.6f})")
