/**
 * pi_controller.c — Custom PI controller CBlock.
 *
 * Implements a discrete-time PI controller:
 *   u[n] = kp * e[n] + ki * integral[n]
 *   integral[n] = integral[n-1] + e[n] * dt
 *
 * Inputs:  in[0] = error signal (setpoint − feedback)
 * Outputs: out[0] = control output (clamped to [output_min, output_max])
 *
 * Parameters (hard-coded for illustration):
 *   kp = 0.5, ki = 50.0, output_min = 0.0, output_max = 1.0
 *
 * Demonstrates:
 *   - Multi-variable state (integral, previous time) in ctx
 *   - Anti-windup via output clamping
 *   - Numeric equivalence with the built-in PI_CONTROLLER block
 */

#include "pulsim/v1/cblock_abi.h"

#include <stdlib.h>

/* Controller parameters */
#define KP          0.5
#define KI          50.0
#define OUTPUT_MIN  0.0
#define OUTPUT_MAX  1.0

struct PulsimCBlockCtx {
    double integral;
    double t_prev;
};


static double clamp(double v, double lo, double hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}


PULSIM_CBLOCK_EXPORT int pulsim_cblock_abi_version = PULSIM_CBLOCK_ABI_VERSION;


PULSIM_CBLOCK_EXPORT int pulsim_cblock_init(
    PulsimCBlockCtx** ctx_out, const PulsimCBlockInfo* info)
{
    (void)info;

    struct PulsimCBlockCtx* s =
        (struct PulsimCBlockCtx*)malloc(sizeof(struct PulsimCBlockCtx));
    if (!s) return -1;

    s->integral = 0.0;
    s->t_prev   = -1.0;

    *ctx_out = (PulsimCBlockCtx*)s;
    return 0;
}


PULSIM_CBLOCK_EXPORT int pulsim_cblock_step(
    PulsimCBlockCtx* ctx, double t, double dt,
    const double* in, double* out)
{
    (void)t;

    struct PulsimCBlockCtx* s = (struct PulsimCBlockCtx*)ctx;

    double error = in[0];
    double effective_dt = (s->t_prev < 0.0) ? 0.0 : dt;

    s->integral += error * effective_dt;
    s->t_prev    = t;

    double u = KP * error + KI * s->integral;
    out[0] = clamp(u, OUTPUT_MIN, OUTPUT_MAX);
    return 0;
}


PULSIM_CBLOCK_EXPORT void pulsim_cblock_destroy(PulsimCBlockCtx* ctx)
{
    free(ctx);
}
