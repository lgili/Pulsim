/**
 * iir_filter.c — First-order IIR low-pass filter CBlock.
 *
 * Transfer function: H(z) = alpha / (1 - (1-alpha)*z^-1)
 * where alpha = dt / (tau + dt) and tau = 1/(2*pi*fc).
 *
 * State: one previous output sample stored in PulsimCBlockCtx.
 *
 * Parameters (hard-coded for simplicity):
 *   fc = 100 Hz   (cutoff frequency)
 *
 * Demonstrates:
 *   - pulsim_cblock_init:    allocate ctx, set initial state
 *   - pulsim_cblock_step:    stateful IIR computation
 *   - pulsim_cblock_destroy: free ctx memory
 */

#include "pulsim/v1/cblock_abi.h"

#include <math.h>
#include <stdlib.h>

/* Filter design constant: cutoff = 100 Hz */
#define CUTOFF_HZ 100.0
#define TWO_PI 6.28318530718

struct PulsimCBlockCtx {
    double y_prev;  /* previous output sample */
    double tau;     /* 1 / (2*pi*fc) */
};


PULSIM_CBLOCK_EXPORT int pulsim_cblock_abi_version = PULSIM_CBLOCK_ABI_VERSION;


PULSIM_CBLOCK_EXPORT int pulsim_cblock_init(
    PulsimCBlockCtx** ctx_out, const PulsimCBlockInfo* info)
{
    (void)info;

    struct PulsimCBlockCtx* s =
        (struct PulsimCBlockCtx*)malloc(sizeof(struct PulsimCBlockCtx));
    if (!s) return -1;

    s->y_prev = 0.0;
    s->tau    = 1.0 / (TWO_PI * CUTOFF_HZ);

    *ctx_out = (PulsimCBlockCtx*)s;
    return 0;
}


PULSIM_CBLOCK_EXPORT int pulsim_cblock_step(
    PulsimCBlockCtx* ctx, double t, double dt,
    const double* in, double* out)
{
    (void)t;

    struct PulsimCBlockCtx* s = (struct PulsimCBlockCtx*)ctx;

    /* Bilinear (or Euler) approximation: alpha = dt / (tau + dt) */
    double alpha = (dt > 0.0) ? (dt / (s->tau + dt)) : 1.0;

    double y = alpha * in[0] + (1.0 - alpha) * s->y_prev;
    s->y_prev = y;
    out[0] = y;
    return 0;
}


PULSIM_CBLOCK_EXPORT void pulsim_cblock_destroy(PulsimCBlockCtx* ctx)
{
    free(ctx);
}
