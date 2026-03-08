/**
 * gain_block.c — passthrough gain C-Block for PulsimCore.
 *
 * Multiplies the single input by a fixed gain.
 * This is the simplest possible CBlock: no state, no init/destroy needed.
 *
 * Compile with:
 *   cc -O2 -shared -fPIC -std=c11 \
 *      -I<pulsimcore>/core/include \
 *      gain_block.c -o gain_block.so
 */

#include "pulsim/v1/cblock_abi.h"

#define GAIN 3.0

PULSIM_CBLOCK_EXPORT int pulsim_cblock_abi_version = PULSIM_CBLOCK_ABI_VERSION;

PULSIM_CBLOCK_EXPORT int pulsim_cblock_step(
    PulsimCBlockCtx* ctx, double t, double dt,
    const double* in, double* out)
{
    (void)ctx; (void)t; (void)dt;
    out[0] = GAIN * in[0];
    return 0;  /* 0 = success */
}
