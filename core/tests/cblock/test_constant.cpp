#include "pulsim/v1/cblock_abi.h"

extern "C" {

PULSIM_CBLOCK_EXPORT int pulsim_cblock_abi_version = PULSIM_CBLOCK_ABI_VERSION;

PULSIM_CBLOCK_EXPORT int pulsim_cblock_step(
    PulsimCBlockCtx* ctx,
    double t,
    double dt,
    const double* in,
    double* out) {
    (void)ctx;
    (void)t;
    (void)dt;
    (void)in;
    out[0] = 0.5;
    return 0;
}

}  // extern "C"
