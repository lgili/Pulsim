#include "pulsim/v1/cblock_abi.h"

#include <cmath>

extern "C" {

PULSIM_CBLOCK_EXPORT int pulsim_cblock_abi_version = PULSIM_CBLOCK_ABI_VERSION;

PULSIM_CBLOCK_EXPORT int pulsim_cblock_step(
    PulsimCBlockCtx* ctx,
    double t,
    double dt,
    const double* in,
    double* out) {
    (void)ctx;
    (void)dt;
    (void)in;

    constexpr double kPi = 3.14159265358979323846;
    constexpr double kFreqHz = 1000.0;

    const double phase_cycles = t * kFreqHz;
    const double phase_fraction = phase_cycles - std::floor(phase_cycles);

    out[0] = std::sin(2.0 * kPi * phase_cycles);
    out[1] = (phase_fraction < 0.5) ? 1.0 : -1.0;
    return 0;
}

}  // extern "C"
