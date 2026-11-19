/* A self-heating resistor as an external device (add-external-device-abi).
 *
 * ILLUSTRATION: the parameter defaults below are round numbers chosen to make
 * the effect visible (a 1 W, 1 s thermal time constant part); they are not a
 * datasheet part. Written in plain C99 against the stable header only
 * (pulsim/xdev/pulsim_xdev.h, `pulsim.get_include()`), so any C compiler
 * builds it; examples/scripts/run_external_self_heating_resistor.py does.
 *
 * Law. Pins a, b; one internal unknown theta = T - T_amb, the temperature
 * rise (K), so the zero start is "at ambient":
 *   R(theta) = R_0 (1 + alpha (theta + T_amb - T_0))
 *   pin rows:      f_a = v / R,  f_b = -v / R            (v = v_a - v_b)
 *   internal row:  f   = theta / R_th - v^2 / R,  q = C_th theta
 * i.e. C_th d(theta)/dt = P - theta / R_th, each row f + dq/dt = 0.
 * The Jacobian is written by hand; the host checks it against central
 * differences when the device is added. */
#include "pulsim/xdev/pulsim_xdev.h"

#define CALL PULSIM_XDEV_CALL

static const char *const PINS[2] = {"a", "b"};
static const pulsim_xdev_field FIELDS[6] = {
    {"R_0", "Ohm", 100.0, 1e-6, 1e9},     /* resistance at T_0           */
    {"alpha", "1/K", 3.9e-3, 0.0, 0.1},   /* temperature coefficient     */
    {"T_0", "degC", 25.0, -273.0, 1e3},   /* reference temperature       */
    {"T_amb", "degC", 25.0, -273.0, 1e3}, /* ambient temperature         */
    {"R_th", "K/W", 50.0, 1e-6, 1e6},     /* thermal resistance to T_amb */
    {"C_th", "J/K", 0.02, 1e-12, 1e6},    /* thermal capacitance         */
};
/* df pattern (row-major): (0,0) (0,1) (0,2) (1,0) (1,1) (1,2) (2,0) (2,1) (2,2) */
static const uint32_t FR[9] = {0, 0, 0, 1, 1, 1, 2, 2, 2};
static const uint32_t FC[9] = {0, 1, 2, 0, 1, 2, 0, 1, 2};
static const uint32_t QR[1] = {2}, QC[1] = {2};
/* the box the registration checks sample: +-50 V, 0..150 K of rise */
static const double LO[3] = {-50, -50, 0}, HI[3] = {50, 50, 150};

static int32_t CALL eval(const pulsim_xdev_ctx *c, const double *x, double *f, double *q,
                         double *df, double *dq) {
    const double *p = c->p;
    const double v = x[0] - x[1], th = x[2];
    const double R = p[0] * (1.0 + p[1] * (th + p[3] - p[2]));
    const double dR = p[0] * p[1]; /* dR/dtheta */
    const double i = v / R;
    const double di_dth = -v * dR / (R * R);
    f[0] = i;
    f[1] = -i;
    f[2] = th / p[4] - v * i;
    q[0] = 0.0;
    q[1] = 0.0;
    q[2] = p[5] * th;
    df[0] = 1.0 / R;
    df[1] = -1.0 / R;
    df[2] = di_dth;
    df[3] = -1.0 / R;
    df[4] = 1.0 / R;
    df[5] = -di_dth;
    df[6] = -2.0 * i;
    df[7] = 2.0 * i;
    df[8] = 1.0 / p[4] - v * di_dth;
    dq[0] = p[5];
    return PULSIM_XDEV_OK;
}

static int32_t CALL validate(const double *p, char *msg, uint32_t len) {
    /* R must stay positive over the probe box (theta up to 150 K) */
    const double lo = 1.0 + p[1] * (p[3] - p[2]);
    if (!(lo > 0.0)) {
        const char *why = "R(T_amb) <= 0: check alpha, T_0 and T_amb";
        uint32_t k = 0;
        for (; why[k] && k + 1 < len; ++k) msg[k] = why[k];
        if (len) msg[k] = '\0';
        return 1;
    }
    return PULSIM_XDEV_OK;
}

static const pulsim_xdev_descriptor MODELS[1] = {{
    PULSIM_XDEV_ABI_VERSION, (uint32_t)sizeof(pulsim_xdev_descriptor), "self_heating_resistor",
    2, 1, 6, 0,         /* pins, internal unknowns, fields, model state */
    9, 1, 0,            /* nnz_f, nnz_q, monotone pairs */
    PULSIM_XDEV_CLAIM_SMOOTH | PULSIM_XDEV_CLAIM_PASSIVE,
    PINS, FIELDS, FR, FC, QR, QC, LO, HI, NULL,
    eval, validate, NULL, NULL, NULL, NULL, {NULL, NULL, NULL, NULL},
}};

PULSIM_XDEV_EXTERN_C PULSIM_XDEV_EXPORT const pulsim_xdev_descriptor *CALL
pulsim_xdev_get(uint32_t index) {
    return index == 0 ? &MODELS[0] : NULL;
}
