/* External-device test models written in plain C99 against pulsim_xdev.h
 * (add-external-device-abi): one honest resistor, one honest stateful
 * model, and models that LIE about their contract in one way each. The
 * registration checks must refuse every liar whose lie is visible inside
 * its probe box; the run-time checks catch the two whose lie is outside it.
 * ILLUSTRATIONS for the tests, not device models. */
#include "pulsim/xdev/pulsim_xdev.h"

#include <math.h>
#include <stddef.h>
#include <stdio.h>

#define CALL PULSIM_XDEV_CALL

static const char *const PINS[2] = {"a", "b"};
static const pulsim_xdev_field FG[1] = {{"G", "S", 1e-3, 1e-9, 1e3}};
static const uint32_t FR[4] = {0, 0, 1, 1}, FC[4] = {0, 1, 0, 1};
static const double LO[2] = {-1, -1}, HI[2] = {1, 1};
static const pulsim_xdev_monotone MONO[1] = {{0, 0}};

/* i = g·v into pin a out of pin b, with its exact Jacobian */
static void cond(double g, double i, const double *x, double *f, double *q, double *df) {
    (void)x;
    f[0] = i;
    f[1] = -i;
    q[0] = 0;
    q[1] = 0;
    df[0] = g;
    df[1] = -g;
    df[2] = -g;
    df[3] = g;
}
#define V (x[0] - x[1])
#define G (c->p[0])

static int32_t CALL good(const pulsim_xdev_ctx *c, const double *x, double *f, double *q,
                         double *df, double *dq) {
    (void)dq;
    cond(G, G * V, x, f, q, df);
    return 0;
}
static int32_t CALL pattern3(const pulsim_xdev_ctx *c, const double *x, double *f, double *q,
                             double *df, double *dq) {
    double d4[4];
    good(c, x, f, q, d4, dq);
    df[0] = d4[0];
    df[1] = d4[1];
    df[2] = d4[2]; /* (1, 1) left out of the declared pattern */
    return 0;
}
static int32_t CALL liar_jacobian(const pulsim_xdev_ctx *c, const double *x, double *f,
                                  double *q, double *df, double *dq) {
    good(c, x, f, q, df, dq);
    df[0] *= 1.01; /* one entry scaled by 1.01 */
    return 0;
}
static int32_t CALL liar_kink(const pulsim_xdev_ctx *c, const double *x, double *f, double *q,
                              double *df, double *dq) {
    const double g = V > 0 ? G : 0.5 * G; /* exact slopes, a kink at 0 */
    (void)dq;
    cond(g, g * V, x, f, q, df);
    return 0;
}
static int32_t CALL liar_nonmonotone(const pulsim_xdev_ctx *c, const double *x, double *f,
                                     double *q, double *df, double *dq) {
    /* Langevin coth(u) − 1/u without its series: cancellation at small u */
    const double u = V;
    double i = 0, g = G / 3;
    (void)dq;
    if (u != 0) {
        i = G * (cosh(u) / sinh(u) - 1 / u);
        g = G * (1 / (u * u) - 1 / (sinh(u) * sinh(u)));
    }
    cond(g, i, x, f, q, df);
    return 0;
}
static int32_t CALL liar_active(const pulsim_xdev_ctx *c, const double *x, double *f, double *q,
                                double *df, double *dq) {
    (void)dq;
    cond(-G, -G * V, x, f, q, df); /* a negative resistance claiming PASSIVE */
    return 0;
}
static const uint32_t QR[4] = {0, 0, 1, 1}, QC[4] = {0, 1, 0, 1};
static int32_t CALL liar_reciprocal(const pulsim_xdev_ctx *c, const double *x, double *f,
                                    double *q, double *df, double *dq) {
    const double C = 1e-9, K = 0.5e-9; /* q0 = C·(v0 − v1) + K·v1 */
    good(c, x, f, q, df, dq);
    q[0] = C * V + K * x[1];
    q[1] = -q[0];
    dq[0] = C;
    dq[1] = -C + K;
    dq[2] = -C;
    dq[3] = C - K;
    return 0;
}
static int32_t CALL liar_impure(const pulsim_xdev_ctx *c, const double *x, double *f, double *q,
                                double *df, double *dq) {
    static double calls = 0; /* hidden state: eval is not a function of x */
    calls += 1;
    good(c, x, f, q, df, dq);
    f[0] += calls * 1e-15;
    f[1] -= calls * 1e-15;
    return 0;
}
static int32_t CALL liar_kcl(const pulsim_xdev_ctx *c, const double *x, double *f, double *q,
                             double *df, double *dq) {
    good(c, x, f, q, df, dq);
    f[1] = 0; /* current enters a and goes nowhere */
    df[2] = 0;
    df[3] = 0;
    return 0;
}
static int32_t CALL liar_overrun(const pulsim_xdev_ctx *c, const double *x, double *f, double *q,
                                 double *df, double *dq) {
    good(c, x, f, q, df, dq);
    df[4] = 0; /* one past nnz_f */
    return 0;
}
static int32_t CALL nan_outside(const pulsim_xdev_ctx *c, const double *x, double *f, double *q,
                                double *df, double *dq) {
    good(c, x, f, q, df, dq);
    if (fabs(V) > 5) f[0] = NAN; /* outside the probe box: only a run sees it */
    return 0;
}
static int32_t CALL liar_outside(const pulsim_xdev_ctx *c, const double *x, double *f, double *q,
                                 double *df, double *dq) {
    good(c, x, f, q, df, dq);
    if (V > 3) df[0] *= 2; /* wrong Jacobian outside the probe box */
    return 0;
}
static int32_t CALL liar_writes_x(const pulsim_xdev_ctx *c, const double *x, double *f,
                                  double *q, double *df, double *dq) {
    good(c, x, f, q, df, dq);
    ((double *)x)[0] = -x[0]; /* casts const away: at x = 0 only the sign bit moves */
    return 0;
}
static int32_t CALL liar_writes_p(const pulsim_xdev_ctx *c, const double *x, double *f,
                                  double *q, double *df, double *dq) {
    good(c, x, f, q, df, dq);
    ((double *)c->p)[0] = -G; /* casts const away: negates the parameter */
    return 0;
}
static int32_t CALL liar_offaxis(const pulsim_xdev_ctx *c, const double *x, double *f,
                                 double *q, double *df, double *dq) {
    good(c, x, f, q, df, dq);
    /* wrong only near the diagonal x0 = x1 away from the centre: no axis
     * line through the centre reaches it, a random probe almost never */
    if (fabs(x[0] - x[1]) < 1e-3 && fabs(x[0]) > 0.25) df[0] *= 2;
    return 0;
}
/* a 0.1 pF capacitor (q = C·V) beside the resistor; dq doubled outside the box */
static int32_t CALL tiny_cap_outside(const pulsim_xdev_ctx *c, const double *x, double *f,
                                     double *q, double *df, double *dq) {
    const double C = 1e-13, k = V > 3 ? 2.0 : 1.0;
    good(c, x, f, q, df, dq);
    q[0] = C * V;
    q[1] = -q[0];
    dq[0] = k * C;
    dq[1] = -k * C;
    dq[2] = -k * C;
    dq[3] = k * C;
    return 0;
}
#ifndef PULSIM_XDEV_HOST_TIME
#define PULSIM_XDEV_HOST_TIME 1u /* the header as of c90fc07f had no name for bit 0 */
#endif
/* g(t) = G·(1 + t / 1 ms): a law that reads the time (ctx->t) */
static int32_t CALL ramp(const pulsim_xdev_ctx *c, const double *x, double *f, double *q,
                         double *df, double *dq) {
    const double t = (c->host_features & PULSIM_XDEV_HOST_TIME) ? c->t : 0.0;
    const double g = G * (1 + 1e3 * t);
    (void)dq;
    cond(g, g * V, x, f, q, df);
    return 0;
}
/* HONEST about a kink: exact one-sided slopes (G above 0, G/2 below; the
 * backward slope AT 0) and no SMOOTH claim */
static int32_t CALL honest_kink(const pulsim_xdev_ctx *c, const double *x, double *f, double *q,
                                double *df, double *dq) {
    const double g = V > 0 ? G : 0.5 * G;
    (void)dq;
    cond(g, g * V, x, f, q, df);
    return 0;
}
/* pure when called serially, a data race when called concurrently: every
 * call of every instance shares one static scratch (a liar about reentrancy) */
static volatile double racy_scratch; /* volatile: -O2 would forward the store and hide the race */
static int32_t CALL liar_racy(const pulsim_xdev_ctx *c, const double *x, double *f, double *q,
                              double *df, double *dq) {
    volatile double window = 0;
    int k;
    (void)dq;
    racy_scratch = V;
    for (k = 0; k < 4000; ++k) window += 1e-300; /* another call overwrites it here */
    cond(G, G * racy_scratch + 0 * window, x, f, q, df);
    return 0;
}
/* finite outputs everywhere; outside the box (V > 3) two of them are 1e308,
 * whose sum overflows: not a reason to refuse */
static int32_t CALL huge_outside(const pulsim_xdev_ctx *c, const double *x, double *f, double *q,
                                 double *df, double *dq) {
    good(c, x, f, q, df, dq);
    if (V > 3) {
        df[0] = 1e308;
        df[3] = 1e308;
    }
    return 0;
}
static int32_t CALL meter_init(const pulsim_xdev_ctx *c, const double *x, double *s) {
    (void)c;
    (void)x;
    s[0] = 0;
    return 0;
}
static int32_t CALL meter_accept(const pulsim_xdev_ctx *c, const double *x, double h, double *s) {
    s[0] += h * G * V * V; /* energy absorbed (rectangle rule) */
    return 0;
}
static int32_t CALL accept_via_ctx(const pulsim_xdev_ctx *c, const double *x, double h,
                                   double *s) {
    (void)s; /* updates the COMMITTED state through const ctx->state instead */
    ((double *)c->state)[0] += h * (1 + G * V * V);
    return 0;
}
static int32_t CALL state_overrun(const pulsim_xdev_ctx *c, const double *x, double h, double *s) {
    meter_accept(c, x, h, s);
    s[1] = 1; /* declared n_state = 1 */
    return 0;
}

#define SMOOTH PULSIM_XDEV_CLAIM_SMOOTH
#define PASSIVE PULSIM_XDEV_CLAIM_PASSIVE
#define DESC(nm, fn, claims, nmono, nf, nq, qr, qc, ns, init, acc, ver)                         \
    {(ver), (uint32_t)sizeof(pulsim_xdev_descriptor), (nm), 2, 0, 1, (ns), (nf), (nq), (nmono),   \
     (claims), PINS, FG, FR, FC, (qr), (qc), LO, HI, MONO, (fn), NULL, NULL, NULL, (init), (acc), \
     {NULL, NULL, NULL, NULL}}
#define PLAIN(nm, fn, claims, nmono) \
    DESC(nm, fn, claims, nmono, 4, 0, NULL, NULL, 0, NULL, NULL, PULSIM_XDEV_ABI_VERSION)

static const pulsim_xdev_descriptor MODELS[] = {
    PLAIN("good_resistor", good, SMOOTH | PASSIVE, 1),
    PLAIN("liar_jacobian", liar_jacobian, SMOOTH, 0),
    PLAIN("liar_pattern", pattern3, SMOOTH, 0), /* nnz_f patched to 3 below */
    PLAIN("liar_kink", liar_kink, SMOOTH, 0),
    PLAIN("liar_nonmonotone", liar_nonmonotone, SMOOTH, 1),
    PLAIN("liar_active", liar_active, SMOOTH | PASSIVE, 0),
    DESC("liar_reciprocal", liar_reciprocal, SMOOTH | PULSIM_XDEV_CLAIM_RECIPROCAL, 0, 4, 4, QR,
         QC, 0, NULL, NULL, PULSIM_XDEV_ABI_VERSION),
    PLAIN("liar_impure", liar_impure, SMOOTH, 0),
    PLAIN("liar_kcl", liar_kcl, SMOOTH, 0),
    PLAIN("liar_overrun", liar_overrun, SMOOTH, 0),
    DESC("liar_state", good, SMOOTH, 0, 4, 0, NULL, NULL, 1, meter_init, state_overrun,
         PULSIM_XDEV_ABI_VERSION),
    DESC("liar_abi", good, SMOOTH, 0, 4, 0, NULL, NULL, 0, NULL, NULL, 99u),
    PLAIN("nan_outside", nan_outside, SMOOTH, 0),
    PLAIN("liar_outside", liar_outside, SMOOTH, 0),
    DESC("energy_meter", good, SMOOTH | PASSIVE, 0, 4, 0, NULL, NULL, 1, meter_init, meter_accept,
         PULSIM_XDEV_ABI_VERSION),
    PLAIN("liar_writes_x", liar_writes_x, SMOOTH, 0),
    DESC("liar_state_via_ctx", good, SMOOTH, 0, 4, 0, NULL, NULL, 1, meter_init, accept_via_ctx,
         PULSIM_XDEV_ABI_VERSION),
    PLAIN("liar_writes_p", liar_writes_p, SMOOTH, 0),
    PLAIN("liar_offaxis", liar_offaxis, 0, 0),
    DESC("tiny_cap_outside", tiny_cap_outside, SMOOTH | PASSIVE, 0, 4, 4, QR, QC, 0, NULL, NULL,
         PULSIM_XDEV_ABI_VERSION),
    PLAIN("ramp_conductance", ramp, SMOOTH | PASSIVE, 1),
    PLAIN("honest_kink", honest_kink, PASSIVE, 0),
    PLAIN("liar_racy", liar_racy, SMOOTH | PASSIVE, 0),
    PLAIN("huge_outside", huge_outside, SMOOTH, 0),
};

PULSIM_XDEV_EXTERN_C PULSIM_XDEV_EXPORT const pulsim_xdev_descriptor *CALL
pulsim_xdev_get(uint32_t index) {
    static pulsim_xdev_descriptor pattern_liar;
    if (index >= sizeof MODELS / sizeof MODELS[0]) return NULL;
    if (index == 2) { /* declares 3 entries: (1, 1) is missing */
        pattern_liar = MODELS[2];
        pattern_liar.nnz_f = 3;
        return &pattern_liar;
    }
    return &MODELS[index];
}
