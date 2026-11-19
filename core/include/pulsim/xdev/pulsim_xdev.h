/*
 * Pulsim external device ABI, version 1 (add-external-device-abi).
 *
 * THIS FILE IS THE STABLE CONTRACT. It compiles as C99 and as C++. A model
 * library built against version 1 keeps loading in every later Pulsim that
 * speaks version 1: fields are only ever APPENDED to the structs below
 * (before `reserved`'s successors), and the host reads a field only when
 * `struct_size` says the library's struct has it. Nothing else is stable:
 * the C++ helper (pulsim/xdev/adapter.hpp) and the kernel's host code may
 * change in any release; a library that uses the helper is still bound only
 * by this header, because the helper compiles INTO the library.
 *
 * Rules of the boundary: plain C types and fixed-width integers only; no
 * exceptions cross it (every function returns a status, 0 = OK); no memory
 * changes owner across it (the host owns every array it passes, the library
 * owns what `create` returns and frees it in `destroy`); explicit sizes
 * everywhere. Every function pointer uses PULSIM_XDEV_CALL (__cdecl on
 * Windows), so 32-bit Windows builds agree on the calling convention.
 *
 * TRUST: a shared library is arbitrary native code running with the user's
 * privileges. Pulsim checks the model's MATHEMATICAL contract at load time
 * (Jacobian, finiteness, declared claims); it does not and cannot sandbox it.
 *
 * The device law (the device interface's row convention): rows 0..n_pins-1
 * are the pins' KCL rows (f = current leaving the node into the device,
 * q = charge whose time derivative is such a current); rows n_pins..R-1,
 * R = n_pins + n_internal, are the device's own equations for its internal
 * unknowns. x = [v_pin..., internal...]. Each row contributes
 * f(x) + d/dt q(x) = 0; the integrator owns d/dt.
 */
#ifndef PULSIM_XDEV_H
#define PULSIM_XDEV_H

#include <stddef.h>
#include <stdint.h>

#define PULSIM_XDEV_ABI_VERSION 1u

#if defined(_WIN32)
#define PULSIM_XDEV_CALL __cdecl
#define PULSIM_XDEV_EXPORT __declspec(dllexport)
#else
#define PULSIM_XDEV_CALL
#define PULSIM_XDEV_EXPORT __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
#define PULSIM_XDEV_EXTERN_C extern "C"
#else
#define PULSIM_XDEV_EXTERN_C
#endif

/* Claims a model makes about its law; the host CHECKS each one it is given
 * at registration and refuses a model whose claim is false. */
#define PULSIM_XDEV_CLAIM_SMOOTH     (1u << 0) /* f, q are C1 (no Jacobian jump) */
#define PULSIM_XDEV_CLAIM_PASSIVE    (1u << 1) /* sum over pins f_p * v_p >= 0 */
#define PULSIM_XDEV_CLAIM_RECIPROCAL (1u << 2) /* dq/dx symmetric per block    */

#define PULSIM_XDEV_OK 0

typedef struct pulsim_xdev_field {
    const char *name; /* parameter name, as the user spells it */
    const char *unit; /* SI unit text ("H", "A", "" for none) */
    double def, lo, hi; /* default and closed bounds          */
} pulsim_xdev_field;

/* "f[row] is non-decreasing in x[col]": checked along a scan. */
typedef struct pulsim_xdev_monotone {
    uint32_t row, col;
} pulsim_xdev_monotone;

typedef struct pulsim_xdev_ctx {
    uint32_t struct_size; /* sizeof(pulsim_xdev_ctx) of the HOST */
    uint32_t host_features; /* version 1: 0 (bit set = an optional ctx field is filled) */
    double t;             /* not filled in version 1 (host_features bit 0 clear): 0 */
    const double *p;      /* n_fields parameter values, field order */
    const double *state;  /* n_state committed values (NULL if n_state == 0) */
    const void *instance; /* what `create` returned, or NULL */
} pulsim_xdev_ctx;

/* REQUIRED. A pure function of (ctx, x): no hidden state, reentrant (the
 * host calls it for Newton iterates, rejected steps, probe solves, from
 * several threads). Writes f[R], q[R], df[nnz_f], dq[nnz_q]; df[k] is
 * d f[f_rows[k]] / d x[f_cols[k]], likewise dq. */
typedef int32_t(PULSIM_XDEV_CALL *pulsim_xdev_eval_fn)(const pulsim_xdev_ctx *c,
                                                      const double *x, double *f,
                                                      double *q, double *df,
                                                      double *dq);
/* The buffers passed to one call never alias each other, x, p or state.
 * Optional: refuse a parameter set (write a NUL-terminated reason). */
typedef int32_t(PULSIM_XDEV_CALL *pulsim_xdev_validate_fn)(const double *p, char *msg,
                                                          uint32_t msg_len);
/* Optional: per-instance data the library allocates and later frees. */
typedef int32_t(PULSIM_XDEV_CALL *pulsim_xdev_create_fn)(const double *p,
                                                        void **instance);
typedef void(PULSIM_XDEV_CALL *pulsim_xdev_destroy_fn)(void *instance);
/* Optional (required when n_state > 0): write state[n_state] at x / after an
 * accepted step of length h. The host owns `state`. */
typedef int32_t(PULSIM_XDEV_CALL *pulsim_xdev_init_fn)(const pulsim_xdev_ctx *c,
                                                      const double *x, double *state);
typedef int32_t(PULSIM_XDEV_CALL *pulsim_xdev_accept_fn)(const pulsim_xdev_ctx *c,
                                                        const double *x, double h,
                                                        double *state);
typedef void(PULSIM_XDEV_CALL *pulsim_xdev_reserved_fn)(void);

typedef struct pulsim_xdev_descriptor {
    uint32_t abi_version; /* PULSIM_XDEV_ABI_VERSION the library was built with */
    uint32_t struct_size; /* sizeof(pulsim_xdev_descriptor) of the library */
    const char *name;     /* model name */
    uint32_t n_pins, n_internal, n_fields, n_state;
    uint32_t nnz_f, nnz_q, n_monotone, claims;
    const char *const *pin_names;      /* n_pins */
    const pulsim_xdev_field *fields;   /* n_fields */
    const uint32_t *f_rows, *f_cols;   /* nnz_f each: the FIXED pattern of df */
    const uint32_t *q_rows, *q_cols;   /* nnz_q each: the FIXED pattern of dq */
    const double *probe_lo, *probe_hi; /* R each: the box the checks sample */
    const pulsim_xdev_monotone *monotone; /* n_monotone */
    pulsim_xdev_eval_fn eval;
    pulsim_xdev_validate_fn validate;
    pulsim_xdev_create_fn create;
    pulsim_xdev_destroy_fn destroy;
    pulsim_xdev_init_fn init;
    pulsim_xdev_accept_fn accept;
    pulsim_xdev_reserved_fn reserved[4]; /* parameter derivatives (later); zero */
} pulsim_xdev_descriptor;

/* The version-1 sizes, FROZEN: a host refuses a library struct smaller than
 * these and reads an appended field only when struct_size covers it. */
#define PULSIM_XDEV_DESCRIPTOR_V1_SIZE \
    (offsetof(pulsim_xdev_descriptor, reserved) + 4 * sizeof(pulsim_xdev_reserved_fn))
#define PULSIM_XDEV_CTX_V1_SIZE (offsetof(pulsim_xdev_ctx, instance) + sizeof(const void *))

/* The ONE exported symbol: the index-th model of the library (0, 1, ...),
 * NULL past the last. The returned descriptor lives as long as the library. */
#define PULSIM_XDEV_ENTRY "pulsim_xdev_get"
typedef const pulsim_xdev_descriptor *(PULSIM_XDEV_CALL *pulsim_xdev_get_fn)(uint32_t index);

/* A hand-written C library defines PULSIM_XDEV_DECLARE_ENTRY before including
 * this header to get the entry point's exact declaration checked. */
#ifdef PULSIM_XDEV_DECLARE_ENTRY
PULSIM_XDEV_EXTERN_C PULSIM_XDEV_EXPORT const pulsim_xdev_descriptor *PULSIM_XDEV_CALL
pulsim_xdev_get(uint32_t index);
#endif

#endif /* PULSIM_XDEV_H */
