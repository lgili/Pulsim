/*
 * Pulsim external device ABI, version 1 (add-external-device-abi).
 *
 * THIS FILE IS THE STABLE CONTRACT. It compiles as C99 and as C++. A model
 * library built against version 1 keeps loading in every later Pulsim that
 * speaks version 1: fields are only ever APPENDED to the end of
 * pulsim_xdev_descriptor and pulsim_xdev_ctx (after `reserved` / `instance`),
 * and the host reads a field only when `struct_size` says the library's
 * struct has it. The ARRAY ELEMENT structs pulsim_xdev_field and
 * pulsim_xdev_monotone carry no size and are FROZEN: a new member there is
 * a new array pointer appended to the descriptor. Nothing else is stable:
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

/* host_features bits: a set bit says the host fills that ctx field. A model
 * reads the field only when the bit is set (a host built before the bit
 * existed leaves it clear and passes 0). */
#define PULSIM_XDEV_HOST_TIME (1u << 0) /* ctx->t is the time of the solve, s */

typedef struct pulsim_xdev_ctx {
    uint32_t struct_size; /* sizeof(pulsim_xdev_ctx) of the HOST */
    uint32_t host_features; /* PULSIM_XDEV_HOST_* bits */
    double t;             /* the time the solve is at (the stage time; t_eval at DC) when
                           * host_features & PULSIM_XDEV_HOST_TIME, else 0 */
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

/* A named output (appended block, see below). FROZEN like pulsim_xdev_field. */
typedef struct pulsim_xdev_output {
    const char *name; /* the key of res.device_output(device, name) */
    const char *unit; /* SI unit text, "" for none */
} pulsim_xdev_output;
/* Optional: write out[n_outputs] at x with the committed state (ctx->state
 * after the accepted step). Pure, like eval. Called after accepted steps
 * only when the caller asks for the outputs, never inside a Newton solve. */
typedef int32_t(PULSIM_XDEV_CALL *pulsim_xdev_outputs_fn)(const pulsim_xdev_ctx *c,
                                                         const double *x, double *out);

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
    /* ---- appended block 1: named outputs (still ABI version 1). A host
     * reads it only when struct_size >= PULSIM_XDEV_DESCRIPTOR_OUTPUTS_SIZE;
     * an older library's struct ends above and has no outputs. */
    uint32_t n_outputs;
    uint32_t n_events;   /* reserved for event functions: must be 0 (refused otherwise) */
    const pulsim_xdev_output *outputs; /* n_outputs */
    pulsim_xdev_outputs_fn outputs_fn; /* required when n_outputs > 0 */
} pulsim_xdev_descriptor;

/* The version-1 sizes, FROZEN: a host refuses a library struct smaller than
 * these and reads an appended field only when struct_size covers it. */
#define PULSIM_XDEV_DESCRIPTOR_V1_SIZE \
    (offsetof(pulsim_xdev_descriptor, reserved) + 4 * sizeof(pulsim_xdev_reserved_fn))
/* The size with appended block 1 (named outputs); also the feature test a
 * library compiled against either header can use. FROZEN. */
#define PULSIM_XDEV_DESCRIPTOR_OUTPUTS_SIZE \
    (offsetof(pulsim_xdev_descriptor, outputs_fn) + sizeof(pulsim_xdev_outputs_fn))
#define PULSIM_XDEV_CTX_V1_SIZE (offsetof(pulsim_xdev_ctx, instance) + sizeof(const void *))

/* The layout, asserted wherever this header is compiled (the host AND every
 * library): no `long`, `size_t` or enum field, so LP64 (Linux, macOS) and
 * LLP64 (Windows) share the 64-bit offsets; 32-bit targets share theirs.
 * Every field is naturally aligned (no padding, so packing pragmas change
 * nothing); a compiler laying a struct out differently fails to compile
 * here instead of loading and reading garbage. */
#if defined(__cplusplus)
#define PULSIM_XDEV_LAYOUT(cond, tag) static_assert(cond, "pulsim_xdev.h layout: " #tag)
#elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
#define PULSIM_XDEV_LAYOUT(cond, tag) _Static_assert(cond, "pulsim_xdev.h layout: " #tag)
#else /* C99 / MSVC's default C: a negative array size fails */
#define PULSIM_XDEV_LAYOUT(cond, tag) typedef char pulsim_xdev_layout_##tag[(cond) ? 1 : -1]
#endif
#define PULSIM_XDEV_P8_P4(a, b) (sizeof(void *) == 8 ? (size_t)(a) : (size_t)(b))
PULSIM_XDEV_LAYOUT(sizeof(void *) == 8 || sizeof(void *) == 4, pointer_width);
PULSIM_XDEV_LAYOUT(sizeof(pulsim_xdev_field) == PULSIM_XDEV_P8_P4(40, 32), field_size);
PULSIM_XDEV_LAYOUT(offsetof(pulsim_xdev_field, def) == PULSIM_XDEV_P8_P4(16, 8), field_def);
PULSIM_XDEV_LAYOUT(sizeof(pulsim_xdev_monotone) == 8, monotone_size);
PULSIM_XDEV_LAYOUT(offsetof(pulsim_xdev_ctx, t) == 8, ctx_t);
PULSIM_XDEV_LAYOUT(offsetof(pulsim_xdev_ctx, p) == 16, ctx_p);
PULSIM_XDEV_LAYOUT(offsetof(pulsim_xdev_ctx, state) == PULSIM_XDEV_P8_P4(24, 20), ctx_state);
PULSIM_XDEV_LAYOUT(PULSIM_XDEV_CTX_V1_SIZE == PULSIM_XDEV_P8_P4(40, 28), ctx_v1_size);
PULSIM_XDEV_LAYOUT(offsetof(pulsim_xdev_descriptor, name) == 8, desc_name);
PULSIM_XDEV_LAYOUT(offsetof(pulsim_xdev_descriptor, n_pins) == PULSIM_XDEV_P8_P4(16, 12), desc_n_pins);
PULSIM_XDEV_LAYOUT(offsetof(pulsim_xdev_descriptor, claims) == PULSIM_XDEV_P8_P4(44, 40), desc_claims);
PULSIM_XDEV_LAYOUT(offsetof(pulsim_xdev_descriptor, pin_names) == PULSIM_XDEV_P8_P4(48, 44), desc_pin_names);
PULSIM_XDEV_LAYOUT(offsetof(pulsim_xdev_descriptor, monotone) == PULSIM_XDEV_P8_P4(112, 76), desc_monotone);
PULSIM_XDEV_LAYOUT(offsetof(pulsim_xdev_descriptor, eval) == PULSIM_XDEV_P8_P4(120, 80), desc_eval);
PULSIM_XDEV_LAYOUT(offsetof(pulsim_xdev_descriptor, accept) == PULSIM_XDEV_P8_P4(160, 100), desc_accept);
PULSIM_XDEV_LAYOUT(offsetof(pulsim_xdev_descriptor, reserved) == PULSIM_XDEV_P8_P4(168, 104), desc_reserved);
PULSIM_XDEV_LAYOUT(PULSIM_XDEV_DESCRIPTOR_V1_SIZE == PULSIM_XDEV_P8_P4(200, 120), desc_v1_size);
PULSIM_XDEV_LAYOUT(sizeof(pulsim_xdev_output) == PULSIM_XDEV_P8_P4(16, 8), output_size);
PULSIM_XDEV_LAYOUT(offsetof(pulsim_xdev_descriptor, n_outputs) == PULSIM_XDEV_P8_P4(200, 120), desc_n_outputs);
PULSIM_XDEV_LAYOUT(offsetof(pulsim_xdev_descriptor, outputs) == PULSIM_XDEV_P8_P4(208, 128), desc_outputs);
PULSIM_XDEV_LAYOUT(PULSIM_XDEV_DESCRIPTOR_OUTPUTS_SIZE == PULSIM_XDEV_P8_P4(224, 136), desc_outputs_size);
PULSIM_XDEV_LAYOUT(sizeof(pulsim_xdev_descriptor) == PULSIM_XDEV_DESCRIPTOR_OUTPUTS_SIZE, desc_size);

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
