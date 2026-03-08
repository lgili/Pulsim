/**
 * @file cblock_abi.h
 * @brief C ABI for user-defined custom computation blocks (CBlock).
 *
 * A CBlock shared library must export three symbols:
 *
 *   pulsim_cblock_init     (optional)
 *   pulsim_cblock_step     (required)
 *   pulsim_cblock_destroy  (optional)
 *
 * The loader validates ``PULSIM_CBLOCK_ABI_VERSION`` against the value stored
 * in the exported ``pulsim_cblock_abi_version`` integer symbol.
 *
 * Usage – minimal example (gain block, 1-in 1-out):
 * @code
 *   #include "pulsim/v1/cblock_abi.h"
 *
 *   PULSIM_CBLOCK_EXPORT int pulsim_cblock_abi_version = PULSIM_CBLOCK_ABI_VERSION;
 *
 *   PULSIM_CBLOCK_EXPORT int pulsim_cblock_step(
 *       PulsimCBlockCtx* ctx, double t, double dt,
 *       const double* in, double* out)
 *   {
 *       (void)ctx; (void)t; (void)dt;
 *       out[0] = 2.0 * in[0];   // gain = 2
 *       return 0;
 *   }
 * @endcode
 */

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------------- */
/* ABI versioning                                                             */
/* ------------------------------------------------------------------------- */

/** Current CBlock ABI version.  Increment when the interface changes in a
 *  binary-incompatible way.                                                  */
#define PULSIM_CBLOCK_ABI_VERSION 1

/* ------------------------------------------------------------------------- */
/* Platform export macro                                                      */
/* ------------------------------------------------------------------------- */

#if defined(_WIN32) || defined(__CYGWIN__)
#  ifdef PULSIM_BUILDING_CBLOCK
#    define PULSIM_CBLOCK_EXPORT __declspec(dllexport)
#  else
#    define PULSIM_CBLOCK_EXPORT __declspec(dllimport)
#  endif
#else
#  if __GNUC__ >= 4
#    define PULSIM_CBLOCK_EXPORT __attribute__((visibility("default")))
#  else
#    define PULSIM_CBLOCK_EXPORT
#  endif
#endif

/* ------------------------------------------------------------------------- */
/* Opaque context                                                             */
/* ------------------------------------------------------------------------- */

/**
 * Opaque per-block state container.
 *
 * Allocate inside ``pulsim_cblock_init`` (heap only) and free inside
 * ``pulsim_cblock_destroy``.  Pass context by pointer throughout the
 * simulation; the simulator never inspects its contents.
 */
typedef struct PulsimCBlockCtx PulsimCBlockCtx;

/* ------------------------------------------------------------------------- */
/* Block metadata passed to init                                              */
/* ------------------------------------------------------------------------- */

/**
 * Read-only block information passed to ``pulsim_cblock_init``.
 *
 * The pointers are valid for the duration of the ``init`` call only.
 * Do not cache them beyond ``init``.
 */
typedef struct {
    int abi_version;    /**< Equals PULSIM_CBLOCK_ABI_VERSION at call time.  */
    int n_inputs;       /**< Number of scalar input channels.                */
    int n_outputs;      /**< Number of scalar output channels.               */
    const char* name;   /**< Block name from the netlist; may be NULL.       */
} PulsimCBlockInfo;

/* ------------------------------------------------------------------------- */
/* Function pointer types                                                     */
/* ------------------------------------------------------------------------- */

/**
 * Optional initialisation callback.
 *
 * Called once before the first call to ``pulsim_cblock_step``.
 * Allocate any persistent state inside ``*ctx_out``.
 *
 * @param ctx_out  Output: set ``*ctx_out`` to the allocated context, or NULL
 *                 if no per-block state is required.
 * @param info     Read-only metadata about the block.
 * @return         0 on success, nonzero to abort simulation.
 */
typedef int (*pulsim_cblock_init_fn)(
    PulsimCBlockCtx**       ctx_out,
    const PulsimCBlockInfo* info
);

/**
 * Required per-step callback.
 *
 * Called once per accepted simulation time step (after electrical network
 * convergence).  Implementations must be re-entrant with respect to
 * different ``ctx`` instances.  Side-effects via ``out`` only.
 *
 * @param ctx  Opaque state (may be NULL if ``init`` was not exported).
 * @param t    Current simulation time [s].
 * @param dt   Elapsed time since previous step [s]; 0 at the first step.
 * @param in   Input signals; ``info.n_inputs`` elements (read-only).
 * @param out  Output signals; ``info.n_outputs`` elements (write-only).
 * @return     0 on success, nonzero to abort simulation.
 */
typedef int (*pulsim_cblock_step_fn)(
    PulsimCBlockCtx* ctx,
    double           t,
    double           dt,
    const double*    in,
    double*          out
);

/**
 * Optional teardown callback.
 *
 * Called once after the simulation ends (successfully or with an error).
 * Must free any memory allocated by ``pulsim_cblock_init``.
 *
 * @param ctx  Context allocated by ``init``; never NULL when called.
 */
typedef void (*pulsim_cblock_destroy_fn)(PulsimCBlockCtx* ctx);

/* ------------------------------------------------------------------------- */
/* Well-known symbol names (use these strings with dlsym / GetProcAddress)   */
/* ------------------------------------------------------------------------- */

/** Version integer symbol name: ``int pulsim_cblock_abi_version;``          */
#define PULSIM_CBLOCK_SYM_VERSION  "pulsim_cblock_abi_version"
/** Optional init function symbol name.                                       */
#define PULSIM_CBLOCK_SYM_INIT     "pulsim_cblock_init"
/** Required step function symbol name.                                       */
#define PULSIM_CBLOCK_SYM_STEP     "pulsim_cblock_step"
/** Optional destroy function symbol name.                                    */
#define PULSIM_CBLOCK_SYM_DESTROY  "pulsim_cblock_destroy"

#ifdef __cplusplus
} /* extern "C" */
#endif
