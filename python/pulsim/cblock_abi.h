/* Pulsim C-block ABI — write a custom block in C or C++.
 *
 * Compile your block into a shared library and pass its path to
 * `pulsim.add_c_block(builder, ..., lib="your_block.so")`. C++ users:
 * keep the `extern "C"` linkage so the symbols are not name-mangled
 * (this header does it for you).
 *
 *   cc  -shared -fPIC -O2 my_block.c   -o my_block.so      # C
 *   c++ -shared -fPIC -O2 my_block.cpp -o my_block.so      # C++
 *
 * Contract per call to `pulsim_cblock_step`:
 *   in    : read-only input buffer, length n_in  (the block's input wires)
 *   out   : write the M outputs here, length n_out (the block's output wires)
 *   t, dt : current time and the block sample time (seconds)
 *   state : opaque per-block pointer you own — allocate in
 *           `pulsim_cblock_init` (or lazily via `*state`) and free it in
 *           `pulsim_cblock_term`.
 */
#ifndef PULSIM_CBLOCK_H
#define PULSIM_CBLOCK_H

#ifdef __cplusplus
extern "C" {
#endif

/* Required: one step of the block at its sample time. */
void pulsim_cblock_step(const double *in, int n_in,
                        double *out, int n_out,
                        double t, double dt, void **state);

/* Optional: allocate per-block state once (returned pointer is passed
 * back as *state to every step / term). Omit to manage state lazily
 * through *state inside the step function. */
void *pulsim_cblock_init(int n_in, int n_out);

/* Optional: free whatever pulsim_cblock_init allocated. */
void pulsim_cblock_term(void *state);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* PULSIM_CBLOCK_H */
