## Context

Review: `raw/models.md` ranked item 11 and the unified-interface
sketch; synthesis M9. The device contract this ABI mirrors is defined
in `add-unified-device-interface` (design D2, D3): rows contribute
`f(x, t) + d/dt q(x) = 0`; rows `0 … n_pins−1` are the pins' KCL rows
(`f` = current leaving the node into the device, `q` = charge whose
derivative is such a current); rows `n_pins …` are the device's own
equations for its internal unknowns; `x = [v_pin…, internal…]`; the
integrator owns the time formula. This change does not redefine any of
that.

Owner notes that bind this design:

- *newton-devices-need-smooth-laws-to-1e-12*: applies to user devices
  as much as built-in ones; the tooling has to say so and check what it
  can.
- *nonlinear-refresh-wrapper-chain*: the reason an external device is
  not bolted onto today's chain.
- Roadmap rule: refusals instead of silent fallbacks.

## Goals / Non-Goals

- Goals: implicit user devices on every engine that solves Newton
  devices, in DC and AC, without rebuilding Pulsim; three authoring
  levels (C, equations, Verilog-A); the same result as an equivalent
  built-in.
- Non-Goals: external PWL mode devices and external static stamps;
  sandboxing native code; bundling a C or Verilog-A compiler; noise;
  SPICE-style voltage limiting callbacks; parameter sensitivities
  (reserved slot only); GPU or vectorised evaluation.

## Decisions

### Stage 1 — the C ABI

```c
#define PULSIM_XDEV_ABI_VERSION 1

typedef struct { const char *name, *unit; double def, lo, hi; } pulsim_xdev_field;
typedef struct { const char *name, *unit; } pulsim_xdev_output;

typedef struct {
    double        t;      /* time at which f is evaluated            */
    const double *p;      /* n_fields parameter values               */
    const double *state;  /* n_state committed values, or NULL       */
} pulsim_xdev_ctx;

typedef struct {
    uint32_t abi_version, struct_size;
    const char *name;
    uint32_t n_pins, n_internal, n_fields, n_state, n_events, n_outputs;
    const char *const *pin_names;
    const pulsim_xdev_field  *fields;
    const pulsim_xdev_output *outputs;
    uint32_t nnz_f, nnz_q;                    /* fixed patterns        */
    const uint32_t *f_rows, *f_cols, *q_rows, *q_cols;

    void (*eval)(const double *x, const pulsim_xdev_ctx *c,
                 double *f, double *q, double *df, double *dq);   /* required */

    int  (*validate)(const double *p, char *msg, size_t msg_len); /* optional */
    void (*init)(const double *x, const pulsim_xdev_ctx *c, double *state);
    void (*accept)(const double *x, const pulsim_xdev_ctx *c, double h, double *state);
    void (*events)(const double *x, const pulsim_xdev_ctx *c, double *g);
    void (*outputs_fn)(const double *x, const pulsim_xdev_ctx *c, double *out);
    void *reserved[4];    /* parameter derivatives: add-parameter-sensitivities */
} pulsim_xdev_descriptor;

const pulsim_xdev_descriptor *pulsim_xdev_get(void);   /* exported symbol */
```

- `R = n_pins + n_internal`. `x`, `f`, `q` have length `R`. `df[k]` is
  `∂f[f_rows[k]]/∂x[f_cols[k]]`; likewise `dq`. Patterns are fixed at
  load, so the kernel reserves matrix slots once.
- `eval` is a pure function of `(x, ctx)`: no hidden state, reentrant.
  The solver calls it for Newton iterates, rejected steps, TR-BDF2
  probe solves and from several threads in a sweep. State changes only
  in `accept`, after an accepted step, into the kernel-owned array.
- `events` returns `n_events` zero-crossing functions; a sign change
  makes the engine locate the crossing as for a built-in device.
- `fields` is the device's field table: it drives the Python keyword
  arguments, YAML keys, bounds validation and documentation.
- A non-finite value returned in `f`, `q`, `df` or `dq` is a Newton
  failure attributed to the device by name (the step is retried
  smaller where the engine can, otherwise the run aborts).

Kernel side: `ExternalDevice` is one entry of the built-in registry
whose instance holds the descriptor pointer, the pin and internal
indices, and offsets into the flat parameter and device-state vectors.
It is stamped by a span-based overload of `stamp_fq` that shares the
time-formula coefficients (`a`, `r`) with the static overload, so the
trapezoidal, BDF2-stage, backward-Euler and DC forms are the same
code. Jacobians come from the library; there is no AD for external
code. One indirect call per device per iteration.

### Registration checks

Refused, naming the device and the reason:

- `abi_version` mismatch, `struct_size` too small, missing `eval`;
- a pattern index `≥ R`, a duplicate pattern entry;
- a parameter outside its declared `[lo, hi]`, or `validate` failing;
- non-finite `f`, `q`, `df`, `dq` at the test points;
- a Jacobian that disagrees with central differences of `f` and `q` at
  16 test points (random within the declared bounds, or supplied by the
  author) by more than `1e-5` of the row scale. The check can be turned
  off with `check_jacobian=False` for a device whose domain the random
  points leave.

The finite-difference check catches a wrong derivative; it cannot
certify the 1e-12 smoothness the Newton tolerance needs. The
documentation states that requirement and provides a helper,
`pulsim.xdev.scan_monotone(dev, pin, lo, hi)`, that evaluates the law
at fine spacing.

### Compilation and trust

- `code=` (inline C or C++) and generated sources compile through the
  C block's cache (`python/pulsim/c_block.py:242-283`), factored into
  `_native_cache.py`.
- The cache today is `tempfile.gettempdir()/pulsim_cblock_cache`
  (`c_block.py:268`) and an existing `cblk_<hash>.so` is loaded as is
  (`:271-272`). On a system with a shared temporary directory another
  local user can pre-place that file. The cache moves to the per-user
  cache directory, created with mode 0700, and a cached file not owned
  by the current user is rebuilt. This applies to C blocks as well.
- A shared library is arbitrary code with the user's privileges. It is
  loaded only when the user names it (`lib=`) or supplies the source.
  A YAML netlist with an `external_device` entry is refused by the
  loader unless the call passes `allow_native_code=True`.

### Stage 2 — equations to C

```python
import sympy as sp
from pulsim import xdev

a, k = xdev.pins("a", "k")
I_S, n, V_T = xdev.params(I_S=(1e-12, "A"), n=(1.0, ""), V_T=(0.025852, "V"))
i = I_S * (sp.exp((a - k) / (n * V_T)) - 1)
dev = xdev.compile("my_diode", pins=[a, k], f=[i, -i], q=[0, 0])
dev.add_to(builder, "D1", "anode", "cathode", I_S=2e-12)
```

- Jacobians by `sympy.diff`; the pattern is the set of structurally
  non-zero derivatives; common sub-expressions are shared; C99 output
  with `expm1` / `log1p` rewriting for accuracy near zero.
- Constructs that are not C¹ (`Piecewise`, `Abs`, `sign`, `Max`,
  `Min`, `Heaviside`, `floor`) are refused with the offending
  sub-expression, unless `allow_nonsmooth=True`.
- Internal unknowns and their equations are declared the same way
  (`xdev.internal("q_M")`, extra rows in `f` and `q`).
- A pure-Python evaluator of the same expressions (`lambdify`) is the
  reference in tests: compiled against symbolic to 1e-13.

### Stage 3 — OSDI host

OSDI is the interface through which OpenVAF-compiled Verilog-A models
are loaded by ngspice. Its descriptor separates resistive and reactive
residuals and Jacobians (general knowledge of OSDI 0.3, to be confirmed
against the header in task 4.1), which is the `f` / `q` split:
`f ← resistive residual`, `q ← reactive residual`, node list ← pins
plus internal nodes, parameters ← the field table, operating-point
variables ← outputs.

- First version: models without voltage limiting and without node
  collapsing options beyond the default; noise ignored; unsupported
  descriptor features are refused by name, never ignored.
- No OSDI or OpenVAF code is bundled until its licence is checked
  against the project rule (no GPL in the core); the host needs only
  the interface header.

### Engines

| Engine | Behaviour |
|---|---|
| `pwl`, `trbdf2` | Newton, through the device interface |
| DC operating point | `f` alone |
| AC | `∂f/∂x + jω·∂q/∂x` at the operating point |
| `dsed` | refused by the conformance error |

### Alternatives considered

- *A Python callable as the device.* One interpreter call per device
  per Newton iteration, no threads; acceptable for prototyping only.
  Stage 2 gives the same convenience with compiled speed.
- *A C++ plugin interface (templates, AD).* No stable ABI across
  compilers and standard libraries.
- *Extend the C block ABI.* Its contract is "sampled, explicit, ZOH";
  an implicit device has a different lifetime and call pattern.
- *Verilog-A first.* The largest stage, with an external tool chain and
  a licence question; the C ABI is its prerequisite anyway.
- *Finite-difference Jacobians in the kernel.* Violates the exact
  Jacobian requirement and stalls Newton at small steps.

## Parameterisation & provenance

A user device's parameters are the user's. The field table carries
units and bounds so that a device is self-describing in
`components()`, YAML and error messages. `components()` reports the
library path and the hash of the source or binary, so a result can be
traced to the code that produced it.

## Validation plan

| Check | Reference | Tolerance |
|---|---|---|
| External Shockley diode (same expression, including the linear extension above `i_lim` and `G_min`) in a 20 V half-wave rectifier, `pwl` and DC operating point | built-in `add_shockley_diode` | 1e-12 relative on every node voltage |
| Same on `trbdf2` | built-in | 1e-9 (the step sequence may differ in the last bits) |
| External junction capacitor (`q` only) | built-in `NonlinearCapacitor` | 1e-12 on `pwl` |
| External Lauritzen diode (one internal unknown) | built-in, after its port to the device interface | 1e-10 |
| External device with `accept` and `events` (a latch that opens above a current) | analytic switching instant | event located within the engine's event tolerance |
| AC of the external Shockley at 1 A | `1/g_d`, `g_d = (I + I_S)/(n·V_T)` | 1e-9 |
| Stage 2: generated C against the SymPy expression | 1000 random points | 1e-13 |
| Stage 2: SymPy-declared Shockley | built-in | 1e-12 |
| Stage 3: a Verilog-A diode through OSDI | the same model in ngspice | 1e-6 |
| Wrong Jacobian (one entry scaled by 1.01) | — | refused at registration |
| Thread safety: 8-thread sweep with an external device | serial run | bit-identical |

## Risks / Trade-offs

- Native code can crash the process or corrupt memory → documented
  trust boundary; opt-in for netlists; no sandbox claimed.
- A user law that is not smooth to 1e-12 stalls Newton exactly as the
  Langevin bug did → smoothness lint in stage 2, the scan helper, and
  an error message that names the external device when Newton stalls
  with a residual at tolerance.
- ABI evolution → `abi_version` plus `struct_size`; fields are only
  appended; a version-1 library keeps loading.
- A C compiler is needed at run time for `code=` and stage 2 → `lib=`
  always works without one; the error says which is missing.
- OSDI coverage is partial → refusal by feature name.
- An indirect call per device per iteration → the review measured
  device evaluation at under 0.5 % of a Newton step.

## Migration Plan

1. After `add-unified-device-interface`: header, `ExternalDevice`,
   loader, checks, cache move, `add_external_device`, YAML; Shockley
   and junction-capacitor acceptance.
2. State, events, outputs; Lauritzen acceptance; conformance rows;
   documentation.
3. `pulsim.xdev` code generation.
4. OSDI host.

Rollback: an external device is opt-in; removing the call restores the
previous circuit.

## Open Questions

- Should the `allow_native_code` gate also apply to C blocks declared
  in YAML (today `wire_c_blocks_from_yaml` compiles inline code when
  called)?
- Is SymPy acceptable as an optional dependency, or should stage 2
  live in a separate package?
- OSDI: is a host that depends on the user's own OpenVAF installation
  acceptable, and may the OSDI interface header be vendored (licence to
  be checked)?
- Should a Python-callable prototype device (slow, single-threaded,
  with finite-difference Jacobians refused) be offered at all?
