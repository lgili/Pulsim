# Design — `pulsim-v2-device-models-ad-driven` (Layer 2)

## Goal

Kill the v1 four-stamp duplication problem permanently. Every v2
device model exposes ONE templated function:

```cpp
template <numeric::FloatingPoint S>
static S current(const S* v, const Params& p) noexcept;
```

Layer 3's generic stamper instantiates the SAME function twice:
- `current<Real>(...)` for forward evaluation (the residual)
- `current<ADRealN<N>>(...)` for the Jacobian (AD-derived)

Manual derivatives become a type error. The drift between paths
that caused the May 2026 diode bug becomes structurally impossible.

## The AD scalar choice

Two AD modes exist; we choose forward mode for Layer 2:

### Forward mode (chosen)

`ADRealN<N>` carries a single value + an array of N derivatives.
Every arithmetic operation updates value AND all N derivatives via
chain rule. After evaluating `current(seed(v0, v1, ...), p)`, the
return type's `derivatives()` array IS the gradient (Jacobian row
for a scalar function).

**Pros**:
- Stack-allocated, no heap traffic. `sizeof(ADRealN<8>) = 72 B`.
- Simple — operator overloads handle propagation transparently.
- Zero state. Trivial to inline.
- Perfect for "few inputs, one output" — exactly Pulsim's device
  shape (≤ 8 terminals, one current per branch).

**Cons**:
- Cost scales with N. For a device with 100 terminals (not a real
  scenario, but...) reverse mode would beat forward mode.

### Reverse mode (rejected for Layer 2)

Carries a tape of operations + back-propagates. Used by deep
learning frameworks where N (inputs) is huge and outputs is small.

**Pros**: Constant cost in N.
**Cons**: Heap-allocated tape, dynamic memory, much more complex
to implement and debug. Total overkill for Pulsim's ≤ 8-terminal
devices.

### Decision

`ADRealN<N>` forward mode. Stack-allocated. Templated on N so
each device gets exactly the size it needs (Resistor uses N=2,
MOSFET uses N=3, future Transformer uses N=4). No heap, no
machinery.

If a future device with N > 32 shows up, we revisit. Until then,
forward mode wins on simplicity AND speed.

## The DeviceModel concept

```cpp
template <typename T>
concept DeviceModel = requires(const typename T::Params& p,
                                const Real* v) {
    typename T::Params;
    { T::kind } -> std::convertible_to<topology::BranchKind>;
    { T::num_terminals } -> std::convertible_to<Size>;
    { T::is_linear } -> std::convertible_to<bool>;
    { T::template current<Real>(v, p) } -> std::convertible_to<Real>;
};
```

Any type satisfying this concept is automatically consumable by
Layer 3's stamping pipeline (when that ships). No registration,
no virtual dispatch, no factory pattern. The compiler verifies
the contract at template instantiation.

### Why static methods, not virtual

Virtual dispatch would cost a v-table lookup per stamping call.
For a 1000-device circuit at 100 kHz switching, that's 10^8
v-table lookups per simulated second — non-trivial overhead.

Static methods + concept = compile-time polymorphism. Each device
type's stamping code is monomorphic, can be inlined fully, and
auto-vectorises in SoA mode (Layer 4's per-device-type stamping).

### Why `current<S>` returns the scalar current

The current convention: positive current flows from terminal 0 to
terminal 1 (anode to cathode for a diode, drain to source for a
MOSFET, positive to negative for a resistor).

Layer 3's stamper takes this single scalar and stamps:
- `+i` on the row for terminal 0's node
- `-i` on the row for terminal 1's node
- For nonlinear models: the AD partials become the off-diagonal
  Jacobian entries via the same generic stamping code.

ONE current value per device. ONE function. The rest is mechanical.

## The three reference models — why these three

### Resistor (linear passive)

Simplest possible test: `i = G·(v[0] - v[1])`. The AD partials
should evaluate to `[+G, -G]` exactly — a closed-form Jacobian
that's trivially verifiable.

If the AD machinery is broken for this case, nothing works.

### VoltageSource (constraint, not current)

Voltage sources don't fit the "stamp a current" pattern — they
impose `v[0] - v[1] = V` as a constraint. The `current<S>` function
returns 0; the source contribution comes via Layer 3 recognising
`kind == Source` and adding a constraint row instead.

The point of including this in Layer 2 is to prove the concept is
flexible enough to handle non-current contributions. Layer 4 will
need similar special-casing for switches.

### IdealDiode (the nonlinear AD killer test)

The diode's smooth-blend behavioral form has all the AD hazards:
- Nonlinear (sigmoid)
- exp() — chain rule with self-multiplication
- Multiplication of intermediate AD values
- Subtraction of constants

If the AD machinery handles the diode correctly (AD partials match
finite-difference to 1e-6), it handles every device in Pulsim's
catalog. The diode IS the canary.

The test `test_ideal_diode.cpp` compares AD-derived partials to
central-difference partials at three op-points: sub-threshold,
at-threshold, forward-biased. Each is a fundamentally different
regime of the smooth-blend, and the AD must work in all three.

## What this layer does NOT do

- No matrix stamping. Layer 3 takes `current<ADRealN<N>>` results
  and stamps them into a sparse matrix.
- No state-space cache. Layer 4 consumes Layers 2 + 3 to build it.
- No event detection, no Newton iteration. Layer 5.
- No reverse-mode AD. Forward mode is sufficient for ≤ 32 terminals.
- No symbolic DSL. We considered ModelingToolkit-style equation
  description for v2, but at the Layer 2 boundary plain C++
  templates give us 95 % of the benefit (compile-time concept
  verification + same function for value and Jacobian) without the
  toolchain investment.

## Validation

`pulsim_v2_layer2_tests` covers each header in isolation:

- **AD scalar**: chain rule for `+`, `-`, `*`, `/`, math functions,
  composition (`exp(x) + y²`). Seeded inputs, derivatives propagate.
- **Concept**: three reference models satisfy `DeviceModel`; a
  broken stub fails the concept.
- **Resistor**: forward eval + AD partials match `[+G, -G]`
  analytically.
- **VoltageSource**: `current` returns 0; `static_voltage` returns
  `V`.
- **IdealDiode**: AD partials match central-difference at three
  op-points within `1e-6`.

Target: ≥ 40 assertions / ≥ 15 test cases. Layer 0 and Layer 1
tests stay green.
