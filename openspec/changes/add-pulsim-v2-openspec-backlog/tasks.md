## 1. Source-helper specs (V5-V10)

- [x] 1.1 `openspec/specs/pulsim-v2-source-helpers/spec.md` covering all 6 helpers with one `#### Scenario:` each (drawn from the existing C++ + Python tests)

## 2. Time-varying source specs (V11, V12)

- [x] 2.1 `openspec/specs/pulsim-v2-sine-source/spec.md`
- [x] 2.2 `openspec/specs/pulsim-v2-pulse-source/spec.md`

## 3. Nonlinear device specs (V13, V14)

- [x] 3.1 `openspec/specs/pulsim-v2-mosfet-level1/spec.md`
- [x] 3.2 `openspec/specs/pulsim-v2-igbt-level1/spec.md`

## 4. Controlled-source spec (V15)

- [x] 4.1 `openspec/specs/pulsim-v2-vcvs-opamp/spec.md`

## 5. Validation

- [x] 5.1 `openspec validate --strict` passes on all 6 new spec files
- [x] 5.2 Commit and push as a single docs-only change
