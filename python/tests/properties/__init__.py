"""Property-based tests for Pulsim.

`add-property-based-testing` Phase 1: each test in this package
generates random Pulsim circuits via Hypothesis strategies and
asserts physical invariants (KCL, Tellegen, energy conservation,
passivity, PWL cache behavior) hold across many randomized runs.

Failures are auto-shrunken to minimal repro circuits and recorded in
`regressions/` so the CI ratchets — once a property finds a bug, the
shrunken example sticks around and runs on every subsequent commit.
"""
