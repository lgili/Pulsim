## ADDED Requirements

### Requirement: Thread Backend For Native-Callback Sweeps

`sweep` and `monte_carlo` SHALL accept `backend="auto"`, `"thread"` or
`"process"` (default `"auto"`), applied when `parallel=True`. With
`"auto"` they SHALL run the first job in the calling thread and SHALL
run the remaining jobs on a thread pool when that job's result reports
no Python callable in the step loop, and on a process pool otherwise.
Results SHALL be ordered by job index and SHALL be identical to those
of sequential execution on every backend. On the thread backend a job
that raises SHALL be recorded in `SweepResult.failed` and SHALL NOT
stop the other jobs. `SweepResult.backend` SHALL name the backend that
ran (`"serial"`, `"thread"` or `"process"`).

#### Scenario: Native schedule selects threads
- **GIVEN** a `builder_factory` for a buck and a `switch_fn` built by
  `make_pwm_switch_fn`
- **WHEN** `sweep(..., parallel=True, n_workers=4)` runs 8 jobs
- **THEN** `result.backend` is `"thread"`
- **AND** every KPI array is `np.array_equal` to the one from
  `parallel=False`

#### Scenario: Python callback keeps processes
- **GIVEN** the same sweep with `switch_fn` a Python lambda
- **WHEN** it runs with `parallel=True`
- **THEN** `result.backend` is `"process"`

#### Scenario: Backend can be forced
- **WHEN** the sweep is run with `backend="process"` and a native
  schedule, and with `backend="thread"` and a Python callback
- **THEN** `result.backend` is `"process"` and `"thread"` respectively
  and the KPIs equal the sequential ones

#### Scenario: Failure isolation on threads
- **GIVEN** a sweep of 8 jobs in which job 3 raises
- **WHEN** it runs on the thread backend
- **THEN** `result.failed` contains index 3 with its message and the
  other seven jobs have KPIs

#### Scenario: Seeded Monte Carlo is backend-independent
- **WHEN** `monte_carlo(..., seed=42, parallel=True)` is run with each
  backend
- **THEN** the sample arrays and the KPI arrays are identical across
  backends
