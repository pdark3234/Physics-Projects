# Project Documentation

## 1. Overview

Modified Gravity Studio is a symbolic computation platform for modified gravity research. It provides a browser interface over a Flask backend and a symbolic core built around SymPy and Pytearcat. The application lets a user move from a theory choice to solved matter-sector expressions with progress feedback and export-friendly output.

Supported workflows include:

- comparing theories across standard backgrounds
- prototyping custom model functions
- applying metric ansätze
- deriving `rho`, `p`, `P_r`, `P_t`, geometric scalars, and diagnostics
- exporting expressions to LaTeX or Mathematica
- solving field-equation residuals numerically over a 1D domain
- evaluating and plotting solved symbolic expressions over a user-defined domain

---

## 2. High-Level Flow

```text
Frontend form
  -> API request
  -> pipeline setup
  -> geometry / tensor preparation
  -> field-equation assembly
  -> solve strategy selection
  -> ansatz application
  -> final symbolic outputs
  -> diagnostics
  -> serialization
  -> streamed UI result
```

Two additional routes run alongside the main symbolic pipeline:

```text
/api/numeric/solve  -> residual export -> pointwise scipy solver -> numeric diagnostics
/api/plot/evaluate  -> symbolic lambdify -> pointwise numpy evaluation -> plot series
```

The important architectural rule is that solving and simplification belong to the solve/finalization stages, while the render stage is reserved for display conversion.

---

## 3. Main Components

### `api/app.py`

Responsible for:

- request handling and routing
- background task execution via threading
- task lifecycle management (queued, running, complete, error, cancelled)
- Server-Sent Events progress streaming
- task pruning, session cleanup, and cache teardown on quit

Task metadata is tracked in an `OrderedDict` keyed by task UUID. Completed tasks are pruned when the pool exceeds `MGS_MAX_COMPLETED_TASKS` or when tasks age past `MGS_MAX_TASK_AGE_SECONDS`.

### `api/routes/numeric.py`

Exposes `/api/numeric/solve` (POST). Receives a JSON payload containing residual expressions (as SymPy-serializable strings), domain bounds, unknown names, and parameter values. Delegates to `core.numerics.solve_residual_system` and returns pointwise solutions, numeric diagnostics, and TOV terms.

### `api/routes/plotting.py`

Exposes `/api/plot/evaluate` (POST). Receives a JSON payload containing symbolic expression groups, a domain specification, and parameter values. Delegates to `core.plotting.evaluate_plot_series` and returns evaluated numeric series for each requested quantity.

### `api/routes/_logging.py`

Shared utilities for route-level logging: payload summarizers (`summarize_plot_payload`, `summarize_numeric_payload`), error truncation (`clean_error`), and a `quiet_route_output` context manager that suppresses symbolic library output on these routes.

### `core/pipeline.py`

Responsible for:

- constructing `PipelineInput` and returning `PipelineResults`
- selecting the background and theory path from the registries
- coordinating geometry and scalar assembly
- assembling field equations
- routing to the appropriate solve strategy
- applying ansätze
- coordinating diagnostics
- creating the serializable result payload

### `core/solver.py`

Responsible for:

- solving matter equations (linear and sequential strategies)
- post-solve cleanup
- ansatz application and finalization
- theory-aware simplification helpers (`fast_simplify`, `solve_final_cleanup`)
- diagnostic construction

### `core/results.py`

Responsible for:

- converting final SymPy expressions to display payloads
- LaTeX export
- Mathematica export using SymPy's Mathematica printer (not derived from LaTeX text)
- optional display compression via filtered kernel extraction

### `core/ansatz.py`

Responsible for:

- parsing ansatz expressions
- building derivative substitutions for metric functions (first and second derivatives)
- keeping ansatz-driven substitutions consistent with pipeline inputs

### `core/numerics/solve.py`

Provides `solve_residual_system(payload)`. Compiles residual strings to numpy-compatible lambdas via SymPy, then solves the resulting nonlinear system pointwise across the domain using `scipy.optimize`. Results are returned as named lists of floats (or `None` for non-convergent points), along with convergence metadata and warnings.

### `core/numerics/diagnostics.py`

Provides `compute_numeric_diagnostics` and `compute_numeric_tov`. Both operate on pointwise solution arrays from the numeric solver.

`compute_numeric_diagnostics` computes energy conditions (NEC, WEC, SEC, DEC), EoS parameters, and sound-speed estimates using finite-difference derivatives. Output format adapts automatically between isotropic and anisotropic labeling.

`compute_numeric_tov` computes TOV equilibrium terms for wormhole and black-hole backgrounds. For wormholes, it evaluates `b(r)` and `Phi(r)` from metric-function strings to derive mass, compactness, and redshift gradient. For black holes, it evaluates `nu_bh(r)` and `lam_bh(r)`. All metric-function expressions are compiled from the symbolic ansatz strings, so they stay consistent with the symbolic solve.

### `core/plotting/evaluate.py`

Provides `evaluate_plot_series(payload)`. Accepts a `groups` dictionary mapping display names to symbolic expression strings, lambdifies each expression using numpy, evaluates over the requested domain, and returns the series data with metadata and warnings. Expressions that fail to evaluate (complex, non-finite, or missing parameter) are returned as empty series with a per-series warning.

### `core/config.py`

Defines per-stage symbolic timeouts (`STAGE_TIMEOUTS`) and reads environment flags (`ALLOW_SHUTDOWN`, `MAX_WORKERS`, `VERBOSE_LOGS`).

### Frontend (`templates/` + `static/`)

Responsible for:

- theory/background/model/matter/ansatz/parameter selection
- streaming progress display (SSE)
- result card rendering with KaTeX
- clipboard export for LaTeX and Mathematica
- warning banners and user-facing messaging
- numeric solve UI: domain/parameter input, result table, energy-condition and TOV charts
- symbolic plot UI: domain/parameter input, chart rendering

---

## 4. Supported Problem Classes

Curvature theories: `f(R)`, `f(R,T,Lm)`

Torsion theories: `f(T)`, `f(T,B)`

Non-metricity theories: `f(Q)`, `f(Q,C)`

Backgrounds: FRW (flat/closed/open), anisotropic cosmologies (Bianchi I, Bianchi III, Kantowski–Sachs), and static spherically symmetric systems (wormhole, black hole).

---

## 5. Parameters and Symbol Management

One recurring issue in symbolic gravity workflows is uncontrolled growth in free constants. The project supports user-supplied substitutions for theory/model parameters and ansatz constants.

Examples: `C0`, `Q0`, `T0`, `r0`, `n`, `a0`, `H0`, `Phi0`, `M`, `Q`.

These values are substituted before solving where possible and are included in cache keys so reruns do not reuse stale symbolic results. Blank inputs remain symbolic.

---

## 6. Simplification Strategy

The project intentionally separates simplification concerns:

- equation reduction uses light structural cleanup
- the main symbolic cleanup occurs during solving and ansatz finalization
- diagnostics are computed after solved matter variables are available
- rendering performs display conversion only

This avoids a common failure mode where downstream display stages become mathematically active and slow.

For difficult families such as transcendental, power-like, and sqrt-heavy models, the display layer can apply conservative kernel extraction and filtered compression without mutating the canonical solved expression.

---

## 7. Diagnostics Strategy

Symbolic diagnostics include energy conditions, EoS, speed of sound/stability, and TOV equilibrium balance for static spherical runs.

Where safe, diagnostics are constructed directly from solved `rho`, `p`, `P_r`, and `P_t` to avoid redundant symbolic solving. Stability terms remain the most expensive class because they involve derivative-heavy expressions. Stability diagnostics force evaluation of remaining partial derivatives even in Fast display mode so `c_s^2` expressions are fully evaluated.

For static spherical compact backgrounds, the TOV diagnostic is handled by `core.diagnostics.tov` rather than the main solver. It computes the hydrostatic, gravitational, and anisotropic force terms from the solved matter sector and ansatz-substituted metric functions, with equilibrium represented by a zero total residual.

Numeric diagnostics are computed by `core.numerics.diagnostics` from the pointwise solution arrays returned by the numeric solver. They cover the same diagnostic categories and additionally provide mass/compactness and mass-continuity residuals for compact backgrounds.

---

## 8. Warnings vs Hard Failures

The project favors warning-first behavior for physically meaningful equalities such as:

- `rho = p`
- `P_r = P_t`

These can reflect a reduced or isotropic effective solution rather than a broken solve path. Strict failure can still be enabled when validation is the priority.

---

## 9. Rendering and Export

Every displayed expression is built from a SymPy expression and exported in two directions:

- LaTeX for browser display and copy
- Mathematica syntax using SymPy's Mathematica printer

This avoids invalid workflows where Mathematica export is derived from LaTeX text.

For large expressions, the display layer may keep a small number of filtered named kernels. The goal is to reduce repetition while avoiding unreadable placeholder-heavy output.

---

## 10. Session Stability

Long symbolic sessions can degrade if memory and caches grow without control. The project includes:

- bounded caches with configurable size limits
- task pruning controlled by `MGS_MAX_COMPLETED_TASKS` and `MGS_MAX_TASK_AGE_SECONDS`
- render-cache cleanup
- teardown cleanup after runs
- garbage collection after task completion

These steps reduce the chance that later computations in the same session become slow to start.

---

## 11. Numeric Solve Architecture

The numeric solve feature allows users to step outside the fully symbolic workflow for heavy or non-converging configurations.

After a symbolic pipeline run, the frontend can construct a numeric-solve payload by:

1. Exporting field-equation residuals as SymPy-serializable strings.
2. Declaring the independent variable (e.g., `r` for static or `t` for cosmological), the matter unknowns, and any free parameters.
3. Sending the payload to `/api/numeric/solve`.

The backend compiles residuals to numpy-compatible functions via `sp.lambdify`, then walks the domain grid and calls `scipy.optimize` rootfinding at each point. Convergence metadata is returned for each point.

Numeric TOV terms are derived from the ansatz metric-function strings at the same grid points, so they are fully consistent with the symbolic ansatz choices.

---

## 12. Symbolic Plot Architecture

After a successful symbolic run, solved matter expressions and diagnostic quantities can be plotted over a user-defined domain.

The frontend sends a payload containing:

- a `groups` dictionary mapping display names to solved symbolic expressions (as strings)
- a domain specification (variable, min, max, points)
- parameter values for any remaining free symbols

The backend lambdifies each expression with numpy and evaluates it over the domain. Each group is returned as a named list of floats, with `null` for non-finite points and per-series warnings for expressions that cannot be evaluated.

---

## 13. Windows-Friendly Operation

The repository includes Windows batch files for setup and execution:

- `setup.bat` — creates a virtual environment and installs dependencies
- `run.bat` — starts the server using any Python found on PATH

---

## 14. Practical Advice for Users

To keep expressions compact and runs faster:

- set harmless scaling constants when possible
- set power-law exponents when known
- disable expensive diagnostics during exploratory runs
- use theory-aware parameter choices to avoid unnecessarily large kernels
- inspect warnings rather than assuming every equality indicates a bad solve
- for very heavy configurations, try the numeric solve path after a symbolic run to get pointwise solutions without waiting for full symbolic simplification

---

## 15. Environment Flags

| Variable | Default | Effect |
| --- | --- | --- |
| `MGS_VERBOSE` | `false` | Enable verbose server-side logging |
| `MGS_SYMBOLIC_LOGS` | `false` | Pass SymPy/Pytearcat stdout to the terminal |
| `MGS_LOCAL` | `true` | Allow the Quit Server action |
| `MGS_WORKERS` | `2` | Number of pipeline worker threads |
| `MGS_MAX_COMPLETED_TASKS` | `24` | Maximum completed tasks retained in memory |
| `MGS_MAX_TASK_AGE_SECONDS` | `900` | Age (seconds) before a completed task is pruned |

---

## 16. Quit and Cache Cleanup

The frontend Quit Server action calls `/api/quit`, which clears temporary render caches, simplify caches, pipeline caches, geometry caches, disk caches, and Python bytecode caches before the local process exits.

---

## 17. Repository Readiness

For GitHub presentation, the repository includes:

- a clear README
- this documentation file
- a pipeline strategy document (`pipeline_working_and_solving_strategy.md`)
- Windows batch scripts for setup and launch
- stable frontend messages and export behavior
- reproducible setup instructions

The project can continue to evolve with additional theory modules, perturbative capabilities, and extended numeric/plot workflows.
