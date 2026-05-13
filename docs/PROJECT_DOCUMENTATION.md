# Project Documentation

## 1. Overview

Modified Gravity Studio is a symbolic computation platform for modified gravity research. It provides a browser interface over a Flask backend and a symbolic core built around SymPy and Pytearcat. The application is designed to let a user move from a theory choice to solved matter-sector expressions with progress feedback and export-friendly output.

The system is aimed at workflows such as:

- compare theories across standard backgrounds
- prototype custom model functions
- apply metric ansätze
- derive `rho`, `p`, `P_r`, `P_t`, geometric scalars, and diagnostics
- export expressions to LaTeX or Mathematica

## 2. High-Level Flow

```text
Frontend form
-> API request
-> pipeline setup
-> geometric/tensor preparation
-> field-equation assembly
-> solve strategy selection
-> ansatz application
-> final symbolic outputs
-> diagnostics
-> serialization
-> streamed UI result
```

The important architectural rule is that solving and simplification belong to the solve/finalization stages, while the render stage is reserved for display conversion.

## 3. Main Components

### `api/app.py`
Responsible for:

- request handling
- background execution
- task lifecycle management
- progress streaming
- task pruning and cleanup

### `core/pipeline.py`
Responsible for:

- constructing pipeline inputs
- selecting the background and theory path
- coordinating geometry/scalar assembly
- assembling field equations
- routing to the appropriate solve strategy
- applying ansätze
- coordinating diagnostics
- creating the result payload

### `core/solver.py`
Responsible for:

- solving matter equations
- post-solve cleanup
- ansatz application and finalization
- theory-aware simplification helpers
- diagnostic construction

### `core/results.py`
Responsible for:

- converting final SymPy expressions to display payloads
- LaTeX export
- Mathematica export
- optional display compression via filtered kernel extraction

### `core/ansatz.py`
Responsible for:

- ansatz parsing
- ansatz parameter substitution
- keeping ansatz-driven substitutions consistent with pipeline inputs

### Frontend (`templates/` + `static/`)
Responsible for:

- theory/background/model selection
- parameter input rendering
- streamed progress display
- result card rendering
- clipboard export for LaTeX and Mathematica code
- warning banners and user-facing messaging

## 4. Supported Problem Classes

The current project supports curvature, torsion, and non-metricity families through:

- `f(R)`
- `f(R,T,Lm)`
- `f(T)`
- `f(T,B)`
- `f(Q)`
- `f(Q,C)`

Backgrounds currently include FRW, anisotropic cosmologies, and static spherically symmetric systems such as wormholes and black-hole configurations.

## 5. Parameters and Symbol Management

One recurring issue in symbolic gravity workflows is uncontrolled growth in free constants. The project therefore supports user-supplied substitutions for theory/model parameters and ansatz constants.

Examples:
- `C0`, `Q0`, `T0`
- `r0`
- `n`
- `a0`, `H0`, `Phi0`

These values are substituted before solving where possible and are included in cache keys so reruns do not reuse stale symbolic results.

## 6. Simplification Strategy

The project intentionally separates concerns:

- equation reduction uses light structural cleanup
- the main symbolic cleanup occurs during solving and ansatz finalization
- diagnostics are computed after the solved matter variables are available
- rendering performs display conversion only

This avoids a common failure mode where downstream display stages become mathematically active and slow.

For difficult families such as transcendental, power-like, and sqrt-heavy models, the display layer can apply conservative kernel extraction and filtered compression without mutating the canonical solved expression.

## 7. Diagnostics Strategy

Diagnostics include:

- energy conditions
- equations of state
- speed of sound / stability terms

Where safe, diagnostics are constructed directly from solved `rho`, `p`, `P_r`, and `P_t` to avoid redundant symbolic solving. Stability terms remain the most expensive class because they can involve derivative-heavy expressions.

## 8. Warnings vs Hard Failures

The project now favors warning-first behavior for physically meaningful equalities such as:

- `rho = p`
- `P_r = P_t`

These can reflect a reduced or isotropic effective solution rather than a broken solve path. Strict failure can still be enabled when validation is the priority.

## 9. Rendering and Export

Every displayed expression is built from a SymPy expression and exported in two directions:

- LaTeX for browser display and copy
- Mathematica syntax using SymPy’s Mathematica printer

This avoids invalid workflows where Mathematica export is derived from LaTeX text.

For large expressions, the display layer may keep a small number of filtered named kernels. The goal is to reduce repetition while avoiding unreadable placeholder-heavy output.

## 10. Session Stability

Long symbolic sessions can degrade if memory and caches grow without control. The project now includes:

- bounded caches
- task pruning
- render-cache cleanup
- teardown cleanup after runs
- garbage collection after task completion

These steps reduce the chance that later computations in the same session become slow to start.

## 11. Windows-Friendly Operation

The repository includes Windows batch files for setup and execution:

- `setup.bat`
- `run.bat`

These are meant to make local use and repository onboarding easier for users who do not want to manage the Python environment manually.

## 12. Suggested Development Direction

The current codebase is suitable for production use for the supported workflows, but it is also a research tool. The next natural extensions are:

- more theories
- perturbation modules
- broader background coverage
- more theory-specific simplification/finalization routes
- more diagnostics
- export presets for symbolic post-processing tools

## 13. Practical Advice for Users

To keep expressions compact and runs faster:

- set harmless scaling constants when possible
- set power-law exponents when known
- disable expensive diagnostics during exploratory runs
- use theory-aware parameter choices to avoid unnecessarily large kernels
- inspect warnings rather than assuming every equality indicates a bad solve

## 14. Repository Readiness

For GitHub presentation, the repository should include:

- a clear README
- this documentation file
- polished Windows scripts
- stable frontend messages
- clean export behavior
- reproducible setup instructions

That baseline is now in place, and the project can continue to evolve with additional theory modules and perturbative capabilities.


## 15. Quit and Cache Cleanup

The frontend Quit Server action calls `/api/quit`, which clears temporary render caches, simplify caches, pipeline caches, geometry caches, disk caches, and Python bytecode caches before the local process exits.

## Recent UI/diagnostic fix

- Perfect-fluid diagnostics now use isotropic pressure labels `p`, `\rho + p`, `\rho + 3p`, `\rho - |p|`, and `c_s^2 = dp/d\rho`; anisotropic labels are shown only for anisotropic-fluid runs.
