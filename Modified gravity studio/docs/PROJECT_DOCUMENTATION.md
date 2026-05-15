# Project Documentation

## Overview

Modified Gravity Studio is a local web application for symbolic and numerical modified-gravity workflows. It provides a guided browser interface over a Flask API and a SymPy/Pytearcat symbolic core. The application derives field-equation components, solves matter variables, applies metric ansatz choices, builds diagnostics, evaluates plots, solves nonlinear residual systems numerically, and scans parameter ranges.

The project keeps symbolic solving, numerical post-processing, and browser rendering as separate layers. This keeps expensive algebra inside controlled backend stages while the frontend focuses on selection, progress display, plotting, and export.

## Runtime Flow

```text
Frontend form
  -> Flask task endpoint
  -> PipelineInput
  -> geometry cache
  -> model derivative preparation
  -> stress-energy assembly
  -> theory LHS assembly
  -> selected component extraction
  -> matter solve
  -> ansatz finalization
  -> diagnostics and plot data
  -> serialization
  -> browser result cards
```

Post-solve numerical routes:

```text
/api/numeric/solve  -> residual lambdify -> scipy root solve -> numeric diagnostics
/api/plot/evaluate  -> expression lambdify -> numpy series evaluation
/api/plot/scan      -> expression lambdify -> grid scan -> score and heatmap
```

## API Modules

### `api/app.py`

Owns Flask app setup, task orchestration, progress streaming, cancellation, task pruning, and local shutdown. Pipeline runs execute in worker threads and stream progress through Server-Sent Events.

### `api/routes/numeric.py`

Receives residual expressions, unknown names, parameter values, and a domain specification. It delegates to `core.numerics.solve.solve_residual_system` and returns pointwise matter solutions plus numeric diagnostics.

### `api/routes/plotting.py`

Receives symbolic expression groups and parameter values for plotting and scanning. It delegates to `core.plotting.evaluate.evaluate_plot_series` and `core.plotting.evaluate.scan_parameter_ranges`.

### `api/routes/_logging.py`

Provides compact request summaries, route-level error cleanup, and quiet output contexts for numerical and plotting endpoints.

## Core Modules

### `core/pipeline.py`

Coordinates the complete symbolic workflow. It validates registry choices, loads geometry, prepares theory derivatives, assembles stress-energy tensors, extracts equations, invokes the solver, applies ansatz substitutions, computes diagnostics, and prepares plot data.

### `core/pipeline_cache.py`

Contains bounded in-memory caches for LHS tensors, selected components, ansatz substitutions, reduced/raw equations, and scalar payloads. Each cache has a dedicated dictionary and an environment-configurable size limit.

### `core/solver.py`

Contains matter solve strategies and expression cleanup. Linear matter systems prefer direct `linsolve`; fallback strategies use bounded side-linear isolation where appropriate. Final matter expressions are simplified after ansatz substitution.

### `core/ansatz.py`

Parses metric-function ansatz expressions, builds derivative substitutions, and substitutes user-supplied ansatz constants.

### `core/results.py`

Serializes SymPy expressions for the frontend. It produces rendered LaTeX payloads, copyable LaTeX, and Mathematica syntax from the original SymPy expressions.

### `core/numerics/solve.py`

Compiles residual systems with `sp.lambdify` and solves them pointwise over a one-dimensional domain using SciPy. Compilation is cached by residual structure, variable, unknowns, and parameter names.

### `core/numerics/diagnostics.py`

Computes numeric energy conditions, equation-of-state quantities, sound-speed estimates, and TOV force balance from pointwise arrays.

### `core/plotting/evaluate.py`

Compiles solved symbolic expressions for numpy evaluation, returns plot series data, and scores parameter scans. Expression compilation is cached so repeated plot rebuilds with different numeric values avoid repeated SymPy parsing and lambdify work.

### `core/theories/`

Contains theory-specific LHS construction for `fR`, `fRTLm`, `fT`, `fTB`, `fQ`, and `fQC`.

### `core/registry/`

Defines supported theories, backgrounds, model presets, ansatz presets, matter compatibility, and component selections.

## Frontend Modules

### `templates/index.html`

Defines the application shell, input panels, result tabs, plot controls, numeric solve controls, and scan controls.

### `static/app.js`

Handles theory/background/model interactions, request submission, progress streaming, result rendering, warning display, and browser-side task state.

### `static/js/numericSolve.js`

Handles numeric solve controls, symbolic plot controls, parameter scan controls, plot SVG rendering, PNG export, component selection, and browser-side plot/scan caches.

### `static/style.css`

Provides the dark application shell, control layout, result card styling, plot styling, and scan visualization styling.

## Supported Problem Classes

Curvature:

- `f(R)`
- `f(R,T,Lm)`

Torsion:

- `f(T)`
- `f(T,B)`

Non-metricity:

- `f(Q)`
- `f(Q,C)`

Background categories:

- Cosmology: FRW
- Anisotropic cosmology: Bianchi I, Bianchi III, Kantowski-Sachs
- Static spherical compact backgrounds: wormhole, black hole

## Symbol And Parameter Handling

Model-form parameters and metric ansatz parameters are kept explicit unless the user provides numeric values. Filled inputs are substituted before the relevant solve or evaluation stage and become part of the cache key for that stage.

Common model symbols include `alpha`, `beta`, `gamma`, `lam`, `C0`, `Q0`, and `T0`. Common metric symbols include `a0`, `B0`, `A0`, `H0`, `h`, `r0`, `Phi0`, `M`, and `Q`.

The frontend plot tools expose remaining free numeric parameters after a symbolic run. This allows repeated plotting and parameter scans without rerunning the symbolic pipeline.

## Diagnostics

Symbolic diagnostics are built from solved matter expressions where possible:

- energy conditions
- equation of state
- stability / sound speed
- TOV equilibrium for static spherical compact backgrounds

Numeric diagnostics are computed from pointwise solution arrays:

- energy conditions
- equation of state
- sound-speed estimates
- TOV force terms and residuals

The diagnostic labels adapt to matter content. Perfect fluids use isotropic labels; anisotropic fluids use radial and tangential labels.

## Static Spherical Coordinate Handling

Static spherical equations use the independent radial components selected by the registry, typically `(t,t)`, `(r,r)`, and `(theta,theta)`. The normal pipeline does not replace `sin(theta)` with a numeric angular value. Non-metricity and torsion branches use bounded spherical branch cleanup only for sign and absolute-value artifacts that come from positive angular metric factors. This avoids injecting angular infinities while still keeping physical diagonal components readable.

## Cache Boundaries

The backend cache layers are intentionally separated:

| Cache | Scope |
| --- | --- |
| Geometry cache | background and curvature sign |
| LHS cache | theory, background, stress content, model, model parameters, matter Lagrangian |
| Component cache | pre-ansatz selected diagonal component |
| Ansatz cache | ansatz choice and ansatz parameter values |
| Reduced/raw equation cache | prepared equations, with distinct versioned prefixes |
| Scalar cache | derived scalar payloads |
| Solver simplify cache | expression cleanup results |
| Numeric compile cache | residual lambdify functions |
| Plot compile cache | expression lambdify functions |

Raw equations and reduced equations share a storage helper but use distinct key namespaces. Component caches are pre-ansatz by design. Ansatz-sensitive stages include ansatz choices and ansatz parameter values in their keys.

## Cleanup

The Quit Server action clears render caches, solver caches, pipeline caches, geometry caches, disk cache folders, and Python bytecode caches before stopping the local server. Task pruning also limits retained completed tasks during long sessions.

## Operating Guidance

- Provide numeric values for harmless scale parameters when expression size matters.
- Use `h` for power-law exponents in ansatz presets.
- Use FRW `k = 1, 0, -1` instead of a separate flat-FRW background.
- Use numeric residual solve for nonlinear matter systems that are difficult to solve symbolically.
- Use parameter scans to identify workable parameter windows before producing final plots.
- Keep expensive diagnostics disabled during early exploratory runs.
