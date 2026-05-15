# Modified Gravity Studio

Modified Gravity Studio is a browser-based symbolic and numerical workbench for modified gravity research. It combines a Flask backend, a KaTeX frontend, SymPy algebra, and Pytearcat tensor tooling to assemble field equations, solve matter variables, evaluate diagnostics, build plots, and scan parameter ranges for supported cosmological and compact backgrounds.

The application is designed around a research workflow: choose a gravity theory, choose a spacetime background, select matter content, set model and ansatz parameters, run the symbolic pipeline, and inspect the resulting matter sector through equations, diagnostics, numerical solve tools, plots, and parameter scans.

## Capabilities

- Registry-driven selection for theories, backgrounds, model forms, matter content, and metric ansatz presets
- Symbolic field-equation assembly for curvature, torsion, and non-metricity based theories
- Covariant teleparallel torsion pipeline with inertial spin connections for spherical tetrads
- Matter solving for perfect fluid, anisotropic fluid, dust, radiation, and vacuum cases where supported
- Static spherical TOV diagnostics for compact backgrounds
- Numeric residual solving for nonlinear matter systems
- Plotting of solved matter variables and diagnostics without recomputing the symbolic pipeline
- Parameter scanning for admissible model and metric parameter windows
- Exportable LaTeX and Mathematica expressions
- PNG export for generated plots
- Bounded cache layers for repeated symbolic, numerical, and plotting workloads

## Supported Theories

| ID | Theory | Geometry class | Main scalars |
| --- | --- | --- | --- |
| `fR` | `f(R)` gravity | Curvature | `R` |
| `fRTLm` | `f(R,T,Lm)` gravity | Curvature-matter coupling | `R`, `T_scalar`, `T_mat`, `L` |
| `fT` | `f(T)` teleparallel gravity | Torsion | `T` |
| `fTB` | `f(T,B)` gravity | Torsion plus boundary term | `T`, `B` |
| `fQ` | `f(Q)` symmetric teleparallel gravity | Non-metricity | `Q` |
| `fQC` | `f(Q,C)` non-metricity-boundary gravity | Non-metricity plus boundary term | `Q`, `C` |

## Supported Backgrounds

| ID | Background | Metric functions |
| --- | --- | --- |
| `FRW` | Friedmann-Robertson-Walker with `k = 1, 0, -1` | `a(t)` |
| `Bianchi_I` | LRS Bianchi type I | `A(t)`, `B(t)` |
| `Bianchi_III` | LRS Bianchi type III | `A(t)`, `B(t)` |
| `Kantowski_Sachs` | Kantowski-Sachs | `A(t)`, `B(t)` |
| `SS_wormhole` | Static spherical wormhole | `b(r)`, `Phi(r)` |
| `SS_blackhole` | Static spherical black hole | `nu_bh(r)`, `lam_bh(r)` |

Cosmological backgrounds use time-domain diagnostics and do not expose TOV controls. Static spherical backgrounds use radial-domain diagnostics and expose compact-object tools where the selected matter model supports them.


## Architecture

```text
Browser UI
  -> Flask API
  -> symbolic pipeline
  -> theory module and stress-energy assembly
  -> solver and ansatz finalization
  -> diagnostics and plot data
  -> serializer
  -> browser results
```

Additional post-solve routes:

```text
/api/numeric/solve  -> pointwise residual solve with scipy
/api/plot/evaluate  -> lambdified plot series
/api/plot/scan      -> parameter range scan over solved expressions
```

## Quick Start

### Requirements

- Python 3.10+
- Git
- A JavaScript-enabled browser
- Network access during dependency installation

### Install

```bash
git clone <your-repo-url>
cd modified-gravity-studio
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### Run

```bash
python run.py
```

Open `http://localhost:5000`.

Windows launchers are also included:

```bat
setup.bat
run.bat
```

The server uses Waitress when available and falls back to the Flask development server.

## Working With A Model

1. Select the gravity theory.
2. Select a compatible background.
3. Choose a model preset or enter a custom expression.
4. Select the stress-energy content.
5. Choose ansatz presets for the active metric functions.
6. Provide numeric values for model parameters and ansatz constants where desired.
7. Select diagnostics.
8. Run the symbolic pipeline.
9. Use plotting, numeric solve, and parameter scan tools on the returned expressions.

Blank parameter fields remain symbolic. Filled parameter fields are substituted into the relevant symbolic stage and included in cache keys.

## Diagnostics

Available diagnostics depend on the background and matter model.

Perfect-fluid outputs use isotropic labels:

- `rho`
- `p`
- `rho + p`
- `rho + 3p`
- `rho - |p|`
- `omega = p / rho`
- `c_s^2 = dp / d rho`

Anisotropic outputs use directional labels:

- `P_r`, `P_t`
- `NEC_r = rho + P_r`
- `NEC_t = rho + P_t`
- `DEC_r = rho - |P_r|`
- `DEC_t = rho - |P_t|`
- `omega_r`, `omega_t`, `omega_eff`
- `cs2_r`, `cs2_t`

Static spherical TOV output is represented by the general equilibrium relation and the force balance components:

```text
F_h + F_g + F_a = 0
```

The plotted TOV series focus on hydrostatic force, gravitational force, anisotropic force, and residual.

## Plotting And Parameter Scans

The symbolic plot panel evaluates solved expressions over a chosen domain. It uses the model and ansatz values already supplied in the main input panel and exposes any remaining free numeric parameters. Plots can be regenerated with different parameter values without rerunning the symbolic pipeline.

The parameter scan panel samples selected parameter ranges against finite-value checks, energy-condition checks, stability checks, and optional TOV residual checks. It returns accepted ranges, top-scoring samples, a heatmap for two-parameter scans, and a best-point plotting shortcut.

## Cache And Performance Model

The project uses bounded caches with separate responsibilities:

- Geometry cache for background tensors
- Spin connection and Lorentz-projected superpotential in the teleparallel geometry cache
- LHS cache for theory tensors
- Component cache for selected diagonal field-equation components
- Ansatz cache for metric-function substitutions
- Reduced/raw equation cache for equation preparation
- Scalar cache for derived scalar payloads
- Solver simplification cache for repeated expression cleanup
- Numeric residual compile cache for repeated nonlinear solves
- Plot expression compile cache for repeated plot and scan evaluation
- Browser-side request caches for numeric solve, symbolic plots, and parameter scans

Raw and reduced equation cache entries use distinct versioned key prefixes. Ansatz-dependent caches include ansatz selections and ansatz parameter values. Component caches are kept pre-ansatz, so the same extracted tensor component can be reused safely across different metric-function substitutions.

## Environment Variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `MGS_VERBOSE` | `false` | Enable verbose server logging |
| `MGS_SYMBOLIC_LOGS` | `false` | Allow symbolic library stdout in the terminal |
| `MGS_LOCAL` | `true` | Enable the local Quit Server action |
| `MGS_WORKERS` | `2` | Pipeline worker count |
| `MGS_MAX_COMPLETED_TASKS` | `24` | Completed task retention limit |
| `MGS_MAX_TASK_AGE_SECONDS` | `900` | Completed task retention age |
| `MGS_MAX_REDUCED_CACHE` | `8` | Reduced/raw equation cache limit |
| `MGS_MAX_ANSATZ_CACHE` | `16` | Ansatz cache limit |
| `MGS_MAX_LHS_CACHE` | `8` | Theory LHS cache limit |
| `MGS_MAX_COMPONENT_CACHE` | `48` | Component cache limit |
| `MGS_MAX_SCALAR_CACHE` | `16` | Scalar cache limit |

## Repository Layout

```text
api/
  app.py
  routes/
    numeric.py
    plotting.py
    _logging.py
core/
  pipeline.py
  pipeline_cache.py
  solver.py
  results.py
  ansatz.py
  geometry.py
  config.py
  diagnostics/
  numerics/
  plotting/
  registry/
  stress_energy/
  theories/
docs/
static/
templates/
run.py
requirements.txt
setup.bat
run.bat
```

## Documentation

- `docs/PROJECT_DOCUMENTATION.md` describes the module layout and runtime flow.
- `docs/pipeline_working_and_solving_strategy.md` describes the symbolic pipeline, solving strategy, and cache boundaries.
