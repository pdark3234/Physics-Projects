# Modified Gravity Studio

Modified Gravity Studio is a browser-based symbolic workbench for modified theories of gravity. It combines a Flask backend, a KaTeX-powered frontend, SymPy-based symbolic manipulation, and Pytearcat tensor tooling to derive field-equation components, solve for matter variables, apply metric ansätze, and inspect diagnostics across cosmological and static spherically symmetric backgrounds.

The project is built for exploratory research workflows: select a theory, choose a spacetime background, enter or pick a model, configure matter content, optionally pin model or ansatz parameters such as `C0`, `r0`, `n`, or `a0`, and stream the symbolic pipeline directly in the browser.

---

## Highlights

- Browser UI for theory, background, model, matter, and ansatz selection
- Progressive streaming updates for long symbolic runs
- Registry-driven theory and background metadata
- Built-in model presets and ansatz presets
- Optional parameter injection for model constants and ansatz constants
- Support for perfect-fluid, anisotropic-fluid, dust, radiation, and vacuum workflows
- Rendered LaTeX output with Mathematica copy/export support
- Warning-first result handling for physical equalities such as `rho = p` or `P_r = P_t`
- Faster diagnostics path built from solved matter variables where appropriate
- Numeric solve mode: export field-equation residuals and solve them pointwise over a 1D domain
- Symbolic plot mode: evaluate and plot solved matter expressions and diagnostics over a user-defined domain
- Session cleanup and bounded cache controls for better long-run stability, including full cache deletion on Quit Server

---

## Supported Theories

| ID | Theory | Geometry Class | Typical Scalars |
| --- | --- | --- | --- |
| `fR` | `f(R)` gravity | Curvature | `R` |
| `fRTLm` | `f(R,T,Lm)` gravity | Curvature–matter coupling | `R`, `T_scalar`, `T_mat`, `L` |
| `fT` | `f(T)` teleparallel gravity | Torsion | `T` |
| `fTB` | `f(T,B)` gravity | Torsion + boundary term | `T`, `B` |
| `fQ` | `f(Q)` symmetric teleparallel gravity | Non-metricity | `Q` |
| `fQC` | `f(Q,C)` non-metricity–boundary gravity | Non-metricity + boundary | `Q`, `C` |

---

## Supported Backgrounds

| ID | Spacetime | Metric Functions |
| --- | --- | --- |
| `FRW` | FRW with selectable `k = 1, 0, -1` | `a(t)` |
| `Bianchi_I` | LRS Bianchi type-I | `A(t)`, `B(t)` |
| `Bianchi_III` | LRS Bianchi type-III | `A(t)`, `B(t)` |
| `Kantowski_Sachs` | Kantowski–Sachs | `A(t)`, `B(t)` |
| `SS_wormhole` | Static spherically symmetric wormhole | `b(r)`, `Phi(r)` |
| `SS_blackhole` | Static spherically symmetric black hole | `nu_bh(r)`, `lam_bh(r)` |

---

## Architecture

```text
UI ──► Flask API ──► symbolic pipeline ──► solver / ansatz ──► diagnostics ──► serializer ──► browser
                │
                ├──► /api/numeric/solve   (numeric residual solver)
                └──► /api/plot/evaluate   (symbolic plot evaluator)
```

Core modules:

- `api/app.py` — Flask app, task orchestration, streaming progress, session/task lifecycle
- `api/routes/numeric.py` — numeric residual-solve endpoint (`/api/numeric/solve`)
- `api/routes/plotting.py` — symbolic plot-evaluate endpoint (`/api/plot/evaluate`)
- `core/pipeline.py` — end-to-end orchestration: theory setup, tensor assembly, solving, ansatz application, diagnostics, serialization
- `core/solver.py` — simplification helpers, solve strategies, ansatz application, diagnostic construction
- `core/results.py` — result serialization, LaTeX/Mathematica export, display compression
- `core/ansatz.py` — ansatz parsing and ansatz-parameter substitution
- `core/numerics/` — pointwise residual solver (`solve.py`) and numeric diagnostics (`diagnostics.py`)
- `core/plotting/` — symbolic expression evaluator for plot series
- `core/config.py` — pipeline stage timeouts and environment flags
- `static/` and `templates/` — production frontend assets (KaTeX, app.js, numericSolve.js)
- `docs/` — project documentation

---

## Quick Start

### Prerequisites

- Python 3.10+
- Git
- A browser with JavaScript enabled
- Network access (Pytearcat is installed from GitHub)

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

Then open `http://localhost:5000`.

The server uses [Waitress](https://docs.pylonsproject.org/projects/waitress/) when available, falling back to the Flask development server. For improved thread management on longer symbolic runs, install Waitress:

```bash
pip install waitress
```

#### Windows shortcuts

```bat
setup.bat      # create virtual environment and install dependencies
run.bat        # start the server (uses any Python found on PATH)
```

---

## Usage

### Symbolic pipeline

1. Select a theory.
2. Select a supported spacetime background.
3. Choose a model preset or enter a custom model expression.
4. Select matter content (perfect fluid, anisotropic fluid, dust, radiation, vacuum).
5. Choose ansatz presets for the active metric functions.
6. Optionally set model or ansatz constants such as `n`, `C0`, `r0`, `a0`, `Phi0`, or `H0`.
7. Enable diagnostics (energy conditions, EoS, stability) if needed.
8. Run the pipeline and inspect the streamed results.

### Numeric solve

After a successful symbolic run, the **Numeric** tab becomes available. It exports the field-equation residuals as SymPy-string expressions, accepts pointwise parameter values, and solves the matter unknowns (`rho`, `P_r`, `P_t`, ...) over a user-defined 1D domain using `scipy.optimize` rootfinding. Numeric energy conditions and TOV balance terms are computed from the pointwise solutions.

### Symbolic plot

After a successful symbolic run, the **Plot** tab becomes available. It evaluates solved symbolic matter expressions and diagnostic quantities (energy conditions, EoS, sound speed) over a user-defined domain and renders them as charts directly in the browser.

---

## Parameter Inputs

The UI exposes theory-specific and ansatz-specific symbols so users can reduce symbolic clutter before solving.

| Symbol | Typical use |
| --- | --- |
| `C0`, `Q0`, `T0` | logarithmic or ratio model constants |
| `r0` | length scale in wormhole shape functions |
| `n` | power-law exponent |
| `a0`, `H0` | scale factor and Hubble constant in cosmological ansätze |
| `Phi0` | constant tidal potential in wormhole ansätze |
| `M`, `Q` | mass and charge in black-hole ansätze |

Blank inputs remain symbolic. Filled inputs are substituted during parsing and become part of the cache key.

---

## Output Modes

Each result card supports:

- Rendered LaTeX in the browser (via KaTeX)
- LaTeX copy/export
- Mathematica copy/export generated directly from the SymPy expression (not from LaTeX text)

For large expressions, the renderer applies a small set of filtered display definitions to compress repeated kernels without over-fragmenting the visible output.

---

## Diagnostics

Available diagnostics depend on theory, background, and matter content.

**Symbolic diagnostics:**

- Energy conditions: NEC, WEC, SEC, DEC
- Equation of state: `omega_r`, `omega_t`, `omega_eff`
- Speed of sound / stability: `cs2_r`, `cs2_t`
- TOV equilibrium terms for static spherical perfect-fluid and anisotropic runs

**Numeric diagnostics (computed after numeric solve):**

- Pointwise energy conditions (NEC, WEC, SEC, DEC)
- Pointwise EoS parameters
- TOV balance: hydrostatic force, gravitational force, anisotropic force, total residual, mass/compactness proxy, mass-continuity residual

Where safe, symbolic diagnostics are built directly from solved matter variables to avoid unnecessary symbolic overhead. Stability diagnostics force evaluation of remaining partial derivatives even in Fast display mode, so `c_s^2` expressions do not contain unevaluated partial-derivative artifacts.

Perfect-fluid diagnostics use isotropic pressure labels (`p`, `rho + p`, `rho + 3p`, `rho - |p|`, `c_s^2 = dp/drho`); anisotropic labels are shown only for anisotropic-fluid runs.

---

## Display Modes

The UI exposes two display simplifier modes:

- **Fast mode** — conservative, lower-overhead output shaping for routine runs and large nested solutions.
- **Heavy simplifier mode** — stronger bounded display simplification for harder transcendental, logarithmic, and sqrt-heavy outputs.

These modes only affect display formatting. They do not change the core field-equation solve.

---

## Warnings and Validation

The pipeline surfaces physical equalities such as `rho = p` or `P_r = P_t` as warnings rather than blocking result display. These can reflect physically meaningful reduced or isotropic cases. Strict failure can be enabled through environment flags when hard validation is required.

---

## Environment Flags

| Variable | Default | Effect |
| --- | --- | --- |
| `MGS_VERBOSE` | `false` | Enable verbose server-side logging |
| `MGS_SYMBOLIC_LOGS` | `false` | Pass SymPy/Pytearcat stdout to the terminal |
| `MGS_LOCAL` | `true` | Allow the Quit Server action |
| `MGS_WORKERS` | `2` | Number of pipeline worker threads |
| `MGS_MAX_COMPLETED_TASKS` | `24` | Maximum completed tasks retained in memory |
| `MGS_MAX_TASK_AGE_SECONDS` | `900` | Age (seconds) before a completed task is pruned |

---

## Performance Notes

The pipeline includes several production-oriented safeguards:

- Bounded in-memory caches for LHS tensors, field-equation components, and simplification results
- Session/task pruning after each run
- Explicit teardown cleanup after runs (render caches, solver caches, geometry caches)
- Fast-path diagnostics computed directly from solved matter variables where safe
- Display compression tuned to preserve readability while reducing LaTeX size
- Browser-side KaTeX rendering limited to the final display pass

For more details on solver strategy and caching, see `docs/pipeline_working_and_solving_strategy.md`.

---

## Session Cleanup

Clicking **Quit Server** in the frontend calls `/api/quit`, which clears temporary render caches, solver caches, pipeline caches, geometry caches, disk cache folders, and Python bytecode caches before the local process exits.

---

## Repository Layout

```text
api/
  app.py                    Flask app and task orchestration
  routes/
    numeric.py              /api/numeric/solve endpoint
    plotting.py             /api/plot/evaluate endpoint
    _logging.py             Shared route logging helpers
core/
  pipeline.py               End-to-end symbolic pipeline
  solver.py                 Solve strategies and simplification
  results.py                Serialization and LaTeX/Mathematica export
  ansatz.py                 Ansatz parsing and substitution
  config.py                 Stage timeouts and environment flags
  geometry.py               Geometry cache and metric setup
  display_simplify.py       Display-layer simplification helpers
  pipeline_cache.py         In-memory LHS/component caches
  numerics/
    solve.py                Pointwise residual solver (scipy)
    diagnostics.py          Numeric energy conditions and TOV
  plotting/
    evaluate.py             Symbolic expression evaluator for plots
  registry/
    theories.py             Theory capability registry
    metrics.py              Background/metric registry
    ansatze.py              Ansatz preset registry
    models.py               Model preset registry
  stress_energy/
    stress_energy.py        Shared stress-energy tensor assembly
    perfect.py              Perfect-fluid specialization
  theories/
    fR.py  fRTLm.py  fT.py  fTB.py  fQ.py  fQC.py
  diagnostics/
    tov.py                  TOV equilibrium diagnostics
docs/
  PROJECT_DOCUMENTATION.md
  pipeline_working_and_solving_strategy.md
static/
  app.js                    Main frontend logic
  js/numericSolve.js        Numeric solve and plot UI
  style.css
templates/
  index.html
run.py                      Entry point (Waitress / Flask dev server)
run.bat                     Windows launcher
setup.bat                   Windows first-time setup
requirements.txt
```

---

## Development Notes

- SymPy and Pytearcat drive most of the symbolic workload.
- Frontend rendering uses KaTeX.
- Progress updates are streamed to the browser during long runs via Server-Sent Events.
- Numeric solving uses SciPy (`scipy.optimize`) with pointwise rootfinding.
- The project is intended for symbolic experimentation; heavy models may still require theory-aware parameter choices to produce compact output.

---

## Roadmap

This repository is actively evolving. Planned additions include:

- More gravity theories
- Perturbative modules
- Broader background coverage
- More theory-specific simplification and finalization routes
- Additional diagnostic outputs
- Export presets for symbolic post-processing tools
