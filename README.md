# Modified Gravity Studio

Modified Gravity Studio is a browser-based symbolic workbench for modified theories of gravity. It combines a Flask backend, a KaTeX-powered frontend, SymPy-based symbolic manipulation, and Pytearcat tensor tooling to derive field-equation components, solve for matter variables, apply metric ansätze, and inspect diagnostics across cosmological and static backgrounds.

The project is built for exploratory research workflows: select a theory, choose a spacetime background, enter or pick a model, configure matter content, optionally pin model/ansatz parameters such as `C0`, `r0`, `n`, or `a0`, and stream the symbolic pipeline directly in the browser.

## Highlights

- Browser UI for theory, background, model, matter, and ansatz selection
- Progressive streaming updates for long symbolic runs
- Registry-driven theory and background metadata
- Built-in model presets and ansatz presets
- Optional parameter injection for model constants and ansatz constants
- Support for perfect-fluid, anisotropic-fluid, dust, radiation, and vacuum-compatible workflows
- Rendered LaTeX output with Mathematica copy/export support
- Warning-first result handling for equalities such as `rho = p` or `P_r = P_t`
- Faster diagnostics path built from solved matter variables where appropriate
- Session cleanup and bounded cache controls for better long-run stability, including full temporary-cache deletion on Quit Server

## Supported Theories

| ID | Theory | Geometry Class | Typical Scalars |
| --- | --- | --- | --- |
| `fR` | `f(R)` gravity | Curvature | `R` |
| `fRTLm` | `f(R,T,Lm)` gravity | Curvature-matter coupling | `R`, `T_scalar`, `T_mat`, `L` |
| `fT` | `f(T)` teleparallel gravity | Torsion | `T` |
| `fTB` | `f(T,B)` gravity | Torsion + boundary term | `T`, `B` |
| `fQ` | `f(Q)` symmetric teleparallel gravity | Non-metricity | `Q` |
| `fQC` | `f(Q,C)` nonmetricity-boundary gravity | Non-metricity + boundary | `Q`, `C` |

## Supported Backgrounds

| ID | Spacetime | Metric Functions |
| --- | --- | --- |
| `FRW_flat` | Flat FRW | `a(t)` |
| `FRW_curved` | Curved FRW with symbolic `k` | `a(t)` |
| `Bianchi_I` | LRS Bianchi type-I | `A(t)`, `B(t)` |
| `Bianchi_III` | LRS Bianchi type-III | `A(t)`, `B(t)` |
| `Kantowski_Sachs` | Kantowski-Sachs | `A(t)`, `B(t)` |
| `SS_wormhole` | Static spherically symmetric wormhole | `b(r)`, `Phi(r)` |
| `SS_blackhole` | Static spherically symmetric black hole | `nu_bh(r)`, `lam_bh(r)` |

## Architecture

```text
UI -> Flask API -> symbolic pipeline -> solver/ansatz finalization -> diagnostics -> serializer -> browser results
```

Core modules:

- `api/app.py`: Flask app, task orchestration, streaming progress, session/task lifecycle
- `core/pipeline.py`: end-to-end orchestration for theory setup, tensor assembly, solving, ansatz application, diagnostics, and serialization
- `core/solver.py`: simplification helpers, solve strategies, ansatz application, and diagnostic construction
- `core/results.py`: result serialization, LaTeX/Mathematica export, and display compression
- `core/ansatz.py`: ansatz parsing and ansatz-parameter substitution
- `static/` and `templates/`: production frontend assets
- `docs/`: project documentation

## Quick Start

### Prerequisites

- Python 3.10+
- Git
- A browser with JavaScript enabled

### Install

```bash
git clone <your-repo-url>
cd modified-gravity-studio
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` installs Pytearcat from GitHub, so installation requires network access.

### Run

```bash
python run.py
```

Then open `http://localhost:5000`.

On Windows, you can also use:

```bat
setup.bat
run.bat
```

## Usage

1. Select a theory.
2. Select a supported spacetime background.
3. Choose a model preset or enter a custom model expression.
4. Select matter content.
5. Choose ansatz presets for the active metric functions.
6. Optionally set model or ansatz constants such as `n`, `C0`, `r0`, `a0`, `Phi0`, or `H0`.
7. Choose diagnostics if needed.
8. Run the pipeline and inspect the streamed results.

## Parameter Inputs

The UI can expose theory-specific and ansatz-specific symbols so users can reduce symbolic clutter before solving.

Examples:
- `C0` in logarithmic boundary models
- `r0` in exponential or inverse-power shape functions
- `n` in power-law models
- `a0`, `H0`, `Phi0`, `M`, `Q` in metric ansätze

Blank inputs remain symbolic. Filled inputs are substituted during parsing and become part of the cache key.

## Output Modes

Each result card supports:

- rendered LaTeX in the browser
- LaTeX copy/export
- Mathematica copy/export generated from the SymPy expression, not from LaTeX text

For large expressions, the renderer may expose a small set of filtered display definitions to compress repeated kernels without over-fragmenting the visible output.

## Diagnostics

Available diagnostics depend on the theory, background, and matter content. The current pipeline supports:

- energy conditions
- equation-of-state quantities
- speed-of-sound / stability terms

Where safe, some diagnostics are built directly from solved matter variables to avoid unnecessary symbolic overhead.

## Warnings and Validation

The pipeline no longer blocks result display when equalities such as `rho = p` or `P_r = P_t` appear. These are surfaced as warnings so physically meaningful reduced cases still display.

Strict validation is still available through environment flags where needed.

## Performance Notes

The pipeline includes several production-oriented safeguards:

- bounded caches
- session/task pruning
- explicit teardown cleanup after runs
- fast-path diagnostics where safe
- display compression tuned to preserve readability
- browser-side math rendering limited to the final display pass

For additional details, see `docs/PROJECT_DOCUMENTATION.md`.

## Repository Layout

```text
api/
core/
docs/
static/
templates/
run.py
run.bat
setup.bat
requirements.txt
README.md
```

## Development Notes

- SymPy and Pytearcat drive most of the symbolic workload.
- Frontend rendering uses KaTeX.
- Progress updates are streamed to the browser during long runs.
- The project is intended for symbolic experimentation, so some heavy models may still require theory-aware parameter choices for compact output.

## Roadmap

This repository is actively evolving. Planned additions include more gravity theories, perturbative modules, broader diagnostic support, and further theory-specific optimization layers.

## Launcher scripts

- `run.bat`: original lightweight launcher for existing users with Python already installed
- `bat_old.bat`: backup copy of the original launcher style
- `setup_new_user.bat`: first-time setup for new users
- `run_new_user.bat`: launcher that prefers the project virtual environment when available


## Display simplifier modes

The UI now exposes two display modes:

- `Fast mode`: conservative, lower-overhead output shaping intended for routine runs and large nested solutions.
- `Heavy simplifier mode`: stronger bounded display simplification for harder transcendental, logarithmic, and sqrt-heavy outputs.

These modes only affect display formatting. They do not change the core field-equation solve.

## Recent note

- Stability diagnostics now force evaluation of remaining partial derivatives even in Fast mode, so displayed `c_s^2` expressions do not contain unevaluated `\partial/\partial` artifacts.


## Session Cleanup

When you click **Quit Server** in the frontend, the backend now clears temporary render caches, solver caches, pipeline caches, geometry caches, disk cache folders, and Python bytecode caches before shutdown.

## Recent UI/diagnostic fix

- Perfect-fluid diagnostics now use isotropic pressure labels `p`, `\rho + p`, `\rho + 3p`, `\rho - |p|`, and `c_s^2 = dp/d\rho`; anisotropic labels are shown only for anisotropic-fluid runs.
