# Pipeline And Solving Strategy

This document describes the symbolic pipeline, solving order, diagnostic construction, numerical evaluation paths, and cache boundaries used by Modified Gravity Studio.

## Pipeline Entry Point

The backend receives a `PipelineInput` object and returns a `PipelineResults` object.

```text
Pipeline.run(inp)
  -> Pipeline._run_pipeline(inp)
  -> results_to_dict(results)
```

The frontend streams progress while the backend validates the selection, prepares geometry, constructs field equations, solves matter variables, applies ansatz choices, builds diagnostics, and serializes the result.

## Stage 1: Validation

The pipeline verifies:

- the selected theory exists in `THEORY_REGISTRY`
- the selected background is compatible with that theory
- the selected matter type is compatible with the background
- the selected matter Lagrangian is valid for `f(R,T,Lm)`
- requested diagnostics are compatible with the background category

Invalid combinations stop before tensor construction.

## Stage 2: Geometry

Geometry is loaded through:

```text
get_geometry(background_id, curvature_k)
```

The geometry cache provides the metric, coordinates, curvature tensors, torsion objects, boundary scalars, non-metricity objects, and tracked Pytearcat tensor names. Cosmological backgrounds use a time coordinate. Static spherical backgrounds use a radial coordinate for solved matter profiles and compact-object diagnostics.

## Stage 3: Model Derivatives

The selected model expression is parsed with dummy scalar symbols and differentiated before replacing those symbols with the active geometric scalars.

Examples:

```text
f(R):       f, f_R
f(T):       f, f_T, f_TT
f(T,B):     f, f_T, f_B
f(R,T,Lm):  f, f_R, f_T, f_L
f(Q):       f, f_Q, f_QQ
f(Q,C):     f, f_Q, f_C, f_QQ, f_CC
```

This keeps model differentiation separate from Pytearcat tensor objects and reduces unnecessary symbolic expansion.

## Stage 4: Stress-Energy Assembly

Matter tensors are assembled through the shared stress-energy API. The pipeline chooses the index form required by the theory:

```text
_,_   curvature and non-metricity theories
^,_   torsion theories
```

Supported matter sectors:

- perfect fluid: `rho`, `p`
- anisotropic fluid: `rho`, `P_r`, `P_t`
- dust: `rho`
- radiation: `rho`
- vacuum: `Lambda`

For anisotropic fluids, the spatial direction is supplied by the background context or falls back to the canonical radial-like direction.

## Stage 5: Theory LHS Assembly

Theory modules construct the left-hand side tensor for the selected gravity model:

```text
core/theories/fR.py
core/theories/fRTLm.py
core/theories/fT.py
core/theories/fTB.py
core/theories/fQ.py
core/theories/fQC.py
```

The pipeline extracts only the components required by the selected matter model. A perfect-fluid run usually needs two equations. An anisotropic run usually needs three independent diagonal equations.

For static spherical anisotropic systems, the canonical component set is:

```text
(t,t), (r,r), (theta,theta)
```

The `(phi,phi)` component is not required for the normal solve path.

## Static Spherical Angular Handling

The normal static spherical pipeline does not evaluate the angular coordinate. The selected `(theta,theta)` component avoids the physical need for a separate angular specialization. Non-metricity and torsion theory modules may remove sign or absolute-value wrappers that arise from positive spherical metric factors, but they do not inject singular angular values.

This keeps diagonal component extraction coordinate-aware without altering the field equations by an artificial angular branch.

## Stage 6: Equation Preparation

The pipeline prepares equations as residual-preserving SymPy equations:

```text
Eq(lhs, rhs, evaluate=False)
```

Identity residuals are filtered by checking:

```text
simplify(lhs - rhs) == 0
```

Two preparation paths are used:

```text
_get_raw_equations_cached(...)
_get_reduced_equations_cached(...)
```

Raw equations are pre-ansatz. Reduced equations include ansatz substitutions when the selected matter and background path benefits from early reduction.

## Stage 7: Matter Solve

The solver treats the matter variables as the primary unknowns. Field equations are usually linear in those variables even when their geometric coefficients are large.

Preferred path:

```text
sp.linsolve(selected_equations, unknowns)
```

Typical unknown sets:

```text
perfect fluid:      [rho, p]
anisotropic fluid:  [rho, P_r, P_t]
dust:               [rho]
radiation:          [rho]
vacuum:             [Lambda]
```

Fallback solving uses bounded side-linear isolation where a field equation has the form:

```text
large_geometric_side = coefficient * matter_variable
```

The fallback avoids unbounded generic solving on the full system. Completed solutions are checked for missing matter variables before finalization.

## Stage 8: Ansatz Finalization

The preferred hard-case order is:

```text
solve matter variables
  -> substitute metric ansatz choices
  -> substitute ansatz parameter values
  -> simplify solved matter expressions
```

Ansatz substitution includes metric functions and their first and second derivatives:

```text
b(r), b'(r), b''(r)
Phi(r), Phi'(r), Phi''(r)
A(t), A'(t), A''(t)
B(t), B'(t), B''(t)
```

This order avoids expanding the full equation system before the matter variables have been isolated.

## Stage 9: Scalar Outputs

Scalar outputs are filled after matter solutions are available:

```text
R
T
B
Q
C
T_scalar
Lm
```

The displayed scalar set depends on the selected theory. Heavy trace substitutions are kept symbolic where substituting solved matter expressions would create disproportionate expression growth.

## Stage 10: Diagnostics

Diagnostics are optional and are built after matter solving.

Energy conditions:

```text
NEC, WEC, SEC, DEC
```

Equation of state:

```text
omega
omega_r, omega_t, omega_eff
```

Stability:

```text
c_s^2
cs2_r, cs2_t
```

TOV equilibrium for static spherical compact backgrounds:

```text
F_h + F_g + F_a = 0
residual = F_h + F_g + F_a
```

TOV plotting focuses on hydrostatic force, gravitational force, anisotropic force, and residual.

## Simplification Strategy

The pipeline separates simplification by purpose:

- selected component simplification for extracted tensor entries
- solve cleanup for isolated matter expressions
- ansatz-final cleanup for final matter expressions
- display cleanup for readable result cards

The guiding rule is:

```text
Avoid deep simplification of the full field-equation system.
Simplify each solved matter expression after isolation.
```

This is especially important for logarithmic, exponential, power-law, torsion, and non-metricity models.

## Cache Strategy

The symbolic pipeline uses bounded caches with distinct responsibilities:

```text
Geometry cache
LHS cache
Component cache
Ansatz cache
Reduced/raw equation cache
Scalar cache
Solver simplify cache
```

Key boundaries:

- LHS cache keys include theory, background, stress tensor, model expression, model parameters, and matter Lagrangian.
- Component cache keys identify pre-ansatz selected field-equation components.
- Ansatz cache keys include ansatz choices and ansatz parameter values.
- Reduced equation keys include ansatz data.
- Raw equation keys are pre-ansatz.
- Raw and reduced equation entries use distinct versioned prefixes.
- Scalar cache keys include theory/background context and scalar-relevant substitutions.

The plotting and numerical routes also maintain compile caches:

```text
Numeric residual compile cache
Plot expression compile cache
Scan expression compile cache
```

Browser-side caches use separate maps for numeric solve, symbolic plots, and parameter scans.

## Numeric Residual Solve

The numeric solve path evaluates nonlinear residual systems after a symbolic run.

Payload contents:

- residual expressions
- unknown names
- independent variable
- domain bounds and sample count
- numeric parameter values
- metric-function ansatz strings for compact diagnostics

The backend compiles residuals with `sp.lambdify` and solves each domain point independently with SciPy. Non-convergent points are represented as missing values without preventing other points from returning.

Numeric diagnostics are computed from the pointwise solution arrays.

## Symbolic Plot Evaluation

The plot route receives solved expressions grouped by tab:

```text
matter
energy_conditions
eos
stability
tov
```

Each expression is compiled with numpy and evaluated over the requested domain. Non-finite or complex values are returned as `null`. Missing numeric parameters produce explicit warnings.

The frontend can rebuild plots with new numeric parameter values without rerunning the symbolic pipeline.

## Parameter Scan

The scan route samples parameter grids over solved expressions. It evaluates constraints such as:

- finite-value coverage
- positive density
- selected energy conditions
- causal sound-speed range
- optional TOV residual tolerance

The response includes:

- total samples
- accepted sample count
- accepted parameter ranges
- top-scoring samples
- best point
- heatmap data for two-parameter scans

The frontend can plot the best point directly by copying its parameter values into the active plot controls.

## Performance Guidance

For faster exploratory runs:

- provide numeric values for scale parameters when known
- keep diagnostics disabled until matter variables are solved
- use numeric solve for nonlinear residual systems
- use parameter scans before committing to a final plot window
- prefer compact ansatz choices for first-pass exploration
- avoid unnecessarily large domain sample counts during scans

The pipeline is optimized for repeated local exploration: cache expensive symbolic structures, solve matter variables directly, reuse solved expressions for plotting, and reserve full reruns for inputs that alter the symbolic equations.
