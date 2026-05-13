# Modified Gravity Studio Pipeline and Solving Strategy

This document explains how the current computation pipeline works, where the expensive symbolic steps happen, and how the solver chooses a strategy for different matter systems. It is written against the current implementation in `core/pipeline.py`, `core/solver.py`, the theory modules in `core/theories/`, and the shared stress-energy implementation in `core/stress_energy/stress_energy.py`.

## 1. Pipeline Overview

The pipeline starts from a `PipelineInput` object and returns a `PipelineResults` object. The browser sends choices such as theory, background, model expression, matter type, ansatz, and diagnostics flags. The backend then streams progress events while it builds geometry, assembles field equations, solves the matter variables, applies ansatz substitutions, and formats the results for KaTeX.

The main entry point is:

```text
Pipeline.run(inp)
  -> Pipeline._run_pipeline(inp)
```

The output is serialized through:

```text
results_to_dict(results)
```

That serialization step is also where lazy diagnostics are evaluated if they are requested.

## 2. Main Pipeline Stages

### Stage 0: Setup and Validation

The pipeline first validates that:

- The selected theory exists in `THEORY_REGISTRY`.
- The selected background is supported by that theory.
- The selected stress-energy tensor is allowed for that background.
- For `f(R,T,Lm)`, the matter Lagrangian choice is compatible with the selected matter model.

Important files:

```text
core/pipeline.py
core/registry/theories.py
core/registry/metrics.py
```

This is where invalid combinations are stopped early, before Pytearcat or SymPy spend time on tensor construction.

### Stage 1: Geometry

The pipeline loads or computes the geometry cache for the selected background:

```text
geom = get_geometry(inp.background_id, inp.curvature_k)
```

The geometry cache stores the metric, live symbols, curvature tensors, torsion scalars, boundary scalars, nonmetricity data, and Pytearcat state where available.

For example:

- `f(R)` and `f(R,T,Lm)` use curvature objects such as `R`, `R_mu_nu`, and the Einstein tensor.
- `f(T)` and `f(T,B)` use torsion and boundary objects.
- `f(Q)` and `f(Q,C)` use the nonmetricity scalar `Q`; `f(Q,C)` also uses `C = R - Q`.

The pipeline also cleans tracked Pytearcat tensor names before continuing so repeated runs do not collide with stale tensor names.

### Stage 2: Model Derivatives

The model expression is parsed using dummy scalar symbols first. This avoids mixing user-level symbols such as `R`, `T`, `Q`, or `C` with Pytearcat objects too early.

Examples:

```text
f(R):       f, f_R
f(T):       f, f_T, f_TT
f(T,B):     f, f_T, f_B
f(R,T,Lm):  f, f_R, f_T, f_L
f(Q):       f, f_Q, f_QQ
f(Q,C):     f, f_Q, f_C, f_QQ, f_CC
```

After differentiation, the dummy scalar is replaced by the actual geometry scalar. The scalar replacement is simplified with the notebook-style `fast_simplify` helper.

Important function:

```text
_compute_model_derivatives(inp, geom)
```

### Stage 3: Stress-Energy Tensor Assembly

All matter models now use one shared assembly API:

```text
set_handler.assemble(g, index_form=...)
```

The pipeline decides the correct index form:

```text
_,_   for curvature and nonmetricity theories
^,_   for torsion theories
```

This matters because torsion field equations are naturally mixed-index equations, while curvature and nonmetricity branches use covariant diagonal components.

Supported matter models:

- Perfect fluid: `rho`, `p`
- Anisotropic fluid: `rho`, `P_r`, `P_t`
- Dust: `rho`
- Radiation: `rho`
- Vacuum: `Lambda`

For anisotropic fluids, the spatial vector is selected from the geometry context or defaults to a radial-like direction.

Important function:

```text
_assemble_set(set_handler, g, theory, spatial_vector)
```

### Stage 4: Theory LHS Assembly

The theory module builds the left-hand side tensor for the selected field equation.

Current structure:

- `core/theories/fR.py`
- `core/theories/fRTLm.py`
- `core/theories/fT.py`
- `core/theories/fTB.py`
- `core/theories/fQ.py`
- `core/theories/fQC.py`

Most theories avoid completing every possible tensor index form unless it is necessary. Instead, the pipeline extracts only the canonical components needed for the selected matter model.

For example, anisotropic systems usually need:

```text
(t,t), (r,r), (theta,theta)
```

or the equivalent diagonal directions for Bianchi/Kantowski backgrounds.

The current `f(Q,C)` branch follows the compact Hessian-style construction:

```text
HfC = covariant Hessian of f_C
Box(f_C) = g^ij HfC_ij

LHS =
  -(f_Q + f_C) E_ij
  - 1/2 g_ij (f - R f_C - Q f_Q)
  + HfC_ij
  - g_ij Box(f_C)
```

Only the selected components are simplified for extraction.

The pipeline now also caches extracted component pairs individually. If a repeated run asks for the same theory, background, model, matter type, and component key, the component expression can be reused without re-extracting that diagonal slot from the theory module.

### Stage 5: Component Extraction

The pipeline extracts the canonical field-equation components from the LHS tensor and the stress-energy tensor.

There are two paths:

```text
_get_raw_equations_cached(...)
_get_reduced_equations_cached(...)
```

For anisotropic matter, the pipeline uses the raw equation path because the current strategy is solve first, then apply the ansatz. This matches the notebook behavior that worked best for the hard `f(R,L,T)`, `f(Q)`, and `f(Q,C)` cases.

For non-anisotropic systems, the pipeline can reduce equations with ansatz substitutions before solving.

Static spherical backgrounds also get an angular cleanup step:

```text
sin(theta) -> 1
cos(theta) -> 0
```

This removes angular artifacts from diagonal spherical components.

### Stage 6: Solve

The solver receives a list of `Eq(lhs, rhs, evaluate=False)` equations and a list of unknown matter variables.

The `evaluate=False` part is important. Without it, SymPy can collapse equations into `False` or `True` too early when assumptions are involved, especially around positive `rho`.

Main solver class:

```text
FieldEquationSolver
```

For anisotropic matter, the current global strategy is:

```text
linsolve([eq1, eq2, eq3], [rho, P_r, P_t])
fast_simplify(rho_expr)
fast_simplify(Pr_expr)
fast_simplify(Pt_expr)
```

This is implemented through:

```text
solve_anisotropic_notebook()
  -> _solve_linear_fast()
```

For other linear matter systems, the solver also tries the direct `linsolve` path first.

If a non-anisotropic system cannot be solved by direct linear solve, it falls back to sequential isolation:

1. Solve the first unknown from the first equation.
2. Substitute it into the remaining equations.
3. Solve the remaining unknowns.
4. Use cheap side-linear isolation before expensive `sp.solve`.

The side-linear isolation path is important for hard models because many field equations have this shape:

```text
huge_geometric_expression = coefficient * matter_variable
```

In that case, the solver avoids expanding or collecting the huge geometric side.

For anisotropic systems, the first choice is still the strict notebook-style direct linear solve. If that fails, the solver now has one bounded fallback only: component-wise side-linear isolation. It tries the expected diagonal order:

```text
tt equation    -> rho
rr equation    -> P_r
theta equation -> P_t
```

This fallback does not call a full generic solve on the whole system. It only accepts the result if each isolated expression is complete and no solved expression still contains unresolved matter variables.

### Stage 7: Ansatz Application

After solving, the ansatz is applied to the solved matter expressions.

This is intentional:

```text
solve first -> apply ansatz -> simplify final matter result
```

For hard anisotropic cases, applying ansatz before solving can make the equation system larger or force SymPy to simplify the wrong object. The notebook-style approach solves the linear matter system first, then substitutes metric functions such as:

```text
b(r)   -> r/exp(r-r0)
Phi(r) -> 0
A(t)   -> a0*t**h
B(t)   -> a0*t**h
```

The ansatz application also builds derivative substitutions:

```text
Derivative(b(r), r)    -> derivative of selected b ansatz
Derivative(b(r), r, 2) -> second derivative of selected b ansatz
```

After substitution, each solved expression is simplified again with the notebook-style final cleanup.

Important function:

```text
apply_ansatz()
```

### Stage 8: Scalar Outputs

The pipeline fills scalar results after matter solutions are known:

```text
R
T
B
Q
C
T_scalar
Lm
```

For `f(Q)` and `f(Q,C)`, the UI reuses existing result slots:

```text
results.T = Q
results.B = C
```

For heavy anisotropic `f(R,T,Lm)` non-FRW runs, the pipeline avoids substituting the fully solved matter expressions back into every scalar trace when that would explode expression size.

Important function:

```text
_fill_scalar_results()
```

### Stage 9: Selected Diagnostics

Diagnostics are controlled by user-selectable flags:

```text
compute_energy_conditions
compute_eos
compute_stability
```

If selected, the pipeline creates lazy diagnostic results:

- Energy conditions: `NEC`, `WEC`, `SEC`, `DEC`
- Equation of state: `omega_r`, `omega_t`, `omega_eff`
- Stability/sound speed: `cs2_r`, `cs2_t`

Lazy means they are not computed until serialization asks for them. This prevents diagnostics from blocking the critical solve path when the user did not ask for them.

For heavy anisotropic non-FRW `f(R,T,Lm)`, diagnostics are deferred instead of computed immediately.

## 3. Current Solving Strategy

The solving strategy is built around one observation from the notebooks:

```text
The field equations are usually linear in the matter variables,
even when the geometric side is huge.
```

So the solver tries to avoid symbolic operations that expand or simplify the full geometric expression before matter variables are isolated.

### Preferred Path: Direct Linear Solve

The preferred path is:

```text
sp.linsolve(selected_equations, unknowns)
```

For anisotropic fluids:

```text
selected_equations = [eq_tt, eq_rr, eq_theta_theta]
unknowns = [rho, P_r, P_t]
```

For perfect fluids:

```text
selected_equations = [eq_tt, eq_rr]
unknowns = [rho, p]
```

The returned tuple is converted into a dictionary and every component is passed through:

```text
solve_final_cleanup()
  -> fast_simplify()
```

### Fallback Path: Sequential Isolation

If direct `linsolve` does not produce a complete solution, non-anisotropic systems can fall back to sequential solving.

The fallback order is:

1. Try side-linear isolation on one equation.
2. Try derivative-based linear isolation.
3. Try `sp.solve(..., simplify=False)`.
4. Try manual collect/factor isolation.

The fallback is deliberately conservative because generic `sp.solve` can hang on large transcendental expressions.

For anisotropic systems, the current strict notebook-style path does not use the older rho-first pressure fallback. If the direct linear solve fails, it raises a solve error instead of spending unbounded time on an uncertain strategy.

## 4. Simplification Strategy

The current pipeline separates simplification into different roles.

### Component Simplification

During theory extraction, only selected field-equation components are simplified:

```text
simplify_selected_component(...)
```

This prevents the pipeline from simplifying all 16 tensor components when only 2 or 3 diagonal equations are needed.

The terminal prints cues such as:

```text
[SIMPLIFY] Simplifying selected component f(Q,C) (r,r)
```

### Notebook-Style Matter Simplification

The core simplifier for solved expressions is:

```text
fast_simplify(expr)
```

It performs:

1. `sympify`
2. `.doit()`
3. temporary replacement of derivative atoms with dummy symbols
4. simplification of transcendental arguments
5. `cancel`
6. `powsimp(force=True)`
7. `factor_terms`
8. collection by derivative dummy symbols
9. another transcendental-argument cleanup
10. physical-domain cleanup

This is the method copied from the working notebooks. It is used after solve and after ansatz application, where it gives the most benefit.

After ansatz substitution, the cleanup also compresses repeated transcendental blocks before the final `fast_simplify` pass. Larger atoms such as `exp((r-r0)/r0)`, `log(r/r0)`, or `tanh((r-r0)/r0)` are temporarily replaced by dummy symbols, simplified as algebraic blocks, collected, and then restored. This keeps SymPy from repeatedly manipulating the same long transcendental object during final matter simplification.

### Light Equation Cleanup

Before solving, the pipeline avoids deep simplification on raw hard expressions. That stage is intentionally light because simplifying the full field-equation component before isolating matter variables is often the slowest path.

The guiding rule is:

```text
Do not deeply simplify the giant equation.
Simplify the solved matter expression.
```

That is why the hard anisotropic path is fast enough for models like:

```text
R + alpha*R**2 + beta*T_scalar + lam*L
Q + alpha*log(1 + Q/beta) + gamma*C
```

### Display Simplification

Result rendering has a separate cheap cleanup:

```text
_readable_display_simplify()
```

This is only for display readability before converting to LaTeX. It is not the main mathematical solve simplifier.

## 5. Caching Strategy

The pipeline uses in-memory caches for repeated runs:

```text
LHS cache
Component cache
Ansatz cache
Reduced/raw equation cache
Scalar cache
Simplify cache
```

The important cache keys include theory, background, stress tensor, model expression, component key, ansatz, and matter Lagrangian where relevant.

At the start of each run:

- SymPy internal caches are flushed.
- The solver simplify cache is retained while it is small and pruned only when it grows past its maximum size.
- Pytearcat tensor names tracked by the geometry cache are cleaned.

This is meant to balance two competing needs:

1. Reuse expensive geometry/LHS/equation work across repeated combinations.
2. Avoid memory growth and stale state from Pytearcat or SymPy.

The cache fix is especially important because repeated hard symbolic runs can otherwise become slower even if each individual model is unchanged.

## 6. Why Hard Cases Still Take Time

The slowest cases are usually not slow because of the final linear solve itself. They are slow because of one or more of these:

- Building the theory LHS for a large non-FRW metric.
- Taking Hessians of model derivatives such as `f_R`, `f_B`, or `f_C`.
- Simplifying selected tensor components containing nested `exp`, `log`, powers, and derivatives.
- Rendering very large final expressions to LaTeX.
- Computing diagnostics on already large matter expressions.

For `f(Q)` and `f(Q,C)`, nonmetricity construction and boundary scalar expressions can become large quickly on non-FRW backgrounds. For static wormhole models, transcendental shape functions and logs can make simplification highly sensitive to when it is applied.

The current optimized order is therefore:

```text
build only needed tensors/components
extract only needed equations
solve matter variables directly
apply ansatz
fast_simplify solved expressions
compute diagnostics only when selected
render final KaTeX
```

## 7. Practical Mental Model

A good way to think about the pipeline is:

```text
Registry choices
  -> geometry
  -> model derivatives
  -> stress-energy tensor
  -> theory LHS
  -> selected diagonal equations
  -> linear matter solve
  -> ansatz substitution
  -> notebook-style simplification
  -> optional diagnostics
  -> KaTeX rendering
```

For anisotropic hard cases, the most important design decision is:

```text
Do not simplify the full symbolic system before solving.
Use the field equations as a linear system in rho, P_r, P_t.
Then simplify each solved expression after ansatz substitution.
```

That is the main reason the current pipeline now behaves closer to the notebooks.
