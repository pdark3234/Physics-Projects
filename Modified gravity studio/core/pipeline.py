"""
Computation Pipeline (All Four Theories)

Orchestrates the full computation flow from input to results.
Emits SSE events for progress tracking.
Strictly follows spec v6 pipeline stages 0-9.
"""

import builtins
import sys
if not hasattr(builtins, 'display'):
    builtins.display = lambda *args, **kwargs: None

from typing import Dict, Any, Callable
import sympy as sp
import threading
import io
import time

from core.ansatz import build_ansatz_subs, build_extended_subs
from core.config import VERBOSE_LOGS
from core.pipeline_cache import (
    flush_sympy_caches,
    get_component_cache,
    get_or_compute_ansatz,
    get_or_compute_lhs,
    get_or_compute_reduced,
    get_scalar_cache,
    set_component_cache,
    set_scalar_cache,
)
from core.results import PipelineInput, PipelineResults, results_to_dict
import os


# Per-run tensor registry for cleanup (Fix 2: pytearcat tensor name collisions)
_pt_tensor_registry: list = []

# â”€â”€â”€ Morris-Thorne Performance Fixes Toggle Flags â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
ENABLE_SS_SANITISE = False         # Angular branch cleanup is not used in the normal pipeline.


def _log_debug(message: str):
    """Emit low-value operational logs only when verbose mode is enabled."""
    if VERBOSE_LOGS:
        print(message)


def _flush_sympy_caches():
    """Clear SymPy internal caches to prevent memory bloat across runs."""
    flush_sympy_caches(_log_debug)


class CancelledError(Exception):
    pass


def _auto_confirm():
    """Context manager that patches stdin to auto-answer 'y'."""
    import contextlib

    @contextlib.contextmanager
    def _ctx():
        orig = sys.stdin
        sys.stdin = io.StringIO('y\n')
        try:
            yield
        finally:
            sys.stdin = orig

    return _ctx()


class Pipeline:
    """Main computation pipeline â€” strict spec v6 implementation."""

    def __init__(self, event_callback: Callable[[Dict], None] = None):
        self.event_callback = event_callback or (lambda x: None)
        self._cancelled = threading.Event()

    def emit(self, event_type: str, **kwargs):
        self.event_callback({'type': event_type, **kwargs})

    def _mark(self, timings: Dict[str, float], label: str, start: float) -> float:
        now = time.perf_counter()
        timings[label] = now - start
        _log_debug(f"[TIMING] {label}: {timings[label]:.3f}s")
        return now

    def check_cancelled(self):
        if self._cancelled.is_set():
            raise CancelledError("Computation cancelled")

    def cancel(self):
        self._cancelled.set()

    def run(self, inp: PipelineInput) -> PipelineResults:
        try:
            return self._run_pipeline(inp)
        except CancelledError:
            self.emit('cancelled')
            try:
                from core.solver import flush_simplify_cache
                _flush_sympy_caches()
                flush_simplify_cache()
                import pytearcat as pt  # type: ignore
                _cleanup_pytearcat_tensors(pt)
            except Exception:
                pass
            return PipelineResults(error='cancelled')
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            print(f"[PIPELINE ERROR] {e}\\n{tb}", flush=True)
            self.emit('error', message=str(e))
            return PipelineResults(error=str(e))

    def _run_pipeline(self, inp: PipelineInput) -> PipelineResults:
        timings = {}
        t0 = time.perf_counter()
        last_mark = t0
        # Fix 1: Flush SymPy caches at start of each run to prevent memory bloat
        _flush_sympy_caches()
        
        # Fix 2: Clear tensor registry for this run
        global _pt_tensor_registry
        _pt_tensor_registry = []

        import pytearcat as pt

        from core.registry.metrics import get_metric_context, is_set_allowed
        from core.geometry import get_geometry
        from core.stress_energy.stress_energy import create_stress_energy
        from core.solver import FieldEquationSolver, safe_simplify, SolveError, flush_simplify_cache, get_matter_symbols
        
        # Optimization 1: Flush simplify expression cache (after import)
        flush_simplify_cache()

        results = PipelineResults(warnings=[])
        results.simplify_mode = getattr(inp, 'simplify_mode', 'fast')
        results.diagnostics_requested = {
            'energy_conditions': inp.compute_energy_conditions,
            'eos': inp.compute_eos,
            'stability': inp.compute_stability,
            'tov': inp.compute_tov,
        }

        # â”€â”€ Stage 0: Setup (merged old 0 + 1: validation + metric context) â”€â”€â”€â”€â”€â”€
        self.emit('progress', stage=0, label='Setup', pct=5)
        self.check_cancelled()

        if not is_set_allowed(inp.background_id, inp.stress_tensor):
            raise ValueError(
                f"Stress-energy type '{inp.stress_tensor}' is not compatible "
                f"with background '{inp.background_id}'"
            )

        from core.registry.theories import THEORY_REGISTRY
        if inp.theory not in THEORY_REGISTRY:
            raise ValueError(f"Unknown theory: {inp.theory!r}")
        if inp.background_id not in THEORY_REGISTRY[inp.theory].supports_backgrounds:
            raise ValueError(
                f"Theory '{inp.theory}' is not enabled for background '{inp.background_id}'"
            )

        # Lm compatibility check for fRTLm theory
        if inp.theory == 'fRTLm':
            from core.registry.metrics import check_lm_set_compatibility
            lm_allowed, lm_reason = check_lm_set_compatibility(
                inp.stress_tensor, inp.matter_lag
            )
            if not lm_allowed:
                raise ValueError(
                    f"Incompatible Lm choice '{inp.matter_lag}' "
                    f"for stress-energy type '{inp.stress_tensor}': {lm_reason}"
                )

        from core.registry.ansatze import validate_grouped_ansatz
        ansatz_ok, ansatz_reason = validate_grouped_ansatz(inp.background_id, inp.ansatz)
        if not ansatz_ok:
            raise ValueError(ansatz_reason)

        blackhole_metric_variant = _blackhole_metric_variant_from_ansatz(inp)
        ctx = get_metric_context(inp.background_id, inp.curvature_k)
        ctx.theory = inp.theory

        # â”€â”€ Stage 1: Geometry (unchanged from old Stage 2) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        self.emit('progress', stage=1, label='Geometry', pct=20)
        self.check_cancelled()

        geom = get_geometry(inp.background_id, inp.curvature_k, blackhole_metric_variant)
        last_mark = self._mark(timings, 'geometry', last_mark)

        # Fix 2: Clean up pytearcat tensor names to prevent collisions in next run
        _cleanup_pytearcat_tensors(pt, geom)

        # Populate ctx with live symbols from geometry cache
        live = geom.live_symbols
        coord_name = ctx.independent_coord_name
        ctx.independent_coord = live.get(coord_name)
        ctx.metric_fns = {k: v for k, v in live.items()
                          if k in ctx.metric_fn_names}

        # â”€â”€ Stage 2: Prepare (merged old 3 + 4: pytearcat re-init + model derivatives) â”€
        self.emit('progress', stage=2, label='Preparing field equations', pct=35)
        self.check_cancelled()

        # Fix 5: Use cached pytearcat module if available (eliminates redundant Christoffel recomputation)
        if geom.pt_module is not None:
            pt = geom.pt_module
            _log_debug("[PIPELINE] Stage 2: Using cached pytearcat module from geometry cache")
        else:
            # pytearcat is stateful â€” we must re-define coords/metric in the current thread
            with _auto_confirm():
                _reinit_metric(pt, inp.background_id, inp.curvature_k, geom)
            # Store pt module in cache for future runs
            geom.pt_module = pt

        model_derivatives = _compute_model_derivatives(inp, geom)
        ansatz_subs = None
        extended_subs = None
        if _use_pre_ansatz_model_derivatives(inp):
            ansatz_key = (
                'ansatz', inp.background_id, ctx.independent_coord_name,
                tuple(sorted(inp.ansatz.items())),
                tuple(sorted(getattr(inp, 'ansatz_params', {}).items())),
            )
            ansatz_subs = get_or_compute_ansatz(
                ansatz_key,
                lambda: build_ansatz_subs(inp, ctx, _log_debug),
                _log_debug,
            )
            extended_subs = build_extended_subs(ansatz_subs, ctx.independent_coord)
            model_derivatives = _apply_pre_ansatz_to_model_derivatives(model_derivatives, extended_subs)
        last_mark = self._mark(timings, 'prepare', last_mark)

        # â”€â”€ Stage 3: SET Assembly (unchanged from old Stage 5) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        self.emit('progress', stage=3, label='Assembling stress-energy tensor', pct=50)
        self.check_cancelled()

        set_handler = create_stress_energy(inp.stress_tensor)

        # Spatial vector for anisotropic
        spatial_vector = None
        if inp.stress_tensor == 'anisotropic':
            spatial_vector = (
                geom.live_symbols.get('spatial_vector')
                or ctx.spatial_vector_contravariant
                or [0, 1, 0, 0]
            )

        g = geom.metric_tensor_obj
        T_SET = _assemble_set(set_handler, g, inp.theory, spatial_vector)
        last_mark = self._mark(timings, 'stress_energy', last_mark)

        # â”€â”€ Stage 4: LHS Assembly (unchanged from old Stage 6) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        self.emit('progress', stage=4, label='Assembling field equation LHS', pct=65)
        self.check_cancelled()

        LHS = _assemble_lhs_cached(inp, model_derivatives, geom, ctx, T_SET)
        last_mark = self._mark(timings, 'lhs_assembly', last_mark)

        # â”€â”€ Stage 5: Solve (merged old 7 + 8: extract components + solve + ansatz) â”€
        self.emit('progress', stage=5, label='Preparing solve inputs', pct=80)
        self.check_cancelled()

        unknowns_str = ctx.unknowns_for_set[inp.stress_tensor]
        matter_symbols = get_matter_symbols()
        unknowns = [matter_symbols.get(u, sp.Symbol(u)) for u in unknowns_str]
        notebook_solve = _use_notebook_anisotropic_solve(inp)
        if notebook_solve:
            self.emit('progress', stage=5, label='Using notebook-style anisotropic solve', pct=81)
        self.emit('progress', stage=5, label='Building ansatz substitutions', pct=82)
        if ansatz_subs is None or extended_subs is None:
            ansatz_key = (
                'ansatz', inp.background_id, ctx.independent_coord_name,
                tuple(sorted(inp.ansatz.items())),
                tuple(sorted(getattr(inp, 'ansatz_params', {}).items())),
            )
            ansatz_subs = get_or_compute_ansatz(
                ansatz_key,
                lambda: build_ansatz_subs(inp, ctx, _log_debug),
                _log_debug,
            )
            extended_subs = build_extended_subs(ansatz_subs, ctx.independent_coord)
        if inp.background_id == 'FRW':
            k_sym = geom.live_symbols.get('curvature_k')
            if k_sym is not None:
                extended_subs[k_sym] = sp.Integer(inp.curvature_k)
        self.emit('progress', stage=5, label='Extracting field-equation components', pct=85)
        def _component_progress(label, pct=85):
            self.emit('progress', stage=5, label=label, pct=pct)

        light_reduced = _use_light_reduced_equations(inp)
        if notebook_solve:
            index_pairs, lhs_comps, rhs_comps, equations = _get_raw_equations_cached(
                inp, LHS, T_SET, ctx, geom,
                light=True,
                progress_callback=_component_progress,
            )
            last_mark = self._mark(timings, 'raw_equations', last_mark)
        else:
            index_pairs, lhs_comps, rhs_comps, equations = _get_reduced_equations_cached(
                inp, LHS, T_SET, ctx, geom, extended_subs,
                light=light_reduced,
                progress_callback=_component_progress,
            )
            last_mark = self._mark(timings, 'reduced_equations', last_mark)
        self.emit('progress', stage=5, label='Checking solve strategy', pct=88)

        # â”€â”€ fRTLm Non-linear Model Detection and Early-Exit â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        if inp.theory == 'fRTLm':
            from core.theories.fRTLm import classify_model
            
            # Reconstruct the model expression for classification
            R_sym = sp.Symbol('R_sym')
            Ts_sym = sp.Symbol('T_scalar_sym')
            L_sym = sp.Symbol('L_sym')
            local_map = {
                'R': R_sym, 'T_scalar': Ts_sym, 'T_mat': Ts_sym, 'L': L_sym,
                'exp': sp.exp, 'log': sp.log,
                'sin': sp.sin, 'cos': sp.cos, 'sqrt': sp.sqrt,
                'pi': sp.pi,
            }
            # Auto-detect parameters in model expression
            import re as _re
            tokens = set(_re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', inp.model_expr))
            reserved = {'R', 'T_scalar', 'T_mat', 'L', 'exp', 'log', 'sin', 'cos', 'sqrt', 'pi'}
            for tok in tokens:
                if tok not in reserved and tok not in local_map:
                    local_map[tok] = sp.Symbol(tok)
            
            model_expr_sym = sp.sympify(inp.model_expr, locals=local_map)
            model_class = classify_model(model_expr_sym, R_sym, Ts_sym, L_sym)
            
            if model_class.is_nonlinear:
                _log_debug(f"[PIPELINE] Non-linear fRTLm model detected: {model_class.reason}")
                self.emit('progress', stage=5, label='Non-linear coupling detected - exporting field equations', pct=89)
                
                # Export equations with ansatz fully substituted
                exported_equations = []
                residual_exprs = []
                for (i_str, j_str), lhs_comp in zip(index_pairs, lhs_comps):
                    lhs_sub = lhs_comp.subs(extended_subs)
                    # F5 fix: use a distinct iteration variable 'rhs_c' to avoid the
                    # name clash where the outer loop variable 'r' shadowed the
                    # generator's bound variable, causing all RHSs to be identical.
                    rhs_comp = next(
                        rhs_c
                        for (is_, js), rhs_c in zip(index_pairs, rhs_comps)
                        if is_ == i_str and js == j_str
                    )
                    rhs_sub = rhs_comp.subs(extended_subs)

                    lhs_sub = sp.cancel(lhs_sub)
                    rhs_sub = sp.cancel(rhs_sub)
                    lhs_sub = sp.powsimp(lhs_sub, force=True)
                    rhs_sub = sp.powsimp(rhs_sub, force=True)
                    residual = sp.powsimp(sp.cancel(lhs_sub - rhs_sub), force=True)
                    residual_exprs.append(residual)

                    exported_equations.append({
                        'index': f"({i_str},{j_str})",
                        'lhs_latex': sp.latex(lhs_sub),
                        'rhs_latex': sp.latex(rhs_sub),
                        'equation_latex': f"{sp.latex(lhs_sub)} = {sp.latex(rhs_sub)}",
                        'residual': str(residual),
                    })

                results.exported_equations = exported_equations
                coord_symbol = ctx.independent_coord
                coord_name_for_numeric = str(coord_symbol) if coord_symbol is not None else ctx.independent_coord_name
                unknown_names = [str(u) for u in unknowns]
                reserved_numeric = {coord_name_for_numeric, *unknown_names}
                param_names = sorted({
                    str(sym)
                    for expr in residual_exprs
                    for sym in getattr(expr, 'free_symbols', set())
                    if str(sym) not in reserved_numeric
                })
                defaults = {}
                defaults.update({str(k): str(v) for k, v in getattr(inp, 'model_params', {}).items() if v not in (None, '')})
                defaults.update({str(k): str(v) for k, v in getattr(inp, 'ansatz_params', {}).items() if v not in (None, '')})
                results.numeric_solve = {
                    'available': True,
                    'mode': 'pointwise_residual_continuation',
                    'theory': inp.theory,
                    'background_id': inp.background_id,
                    'stress_tensor': inp.stress_tensor,
                    'metric_functions': dict(inp.ansatz or {}),
                    'variable': coord_name_for_numeric,
                    'unknowns': unknown_names,
                    'parameters': param_names,
                    'parameter_defaults': {k: defaults[k] for k in param_names if k in defaults},
                    'residuals': [str(expr) for expr in residual_exprs],
                    'residual_labels': [eq['index'] for eq in exported_equations],
                }
                results.early_exit = True
                results.early_exit_reason = model_class.reason

                _fill_scalar_results(results, inp, geom, extended_subs)
                total = time.perf_counter() - t0
                timings['total'] = total
                results.timings = timings
                _log_debug(f"[TIMING] total: {total:.3f}s")
                self.emit('progress', stage=6, label='Rendering results and formatting equations', pct=99)
                self.emit('complete', stage=6, results=self._results_to_dict(results))
                return results
        # â”€â”€ End Non-linear Detection â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

        # Filter out trivial equations (False/True = already satisfied or contradictory)
        valid_equations = []
        for eq in equations:
            # Check if eq is a boolean (True/False) rather than an Eq object.
            # Use hasattr to check for Eq attributes as a more robust test.
            if not hasattr(eq, 'lhs') or not hasattr(eq, 'rhs'):
                # Skip trivial identities and contradictions (no .lhs/.rhs means it's not an Eq).
                _log_debug(f"[PIPELINE] Skipping trivial equation (no Eq attributes): {eq}")
                continue
            # F7 fix: the original str(type(eq)) branch was dead code (Python booleans
            # are already caught above by the hasattr guard).  The intended check is for
            # SymPy BooleanTrue/BooleanFalse, which *do* have .lhs/.rhs on some versions.
            if isinstance(eq, (sp.logic.boolalg.BooleanTrue, sp.logic.boolalg.BooleanFalse)):
                _log_debug(f"[PIPELINE] Skipping SymPy boolean equation: {eq}")
                continue
            try:
                if _is_trivial_equation(eq, inp):
                    _log_debug("[PIPELINE] Skipping identity equation with zero residual")
                    continue
            except Exception:
                pass
            valid_equations.append(eq)

        if not valid_equations:
            raise ValueError("No independent field equations to solve - all equations are trivially satisfied. "
                             "This may indicate the model is under-constrained or has no dynamics.")

        # Bug C fix: ensure we have at least as many non-trivial equations as unknowns.
        # If filtering dropped equations (e.g. a component reduced to True), the zip
        # inside _solve_component_linear_fast would silently pair equations to the wrong
        # unknowns, producing rho = P_r = P_t.
        if notebook_solve and len(valid_equations) < len(unknowns):
            raise ValueError(
                f"Anisotropic solve requires {len(unknowns)} independent field equations "
                f"but only {len(valid_equations)} remain after filtering trivial identities. "
                f"Check that the field equations for index pairs "
                f"{ctx.canonical_index_pairs.get(inp.stress_tensor, '?')} are all non-trivial "
                f"for this model."
            )

        solver = FieldEquationSolver(ctx)
        solve_label = 'Solving anisotropic matter with direct linsolve' if notebook_solve else 'Solving matter variables'
        self.emit('progress', stage=5, label=solve_label, pct=90)
        solutions_are_final = False
        try:
            direct_solutions = _try_direct_diagonal_rhs_solve(inp, valid_equations, unknowns, extended_subs)
            if direct_solutions is not None:
                solutions = direct_solutions
                solutions_are_final = True
            elif notebook_solve:
                solutions = solver.solve_anisotropic_notebook(valid_equations, unknowns)
            else:
                solutions = solver.solve_sequential(valid_equations, unknowns)
        except SolveError as e:
            raise ValueError(f"Failed to solve field equations: {e}")

        # Apply ansatz to solutions (returns tuple with derivatives for deferred diff).
        self.emit('progress', stage=5, label='Applying ansatz to matter solution', pct=94)
        def _ansatz_progress(sym, phase):
            self.emit(
                'progress',
                stage=5,
                label=f'Applying ansatz: {sym} {phase}',
                pct=94,
            )
        direct_blackhole_final = solutions_are_final and inp.background_id == 'SS_blackhole'
        if solutions_are_final and (not inp.compute_stability or direct_blackhole_final):
            final_solutions = solutions
            matter_derivatives = None
        else:
            final_solutions, matter_derivatives = solver.apply_ansatz(
                solutions,
                ansatz_subs,
                compute_derivatives=inp.compute_stability,
                progress_callback=_ansatz_progress,
            )
        results.matter_derivatives = matter_derivatives  # Store for deferred diff in speed of sound
        last_mark = self._mark(timings, 'solve_and_ansatz', last_mark)

        solution_names = {str(key) for key in final_solutions}
        missing = [u for u in unknowns if u not in final_solutions and str(u) not in solution_names]
        if missing:
            raise ValueError(f"Incomplete matter solution. Missing: {[str(u) for u in missing]}")

        # Map solutions back to result fields
        unknown_map = {str(u): u for u in unknowns}
        def add_warning(message):
            if results.warnings is None:
                results.warnings = []
            results.warnings.append(message)

        _fill_matter_results(results, inp.stress_tensor, final_solutions, unknown_map, add_warning=add_warning)

        # Determine Pr, Pt for derived quantities
        Pr, Pt = _get_pressures(results, inp.stress_tensor)

        # Theory scalars (ansatz-substituted)
        self.emit('progress', stage=5, label='Preparing scalar outputs', pct=96)
        _fill_scalar_results(results, inp, geom, extended_subs)

        # Diagnostic short-circuit: if finalized matter variables vanish identically,
        # derive diagnostics from that vacuum branch directly instead of reusing any
        # older symbolic path.
        zero_matter_branch = (
            results.rho == sp.S.Zero and
            Pr == sp.S.Zero and
            Pt == sp.S.Zero
        )

        if zero_matter_branch:
            # All matter is identically zero (vacuum solution, e.g. Schwarzschild star).
            # Short-circuit every diagnostic: EoS ratios are undefined (0/0 â†’ NaN),
            # energy conditions are trivially satisfied at zero, and speed-of-sound
            # is meaningless without matter.
            if inp.compute_eos:
                results.omega_r = None
                if inp.stress_tensor == 'anisotropic':
                    results.omega_t = None
                results.omega_eff = None
                timings['diagnostics_eos'] = 0.0

            if inp.compute_energy_conditions:
                results.NEC_r = sp.S.Zero
                results.NEC_t = sp.S.Zero
                results.WEC   = sp.S.Zero
                results.SEC   = sp.S.Zero
                results.DEC_r = sp.S.Zero
                results.DEC_t = sp.S.Zero
                timings['diagnostics_energy_conditions'] = 0.0

            if inp.compute_stability:
                results.cs2_r = None
                if inp.stress_tensor == 'anisotropic':
                    results.cs2_t = None
                timings['diagnostics_stability'] = 0.0

            if inp.compute_tov:
                timings['diagnostics_tov'] = 0.0

        if (not zero_matter_branch) and inp.compute_eos:
            t_diag = time.perf_counter()
            self.emit('progress', stage=5, label='Computing equation-of-state diagnostics', pct=97)
            eos = solver.compute_eos(results.rho, Pr, Pt, lazy=False)
            results.omega_r = eos['omega_r']
            if inp.stress_tensor == 'anisotropic':
                results.omega_t = eos['omega_t']
            results.omega_eff = eos['omega_eff']
            timings['diagnostics_eos'] = time.perf_counter() - t_diag

        if (not zero_matter_branch) and inp.compute_energy_conditions:
            t_diag = time.perf_counter()
            self.emit('progress', stage=5, label='Computing energy conditions', pct=98)
            ec = solver.compute_energy_conditions(results.rho, Pr, Pt, lazy=False)
            results.NEC_r = ec['NEC_r']
            results.NEC_t = ec['NEC_t']
            results.WEC   = ec['WEC']
            results.SEC   = ec['SEC']
            results.DEC_r = ec['DEC_r']
            results.DEC_t = ec['DEC_t']
            timings['diagnostics_energy_conditions'] = time.perf_counter() - t_diag

        if (not zero_matter_branch) and inp.compute_stability:
            t_diag = time.perf_counter()
            self.emit('progress', stage=5, label='Computing stability diagnostics', pct=98)
            if inp.background_id == 'SS_blackhole':
                cs2 = _compute_blackhole_stability_light(results.rho, Pr, Pt, ctx.independent_coord)
            else:
                cs2 = solver.compute_speed_of_sound(
                    results.rho, Pr, Pt, ctx.independent_coord, matter_derivatives, lazy=False
                )
            results.cs2_r = cs2['cs2_r']
            if inp.stress_tensor == 'anisotropic':
                results.cs2_t = cs2['cs2_t']
            timings['diagnostics_stability'] = time.perf_counter() - t_diag

        if (not zero_matter_branch) and inp.compute_tov:
            from core.diagnostics import compute_tov_diagnostics, supports_tov
            if supports_tov(inp.background_id, inp.stress_tensor):
                t_diag = time.perf_counter()
                self.emit('progress', stage=5, label='Computing TOV equilibrium analysis', pct=98)
                tov = compute_tov_diagnostics(
                    background_id=inp.background_id,
                    stress_tensor=inp.stress_tensor,
                    rho=results.rho,
                    radial_pressure=Pr,
                    tangential_pressure=Pt,
                    geom=geom,
                    extended_subs=extended_subs,
                    independent_coord=ctx.independent_coord,
                )
                results.tov_mass = tov.get('mass')
                results.tov_compactness = tov.get('compactness')
                results.tov_redshift_gradient = tov.get('redshift_gradient')
                results.tov_pressure_gradient = tov.get('pressure_gradient')
                results.tov_hydrostatic_force = tov.get('hydrostatic_force')
                results.tov_gravitational_force = tov.get('gravitational_force')
                results.tov_anisotropic_force = tov.get('anisotropic_force')
                results.tov_residual = tov.get('tov_residual')
                results.tov_mass_continuity_residual = tov.get('mass_continuity_residual')
                timings['diagnostics_tov'] = time.perf_counter() - t_diag

        # â”€â”€ Complete â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        total = time.perf_counter() - t0
        timings['total'] = total
        results.timings = timings
        results.plot_data = _build_plot_data(results, inp, ctx)
        _log_debug(f"[TIMING] total: {total:.3f}s")
        self.emit('progress', stage=6, label='Rendering results and formatting equations', pct=99)
        self.emit('complete', stage=6, results=self._results_to_dict(results))
        return results

    def _results_to_dict(self, results: PipelineResults) -> Dict:
        return results_to_dict(results)
# â”€â”€â”€ Helper functions â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _build_plot_data(results: PipelineResults, inp: PipelineInput, ctx) -> Dict[str, Any]:
    """Build raw expression payloads for post-solve numerical plotting."""
    import re

    variable = ctx.independent_coord_name or str(ctx.independent_coord or 't')
    is_anisotropic = inp.stress_tensor == 'anisotropic'
    groups: Dict[str, Dict[str, str]] = {
        'matter': {},
        'energy_conditions': {},
        'eos': {},
        'stability': {},
        'tov': {},
    }

    def add(group: str, key: str, value: Any) -> None:
        if value is None:
            return
        try:
            if hasattr(value, 'evaluate'):
                value = value.evaluate()
        except Exception:
            return
        if value is not None:
            groups[group][key] = str(value)

    add('matter', 'rho', results.rho)
    if is_anisotropic:
        add('matter', 'P_r', results.Pr)
        add('matter', 'P_t', results.Pt)
    else:
        add('matter', 'p', results.p if results.p is not None else results.Pr)

    if inp.compute_energy_conditions:
        if is_anisotropic:
            add('energy_conditions', 'NEC_r', results.NEC_r)
            add('energy_conditions', 'NEC_t', results.NEC_t)
            add('energy_conditions', 'DEC_r', results.DEC_r)
            add('energy_conditions', 'DEC_t', results.DEC_t)
        else:
            add('energy_conditions', 'NEC', results.NEC_r)
            add('energy_conditions', 'DEC', results.DEC_r)
        add('energy_conditions', 'WEC', results.WEC)
        add('energy_conditions', 'SEC', results.SEC)

    if inp.compute_eos:
        if is_anisotropic:
            add('eos', 'omega_r', results.omega_r)
            add('eos', 'omega_t', results.omega_t)
            add('eos', 'omega_eff', results.omega_eff)
        else:
            add('eos', 'omega', results.omega_r if results.omega_r is not None else results.omega_eff)

    if inp.compute_stability:
        if is_anisotropic:
            add('stability', 'cs2_r', results.cs2_r)
            add('stability', 'cs2_t', results.cs2_t)
        else:
            add('stability', 'cs2', results.cs2_r)

    if inp.compute_tov:
        add('tov', 'hydrostatic_force', results.tov_hydrostatic_force)
        add('tov', 'gravitational_force', results.tov_gravitational_force)
        add('tov', 'anisotropic_force', results.tov_anisotropic_force)
        add('tov', 'residual', results.tov_residual)

    groups = {name: series for name, series in groups.items() if series}
    if not groups:
        return {'available': False}

    function_names = {
        'exp', 'log', 'sin', 'cos', 'tan',
        'sinh', 'cosh', 'tanh', 'sqrt', 'Abs',
        'pi', 'E',
    }
    local_map = {
        'exp': sp.exp,
        'log': sp.log,
        'sin': sp.sin,
        'cos': sp.cos,
        'tan': sp.tan,
        'sinh': sp.sinh,
        'cosh': sp.cosh,
        'tanh': sp.tanh,
        'sqrt': sp.sqrt,
        'Abs': sp.Abs,
        'pi': sp.pi,
        'E': sp.E,
        variable: sp.Symbol(variable),
    }
    expr_strings = [
        expr_str
        for series in groups.values()
        for expr_str in series.values()
    ]
    for expr_str in expr_strings:
        for token in re.findall(r'[A-Za-z_][A-Za-z0-9_]*', str(expr_str)):
            if token not in function_names and token not in local_map:
                local_map[token] = sp.Symbol(token)

    free_symbols = set()
    for series in groups.values():
        for expr_str in series.values():
            try:
                expr = sp.sympify(expr_str, locals=local_map)
                free_symbols.update(str(sym) for sym in expr.free_symbols)
            except Exception:
                pass
    parameters = sorted(name for name in free_symbols if name != variable)
    defaults = {}
    defaults.update({str(k): str(v) for k, v in getattr(inp, 'model_params', {}).items() if v not in (None, '')})
    defaults.update({str(k): str(v) for k, v in getattr(inp, 'ansatz_params', {}).items() if v not in (None, '')})
    if inp.background_id == 'FRW':
        defaults['k'] = str(inp.curvature_k)

    return {
        'available': True,
        'background_id': inp.background_id,
        'stress_tensor': inp.stress_tensor,
        'variable': variable,
        'groups': groups,
        'parameters': parameters,
        'parameter_defaults': {k: defaults[k] for k in parameters if k in defaults},
    }


def _reinit_metric(pt, background_id, curvature_k, geom):
    """Re-declare coordinates and metric in the current pytearcat session."""
    ls = geom.live_symbols
    if background_id == 'FRW':
        t, r, th, ph = pt.coords('t,r,theta,phi')
        a = pt.fun('a', 't')
        pt.con('k')
        _frw_c_str = 'ds2 = -dt**2 + a**2/(1 - k*r**2)*dr**2 + a**2*r**2*dtheta**2 + a**2*r**2*sin(theta)**2*dphi**2'
        g = pt.metric(_frw_c_str)
        g.complete('_,_')
        pt.christoffel()
    elif background_id == 'Bianchi_I':
        t, x, y, z = pt.coords('t,x,y,z')
        A = pt.fun('A', 't');  B = pt.fun('B', 't')
        g = pt.metric('ds2 = -dt**2 + A**2*dx**2 + B**2*dy**2 + B**2*dz**2')
        g.complete('_,_');  pt.christoffel()
    elif background_id == 'Bianchi_III':
        t, x, y, z = pt.coords('t,x,y,z')
        A = pt.fun('A', 't');  B = pt.fun('B', 't')
        g = pt.metric('ds2 = -dt**2 + A**2*dx**2 + exp(2*x)*(B**2*dy**2 + B**2*dz**2)')
        g.complete('_,_');  pt.christoffel()
    elif background_id == 'Kantowski_Sachs':
        t, r, th, ph = pt.coords('t,r,theta,phi')
        A = pt.fun('A', 't');  B = pt.fun('B', 't')
        g = pt.metric('ds2 = -dt**2 + A**2*dr**2 + B**2*dtheta**2 + B**2*sin(theta)**2*dphi**2')
        g.complete('_,_');  pt.christoffel()
    elif background_id == 'SS_wormhole':
        t, r, th, ph = pt.coords('t,r,theta,phi')
        b = pt.fun('b', 'r');  Phi = pt.fun('Phi', 'r')
        g = pt.metric('ds2 = -exp(2*Phi)*dt**2 + 1/(1-b/r)*dr**2 + r**2*dtheta**2 + r**2*sin(theta)**2*dphi**2')
        g.complete('_,_');  pt.christoffel()
    elif background_id == 'SS_blackhole':
        t, r, th, ph = pt.coords('t,r,theta,phi')
        variant = (ls or {}).get('_blackhole_metric_variant', 'generic')
        if variant == 'schwarzschild':
            pt.con('M')
            g = pt.metric('ds2 = -(1 - 2*M/r)*dt**2 + 1/(1 - 2*M/r)*dr**2 + r**2*dtheta**2 + r**2*sin(theta)**2*dphi**2')
        elif variant == 'reissner_nordstrom':
            pt.con('M');  pt.con('Q')
            g = pt.metric('ds2 = -(1 - 2*M/r + Q**2/r**2)*dt**2 + 1/(1 - 2*M/r + Q**2/r**2)*dr**2 + r**2*dtheta**2 + r**2*sin(theta)**2*dphi**2')
        else:
            nu_bh = pt.fun('nu_bh', 'r');  lam_bh = pt.fun('lam_bh', 'r')
            g = pt.metric('ds2 = -nu_bh*dt**2 + lam_bh*dr**2 + r**2*dtheta**2 + r**2*sin(theta)**2*dphi**2')
        g.complete('_,_');  pt.christoffel()


def _cleanup_pytearcat_tensors(pt, geom=None):
    """Clean up tensor names from pytearcat's internal registry to prevent collisions."""
    global _pt_tensor_registry
    try:
        # Merge global registry with geometry cache tensor names
        all_tensors = set(_pt_tensor_registry)
        if geom is not None and hasattr(geom, 'tensor_names'):
            all_tensors.update(geom.tensor_names)
        
        # Clean up tensors from pytearcat's internal _tensors dict if it exists
        if hasattr(pt, '_tensors'):
            for name in all_tensors:
                if name in pt._tensors:
                    del pt._tensors[name]
                    _log_debug(f"[PIPELINE] Cleaned up tensor: {name}")
        _pt_tensor_registry.clear()
        _log_debug(f"[PIPELINE] Pytearcat tensor registry cleaned ({len(all_tensors)} tensors)")
    except Exception as e:
        _log_debug(f"[PIPELINE] Tensor cleanup warning: {e}")



def _coerce_numeric_params(raw_params) -> Dict[str, sp.Expr]:
    values = {}
    if not isinstance(raw_params, dict):
        return values
    for key, value in raw_params.items():
        if value in (None, ''):
            continue
        try:
            values[str(key)] = sp.nsimplify(value, rational=True)
        except Exception:
            try:
                values[str(key)] = sp.sympify(value)
            except Exception:
                continue
    return values


def _normalise_metric_text(expr: Any) -> str:
    return str(expr or '').replace(' ', '').replace('^', '**')


def _blackhole_metric_variant_from_ansatz(inp: PipelineInput) -> str | None:
    """Detect standard black-hole presets that can be built as direct metrics."""
    if inp.background_id != 'SS_blackhole':
        return None
    if inp.theory in {'fT', 'fTB'}:
        return 'generic'

    nu_expr = _normalise_metric_text((inp.ansatz or {}).get('nu_bh'))
    lam_expr = _normalise_metric_text((inp.ansatz or {}).get('lam_bh'))

    schwarzschild_f = '1-2*M/r'
    rn_f = '1-2*M/r+Q**2/r**2'
    if nu_expr == schwarzschild_f and lam_expr == f'1/({schwarzschild_f})':
        return 'schwarzschild'
    if nu_expr == rn_f and lam_expr == f'1/({rn_f})':
        return 'reissner_nordstrom'
    return 'generic'


def _sympify_model_expr(expr_str: str, local_map: Dict[str, sp.Expr], reserved: set[str], model_params) -> sp.Expr:
    import re as _re

    param_values = _coerce_numeric_params(model_params)
    for tok in set(_re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', expr_str)):
        if tok in param_values:
            local_map[tok] = param_values[tok]
        elif tok not in reserved and tok not in local_map:
            local_map[tok] = sp.Symbol(tok)
    return sp.sympify(expr_str, locals=local_map)


def _compute_model_derivatives(inp: PipelineInput, geom) -> Dict:
    """Compute theory-specific model derivatives. Returns a dict of actuals."""
    from core.solver import safe_simplify
    R_sym  = sp.Symbol('R_sym')
    T_sym  = sp.Symbol('T_sym')
    B_sym  = sp.Symbol('B_sym')
    Ts_sym = sp.Symbol('Ts_sym')
    L_sym  = sp.Symbol('L_sym')

    theory = inp.theory
    expr_str = inp.model_expr

    def scalar_simplify(expr, name):
        try:
            return safe_simplify(expr)
        except Exception:
            return expr

    if theory == 'fR':
        from core.theories.fR import compute_model_derivatives
        local_map = {
            'R': R_sym,
            'exp': sp.exp, 'log': sp.log,
            'sin': sp.sin, 'cos': sp.cos, 'tan': sp.tan,
            'sinh': sp.sinh, 'cosh': sp.cosh, 'tanh': sp.tanh,
            'sqrt': sp.sqrt, 'pi': sp.pi, 'E': sp.E,
        }
        # Auto-detect other symbols (alpha, etc.) and add them to local_map
        reserved = {'R', 'exp', 'log', 'sin', 'cos', 'tan', 'sinh', 'cosh', 'tanh', 'sqrt', 'pi', 'E'}
        f_dum = _sympify_model_expr(expr_str, local_map, reserved, inp.model_params)
        f, fR = compute_model_derivatives(f_dum, R_sym)
        R_actual  = scalar_simplify(geom.ricci_scalar, 'R')
        f_act     = f.subs(R_sym,  R_actual)
        fR_act    = fR.subs(R_sym, R_actual)
        return {'f': f_act, 'fR': fR_act}

    elif theory == 'fT':
        from core.theories.fT import compute_model_derivatives
        # Use dummy symbol with name 'T' to avoid pytearcat's T in global namespace
        T_dummy = sp.Symbol('T')
        local_map = {
            'T': T_dummy,
            'exp': sp.exp, 'log': sp.log,
            'sin': sp.sin, 'cos': sp.cos, 'tan': sp.tan,
            'sinh': sp.sinh, 'cosh': sp.cosh, 'tanh': sp.tanh,
            'sqrt': sp.sqrt, 'pi': sp.pi, 'E': sp.E,
        }
        # Auto-detect other symbols (alpha, etc.) and add them to local_map
        reserved = {'T', 'exp', 'log', 'sin', 'cos', 'tan', 'sinh', 'cosh', 'tanh', 'sqrt', 'pi', 'E'}
        f_dum = _sympify_model_expr(expr_str, local_map, reserved, inp.model_params)
        f, fT, fTT = compute_model_derivatives(f_dum, T_dummy)
        T_actual   = scalar_simplify(geom.T_scalar_expr, 'T')
        f_act      = f.subs(T_dummy, T_actual)
        fT_act     = fT.subs(T_dummy, T_actual)
        fTT_act    = fTT.subs(T_dummy, T_actual)
        return {'f': f_act, 'fT': fT_act, 'fTT': fTT_act, 'T_actual': T_actual}

    elif theory == 'fTB':
        from core.theories.fTB import compute_model_derivatives
        # Use dummy symbols with names matching the user's expression to avoid global conflicts
        T_dummy = sp.Symbol('T')
        B_dummy = sp.Symbol('B')
        # Parse with custom symbol mapping to avoid pytearcat's T/B in global namespace
        local_map = {
            'T': T_dummy, 'B': B_dummy,
            'exp': sp.exp, 'log': sp.log,
            'sin': sp.sin, 'cos': sp.cos, 'tan': sp.tan,
            'sinh': sp.sinh, 'cosh': sp.cosh, 'tanh': sp.tanh,
            'sqrt': sp.sqrt, 'pi': sp.pi, 'E': sp.E,
        }
        # Auto-detect other symbols (alpha, beta, etc.) and add them to local_map
        reserved = {'T', 'B', 'exp', 'log', 'sin', 'cos', 'tan', 'sinh', 'cosh', 'tanh', 'sqrt', 'pi', 'E'}
        f_dum = _sympify_model_expr(expr_str, local_map, reserved, inp.model_params)
        f, fT, fB = compute_model_derivatives(f_dum, T_dummy, B_dummy)
        T_actual  = scalar_simplify(geom.T_scalar_expr, 'T')
        B_actual  = scalar_simplify(geom.B_scalar_expr, 'B')
        f_act     = f.subs({T_dummy: T_actual, B_dummy: B_actual})
        fT_act    = fT.subs({T_dummy: T_actual, B_dummy: B_actual})
        fB_act    = fB.subs({T_dummy: T_actual, B_dummy: B_actual})
        return {'f': f_act, 'fT': fT_act, 'fB': fB_act,
                'T_actual': T_actual, 'B_actual': B_actual}

    elif theory == 'fRTLm':
        from core.theories.fRTLm import compute_model_derivatives
        local_map = {
            'R': R_sym, 'T_scalar': Ts_sym, 'T_mat': Ts_sym, 'L': L_sym,
            'exp': sp.exp, 'log': sp.log,
            'sin': sp.sin, 'cos': sp.cos, 'tan': sp.tan,
            'sinh': sp.sinh, 'cosh': sp.cosh, 'tanh': sp.tanh,
            'sqrt': sp.sqrt, 'pi': sp.pi, 'E': sp.E,
        }
        # Auto-detect other symbols (alpha, etc.) and add them to local_map
        reserved = {'R', 'T_scalar', 'T_mat', 'L', 'exp', 'log', 'sin', 'cos', 'tan', 'sinh', 'cosh', 'tanh', 'sqrt', 'pi', 'E'}
        f_dum = _sympify_model_expr(expr_str, local_map, reserved, inp.model_params)
        f, fR, fT, fL = compute_model_derivatives(f_dum, R_sym, Ts_sym, L_sym)
        R_actual  = scalar_simplify(geom.ricci_scalar, 'R')
        # T_scalar (SET trace) is not yet known â€” it depends on SET unknowns;
        # we substitute after SET assembly. For now pass symbolic Ts_sym.
        # F11 fix: removed dead 'Ts_sym_live = sp.Symbol("T_scalar_live")' that
        # was created here but never used (the real substitution happens in _assemble_lhs).
        f_act  = f.subs({R_sym: R_actual})
        fR_act = fR.subs({R_sym: R_actual})
        fT_act = fT.subs({R_sym: R_actual})
        fL_act = fL.subs({R_sym: R_actual})
        return {
            'f': f_act, 'fR': fR_act, 'fT': fT_act, 'fL': fL_act,
            'Ts_sym': Ts_sym, 'L_sym': L_sym,
            'R_actual': R_actual,
        }
    elif theory == 'fQ':
        from core.theories.fQ import (
            compute_model_derivatives,
            _Q_expr_for_background,
            Q_expr_from_geometry,
        )
        Q_dummy = sp.Symbol('Q_dummy')
        _reserved_fq = {'Q', 'exp', 'log', 'sin', 'cos', 'tan',
                         'sinh', 'cosh', 'tanh', 'sqrt', 'pi', 'E'}
        _local_fq = {
            'Q':    Q_dummy,
            'exp':  sp.exp, 'log': sp.log,
            'sin':  sp.sin, 'cos': sp.cos, 'tan': sp.tan,
            'sinh': sp.sinh, 'cosh': sp.cosh, 'tanh': sp.tanh,
            'sqrt': sp.sqrt, 'pi': sp.pi, 'E': sp.E,
        }
        f_dum = _sympify_model_expr(expr_str, _local_fq, _reserved_fq, inp.model_params)
        f, fQ, fQQ = compute_model_derivatives(f_dum, Q_dummy)
        # Compute actual Q from geometry.  Prefer the generic metric route so
        # non-FRW backgrounds do not leak a symbolic placeholder.
        Q_actual = scalar_simplify(Q_expr_from_geometry(geom), 'Q')
        if Q_actual == sp.Symbol('Q'):
            Q_actual = scalar_simplify(
                _Q_expr_for_background(inp.background_id, geom.live_symbols),
                'Q',
            )
        f_act   = f.subs(Q_dummy, Q_actual)
        fQ_act  = fQ.subs(Q_dummy, Q_actual)
        fQQ_act = fQQ.subs(Q_dummy, Q_actual)
        return {
            'f': f_act, 'fQ': fQ_act, 'fQQ': fQQ_act, 'Q_actual': Q_actual,
        }
    elif theory == 'fQC':
        from core.theories.fQC import compute_model_derivatives
        from core.theories.fQ import Q_expr_from_geometry, _Q_expr_for_background
        Q_dummy = sp.Symbol('Q_dummy')
        C_dummy = sp.Symbol('C_dummy')
        reserved = {'Q', 'C', 'exp', 'log', 'sin', 'cos', 'tan',
                    'sinh', 'cosh', 'tanh', 'sqrt', 'pi', 'E'}
        local_map = {
            'Q': Q_dummy, 'C': C_dummy,
            'exp': sp.exp, 'log': sp.log,
            'sin': sp.sin, 'cos': sp.cos, 'tan': sp.tan,
            'sinh': sp.sinh, 'cosh': sp.cosh, 'tanh': sp.tanh,
            'sqrt': sp.sqrt, 'pi': sp.pi, 'E': sp.E,
        }
        f_dum = _sympify_model_expr(expr_str, local_map, reserved, inp.model_params)
        f, fQ, fC, fQQ, fCC = compute_model_derivatives(f_dum, Q_dummy, C_dummy)
        Q_actual = scalar_simplify(Q_expr_from_geometry(geom), 'Q')
        if Q_actual == sp.Symbol('Q'):
            Q_actual = scalar_simplify(
                _Q_expr_for_background(inp.background_id, geom.live_symbols),
                'Q',
            )
        C_actual = scalar_simplify(geom.ricci_scalar - Q_actual, 'C')
        subs = {Q_dummy: Q_actual, C_dummy: C_actual}
        return {
            'f': f.subs(subs),
            'fQ': fQ.subs(subs),
            'fC': fC.subs(subs),
            'fQQ': fQQ.subs(subs),
            'fCC': fCC.subs(subs),
            'Q_actual': Q_actual,
            'C_actual': C_actual,
        }
    else:
        raise ValueError(f"Unknown theory: {theory!r}")


def _assemble_set(set_handler, g, theory: str, spatial_vector) -> Any:
    """Assemble the stress-energy tensor in the correct index form for the theory."""
    index_form = '^,_' if theory in ('fT', 'fTB') else '_,_'
    if spatial_vector is not None:
        return set_handler.assemble(g, spatial_vector, index_form=index_form)
    return set_handler.assemble(g, index_form=index_form)


def _assemble_lhs(inp: PipelineInput, md: Dict, geom, ctx, T_SET) -> Any:
    """Call the theory-specific LHS assembly function."""
    theory = inp.theory
    g = geom.metric_tensor_obj

    if theory == 'fR':
        from core.theories.fR import assemble_field_equations
        return assemble_field_equations(
            md['f'], md['fR'],
            geom.ricci_scalar, geom.ricci_tensor,
            g, geom, ctx
        )

    elif theory == 'fT':
        from core.theories.fT import assemble_field_equations
        return assemble_field_equations(
            md['f'], md['fT'], md['fTT'],
            md['T_actual'], geom, ctx
        )

    elif theory == 'fTB':
        from core.theories.fTB import assemble_field_equations
        return assemble_field_equations(
            md['f'], md['fT'], md['fB'],
            md['T_actual'], md['B_actual'],
            geom, ctx
        )

    elif theory == 'fRTLm':
        from core.theories.fRTLm import assemble_field_equations, get_Lm_expression
        from core.solver import safe_simplify
        # Resolve T_scalar (SET trace) and Lm
        set_handler_trace = getattr(T_SET, '_mgs_trace', None)
        # We need actual trace from the SET; approximate using handler
        rho_sym = sp.Symbol('rho', positive=True)
        p_sym   = sp.Symbol('p')
        # F2 fix: for anisotropic, _compute_set_trace uses P_r/P_t internally and
        # ignores p_sym, so Ts_live is correct. However get_Lm_expression must also
        # use the correct trace (Ts_live) for the T_mat matter Lagrangian, not bare p.
        # We pass Ts_live as the effective trace so that matter_lag='T_mat' uses the
        # actual -rho + P_r + 2*P_t expression rather than the ghost p symbol.
        Ts_live = safe_simplify(_compute_set_trace(inp.stress_tensor, rho_sym, p_sym))
        _p_for_lm = Ts_live if inp.stress_tensor == 'anisotropic' else p_sym
        Lm_live = safe_simplify(get_Lm_expression(inp.matter_lag, rho_sym, _p_for_lm, Ts_live))

        # Substitute T_scalar and L into model derivatives
        Ts_sym = md['Ts_sym']
        L_sym  = md['L_sym']
        f_act  = md['f'].subs({Ts_sym: Ts_live, L_sym: Lm_live})
        fR_act = md['fR'].subs({Ts_sym: Ts_live, L_sym: Lm_live})
        fT_act = md['fT'].subs({Ts_sym: Ts_live, L_sym: Lm_live})
        fL_act = md['fL'].subs({Ts_sym: Ts_live, L_sym: Lm_live})

        # Store Lm for scalar display later
        md['Lm_expr']  = Lm_live
        md['T_scalar_expr'] = Ts_live

        # Attach metric object to geom (needed inside fRTLm.py)
        geom.metric_tensor_obj = g

        return assemble_field_equations(
            f_act, fR_act, fT_act, fL_act,
            geom.ricci_scalar, T_SET, Lm_live,
            geom, ctx
        )

    elif theory == 'fQ':
        from core.theories.fQ import assemble_field_equations
        return assemble_field_equations(
            md['f'], md['fQ'], md['fQQ'],
            md['Q_actual'], geom, ctx
        )
    elif theory == 'fQC':
        from core.theories.fQC import assemble_field_equations
        return assemble_field_equations(
            md['f'], md['fQ'], md['fC'],
            md['Q_actual'], md['C_actual'], geom, ctx
        )

    raise ValueError(f"Unknown theory: {theory!r}")


def _assemble_lhs_cached(inp: PipelineInput, md: Dict, geom, ctx, T_SET) -> Any:
    """Cache assembled theory LHS tensors for repeated identical runs."""
    key = (
        'lhs_v3_nonmetricity_pre_ansatz_only', inp.theory, inp.background_id, inp.curvature_k, inp.stress_tensor,
        inp.model_expr, tuple(sorted(inp.model_params.items())),
        tuple(sorted(inp.ansatz.items())), tuple(sorted(getattr(inp, 'ansatz_params', {}).items())),
        inp.matter_lag,
    )
    return get_or_compute_lhs(
        key,
        lambda: _assemble_lhs(inp, md, geom, ctx, T_SET),
        _log_debug,
    )


def _use_light_scalar_outputs(inp: PipelineInput) -> bool:
    """Keep scalar display cheap for solved heavy anisotropic geometry runs."""
    return (
        inp.theory == 'fRTLm'
        and inp.stress_tensor == 'anisotropic'
        and inp.background_id in ('Bianchi_I', 'Bianchi_III', 'Kantowski_Sachs')
    )


def _use_notebook_anisotropic_solve(inp: PipelineInput) -> bool:
    """Prefer direct linsolve for all anisotropic systems."""
    return inp.stress_tensor == 'anisotropic'


def _use_light_reduced_equations(inp: PipelineInput) -> bool:
    """Avoid pre-solve reduction passes that explode symbolic-power models."""
    if inp.theory not in ('fQ', 'fQC'):
        return False
    try:
        locals_map = {
            name: sp.Symbol(name)
            for name in ('Q', 'C', 'R', 'T', 'B', 'alpha', 'beta', 'gamma', 'n', 'm')
        }
        model = sp.sympify(str(inp.model_expr).replace('^', '**'), locals=locals_map)
        return any(pow_expr.exp.free_symbols for pow_expr in model.atoms(sp.Pow))
    except Exception:
        model_text = str(inp.model_expr).replace(' ', '')
        return '**n' in model_text or '^n' in model_text


def _compute_set_trace(stress_tensor: str, rho: sp.Symbol, p: sp.Symbol) -> sp.Expr:
    """Return symbolic SET trace for each SET type (covariant form, -+++ signature)."""
    if stress_tensor == 'perfect_fluid':
        return -rho + 3 * p
    elif stress_tensor == 'dust':
        return -rho
    elif stress_tensor == 'radiation':
        return sp.Integer(0)
    elif stress_tensor == 'vacuum':
        Lambda = sp.Symbol('Lambda')
        return -4 * Lambda  # T = g^{Î¼Î½}(-Î›g_{Î¼Î½}) = -Î›Â·4 = -4Î›  (signature -,+,+,+)
    elif stress_tensor == 'anisotropic':
        P_r = sp.Symbol('P_r')
        P_t = sp.Symbol('P_t')
        return -rho + P_r + 2 * P_t
    return sp.Integer(0)


def _use_pre_ansatz_model_derivatives(inp: PipelineInput) -> bool:
    """Substitute fixed black-hole presets before assembling hard LHS tensors."""
    # This is a win for nonmetricity, especially logarithmic f(Q,C), because
    # Q/C model derivatives collapse before the Hessian is built. It is not a
    # win for torsion theories: pre-substituting Schwarzschild into T/B before
    # teleparallel tensor assembly can expand the LHS dramatically.
    return inp.background_id == 'SS_blackhole' and inp.theory in {'fQ', 'fQC'}


def _apply_pre_ansatz_to_model_derivatives(md: Dict, extended_subs: Dict) -> Dict:
    """Apply metric ansatz to model derivative actuals before LHS assembly."""
    if not extended_subs:
        return md

    cleaned = {}
    try:
        from core.solver import _evaluate_derivative_atoms_locally, _blackhole_post_ansatz_cleanup
    except Exception:
        _evaluate_derivative_atoms_locally = None
        _blackhole_post_ansatz_cleanup = None

    for key, value in md.items():
        if not isinstance(value, sp.Expr):
            cleaned[key] = value
            continue
        try:
            expr = value.subs(extended_subs)
            if _evaluate_derivative_atoms_locally is not None:
                expr = _evaluate_derivative_atoms_locally(expr)
            else:
                expr = expr.doit()
            if _blackhole_post_ansatz_cleanup is not None:
                expr = _blackhole_post_ansatz_cleanup(expr)
            else:
                expr = sp.powsimp(expr, force=True)
            cleaned[key] = expr
        except Exception:
            cleaned[key] = value
    return cleaned


def _extract_components(inp, LHS, T_SET, index_pairs, ctx):
    cached_lhs = []
    cached_rhs = []
    missing_pairs = []
    for pair in index_pairs:
        key = _component_cache_key(inp, pair)
        cached = get_component_cache(key)
        if cached is None:
            missing_pairs.append(pair)
            cached_lhs.append(None)
            cached_rhs.append(None)
        else:
            lhs_comp, rhs_comp = cached
            cached_lhs.append(lhs_comp)
            cached_rhs.append(rhs_comp)

    if not missing_pairs:
        _log_debug(f"[CACHE] Component cache hit: {inp.theory}/{inp.background_id}/{inp.stress_tensor}")
        return cached_lhs, cached_rhs

    theory = inp.theory
    if theory == 'fR':
        from core.theories.fR import extract_components
    elif theory == 'fT':
        from core.theories.fT import extract_components
    elif theory == 'fTB':
        from core.theories.fTB import extract_components
    elif theory == 'fRTLm':
        from core.theories.fRTLm import extract_components
    elif theory == 'fQ':
        from core.theories.fQ import extract_components
    elif theory == 'fQC':
        from core.theories.fQC import extract_components
    else:
        raise ValueError(f"Unknown theory: {theory!r}")
    lhs_comps, rhs_comps = extract_components(LHS, T_SET, missing_pairs, ctx)

    by_pair = {
        pair: (lhs, rhs)
        for pair, lhs, rhs in zip(missing_pairs, lhs_comps, rhs_comps)
    }
    for pair, value in by_pair.items():
        set_component_cache(_component_cache_key(inp, pair), value, _log_debug)

    final_lhs = []
    final_rhs = []
    for pair, lhs_cached, rhs_cached in zip(index_pairs, cached_lhs, cached_rhs):
        if lhs_cached is not None:
            final_lhs.append(lhs_cached)
            final_rhs.append(rhs_cached)
        else:
            lhs, rhs = by_pair[pair]
            final_lhs.append(lhs)
            final_rhs.append(rhs)
    return final_lhs, final_rhs


def _component_cache_key(inp, pair):
    index_form = '^,_' if inp.theory in ('fT', 'fTB') else '_,_'
    return (
        'component_v11_blackhole_light_teleparallel', inp.theory, inp.background_id, inp.curvature_k,
        inp.stress_tensor, inp.model_expr, tuple(sorted(inp.model_params.items())),
        tuple(sorted(inp.ansatz.items())), tuple(sorted(getattr(inp, 'ansatz_params', {}).items())),
        inp.matter_lag, index_form, tuple(pair),
    )


def _get_reduced_equations_cached(
    inp, LHS, T_SET, ctx, geom, extended_subs,
    light=False, progress_callback=None,
):
    """Extract and lightly reduce diagonal equations with a per-process cache."""
    key = (
        'reduced_v10_blackhole_light_teleparallel', inp.theory, inp.background_id, inp.curvature_k,
        inp.stress_tensor, inp.model_expr, tuple(sorted(inp.model_params.items())),
        tuple(sorted(inp.ansatz.items())), tuple(sorted(getattr(inp, 'ansatz_params', {}).items())),
        inp.matter_lag, bool(light),
    )

    def compute():
        index_pairs = ctx.canonical_index_pairs[inp.stress_tensor]
        if progress_callback:
            progress_callback('Extracting and simplifying diagonal tensor components', 85)
        lhs_comps, rhs_comps = _extract_components(inp, LHS, T_SET, index_pairs, ctx)
        return _reduce_component_equations(
            index_pairs, lhs_comps, rhs_comps, extended_subs,
            light=light,
            progress_callback=progress_callback,
        )

    return get_or_compute_reduced(key, compute, _log_debug)


def _get_raw_equations_cached(
    inp, LHS, T_SET, ctx, geom,
    light=False, progress_callback=None,
):
    """Extract raw diagonal equations for solve-first hard anisotropic runs."""
    key = (
        'raw_equations_v11_blackhole_light_teleparallel', inp.theory, inp.background_id, inp.curvature_k,
        inp.stress_tensor, inp.model_expr, tuple(sorted(inp.model_params.items())),
        tuple(sorted(inp.ansatz.items())), tuple(sorted(getattr(inp, 'ansatz_params', {}).items())),
        inp.matter_lag, bool(light),
    )

    def compute():
        index_pairs = ctx.canonical_index_pairs[inp.stress_tensor]
        if progress_callback:
            progress_callback('Extracting and simplifying diagonal tensor components', 85)
        lhs_comps, rhs_comps = _extract_components(inp, LHS, T_SET, index_pairs, ctx)
        return _raw_component_equations(
            index_pairs, lhs_comps, rhs_comps,
            light=light,
            progress_callback=progress_callback,
        )

    return get_or_compute_reduced(key, compute, _log_debug)


def _raw_component_equations(index_pairs, lhs_comps, rhs_comps, light=False, progress_callback=None):
    """Build equations without applying metric ansatz substitutions."""
    reduced_lhs = []
    reduced_rhs = []
    for n, (lhs, rhs) in enumerate(zip(lhs_comps, rhs_comps), start=1):
        if progress_callback:
            progress_callback(f'Simplifying raw component {n}/{len(lhs_comps)}', 85)
        if light:
            reduced_lhs.append(lhs)
            reduced_rhs.append(rhs)
        else:
            reduced_lhs.append(lhs.doit())
            reduced_rhs.append(rhs.doit())
    equations = [sp.Eq(l, r, evaluate=False) for l, r in zip(reduced_lhs, reduced_rhs)]
    return index_pairs, reduced_lhs, reduced_rhs, equations


def _try_direct_diagonal_rhs_solve(inp, equations, unknowns, extended_subs=None):
    """Solve diagonal one-variable equations by reading the RHS coefficient.

    Static black-hole expressions can be huge, but their selected
    diagonal equations still have the simple form LHS = coefficient * matter.
    Avoiding SymPy's general solve/collect path here saves most of the runtime.
    """
    if not (
        inp.background_id == 'SS_blackhole'
        and len(equations) >= len(unknowns)
    ):
        return None

    solutions = {}
    unknown_set = set(unknowns)
    try:
        local_derivative_cleanup = None
        local_result_cleanup = None
        if inp.background_id == 'SS_blackhole':
            try:
                from core.solver import _evaluate_derivative_atoms_locally, _blackhole_post_ansatz_cleanup
                local_derivative_cleanup = _evaluate_derivative_atoms_locally
                if inp.theory in {'fQ', 'fQC'}:
                    local_result_cleanup = _blackhole_post_ansatz_cleanup
            except Exception:
                pass
        for eq, sym in zip(equations[:len(unknowns)], unknowns):
            if not hasattr(eq, 'lhs') or not hasattr(eq, 'rhs'):
                return None
            if any(eq.lhs.has(other) for other in unknown_set):
                return None
            coeff = eq.rhs.coeff(sym)
            if coeff == 0:
                return None
            rest = eq.rhs - coeff * sym
            if any(rest.has(other) for other in unknown_set):
                return None
            lhs = eq.lhs
            if extended_subs:
                lhs = lhs.subs(extended_subs)
                coeff = coeff.subs(extended_subs)
                rest = rest.subs(extended_subs)
            if local_derivative_cleanup is not None:
                lhs = local_derivative_cleanup(lhs)
                coeff = local_derivative_cleanup(coeff)
                rest = local_derivative_cleanup(rest)
            sol = (lhs - rest) / coeff
            if local_result_cleanup is not None:
                sol = local_result_cleanup(sol)
            solutions[sym] = sol
        print("[SOLVE_COMPONENT] Direct diagonal RHS isolation succeeded", flush=True)
        return solutions
    except Exception as exc:
        _log_debug(f"[SOLVE_COMPONENT] Direct diagonal RHS isolation skipped: {exc}")
        return None


def _is_trivial_equation(eq, inp) -> bool:
    """Detect Eq identities without hanging on hard symbolic log components."""
    try:
        residual = eq.lhs - eq.rhs
        if residual == 0:
            return True
        ops = sp.count_ops(residual)
        hard_static_log = (
            inp.background_id == 'SS_blackhole'
            and inp.theory in {'fQ', 'fQC'}
            and residual.has(sp.log)
        )
        if hard_static_log or ops > 1600:
            light = sp.powsimp(residual, force=True)
            if light == 0:
                return True
            if sp.count_ops(light) <= 350:
                return sp.cancel(light) == 0
            return False
        return sp.simplify(residual) == 0
    except Exception:
        return False


def _compute_blackhole_stability_light(rho, Pr, Pt, coord):
    """Fast sound-speed diagnostics for static black-hole expressions."""
    if coord is None or rho is None or Pr is None:
        return {'cs2_r': None, 'cs2_t': None}
    try:
        from core.solver import lightweight_diagnostic_simplify, _force_eval_partials
    except Exception:
        lightweight_diagnostic_simplify = None
        _force_eval_partials = None

    def deriv(expr):
        try:
            d = sp.diff(expr, coord)
            if _force_eval_partials is not None:
                d = _force_eval_partials(d)
            return d
        except Exception:
            return None

    def ratio(num, den):
        if num is None or den is None or den == sp.S.Zero:
            return None
        expr = num / den
        try:
            expr = sp.powsimp(expr, force=True)
            if sp.count_ops(expr) <= 600:
                expr = sp.cancel(expr)
            if lightweight_diagnostic_simplify is not None:
                return lightweight_diagnostic_simplify(expr, is_ratio=True)
            return expr
        except Exception:
            return expr

    dRho = deriv(rho)
    if dRho is None or dRho == sp.S.Zero:
        return {'cs2_r': None, 'cs2_t': None}
    dPr = deriv(Pr)
    dPt = deriv(Pt) if Pt is not None else None
    cs2_r = ratio(dPr, dRho)
    cs2_t = ratio(dPt, dRho) if dPt is not None else cs2_r
    return {'cs2_r': cs2_r, 'cs2_t': cs2_t}


def _reduce_component_equations(
    index_pairs, lhs_comps, rhs_comps, extended_subs,
    light=False, progress_callback=None,
):
    """Apply ansatz substitutions and weighted simplification to components."""
    reduced_lhs = []
    reduced_rhs = []
    for n, (lhs, rhs) in enumerate(zip(lhs_comps, rhs_comps), start=1):
        if progress_callback:
            progress_callback(f'Simplifying reduced component {n}/{len(lhs_comps)}', 85)
        if light:
            lhs_r = lhs.subs(extended_subs)
            rhs_r = rhs.subs(extended_subs)
        else:
            lhs_r = _light_reduce(lhs.subs(extended_subs).doit())
            rhs_r = _light_reduce(rhs.subs(extended_subs).doit())
        reduced_lhs.append(lhs_r)
        reduced_rhs.append(rhs_r)
    equations = [sp.Eq(l, r, evaluate=False) for l, r in zip(reduced_lhs, reduced_rhs)]
    return index_pairs, reduced_lhs, reduced_rhs, equations




def _light_equation_cleanup(expr, context='equation', light=False):
    """
    Light equation-boundary cleanup.

    The notebooks simplify tensors/components interactively, but applying full
    simplify here can hang on f(Q)/f(Q,C) and non-FRW component expressions.
    Keep this stage cheap; the real notebook-style simplify happens after
    matter variables are isolated and after ansatz substitution.
    """
    from core.solver import physical_domain_simplify
    if expr is None:
        return None
    try:
        ops = sp.count_ops(expr)
        has_trans = expr.has(sp.exp, sp.log, sp.sin, sp.cos, sp.tan, sp.sinh, sp.cosh, sp.tanh)
        if light:
            return expr
        if ops > 1200 or (has_trans and ops > 250):
            return physical_domain_simplify(expr)
        cleaned = sp.powsimp(expr, force=True)
        if ops <= 250 and not has_trans:
            cleaned = sp.cancel(sp.factor_terms(cleaned))
        elif ops <= 600 and not has_trans:
            cleaned = sp.factor_terms(cleaned)
        return physical_domain_simplify(cleaned)
    except Exception:
        return expr


def _light_reduce(expr):
    """Cheap reduction for large intermediate equations before solving."""
    from core.solver import physical_domain_simplify
    expr = _light_equation_cleanup(expr, context='reduced-equation')
    try:
        score, meta = _simplification_weight(expr)
        if meta['ops'] > 1200 or (meta['transcendentals'] and meta['ops'] > 250):
            return physical_domain_simplify(expr)
        if score > 2500:
            _log_debug(
                "[REDUCE] very heavy expression "
                f"ops={meta['ops']} weight={score}; using powsimp only"
            )
            return physical_domain_simplify(sp.powsimp(expr, force=True))
        if score > 900:
            _log_debug(
                "[REDUCE] heavy expression "
                f"ops={meta['ops']} weight={score}; using cancel+powsimp"
            )
            return physical_domain_simplify(sp.cancel(sp.powsimp(expr, force=True)))
        if score > 350:
            base = sp.powsimp(expr, force=True)
            if meta['transcendentals']:
                return physical_domain_simplify(sp.cancel(base))
            return physical_domain_simplify(sp.cancel(sp.together(base)))
        return physical_domain_simplify(sp.cancel(sp.factor_terms(expr)))
    except Exception:
        return expr


def _simplification_weight(expr):
    """Score symbolic expressions so costly forms avoid deep simplification."""
    try:
        ops = int(sp.count_ops(expr))
    except Exception:
        ops = 0
    derivative_count = len(expr.atoms(sp.Derivative))
    function_atoms = expr.atoms(sp.Function)
    transcendental_funcs = (
        sp.exp, sp.log, sp.sin, sp.cos, sp.tan,
        sp.sinh, sp.cosh, sp.tanh,
    )
    transcendental_count = sum(
        1 for atom in function_atoms if atom.func in transcendental_funcs
    )
    piecewise_count = len(expr.atoms(sp.Piecewise))
    weight = ops + 80 * derivative_count + 120 * transcendental_count + 200 * piecewise_count
    return weight, {
        'ops': ops,
        'derivatives': derivative_count,
        'transcendentals': transcendental_count,
        'piecewise': piecewise_count,
    }





def _fill_matter_results(results: PipelineResults, stress_tensor: str,
                          final_solutions: Dict, unknown_map: Dict, add_warning=None):
    """Map symbolic solutions into PipelineResults fields with enhanced simplification."""
    by_name = {}
    for key, value in final_solutions.items():
        by_name[str(key)] = value
        if hasattr(key, 'name'):
            by_name[key.name] = value

    def get(name):
        sym = unknown_map.get(name)
        if sym in final_solutions:
            return final_solutions[sym]
        return by_name.get(name)

    if add_warning is None:
        def add_warning(message):
            if results.warnings is None:
                results.warnings = []
            results.warnings.append(message)

    # Conservative anisotropic duplicate check.
    # A collapse P_r == P_t can be physically legitimate for a given model/ansatz,
    # so only suspicious duplicates remain fatal by default.
    if stress_tensor == 'anisotropic':
        matter_values = {
            'rho': get('rho'),
            'P_r': get('P_r'),
            'P_t': get('P_t'),
        }
        non_none = {k: v for k, v in matter_values.items() if v is not None}
        items = list(non_none.items())
        duplicate_pairs = []
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                name_a, val_a = items[i]
                name_b, val_b = items[j]
                try:
                    if val_a == val_b:
                        duplicate_pairs.append((name_a, name_b))
                except Exception:
                    if str(val_a) == str(val_b):
                        duplicate_pairs.append((name_a, name_b))
        if duplicate_pairs:
            strict_duplicates = os.getenv('MGS_STRICT_DUPLICATE_MATTER_CHECK', 'false').lower() in ('1', 'true', 'yes', 'on')
            duplicate_messages = []
            for name_a, name_b in duplicate_pairs:
                if {name_a, name_b} == {'P_r', 'P_t'}:
                    duplicate_messages.append(
                        "Anisotropic solution collapsed to isotropic pressure: P_r = P_t. "
                        "Results are kept and displayed."
                    )
                elif {name_a, name_b} == {'rho', 'p'}:
                    duplicate_messages.append(
                        "Equation-of-state collapse detected: rho = p. "
                        "Results are kept and displayed."
                    )
                else:
                    duplicate_messages.append(
                        f"Duplicate matter solution detected: {name_a} = {name_b}. "
                        "Results are kept and displayed."
                    )

            if strict_duplicates:
                print("[PIPELINE] ERROR: Symbolically identical matter solutions detected in strict mode:")
                for name, val in non_none.items():
                    print(f"[PIPELINE]   {name} = {val}")
                for na, nb in duplicate_pairs:
                    print(f"[PIPELINE]   {na} == {nb}")
                raise ValueError(
                    "Duplicate matter solutions detected for anisotropic stress-energy. "
                    f"Identical pairs: {duplicate_pairs}. "
                    "Disable MGS_STRICT_DUPLICATE_MATTER_CHECK to keep the result and show warnings instead."
                )

            for message in duplicate_messages:
                print(f"[PIPELINE] WARNING: {message}")
                add_warning(message)

    if stress_tensor == 'perfect_fluid':
        results.rho = get('rho')
        results.p   = get('p')
        results.Pr  = None  # Collapse to isotropic: no separate Pr/Pt
        results.Pt  = None  # Collapse to isotropic: no separate Pr/Pt
        try:
            rho_equals_p = results.rho is not None and results.p is not None and results.rho == results.p
        except Exception:
            rho_equals_p = str(results.rho) == str(results.p) if results.rho is not None and results.p is not None else False
        if rho_equals_p:
            add_warning("Equation-of-state collapse detected: rho = p. Results are kept and displayed.")
    elif stress_tensor == 'anisotropic':
        results.rho = get('rho')
        results.Pr  = get('P_r')
        results.Pt  = get('P_t')
    elif stress_tensor == 'dust':
        results.rho = get('rho')
        results.p   = sp.Integer(0)
    elif stress_tensor == 'radiation':
        results.rho = get('rho')
        results.p   = results.rho / 3 if results.rho is not None else None
    elif stress_tensor == 'vacuum':
        Lam = get('Lambda')
        results.p   = -Lam if Lam is not None else None
        results.rho = -results.p if results.p is not None else None


def _get_pressures(results: PipelineResults, stress_tensor: str):
    """Return (Pr, Pt) from results for derived quantity computation."""
    if stress_tensor == 'anisotropic':
        return results.Pr, results.Pt
    else:
        p = results.p
        return p, p


def _trace_from_results(results: PipelineResults, stress_tensor: str) -> sp.Expr:
    """Return the SET trace using solved matter values when they are available."""
    rho = results.rho if results.rho is not None else sp.Symbol('rho', positive=True)
    p = results.p if results.p is not None else sp.Symbol('p')

    if stress_tensor == 'perfect_fluid':
        return -rho + 3 * p
    if stress_tensor == 'dust':
        return -rho
    if stress_tensor == 'radiation':
        return sp.Integer(0)
    if stress_tensor == 'vacuum':
        return -rho + 3 * p
    if stress_tensor == 'anisotropic':
        Pr = results.Pr if results.Pr is not None else sp.Symbol('P_r')
        Pt = results.Pt if results.Pt is not None else sp.Symbol('P_t')
        return -rho + Pr + 2 * Pt
    return sp.Integer(0)


def _fill_scalar_results(results: PipelineResults, inp: PipelineInput,
                          geom, extended_subs: Dict):
    """Compute fully ansatz-substituted theory scalars."""
    from core.solver import safe_simplify
    theory = inp.theory

    matter_cache_key = ()
    if theory == 'fRTLm':
        if (
            inp.stress_tensor == 'anisotropic'
            and not inp.background_id.startswith('FRW')
        ):
            matter_cache_key = (
                inp.stress_tensor,
                inp.matter_lag,
                'symbolic-heavy-matter-scalars',
            )
        else:
            matter_cache_key = (
                inp.stress_tensor,
                inp.matter_lag,
                str(results.rho),
                str(results.p),
                str(results.Pr),
                str(results.Pt),
            )

    cache_key = (
        'scalars', theory,
        geom.live_symbols.get('_background_id', ''),
        # F10 fix: sort by sp.srepr rather than str() â€” srepr is deterministic across
        # Python/SymPy versions because it uses the canonical internal repr, whereas
        # str(expr) can change order with internal hash randomisation.
        tuple(sorted(
            (sp.srepr(k) if hasattr(k, 'free_symbols') else str(k),
             sp.srepr(v) if hasattr(v, 'free_symbols') else str(v))
            for k, v in extended_subs.items()
        )),
        matter_cache_key,
    )
    cached = get_scalar_cache(cache_key)
    if cached is not None:
        _log_debug(f"[CACHE] Scalar cache hit: {cache_key[1:3]}")
        results.R = cached.get('R')
        results.T = cached.get('T')
        results.B = cached.get('B')
        results.T_scalar = cached.get('T_scalar')
        results.Lm = cached.get('Lm')
        return

    def sub(expr):
        if expr is None:
            return None
        try:
            return safe_simplify(expr.subs(extended_subs).doit())
        except Exception:
            return expr

    def sub_light(expr):
        if expr is None:
            return None
        try:
            return safe_simplify(expr.xreplace(extended_subs))
        except Exception:
            return expr

    if theory in ('fR', 'fRTLm'):
        scalar_sub = sub_light if _use_light_scalar_outputs(inp) else sub
        results.R = scalar_sub(geom.ricci_scalar)
    if theory in ('fT', 'fTB'):
        results.T = sub(geom.T_scalar_expr)
    if theory == 'fTB':
        results.B = sub(geom.B_scalar_expr)
        T = results.T
        B = results.B
        results.T_scalar = sub(T - B) if (T is not None and B is not None) else None
    if theory == 'fRTLm':
        heavy_matter_scalars = (
            inp.stress_tensor == 'anisotropic'
            and not inp.background_id.startswith('FRW')
        )
        if heavy_matter_scalars:
            _log_debug("[SCALARS] Skipping solved matter scalar substitution for heavy anisotropic fRTLm run")
            if results.warnings is None:
                results.warnings = []
            results.warnings.append(
                "Scalar display shortcut: anisotropic f(R,T,Lm) T_scalar and Lm are shown with symbolic matter variables "
                "to avoid an expensive post-solve substitution."
            )
            results.T_scalar = sub(_compute_set_trace(
                inp.stress_tensor,
                sp.Symbol('rho', positive=True),
                sp.Symbol('p'),
            ))
        else:
            results.T_scalar = sub(_trace_from_results(results, inp.stress_tensor))
        from core.theories.fRTLm import get_Lm_expression
        if heavy_matter_scalars:
            rho = sp.Symbol('rho', positive=True)
            p = sp.Symbol('p')
        else:
            rho = results.rho if results.rho is not None else sp.Symbol('rho', positive=True)
            p = results.p if results.p is not None else sp.Symbol('p')
        results.Lm = sub(get_Lm_expression(inp.matter_lag, rho, p, results.T_scalar))
        scalar_sub = sub_light if _use_light_scalar_outputs(inp) else sub
        results.T  = scalar_sub(geom.T_scalar_expr)
        results.B  = scalar_sub(geom.B_scalar_expr)
    if theory in ('fQ', 'fQC'):
        from core.theories.fQ import Q_expr_from_geometry, _Q_expr_for_background
        Q_expr = getattr(geom, 'Q_scalar_expr', None)
        if Q_expr is None:
            Q_expr = Q_expr_from_geometry(geom)
        if Q_expr == sp.Symbol('Q'):
            bg_id = geom.live_symbols.get('_background_id', '')
            Q_expr = _Q_expr_for_background(bg_id, geom.live_symbols)
        results.T = sub(Q_expr)   # Store Q in the 'T' slot for display (reused key)
        if theory == 'fQC':
            results.B = sub(geom.ricci_scalar - Q_expr)  # Store C in the 'B' slot.

    set_scalar_cache(cache_key, {
        'R': results.R,
        'T': results.T,
        'B': results.B,
        'T_scalar': results.T_scalar,
        'Lm': results.Lm,
    }, _log_debug)
