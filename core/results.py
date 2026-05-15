"""Pipeline input/output data structures and serialization helpers."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import sympy as sp
from sympy.printing.mathematica import mathematica_code

# Environment-controlled rendering options
import os
SKIP_RENDERING = os.environ.get('MGS_SKIP_RENDERING', 'false').lower() == 'true'
ULTRA_LIGHT_MODE = os.environ.get('MGS_ULTRA_LIGHT', 'true').lower() == 'true'
AUTO_LATEX_OPS_LIMIT = int(os.environ.get('MGS_OPS_LIMIT', '3000'))
RENDER_CACHE_SIZE = int(os.environ.get('MGS_CACHE_SIZE', '50'))
ENABLE_DISPLAY_COMPRESSION = os.environ.get('MGS_ENABLE_DISPLAY_COMPRESSION', 'true').lower() == 'true'
DISPLAY_CSE_MIN_OPS = int(os.environ.get('MGS_DISPLAY_CSE_MIN_OPS', '120'))
DISPLAY_CSE_MAX_OPS = int(os.environ.get('MGS_DISPLAY_CSE_MAX_OPS', '5000'))
DISPLAY_MAX_DEFS = int(os.environ.get('MGS_DISPLAY_MAX_DEFS', '6'))
DISPLAY_KERNEL_LIMIT = int(os.environ.get('MGS_DISPLAY_KERNEL_LIMIT', '8'))
DISPLAY_VISIBLE_DEFS = int(os.environ.get('MGS_DISPLAY_VISIBLE_DEFS', '3'))
DISPLAY_DEF_MIN_OPS = int(os.environ.get('MGS_DISPLAY_DEF_MIN_OPS', '18'))
DISPLAY_MIN_IMPROVEMENT_PCT = float(os.environ.get('MGS_DISPLAY_MIN_IMPROVEMENT_PCT', '0.18'))
DISPLAY_ROUTE_OP_LIMIT = int(os.environ.get('MGS_DISPLAY_ROUTE_OP_LIMIT', '3500'))
DISPLAY_HARD_BLOWUP_PCT = float(os.environ.get('MGS_DISPLAY_HARD_BLOWUP_PCT', '0.35'))
DISPLAY_MAX_RENDER_DEFS = int(os.environ.get('MGS_DISPLAY_MAX_RENDER_DEFS', '2'))
DEFAULT_SIMPLIFY_MODE = os.environ.get('MGS_SIMPLIFY_MODE', 'fast').strip().lower() or 'fast'

# Global cache registry for cleanup
_all_render_caches = []


def _maintenance_logs_enabled() -> bool:
    return os.environ.get('MGS_SYMBOLIC_LOGS', 'false').lower() == 'true'


def _cache_log(message: str) -> None:
    if _maintenance_logs_enabled():
        print(message, flush=True)


def clear_all_render_caches():
    """Clear all temporary render caches. Call this at session end."""
    global _all_render_caches
    for cache in _all_render_caches:
        cache.clear()
    _all_render_caches.clear()


def clear_all_temporary_caches():
    """Clear all temporary in-memory and on-disk caches."""
    try:
        clear_all_render_caches()

        try:
            from core.solver import flush_simplify_cache
            flush_simplify_cache(clear_all=True)
        except Exception as e:
            _cache_log(f"[CACHE] Error clearing solver cache: {e}")

        try:
            from core.pipeline_cache import clear_all_pipeline_caches
            clear_all_pipeline_caches(lambda _msg: None)
        except Exception as e:
            _cache_log(f"[CACHE] Error clearing pipeline cache: {e}")

        try:
            from core.geometry import clear_cache
            clear_cache()
        except Exception as e:
            _cache_log(f"[CACHE] Error clearing geometry cache: {e}")

        _clear_disk_caches()
        _cache_log("[CACHE] All temporary caches cleared")
    except Exception as e:
        _cache_log(f"[CACHE] Error during cache cleanup: {e}")


def _clear_disk_caches():
    """Clear all disk-based caches to ensure temporary-only storage."""
    import os
    import shutil
    import glob
    
    try:
        # Clear geometry cache
        geometry_cache_dir = os.path.join(os.path.dirname(__file__), '.geometry_cache')
        if os.path.exists(geometry_cache_dir):
            try:
                shutil.rmtree(geometry_cache_dir)
                _cache_log(f"[CACHE] Removed geometry cache: {geometry_cache_dir}")
            except Exception as e:
                _cache_log(f"[CACHE] Error removing geometry cache: {e}")
        
        # Clear Python bytecode caches
        base_dir = os.path.dirname(os.path.dirname(__file__))  # Project root
        pycache_patterns = [
            os.path.join(base_dir, '**', '__pycache__'),
            os.path.join(base_dir, '**', '*.pyc'),
            os.path.join(base_dir, '**', '*.pyo'),
        ]
        
        for pattern in pycache_patterns:
            for cache_path in glob.glob(pattern, recursive=True):
                try:
                    if os.path.isdir(cache_path):
                        shutil.rmtree(cache_path)
                        _cache_log(f"[CACHE] Removed bytecode cache dir: {cache_path}")
                    else:
                        os.remove(cache_path)
                        _cache_log(f"[CACHE] Removed bytecode cache file: {cache_path}")
                except Exception as e:
                    _cache_log(f"[CACHE] Error removing {cache_path}: {e}")
        
        # Clear any temporary files in core directory
        temp_patterns = [
            os.path.join(os.path.dirname(__file__), '*.tmp'),
            os.path.join(os.path.dirname(__file__), 'temp_*'),
        ]
        
        for pattern in temp_patterns:
            for temp_path in glob.glob(pattern):
                try:
                    if os.path.isfile(temp_path):
                        os.remove(temp_path)
                        _cache_log(f"[CACHE] Removed temp file: {temp_path}")
                    elif os.path.isdir(temp_path):
                        shutil.rmtree(temp_path)
                        _cache_log(f"[CACHE] Removed temp dir: {temp_path}")
                except Exception as e:
                    _cache_log(f"[CACHE] Error removing temp {temp_path}: {e}")
                    
    except Exception as e:
        _cache_log(f"[CACHE] Error during disk cache cleanup: {e}")


def _register_cleanup_handler():
    """Register cleanup handler for automatic cache clearing on session end."""
    import atexit
    import signal
    import sys
    import threading
    
    # Register for normal exit (works in all environments)
    atexit.register(clear_all_temporary_caches)
    
    # Register for common termination signals only in main thread
    # This prevents the "signal only works in main thread" error in multi-threaded servers
    if threading.current_thread() == threading.main_thread():
        try:
            def cleanup_handler(signum, frame):
                clear_all_temporary_caches()
                sys.exit(0)
            
            signal.signal(signal.SIGTERM, cleanup_handler)
            signal.signal(signal.SIGINT, cleanup_handler)
            _cache_log("[CACHE] Signal handlers registered (main thread)")
        except (ValueError, OSError) as e:
            _cache_log(f"[CACHE] Could not register signal handlers: {e}")
    else:
        _cache_log("[CACHE] Signal handlers skipped (non-main thread)")


# Register cleanup handlers when module is imported
try:
    _register_cleanup_handler()
except Exception as e:
    _cache_log(f"[CACHE] Error during cleanup registration: {e}")


@dataclass
class PipelineInput:
    """Input parameters for pipeline."""
    background_id: str
    theory: str
    model_expr: str
    model_params: Dict[str, float]
    stress_tensor: str
    ansatz: Dict[str, str]
    ansatz_params: Dict[str, float]
    curvature_k: int = 0
    matter_lag: str = 'rho'
    compute_energy_conditions: bool = True
    compute_eos: bool = True
    compute_stability: bool = True
    compute_tov: bool = True
    simplify_mode: str = 'fast'


@dataclass
class PipelineResults:
    """Complete results from pipeline."""
    R: Optional[sp.Expr] = None
    T: Optional[sp.Expr] = None
    B: Optional[sp.Expr] = None
    T_scalar: Optional[sp.Expr] = None
    Lm: Optional[sp.Expr] = None

    rho: Optional[sp.Expr] = None
    p: Optional[sp.Expr] = None
    Pr: Optional[sp.Expr] = None
    Pt: Optional[sp.Expr] = None

    NEC_r: Optional[Any] = None
    NEC_t: Optional[Any] = None
    WEC: Optional[Any] = None
    SEC: Optional[Any] = None
    DEC_r: Optional[Any] = None
    DEC_t: Optional[Any] = None

    omega_r: Optional[Any] = None
    omega_t: Optional[Any] = None
    omega_eff: Optional[Any] = None

    cs2_r: Optional[Any] = None
    cs2_t: Optional[Any] = None

    tov_mass: Optional[Any] = None
    tov_compactness: Optional[Any] = None
    tov_redshift_gradient: Optional[Any] = None
    tov_pressure_gradient: Optional[Any] = None
    tov_hydrostatic_force: Optional[Any] = None
    tov_gravitational_force: Optional[Any] = None
    tov_anisotropic_force: Optional[Any] = None
    tov_residual: Optional[Any] = None
    tov_mass_continuity_residual: Optional[Any] = None

    matter_derivatives: Optional[Dict[sp.Symbol, sp.Expr]] = None

    exported_equations: Optional[List[Dict]] = None
    numeric_solve: Optional[Dict[str, Any]] = None
    plot_data: Optional[Dict[str, Any]] = None
    early_exit: bool = False
    early_exit_reason: Optional[str] = None
    derived_deferred: bool = False
    derived_deferred_reason: Optional[str] = None
    diagnostics_requested: Optional[Dict[str, bool]] = None
    warnings: Optional[List[str]] = None
    simplify_mode: str = 'fast'

    error: Optional[str] = None



_TRANS_FUNCS = (
    sp.exp, sp.log, sp.sin, sp.cos, sp.tan,
    sp.sinh, sp.cosh, sp.tanh, sp.asin, sp.acos, sp.atan,
)


def _safe_count_ops(expr: Any) -> int:
    try:
        return int(sp.count_ops(expr))
    except Exception:
        return 10**9


def _radical_atoms(expr: sp.Expr) -> List[sp.Expr]:
    atoms: List[sp.Expr] = []
    try:
        for atom in expr.atoms(sp.Pow):
            exp = getattr(atom, 'exp', None)
            if isinstance(exp, sp.Rational) and exp.q == 2:
                atoms.append(atom)
    except Exception:
        return []
    return atoms


def _classify_expr(expr: sp.Expr) -> str:
    try:
        if expr.has(*_TRANS_FUNCS):
            return 'transcendental'
        if _radical_atoms(expr):
            return 'sqrt_heavy'
    except Exception:
        pass
    return 'rational_power'


def _protect_kernels(expr: sp.Expr, family: str) -> Tuple[sp.Expr, Dict[sp.Symbol, sp.Expr]]:
    if family not in {'transcendental', 'sqrt_heavy'}:
        return expr, {}
    try:
        atoms: List[sp.Expr] = []
        if family == 'transcendental':
            for atom in expr.atoms(*_TRANS_FUNCS):
                if _safe_count_ops(atom) >= 2:
                    atoms.append(atom)
        else:
            atoms.extend(atom for atom in _radical_atoms(expr) if _safe_count_ops(atom) >= 2)
        if not atoms:
            return expr, {}
        atoms = sorted(set(atoms), key=lambda item: (-_safe_count_ops(item), len(str(item))))
        atoms = atoms[:DISPLAY_KERNEL_LIMIT]
        placeholders = sp.symbols(f'_K0:{len(atoms)}')
        to_dummy = dict(zip(atoms, placeholders))
        from_dummy = dict(zip(placeholders, atoms))
        return expr.xreplace(to_dummy), from_dummy
    except Exception:
        return expr, {}


def _restore_kernels(expr: sp.Expr, reverse_map: Dict[sp.Symbol, sp.Expr]) -> sp.Expr:
    if not reverse_map:
        return expr
    try:
        return expr.xreplace(reverse_map)
    except Exception:
        return expr


def _score_expr(expr: sp.Expr) -> int:
    ops = _safe_count_ops(expr)
    try:
        length = len(str(expr))
    except Exception:
        length = ops
    radical_penalty = len(_radical_atoms(expr)) * 12
    trans_penalty = sum(1 for _ in expr.atoms(*_TRANS_FUNCS)) * 10 if hasattr(expr, 'atoms') else 0
    return 4 * ops + length + radical_penalty + trans_penalty


def _dedupe_candidates(candidates: List[Tuple[str, sp.Expr]]) -> List[Tuple[str, sp.Expr]]:
    uniq: List[Tuple[str, sp.Expr]] = []
    seen = set()
    for route, cand in candidates:
        if cand is None:
            continue
        marker = str(cand)
        if marker in seen:
            continue
        seen.add(marker)
        uniq.append((route, cand))
    return uniq


def _choose_best_expr(candidates: List[Tuple[str, sp.Expr]], original: sp.Expr) -> Tuple[str, sp.Expr]:
    uniq = _dedupe_candidates(candidates)
    if not uniq:
        return 'original', original
    original_ops = max(1, _safe_count_ops(original))
    blowup_limit = max(64, int(original_ops * (1.0 + DISPLAY_HARD_BLOWUP_PCT)))
    best_route = 'original'
    best_expr = original
    best_score = _score_expr(original)
    for route, cand in uniq:
        cand_ops = _safe_count_ops(cand)
        if cand_ops > blowup_limit:
            continue
        score = _score_expr(cand)
        if score < best_score:
            best_route = route
            best_expr = cand
            best_score = score
    return best_route, best_expr


def _run_bounded_cse(expr: sp.Expr) -> Tuple[List[Tuple[sp.Symbol, sp.Expr]], sp.Expr]:
    ops = _safe_count_ops(expr)
    if ops < DISPLAY_CSE_MIN_OPS or ops > DISPLAY_CSE_MAX_OPS:
        return [], expr
    try:
        symbols = sp.numbered_symbols(prefix='X')
        replacements, reduced = sp.cse(expr, symbols=symbols)
        if not reduced:
            return [], expr
        replacements = replacements[:DISPLAY_MAX_DEFS]
        return replacements, reduced[0]
    except Exception:
        return [], expr


def _is_high_value_def(symbol: sp.Symbol, subexpr: sp.Expr, family: str) -> bool:
    ops = _safe_count_ops(subexpr)
    if ops < DISPLAY_DEF_MIN_OPS:
        return False
    if family == 'transcendental' and subexpr.has(*_TRANS_FUNCS):
        return True
    if family == 'sqrt_heavy' and _radical_atoms(subexpr):
        return True
    if family == 'rational_power':
        return ops >= max(DISPLAY_DEF_MIN_OPS + 4, 24)
    return True


def _substitute_dropped_defs(
    replacements: List[Tuple[sp.Symbol, sp.Expr]],
    reduced: sp.Expr,
    keep_symbols: set[sp.Symbol],
) -> Tuple[List[Tuple[sp.Symbol, sp.Expr]], sp.Expr]:
    active_reduced = reduced
    kept: List[Tuple[sp.Symbol, sp.Expr]] = []
    kept_map: Dict[sp.Symbol, sp.Expr] = {}
    for sym, subexpr in reversed(replacements):
        expanded = subexpr.xreplace(kept_map)
        if sym in keep_symbols:
            kept_map[sym] = expanded
            kept.append((sym, expanded))
        else:
            active_reduced = active_reduced.xreplace({sym: expanded})
            kept_map[sym] = expanded
    kept.reverse()
    if kept:
        inline_map = {sym: subexpr for sym, subexpr in kept}
        normalized_kept = []
        for sym, subexpr in kept:
            normalized_kept.append((sym, subexpr.xreplace({k: v for k, v in inline_map.items() if k != sym})))
        kept = normalized_kept
    return kept, active_reduced


def _cse_cost(expr: sp.Expr, defs: List[Tuple[sp.Symbol, sp.Expr]]) -> int:
    total = len(str(expr))
    for sym, subexpr in defs:
        total += len(str(sym)) + 3 + len(str(subexpr))
    return total


def _prune_cse_defs(
    original: sp.Expr,
    replacements: List[Tuple[sp.Symbol, sp.Expr]],
    reduced: sp.Expr,
    family: str,
) -> Tuple[List[Tuple[sp.Symbol, sp.Expr]], sp.Expr]:
    if not replacements:
        return [], reduced

    high_value = [(sym, subexpr) for sym, subexpr in replacements if _is_high_value_def(sym, subexpr, family)]
    if not high_value:
        return [], original

    high_value = sorted(
        high_value,
        key=lambda item: (-_safe_count_ops(item[1]), -len(str(item[1])), str(item[0]))
    )[:max(0, DISPLAY_VISIBLE_DEFS)]

    keep_symbols = {sym for sym, _ in high_value}
    kept_defs, pruned_reduced = _substitute_dropped_defs(replacements, reduced, keep_symbols)

    original_cost = len(str(original))
    compressed_cost = _cse_cost(pruned_reduced, kept_defs)
    improvement = (original_cost - compressed_cost) / max(1, original_cost)
    if improvement < DISPLAY_MIN_IMPROVEMENT_PCT:
        return [], original

    return kept_defs, pruned_reduced




def _finalize_display_expr(
    expr: sp.Expr,
    simplify_mode: str = DEFAULT_SIMPLIFY_MODE,
) -> Tuple[sp.Expr, List[Tuple[sp.Symbol, sp.Expr]]]:
    if not ENABLE_DISPLAY_COMPRESSION or expr is None:
        return expr, []

    mode = (simplify_mode or DEFAULT_SIMPLIFY_MODE or 'fast').strip().lower()
    if mode not in ('fast', 'heavy'):
        mode = 'fast'

    family = _classify_expr(expr)
    ops = _safe_count_ops(expr)
    protected, reverse_map = _protect_kernels(expr, family)

    route_limit = DISPLAY_ROUTE_OP_LIMIT if mode == 'heavy' else max(1200, DISPLAY_ROUTE_OP_LIMIT // 2)
    max_defs = DISPLAY_MAX_RENDER_DEFS if mode == 'heavy' else min(1, DISPLAY_MAX_RENDER_DEFS)
    min_improvement = DISPLAY_MIN_IMPROVEMENT_PCT if mode == 'heavy' else max(0.28, DISPLAY_MIN_IMPROVEMENT_PCT + 0.10)

    candidates: List[Tuple[str, sp.Expr]] = [('original', protected)]
    if ops <= route_limit:
        try:
            if family == 'rational_power':
                together_cancel = sp.cancel(sp.together(protected))
                candidates.append(('factor_terms', sp.factor_terms(protected)))
                candidates.append(('cancel_together', together_cancel))
                if mode == 'heavy':
                    candidates.append(('powsimp', sp.powsimp(protected, force=False)))
                    candidates.append(('powsimp_factor_terms', sp.factor_terms(sp.powsimp(together_cancel, force=False))))
            elif family == 'transcendental':
                candidates.append(('factor_terms', sp.factor_terms(protected)))
                if mode == 'heavy':
                    together_cancel = sp.cancel(sp.together(protected))
                    candidates.append(('cancel_together', together_cancel))
                    candidates.append(('powsimp_factor_terms', sp.factor_terms(sp.powsimp(together_cancel, force=False))))
            elif family == 'sqrt_heavy':
                candidates.append(('factor_terms', sp.factor_terms(protected)))
                if mode == 'heavy':
                    together_cancel = sp.cancel(sp.together(protected))
                    candidates.append(('cancel_together', together_cancel))
                    candidates.append(('factor_terms_cancel', sp.factor_terms(together_cancel)))
        except Exception:
            pass

    _route, best = _choose_best_expr(candidates, protected)
    replacements, reduced = _run_bounded_cse(best)
    replacements, reduced = _prune_cse_defs(best, replacements, reduced, family)
    if len(replacements) > max_defs:
        replacements = replacements[:max_defs]

    if replacements:
        original_cost = len(str(best))
        compressed_cost = _cse_cost(reduced, replacements)
        improvement = (original_cost - compressed_cost) / max(1, original_cost)
        if improvement < min_improvement:
            replacements = []
            reduced = best

    restored_defs = [
        (sym, _restore_kernels(val, reverse_map))
        for sym, val in replacements
    ]
    restored_expr = _restore_kernels(reduced, reverse_map)
    return restored_expr, restored_defs


def _expr_to_mathematica(expr: sp.Expr, label: str) -> str:
    """Convert expression to Mathematica syntax with a conservative fallback."""
    if expr is None:
        return ''
    try:
        return mathematica_code(expr)
    except Exception:
        try:
            return str(sp.sstr(expr))
        except Exception:
            return f"(* Error converting {label} to Mathematica *)"


def _defs_to_mathematica_lines(defs: List[Tuple[sp.Symbol, sp.Expr]]) -> List[str]:
    lines: List[str] = []
    for sym, subexpr in defs:
        try:
            lines.append(f"{mathematica_code(sym)} = {mathematica_code(subexpr)}")
        except Exception:
            lines.append(f"{str(sym)} = {str(subexpr)}")
    return lines

def _format_latex_payload(expr: sp.Expr, label: str, cache: Dict[Any, Any], simplify_mode: str = DEFAULT_SIMPLIFY_MODE):
    cache_key = ('display', expr)
    if cache_key in cache:
        return cache[cache_key]

    display_expr, defs = _finalize_display_expr(expr, simplify_mode=simplify_mode)
    main_latex = _expr_str_with_timeout(display_expr, label)
    main_mathematica = _expr_to_mathematica(display_expr, label)
    if defs:
        rendered_defs = []
        latex_copy_lines = []
        mathematica_copy_lines = []
        for sym, subexpr in defs:
            sym_latex = _expr_str_with_timeout(sym, f'{label} symbol')
            sub_latex = _expr_str_with_timeout(subexpr, f'{label} definition')
            sym_mathematica = _expr_to_mathematica(sym, f'{label} symbol')
            sub_mathematica = _expr_to_mathematica(subexpr, f'{label} definition')
            rendered_defs.append({
                'name': str(sym),
                'latex_name': sym_latex,
                'latex': sub_latex,
                'mathematica_name': sym_mathematica,
                'mathematica': sub_mathematica,
            })
            latex_copy_lines.append(f'{sym_latex} = {sub_latex}')
            mathematica_copy_lines.append(f'{sym_mathematica} = {sub_mathematica}')
        latex_copy_lines.append(main_latex)
        mathematica_copy_lines.append(main_mathematica)
        payload = {
            'latex': main_latex,
            'mathematica': main_mathematica,
            'defs': rendered_defs,
            'copy_latex': '\n'.join(latex_copy_lines),
            'copy_mathematica': '\n'.join(mathematica_copy_lines),
            'render_family': _classify_expr(expr),
        }
    else:
        payload = {
            'latex': main_latex,
            'mathematica': main_mathematica,
            'copy_latex': main_latex,
            'copy_mathematica': main_mathematica,
            'render_family': _classify_expr(expr),
        }

    cache[cache_key] = payload
    return payload


def _expr_str_with_timeout(e, label: str):
    """Convert expression to string with portable threading timeout."""
    if e is None:
        return None
    
    import threading
    import queue
    
    result_queue = queue.Queue()
    exception_queue = queue.Queue()
    
    def worker():
        try:
            result = sp.latex(e)
            result_queue.put(result)
        except Exception as ex:
            exception_queue.put(ex)
    
    thread = threading.Thread(target=worker)
    thread.daemon = True
    thread.start()
    thread.join(timeout=5.0)
    
    if thread.is_alive():
        # Thread is still running, timeout occurred
        return f"[TIMEOUT] {label} (LaTeX conversion >5s)"
    
    # Check for exceptions
    if not exception_queue.empty():
        ex = exception_queue.get()
        try:
            return sp.latex(sp.sympify(str(e), evaluate=False))
        except Exception:
            return f"[Error converting {label}]"
    
    # Get successful result
    if not result_queue.empty():
        return result_queue.get()
    
    return f"[Error converting {label}]"



def results_to_dict(results: PipelineResults) -> Dict:
    """Serialize symbolic pipeline results for API/UI payload."""
    if SKIP_RENDERING:
        return _minimal_results_dict(results)

    from core.solver import LazyResult

    _render_cache: Dict[Any, Any] = {}
    _all_render_caches.append(_render_cache)

    simplify_mode = getattr(results, 'simplify_mode', DEFAULT_SIMPLIFY_MODE)
    def expr_str(e, label: str):
        if e is None:
            return None
        if isinstance(e, LazyResult):
            try:
                e = e.evaluate()
            except Exception as lazy_ex:
                print(f"[RENDER] LazyResult evaluation failed for {label}: {lazy_ex}", flush=True)
                return f"[Error evaluating {label}]"
        if e is None:
            return None
        try:
            return _format_latex_payload(e, label, _render_cache, simplify_mode=simplify_mode)
        except Exception as ex:
            print(f"[RENDER] Error formatting {label}: {ex}", flush=True)
            return _expr_str_with_timeout(e, label)

    dr = results.diagnostics_requested or {}

    def maybe_ec(field, label):
        if not dr.get('energy_conditions', True):
            return None
        return expr_str(getattr(results, field, None), label)

    def maybe_eos(field, label):
        if not dr.get('eos', True):
            return None
        return expr_str(getattr(results, field, None), label)

    def maybe_stab(field, label):
        if not dr.get('stability', True):
            return None
        return expr_str(getattr(results, field, None), label)

    def maybe_tov(field, label):
        if not dr.get('tov', True):
            return None
        return expr_str(getattr(results, field, None), label)

    return {
        'scalars': {
            'R':        expr_str(results.R,        'scalar R'),
            'T':        expr_str(results.T,        'scalar T/Q'),
            'B':        expr_str(results.B,        'scalar B/C'),
            'T_scalar': expr_str(results.T_scalar, 'scalar matter_trace'),
            'Lm':       expr_str(results.Lm,       'scalar matter_lagrangian'),
        },
        'matter': {
            'rho': expr_str(results.rho, 'rho'),
            'p':   expr_str(results.p,   'p'),
            'Pr':  expr_str(results.Pr,  'P_r'),
            'Pt':  expr_str(results.Pt,  'P_t'),
        },
        'energy_conditions': {
            'NEC_r': maybe_ec('NEC_r', 'NEC_r'),
            'NEC_t': maybe_ec('NEC_t', 'NEC_t'),
            'WEC':   maybe_ec('WEC',   'WEC'),
            'SEC':   maybe_ec('SEC',   'SEC'),
            'DEC_r': maybe_ec('DEC_r', 'DEC_r'),
            'DEC_t': maybe_ec('DEC_t', 'DEC_t'),
        },
        'eos': {
            'omega_r':   maybe_eos('omega_r',   'omega_r'),
            'omega_t':   maybe_eos('omega_t',   'omega_t'),
            'omega_eff': maybe_eos('omega_eff', 'omega_eff'),
        },
        'speed_of_sound': {
            'cs2_r': maybe_stab('cs2_r', 'cs2_r'),
            'cs2_t': maybe_stab('cs2_t', 'cs2_t'),
        },
        'tov': {
            'mass': maybe_tov('tov_mass', 'm(r)'),
            'compactness': maybe_tov('tov_compactness', '2m/r'),
            'redshift_gradient': maybe_tov('tov_redshift_gradient', 'Phi_prime'),
            'pressure_gradient': maybe_tov('tov_pressure_gradient', 'dP_r/dr'),
            'hydrostatic_force': maybe_tov('tov_hydrostatic_force', 'F_h'),
            'gravitational_force': maybe_tov('tov_gravitational_force', 'F_g'),
            'anisotropic_force': maybe_tov('tov_anisotropic_force', 'F_a'),
            'residual': maybe_tov('tov_residual', 'TOV residual'),
            'mass_continuity_residual': maybe_tov('tov_mass_continuity_residual', 'mass continuity residual'),
        },
        'early_exit':              results.early_exit,
        'early_exit_reason':       results.early_exit_reason,
        'exported_equations':      results.exported_equations,
        'numeric_solve':           results.numeric_solve,
        'plot_data':               results.plot_data,
        'derived_deferred': results.derived_deferred,
        'derived_deferred_reason': results.derived_deferred_reason,
        'diagnostics_requested': results.diagnostics_requested,
        'warnings': list(results.warnings or []),
        'timings': getattr(results, 'timings', {}),
    }


def _minimal_results_dict(results: PipelineResults) -> Dict:
    """Return minimal results dict with no LaTeX rendering for maximum speed."""
    def simple_repr(e):
        if e is None:
            return None
        if isinstance(e, (int, float, str)):
            return e
        try:
            from core.solver import LazyResult
            if isinstance(e, LazyResult):
                try:
                    e = e.evaluate()
                except Exception as lazy_ex:
                    return f"[Error evaluating lazy: {lazy_ex}]"
            return str(e)[:200] + ("..." if len(str(e)) > 200 else "")
        except Exception:
            return "[Error converting to string]"
    
    return {
        'scalars': {
            'R': simple_repr(results.R),
            'T': simple_repr(results.T),
            'B': simple_repr(results.B),
            'T_scalar': simple_repr(results.T_scalar),
            'Lm': simple_repr(results.Lm),
        },
        'matter': {
            'rho': simple_repr(results.rho),
            'p': simple_repr(results.p),
            'Pr': simple_repr(results.Pr),
            'Pt': simple_repr(results.Pt),
        },
        'energy_conditions': {
            'NEC_r': simple_repr(results.NEC_r),
            'NEC_t': simple_repr(results.NEC_t),
            'WEC': simple_repr(results.WEC),
            'SEC': simple_repr(results.SEC),
            'DEC_r': simple_repr(results.DEC_r),
            'DEC_t': simple_repr(results.DEC_t),
        },
        'eos': {
            'omega_r': simple_repr(results.omega_r),
            'omega_t': simple_repr(results.omega_t),
            'omega_eff': simple_repr(results.omega_eff),
        },
        'speed_of_sound': {
            'cs2_r': simple_repr(results.cs2_r),
            'cs2_t': simple_repr(results.cs2_t),
        },
        'tov': {
            'mass': simple_repr(results.tov_mass),
            'compactness': simple_repr(results.tov_compactness),
            'redshift_gradient': simple_repr(results.tov_redshift_gradient),
            'pressure_gradient': simple_repr(results.tov_pressure_gradient),
            'hydrostatic_force': simple_repr(results.tov_hydrostatic_force),
            'gravitational_force': simple_repr(results.tov_gravitational_force),
            'anisotropic_force': simple_repr(results.tov_anisotropic_force),
            'residual': simple_repr(results.tov_residual),
            'mass_continuity_residual': simple_repr(results.tov_mass_continuity_residual),
        },
        'early_exit': results.early_exit,
        'early_exit_reason': results.early_exit_reason,
        'exported_equations': results.exported_equations,
        'numeric_solve': results.numeric_solve,
        'plot_data': results.plot_data,
        'derived_deferred': results.derived_deferred,
        'derived_deferred_reason': results.derived_deferred_reason,
        'diagnostics_requested': results.diagnostics_requested,
        'warnings': list(results.warnings or []),
        'timings': getattr(results, 'timings', {}),
        'render_mode': 'SKIPPED' if SKIP_RENDERING else ('ULTRA_LIGHT' if ULTRA_LIGHT_MODE else 'NORMAL'),
    }


# Display simplification functions completely removed
# All rendering now uses direct LaTeX conversion for maximum speed
