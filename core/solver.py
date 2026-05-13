"""
Field Equation Solver

Solves the field equations for matter variables (ρ, p, etc.)
and computes derived quantities (energy conditions, EOS, c_s²).
"""

from typing import Dict, List, Tuple, Any, Optional
import sympy as sp
import os
import random

# ─── Optimization Toggle Flags ───────────────────────────────────────────────
# All default to True; disable individually for debugging
ENABLE_SIMPLIFY_CACHE = True
ENABLE_LAZY_DERIVED = True
ENABLE_DEFERRED_DIFF = True
ENABLE_SHARED_DENOM = True
ENABLE_COLLECT_FIRST = True
ENABLE_SKIP_FACTOR_TRANSCENDENTAL = True
ENABLE_LINEAR_FAST_SOLVE = True
ENABLE_COMPONENT_LINEAR_SOLVE = True
ENABLE_EQUATORIAL_BRANCH = False  # Guard sin(theta) positivity assumption
SIMPLIFY_CACHE_MAX = 512

try:
    from core.config import VERBOSE_LOGS
except Exception:
    VERBOSE_LOGS = False


FAST_DIAGNOSTICS = os.getenv('MGS_FAST_DIAGNOSTICS', 'true').lower() in ('1', 'true', 'yes', 'on')
FAST_DIAGNOSTIC_OPS_LIMIT = int(os.getenv('MGS_FAST_DIAGNOSTIC_OPS_LIMIT', '400'))

# ─── Expression Fingerprint Cache for safe_simplify() ───────────────────────────
# Keys: (hash(expr), ops, free_symbols_tuple, context) — structure_key throughout.
# All write paths must use _store_simplify_cache() so the key scheme stays
# consistent with the lookup performed at the top of safe_simplify().
_simplify_cache: Dict[Tuple, sp.Expr] = {}


def _store_simplify_cache(original_expr: sp.Expr, result: sp.Expr,
                          context: str) -> None:
    """Write *result* into _simplify_cache under the canonical structure_key.

    Using this helper in every write path guarantees the key scheme matches the
    lookup in safe_simplify() exactly.  The previous code wrote bare
    ``hash(original_expr)`` integers in most branches while the lookup used a
    4-tuple, making the cache effectively write-only.
    """
    try:
        ops = sp.count_ops(original_expr)
        free_symbols = tuple(sorted(str(s) for s in original_expr.free_symbols))
        structure_key = (hash(original_expr), ops, free_symbols, context)
        _simplify_cache[structure_key] = result
    except Exception:
        pass  # never let cache bookkeeping crash the solver


# ─── Fix 2: Module-level matter symbols to stop assumption cache accumulation ───
# These are reused across all pipeline runs to prevent SymPy assumption context bloat
_SYM_RHO = sp.Symbol('rho', positive=True)
_SYM_P = sp.Symbol('p')
_SYM_PR = sp.Symbol('P_r')
_SYM_PT = sp.Symbol('P_t')
_SYM_LAMBDA = sp.Symbol('Lambda')

_POSITIVE_SYMBOL_NAMES = {
    # Coordinates / chart branch
    'r',  # Removed 't' - time should be context-dependent
    # Cosmological and spherical ansatz constants commonly used by presets
    'a0', 'A0', 'B0', 'H0', 'R0', 'r0', 'M', 'M0', 'Q0',
}
_POSITIVE_FUNCTION_NAMES = {'a', 'A', 'B'}


def get_matter_symbols() -> Dict[str, sp.Symbol]:
    """Return module-level matter symbols as a dict keyed by name."""
    return {
        'rho': _SYM_RHO,
        'p': _SYM_P,
        'P_r': _SYM_PR,
        'P_t': _SYM_PT,
        'Lambda': _SYM_LAMBDA,
    }


def flush_simplify_cache(clear_all: bool = False):
    """Prune or clear simplify caches and related SymPy assumptions."""
    global _simplify_cache
    if clear_all:
        before = len(_simplify_cache)
        _simplify_cache.clear()
        _solver_log(f"[SOLVER] Simplify cache cleared ({before} -> 0 entries)")
    elif len(_simplify_cache) > SIMPLIFY_CACHE_MAX:
        overflow = len(_simplify_cache) - SIMPLIFY_CACHE_MAX
        for key in list(_simplify_cache.keys())[:overflow]:
            _simplify_cache.pop(key, None)
        _solver_log(f"[SOLVER] Simplify cache pruned to {len(_simplify_cache)} entries")
    else:
        _solver_log(f"[SOLVER] Simplify cache retained ({len(_simplify_cache)} entries)")
    
    # Fix 2: Also clear SymPy assumptions cache to prevent accumulation
    try:
        import sympy.core.assumptions as _sa
        if hasattr(_sa, 'assumptions_cache'):
            _sa.assumptions_cache.clear()
        if hasattr(_sa, '_assume_defined'):
            _sa._assume_defined.clear()
        _solver_log("[SOLVER] SymPy assumptions cache cleared")
    except Exception as e:
        _solver_log(f"[SOLVER] Failed to clear assumptions cache: {e}")


def _solver_log(message: str):
    """Avoid stringifying huge symbolic expressions unless verbose logs are on."""
    if VERBOSE_LOGS:
        print(message)


def _solver_log_ops(label: str, expr: sp.Expr):
    """Log operation counts only when verbose logs are enabled."""
    if VERBOSE_LOGS:
        try:
            print(f"{label}: {sp.count_ops(expr)}")
        except Exception:
            print(f"{label}: unavailable")


def physical_domain_simplify(expr: sp.Expr) -> sp.Expr:
    """
    Apply the physical coordinate/metric branch used by the pipeline.

    Cosmological scale factors and radial chart factors are taken on their
    positive branches. This removes symbolic artifacts like Abs(a(t)) and
    sign(a0*t**h) without changing genuine matter inequalities such as Abs(p).
    """
    if expr is None:
        return None
    try:
        replacements = _physical_branch_replacements(expr)
        cleaned = expr
        if replacements:
            cleaned = cleaned.xreplace(replacements)
            cleaned = cleaned.subs(replacements, simultaneous=True)
        cleaned = sp.powsimp(cleaned, force=True)
        
        # Additional pass to handle nested Abs(t) structures
        cleaned = _strip_physical_abs(cleaned)
        
        replacements = _physical_branch_replacements(cleaned)
        if replacements:
            cleaned = cleaned.xreplace(replacements)
            cleaned = cleaned.subs(replacements, simultaneous=True)
        return cleaned
    except Exception as e:
        print(f"[DOMAIN] Physical-domain cleanup failed: {e}")
        return expr


def _physical_branch_replacements(expr: sp.Expr) -> Dict[sp.Expr, sp.Expr]:
    replacements = {}
    for node in expr.atoms(sp.Abs):
        if node.args and _is_physical_positive(node.args[0]):
            replacements[node] = node.args[0]
    for node in expr.atoms(sp.sign):
        if node.args and _is_physical_positive(node.args[0]):
            replacements[node] = sp.Integer(1)
    return replacements


def _is_physical_positive(expr: sp.Expr) -> bool:
    """Conservative positivity predicate for metric/chart branch cleanup."""
    if expr is None:
        return False
    if expr.is_positive is True:
        return True
    if expr.is_number:
        try:
            return bool(expr > 0)
        except TypeError:
            return False
    if isinstance(expr, sp.Symbol):
        # Special handling for time t - only positive in specific contexts
        if str(expr) == 't':
            return False  # Time is not universally positive - context dependent
        return expr.name in _POSITIVE_SYMBOL_NAMES or str(expr) in _POSITIVE_SYMBOL_NAMES
    if expr.func == sp.sin and expr.args and str(expr.args[0]) in ('theta', 'th'):
        return ENABLE_EQUATORIAL_BRANCH
    if expr.is_Function:
        if expr.func in (sp.exp, sp.sqrt):
            return True
        return getattr(expr.func, '__name__', '') in _POSITIVE_FUNCTION_NAMES
    if expr.is_Pow:
        return _is_physical_positive(expr.base)
    if expr.is_Mul:
        return all(_is_physical_positive(arg) for arg in expr.args)
    return False


# ─── LazyResult Class for Deferred Computation ───────────────────────────────
class LazyResult:
    """
    Wrapper for deferred computation of derived quantities.

    Evaluates the actual computation only on first access, caching the result.
    Picklable for compatibility with disk cache — but ONLY after evaluation:
    the compute_fn closure is not picklable (it captures live SymPy objects and
    method references).  __getstate__ evaluates eagerly before serialising so
    the result is always concrete on the other side.
    """

    def __init__(self, compute_fn, *args, **kwargs):
        self.compute_fn = compute_fn
        self.args = args
        self.kwargs = kwargs
        self._result = None
        self._evaluated = False
        self._name = compute_fn.__name__ if hasattr(compute_fn, '__name__') else 'lazy'

    def evaluate(self) -> Any:
        """Execute the deferred computation and cache the result."""
        if not self._evaluated:
            if self.compute_fn is None:
                # Bug E fix: deserialised without a result means something went
                # wrong during pickling — surface a clear error instead of
                # crashing with 'NoneType is not callable'.
                raise RuntimeError(
                    f"LazyResult '{self._name}' was deserialised without an evaluated "
                    "result. This should not happen — __getstate__ always evaluates "
                    "before pickling. Re-run the pipeline to regenerate the result."
                )
            print(f"[LAZY] Evaluating {self._name}...", flush=True)
            self._result = self.compute_fn(*self.args, **self.kwargs)
            self._evaluated = True
            print(f"[LAZY] {self._name} evaluated", flush=True)
        return self._result

    def __getstate__(self):
        """Support for pickle serialization.
        Bug E fix: evaluate eagerly here so the pickled payload always contains
        a concrete result. Closures captured by compute_fn are not picklable.
        """
        if not self._evaluated:
            try:
                self.evaluate()
            except Exception as exc:
                # If evaluation fails at pickle time, store None and the error message.
                self._result = None
                self._evaluated = True
                print(f"[LAZY] WARNING: could not evaluate '{self._name}' before pickling: {exc}", flush=True)
        return {
            '_result': self._result,
            '_evaluated': self._evaluated,
            '_name': self._name,
        }

    def __setstate__(self, state):
        """Support for pickle deserialization."""
        self._result = state['_result']
        self._evaluated = state['_evaluated']
        self._name = state['_name']
        # compute_fn is intentionally not restored; the result is already concrete.
        self.compute_fn = None
        self.args = ()
        self.kwargs = {}

    def __repr__(self):
        if self._evaluated:
            return f"LazyResult(evaluated, result={self._result})"
        return f"LazyResult(pending, fn={self._name})"



def _safe_count_ops(expr: sp.Expr) -> int:
    try:
        return int(sp.count_ops(expr))
    except Exception:
        return 10**9


def _score_expr(expr: sp.Expr) -> int:
    ops = _safe_count_ops(expr)
    try:
        length = len(str(expr))
    except Exception:
        length = ops
    try:
        radical_penalty = len(_radical_atoms(expr)) * 12
    except Exception:
        radical_penalty = 0
    trans_funcs = (
        sp.exp, sp.log,
        sp.sin, sp.cos, sp.tan,
        sp.sinh, sp.cosh, sp.tanh,
        sp.asin, sp.acos, sp.atan,
        sp.asinh, sp.acosh, sp.atanh,
    )
    try:
        trans_penalty = sum(1 for _ in expr.atoms(*trans_funcs))
    except Exception:
        trans_penalty = 0
    return 4 * ops + length + radical_penalty + 10 * trans_penalty


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


def _numerically_equivalent(a: sp.Expr, b: sp.Expr, trials: int = 3) -> bool:
    if a == b:
        return True
    try:
        symbols = sorted(a.free_symbols.union(b.free_symbols), key=lambda s: str(s))
    except Exception:
        return True
    if not symbols:
        try:
            return abs(complex(sp.N(a - b))) < 1e-8
        except Exception:
            return True

    positive_names = {'r', 'a0', 'A0', 'B0', 'H0', 'R0', 'r0', 'M', 'M0', 'Q0'}
    nonzero_names = positive_names | {'alpha', 'beta', 'gamma', 'lam', 'lamc', 'lambda', 'n', 'm', 'h', 'k'}

    for _ in range(trials):
        subs = {}
        for sym in symbols:
            name = str(sym)
            num = random.randint(2, 9)
            den = random.randint(2, 9)
            value = sp.Rational(num, den)
            if name in positive_names:
                subs[sym] = value
            elif name in nonzero_names:
                subs[sym] = value if random.randint(0, 1) else -value
            else:
                subs[sym] = value if random.randint(0, 1) else -value
        try:
            delta = sp.N((a - b).subs(subs))
            if delta.has(sp.zoo, sp.oo, sp.nan):
                continue
            if abs(complex(delta)) > 1e-6:
                return False
        except Exception:
            continue
    return True


def _choose_best_expr(candidates: List[Tuple[str, sp.Expr]], original: sp.Expr) -> Tuple[str, sp.Expr]:
    uniq = _dedupe_candidates(candidates)
    if not uniq:
        return 'original', original
    original_ops = max(1, _safe_count_ops(original))
    blowup_limit = max(64, int(original_ops * 1.35))
    best_route = 'original'
    best_expr = original
    best_score = _score_expr(original)
    for route, cand in uniq:
        cand_ops = _safe_count_ops(cand)
        if cand_ops > blowup_limit:
            continue
        try:
            if not _numerically_equivalent(original, cand):
                continue
        except Exception:
            pass
        score = _score_expr(cand)
        if score < best_score:
            best_route = route
            best_expr = cand
            best_score = score
    return best_route, best_expr


def safe_simplify(expr: sp.Expr, timeout: int = 5, *, context: str = "general") -> sp.Expr:
    """
    Structure-aware simplification router with intelligent expression handling.

    Routes expressions by structure first, then by operation count.
    Uses model-kernel structured cleanup for symbolic powers and heavy expressions.

    Context options:
    - "general": standard simplification
    - "display": display-focused simplification
    - "post_solve": post-solving cleanup
    """
    global _simplify_cache

    if expr is None:
        return None

    original_expr = expr

    expr = physical_domain_simplify(expr)
    expr = _strip_physical_abs(expr)
    expr = sp.powsimp(expr, force=True)

    if expr.is_number or expr.is_symbol:
        return expr

    if ENABLE_SIMPLIFY_CACHE:
        try:
            ops = sp.count_ops(expr)
            free_symbols = tuple(sorted(str(s) for s in expr.free_symbols))
            structure_key = (hash(expr), ops, free_symbols, context)
            if structure_key in _simplify_cache:
                print(f"[SIMPLIFY] Cache hit, returning cached result")
                return physical_domain_simplify(_simplify_cache[structure_key])
        except Exception:
            pass

    ops = sp.count_ops(expr)
    has_symbolic_power = _has_symbolic_power(expr)
    is_heavy_transcendental = _is_heavy_transcendental_expr(expr)

    if ops <= 3000:
        try:
            if len(_radical_atoms(expr)) > 1:
                expr = _accept_if_better(expr, sp.sqrtdenest(sp.powsimp(expr, force=True)))
        except Exception:
            pass

    if context == "display":
        print(f"[SIMPLIFY] display context, ops={ops}")
        result = model_kernel_structured_cleanup(expr, display=True)
        if ENABLE_SIMPLIFY_CACHE:
            _store_simplify_cache(original_expr, result, context)
        return result

    if context == "post_solve":
        print(f"[SIMPLIFY] post_solve context, ops={ops}")
        try:
            from core.solver import solve_final_cleanup
            result = _accept_if_better(expr, solve_final_cleanup(expr))
            if ENABLE_SIMPLIFY_CACHE:
                _store_simplify_cache(original_expr, result, context)
            return result
        except Exception:
            return expr

    candidates: List[Tuple[str, sp.Expr]] = [('original', expr)]

    if has_symbolic_power and ops > 400:
        print(f"[SIMPLIFY] symbolic power detected, ops={ops} -> model_kernel_structured_cleanup")
        try:
            candidates.append(('model_kernel', model_kernel_structured_cleanup(expr, display=False)))
        except Exception as e:
            print(f"[SIMPLIFY] Model-kernel path failed: {e}")

    if is_heavy_transcendental:
        print(f"[SIMPLIFY] heavy transcendental, ops={ops} -> rational_transcendental_simplify")
        try:
            candidates.append(('transcendental', rational_transcendental_simplify(expr)))
        except Exception as e:
            print(f"[SIMPLIFY] Heavy transcendental path failed: {e}")

    if ops > 2000:
        print(f"[SIMPLIFY] large expression, ops={ops} -> structured_expression_cleanup")
        try:
            candidates.append(('structured', structured_expression_cleanup(expr, display=False)))
        except Exception as e:
            print(f"[SIMPLIFY] Structured cleanup failed: {e}")
        try:
            candidates.append(('enhanced_cancel', _enhanced_cancel_simplify(expr)))
        except Exception:
            pass
    elif ops > 800:
        print(f"[SIMPLIFY] complex expression, ops={ops} -> enhanced cancel")
        try:
            candidates.append(('enhanced_cancel', _enhanced_cancel_simplify(expr)))
        except Exception as e:
            print(f"[SIMPLIFY] Enhanced cancel failed: {e}")

    if ops > 200:
        print(f"[SIMPLIFY] moderate expression, ops={ops} -> together+cancel+readability")
        try:
            canceled = sp.cancel(sp.together(expr))
            canceled = sp.powsimp(canceled, force=True)
            canceled = _collect_model_params(canceled)
            canceled = physical_domain_simplify(canceled)
            candidates.append(('together_cancel', canceled))
        except Exception as e:
            print(f"[SIMPLIFY] Together+cancel failed: {e}")

    if ops <= 80:
        print(f"[SIMPLIFY] small expression, ops={ops} -> conservative full simplify")
        try:
            candidates.append(('simplify', physical_domain_simplify(sp.simplify(expr))))
        except Exception as e:
            print(f"[SIMPLIFY] Conservative simplify failed: {e}")

    try:
        default_result = expr
        if ops <= 300:
            default_result = sp.cancel(sp.factor(default_result))
        if default_result.has(sp.Symbol):
            default_result = sp.together(default_result)
        if default_result.has(sp.Pow):
            default_result = sp.powsimp(default_result)
        default_result = _collect_model_params(default_result)
        default_result = _factor_common_display_terms(default_result)
        if ops <= 50:
            default_result = sp.simplify(default_result)
        default_result = physical_domain_simplify(default_result)
        candidates.append(('default', default_result))
    except Exception:
        pass

    route, result = _choose_best_expr(candidates, expr)
    print(f"[SIMPLIFY] selected route={route}, ops={_safe_count_ops(result)}")
    if ENABLE_SIMPLIFY_CACHE:
        _store_simplify_cache(original_expr, result, context)
    return result

def solve_light_simplify(expr: sp.Expr) -> sp.Expr:
    """Low-risk cleanup for expressions created while isolating variables."""
    if expr is None:
        return None
    try:
        expr = physical_domain_simplify(expr)
        ops = sp.count_ops(expr)
        base = sp.powsimp(expr, force=True)
        if ops > 1500:
            return _common_factor_cleanup(physical_domain_simplify(base))
        if ops > 250:
            return _common_factor_cleanup(physical_domain_simplify(sp.cancel(base)))
        return _common_factor_cleanup(physical_domain_simplify(sp.cancel(sp.factor_terms(base))))
    except Exception:
        return expr


def bounded_fraction_cleanup(expr: sp.Expr) -> sp.Expr:
    """
    Bounded cleanup for hard solved expressions.

    This deliberately avoids a whole-expression cancel/simplify pass.  Hard
    f(Q)/f(Q,C) models often become readable enough when numerator and
    denominator are cleaned separately, while global cancellation can dominate
    the entire pipeline runtime.
    """
    if expr is None:
        return None

    def clean_piece(piece: sp.Expr) -> sp.Expr:
        try:
            ops = sp.count_ops(piece)
            piece = sp.powsimp(piece, force=True)
            if ops <= 700:
                piece = sp.cancel(piece)
            if ops <= 1400:
                piece = sp.factor_terms(piece)
            return piece
        except Exception:
            return piece

    try:
        expr = sp.sympify(expr)
        expr = sp.powsimp(expr, force=True)
        num, den = sp.fraction(expr)

        if isinstance(num, sp.Add):
            num = sp.Add(*(clean_piece(arg) for arg in num.args), evaluate=False)
        else:
            num = clean_piece(num)
        den = clean_piece(den)

        return _common_factor_cleanup(physical_domain_simplify(num / den))
    except Exception as e:
        _solver_log(f"[SIMPLIFY] bounded_fraction_cleanup skipped: {e}")
        return expr


def structured_expression_cleanup(expr: sp.Expr, *, display: bool = False) -> sp.Expr:
    """
    Simplify large formulas by local structure instead of whole-expression simplify.

    Uses a two-pass strategy:
    1. a bounded global numerator/denominator pass to preserve cross-term cancellation
    2. chunked local cleanup only when pieces are still large
    """
    if expr is None:
        return None

    def opcount(e: sp.Expr) -> int:
        try:
            return int(sp.count_ops(e))
        except Exception:
            return 10**9

    def clean_piece(piece: sp.Expr) -> sp.Expr:
        try:
            ops = opcount(piece)
            piece = sp.powsimp(piece, force=True)
            if ops <= 1200:
                piece = sp.cancel(piece)
            if ops <= 2200:
                piece = sp.factor_terms(piece)
            if ops <= 2800:
                piece = _collect_display_kernels(piece)
            return piece
        except Exception:
            return piece

    def maybe_chunk(piece: sp.Expr) -> sp.Expr:
        if not isinstance(piece, sp.Add):
            return clean_piece(piece)
        terms = list(piece.args)
        if len(terms) <= 8:
            return clean_piece(piece)
        chunks = _cost_balanced_chunks(terms, max_chunk_ops=1800)
        cleaned_chunks = [clean_piece(chunk) for chunk in chunks]
        return sp.Add(*cleaned_chunks, evaluate=False)

    try:
        expr = physical_domain_simplify(sp.sympify(expr))
        expr = _strip_physical_abs(expr)
        expr = sp.powsimp(expr, force=True)

        expr_ops = opcount(expr)
        if expr_ops <= 900:
            return fast_simplify(expr)

        num, den = sp.fraction(expr)

        if expr_ops <= 6000:
            num = clean_piece(num)
            den = clean_piece(den)

        num = maybe_chunk(num)
        den = maybe_chunk(den)

        rebuilt = num / den
        rebuilt = sp.powsimp(rebuilt, force=True)
        rebuilt = _strip_physical_abs(rebuilt)
        rebuilt = _collect_display_kernels(rebuilt)

        final_ops = opcount(rebuilt)
        if final_ops <= (4200 if display else 2600):
            rebuilt = sp.factor_terms(rebuilt)
            rebuilt = sp.powsimp(rebuilt, force=True)
        if final_ops <= 1800:
            rebuilt = sp.cancel(rebuilt)
        return _common_factor_cleanup(physical_domain_simplify(rebuilt))
    except Exception as e:
        _solver_log(f"[SIMPLIFY] structured_expression_cleanup skipped: {e}")
        return expr

def _common_factor_cleanup(expr: sp.Expr) -> sp.Expr:
    """Apply bounded common-kernel factoring shared with display rendering."""
    if expr is None:
        return None
    try:
        ops = sp.count_ops(expr)
        if ops > 3500:
            return expr
        factored = _factor_common_display_terms(expr)
        if factored is None:
            return expr
        factored_ops = sp.count_ops(factored)
        if factored_ops > max(ops * 3, ops + 5000):
            return expr
        return physical_domain_simplify(factored)
    except Exception:
        return expr


def _collect_display_kernels(expr: sp.Expr) -> sp.Expr:
    """Collect model parameters and repeated symbolic-power kernels."""
    try:
        kernels = []
        for name in ('alpha', 'beta', 'gamma', 'lam', 'lamc', 'lambda', 'n', 'm', 'h'):
            sym = sp.Symbol(name)
            if sym in expr.free_symbols:
                kernels.append(sym)
        symbolic_powers = [
            pow_expr for pow_expr in expr.atoms(sp.Pow)
            if getattr(pow_expr.exp, 'free_symbols', set())
        ]
        symbolic_powers = sorted(
            symbolic_powers,
            key=lambda item: (sp.count_ops(item), str(item)),
        )[:8]
        kernels.extend(symbolic_powers)
        if kernels:
            return sp.collect(expr, kernels, evaluate=True)
    except Exception:
        return expr
    return expr


def _strip_physical_abs(expr: sp.Expr) -> sp.Expr:
    """Remove Abs around powers/products already assumed positive by the branch."""
    try:
        replacements = {}
        for node in expr.atoms(sp.Abs):
            arg = node.args[0] if node.args else None
            if arg is not None:
                # Special handling for time t - be more aggressive in removing Abs(t)
                if isinstance(arg, sp.Symbol) and str(arg) == 't':
                    # For FRW cosmology, t is typically positive, so remove Abs(t)
                    replacements[node] = arg
                    _solver_log("[ABS] Removing Abs(t) -> t")
                elif _is_physical_positive(arg):
                    replacements[node] = arg
        if replacements:
            result = expr.xreplace(replacements)
            _solver_log(f"[ABS] Applied {len(replacements)} Abs replacements")
            return result
    except Exception as e:
        _solver_log(f"[ABS] Abs stripping failed: {e}")
    return expr


def diagnostic_simplify(expr: sp.Expr) -> sp.Expr:
    """Cheap cleanup for secondary diagnostic expressions.

    Three specialised fast paths are tried before falling back to the generic
    structured_expression_cleanup:

      1. **Transcendental forms** – expressions containing exp/log/trig/hyp.
         Arguments of transcendental functions are simplified independently so
         that cancellations inside the argument are found without a global
         simplify call that would be extremely expensive.

      2. **Power-like model forms** – expressions whose only non-rational
         structure is symbolic Pow nodes (i.e. n-th power ansätze such as
         r**n, a**n, (1+r**2)**n …).  A collect pass on the repeated symbolic-
         power kernels is applied after powsimp, which is much cheaper than
         cancel/together when no transcendentals are involved.

      3. **Sqrt-paired anisotropic forms** – expressions containing sqrt (or
         Pow(·, 1/2)) together with two distinct pressure-like terms (as
         occurs in NEC_r/NEC_t for non-FRW anisotropic wormhole backgrounds).
         sqrtdenest is called on numerator and denominator separately so that
         paired radicals collapse before the final factor pass.

    All three paths guard on ops-count so they degrade gracefully for very
    large expressions.
    """
    if expr is None:
        return None
    try:
        ops = sp.count_ops(expr)

        # ── 1. Transcendental form ──────────────────────────────────────────
        _TRANS = (sp.exp, sp.log,
                  sp.sin, sp.cos, sp.tan,
                  sp.sinh, sp.cosh, sp.tanh,
                  sp.asin, sp.acos, sp.atan,
                  sp.asinh, sp.acosh, sp.atanh)
        if expr.has(*_TRANS):
            if ops <= 2500 and expr.has(sp.sin, sp.cos, sp.tan, sp.sinh, sp.cosh, sp.tanh):
                try:
                    from sympy.simplify.fu import fu
                    expr = _accept_if_better(expr, fu(expr))
                    ops = sp.count_ops(expr)
                except Exception:
                    pass
            return _diagnostic_transcendental(expr, ops)

        # ── 2. Power-like model form ────────────────────────────────────────
        symbolic_powers = [
            pw for pw in expr.atoms(sp.Pow)
            if getattr(pw.exp, 'free_symbols', set())
        ]
        if symbolic_powers:
            return _diagnostic_power_like(expr, ops, symbolic_powers)

        # ── 3. Sqrt-paired anisotropic form ────────────────────────────────
        sqrt_atoms = [
            pw for pw in expr.atoms(sp.Pow)
            if pw.exp == sp.Rational(1, 2)
        ]
        # Extended to handle both sp.Add and sp.Mul top-level expressions
        # since fQ pressure fractions commonly aren't Add at top level
        if sqrt_atoms and isinstance(expr, (sp.Add, sp.Mul)) and len(expr.args) >= 2:
            return _diagnostic_sqrt_anisotropic(expr, ops, sqrt_atoms)

        # ── Generic fallback ────────────────────────────────────────────────
        hard = ops > 800 or expr.has(sp.sqrt, sp.Abs)
        if hard:
            return structured_expression_cleanup(expr, display=True)
        return _common_factor_cleanup(fast_simplify(expr))
    except Exception:
        return expr


def _diagnostic_transcendental(expr: sp.Expr, ops: int) -> sp.Expr:
    """Simplify transcendental expressions by cleaning function arguments first.

    Strategy:
    - Simplify each transcendental argument in isolation (cheap cancel/powsimp).
    - Rebuild the expression with cleaner arguments.
    - Apply factor_terms / collect on the result.
    - For very large expressions fall back immediately to structured_cleanup.
    """
    _TRANS_FUNCS = (sp.exp, sp.log,
                    sp.sin, sp.cos, sp.tan,
                    sp.sinh, sp.cosh, sp.tanh,
                    sp.asin, sp.acos, sp.atan,
                    sp.asinh, sp.acosh, sp.atanh)
    if ops > 6000:
        # Too large for local argument cleanup to help; use the generic path.
        return structured_expression_cleanup(expr, display=True)

    try:
        replacements: Dict[sp.Expr, sp.Expr] = {}
        for node in expr.atoms(*_TRANS_FUNCS):
            if not node.args:
                continue
            arg = node.args[0]
            arg_ops = sp.count_ops(arg)
            if arg_ops == 0:
                continue
            try:
                if arg_ops <= 1200:
                    cleaned_arg = sp.factor_terms(
                        sp.powsimp(sp.cancel(arg), force=True)
                    )
                else:
                    cleaned_arg = sp.factor_terms(sp.powsimp(arg, force=True))
                if cleaned_arg != arg:
                    replacements[node] = node.func(cleaned_arg)
            except Exception:
                pass

        result = expr.xreplace(replacements) if replacements else expr
        result = sp.powsimp(result, force=True)
        result = _strip_physical_abs(result)

        if ops <= 2500:
            result = sp.factor_terms(result)

        # Collect on any surviving model-parameter symbols.
        result = _collect_display_kernels(result)
        return result
    except Exception:
        return expr


def _diagnostic_power_like(
    expr: sp.Expr, ops: int, symbolic_powers: list
) -> sp.Expr:
    """Simplify power-law ansatz expressions (r**n, a**n, (1+r^2)**n …).

    The key insight: powsimp + collect on the dominant symbolic-power kernel
    is far cheaper than cancel/together and handles most practical model forms
    produced by the pipeline (polynomial-in-r times a power-law factor).
    """
    if ops > 8000:
        return _common_factor_cleanup(expr)

    try:
        result = sp.powsimp(expr, force=True)
        result = _strip_physical_abs(result)

        # Rank kernels by ops-count descending; use the top few for collect.
        kernels = sorted(
            symbolic_powers,
            key=lambda pw: (-sp.count_ops(pw), str(pw)),
        )[:6]

        if ops <= 1800:
            result = sp.cancel(result)

        result = sp.factor_terms(result)

        try:
            result = sp.collect(result, kernels, evaluate=True)
        except Exception:
            pass

        result = _collect_display_kernels(result)
        return physical_domain_simplify(result)
    except Exception:
        return expr


def _diagnostic_sqrt_anisotropic(
    expr: sp.Expr, ops: int, sqrt_atoms: list
) -> sp.Expr:
    """Simplify expressions with paired sqrt factors (NEC/DEC anisotropic forms).

    Anisotropic wormhole models often produce energy-condition expressions of
    the form  A/sqrt(f(r))  ±  B/sqrt(g(r)).  sqrtdenest on num/den
    independently collapses paired radicals before the standard factor pass.
    
    Now uses tiered strategy for different ops ranges to avoid freezes.
    """
    if ops > 15000:
        # Very large expressions: skip expensive sqrtdenest entirely
        return structured_expression_cleanup(expr, display=True)
    
    if ops > 5000:
        # Intermediate range: structured lightweight approach
        try:
            num, den = sp.fraction(expr)
            
            # Apply powsimp first to reduce complexity
            num = sp.powsimp(num, force=True)
            den = sp.powsimp(den, force=True)
            
            # Limited sqrtdenest only on small pieces
            def _safe_denest(e: sp.Expr) -> sp.Expr:
                if sp.count_ops(e) <= 2000:
                    try:
                        return sp.sqrtdenest(e)
                    except Exception:
                        pass
                return e
            
            # Apply only to numerator/denominator if they're not too large
            if sp.count_ops(num) <= 3000:
                num = _safe_denest(num)
            if sp.count_ops(den) <= 3000:
                den = _safe_denest(den)
            
            # Light factor pass
            result = sp.factor_terms(num / den)
            result = sp.powsimp(result, force=True)
            result = _strip_physical_abs(result)
            
            # Only apply expensive operations if still reasonable
            if sp.count_ops(result) <= 4000:
                result = sp.cancel(result)
                result = _collect_display_kernels(result)
            
            return physical_domain_simplify(result)
        except Exception:
            return structured_expression_cleanup(expr, display=True)

    # Original path for ops <= 5000
    try:
        num, den = sp.fraction(expr)

        def _denest(e: sp.Expr) -> sp.Expr:
            try:
                e = sp.sqrtdenest(sp.powsimp(e, force=True))
            except Exception:
                e = sp.powsimp(e, force=True)
            return e

        num = _denest(num)
        den = _denest(den)

        result = sp.factor_terms(num / den)
        result = sp.powsimp(result, force=True)
        result = _strip_physical_abs(result)

        if sp.count_ops(result) <= 2000:
            result = sp.cancel(result)

        result = _collect_display_kernels(result)
        return physical_domain_simplify(result)
    except Exception:
        return expr


def simplify_transcendental_args(expr: sp.Expr) -> sp.Expr:
    """Original notebook helper: simplify arguments inside log/exp/trig functions."""
    trans_funcs = (
        sp.log, sp.exp,
        sp.sin, sp.cos, sp.tan,
        sp.sinh, sp.cosh, sp.tanh,
        sp.asin, sp.acos, sp.atan,
    )

    def _visit(e):
        if not getattr(e, 'args', None):
            return e

        new_args = tuple(_visit(arg) for arg in e.args)

        if e.func in trans_funcs:
            new_args = tuple(
                sp.factor_terms(sp.powsimp(sp.cancel(arg), force=True))
                for arg in new_args
            )

        if new_args != e.args:
            return e.func(*new_args)
        return e

    return _visit(expr)


def lightweight_diagnostic_simplify(expr: sp.Expr, *, is_ratio: bool = False) -> sp.Expr:
    """Bounded cleanup for derived diagnostics built from solved rho/pressures.

    This intentionally avoids broad simplification. Energy conditions are linear
    combinations and EoS terms are simple ratios of already-solved expressions,
    so a tiny cleanup is usually enough and far cheaper than the full
    diagnostic_simplify() pipeline.
    """
    if expr is None:
        return None
    try:
        expr = sp.sympify(expr).doit()
        ops = int(sp.count_ops(expr))
        if ops <= FAST_DIAGNOSTIC_OPS_LIMIT:
            expr = sp.powsimp(expr, force=False)
            expr = sp.factor_terms(expr)
            if is_ratio and ops <= max(80, FAST_DIAGNOSTIC_OPS_LIMIT // 2):
                expr = sp.cancel(expr)
        return expr
    except Exception:
        return expr


def fast_simplify(expr: sp.Expr) -> sp.Expr:
    """Original notebook fast_simplify method."""
    if expr is None:
        return None
    try:
        expr = sp.sympify(expr)

        # First evaluate derivatives if an ansatz has already been substituted.
        expr = expr.doit()

        # Temporarily replace derivative objects by short dummy symbols.
        derivs = sorted(expr.atoms(sp.Derivative), key=str)
        dummies = sp.symbols('D0:%d' % len(derivs))

        to_dummy = dict(zip(derivs, dummies))
        from_dummy = dict(zip(dummies, derivs))

        temp = expr.xreplace(to_dummy)

        # Simplify arguments inside transcendental functions before simplifying the outside.
        temp = simplify_transcendental_args(temp)
        temp = sp.logcombine(temp, force=True)

        temp = sp.cancel(temp)
        temp = sp.powsimp(temp, force=True)
        temp = sp.factor_terms(temp)

        if len(dummies) != 0:
            temp = sp.collect(temp, list(dummies))

        # One more pass, because cancel/factor can expose simpler log(...) or exp(...) arguments.
        temp = simplify_transcendental_args(temp)
        temp = sp.logcombine(temp, force=True)

        return _common_factor_cleanup(physical_domain_simplify(temp.xreplace(from_dummy)))
    except Exception as e:
        _solver_log(f"[SIMPLIFY] fast_simplify skipped: {e}")
        return expr


def rational_transcendental_simplify(expr: sp.Expr) -> sp.Expr:
    """
    Balanced cleanup for expressions with large transcendental blocks.

    This keeps exp/log/trig atoms intact, simplifies only their arguments, then
    works on numerator and denominator separately with dummy replacements for
    repeated transcendental atoms.
    """
    if expr is None:
        return None
    try:
        expr = sp.sympify(expr).doit()
        expr = simplify_transcendental_args(expr)
        expr = sp.logcombine(expr, force=True)

        to_dummy, from_dummy, dummy_symbols = _build_transcendental_subs(expr)
        if to_dummy:
            print(
                f"[SIMPLIFY] Compressing {len(to_dummy)} transcendental block(s) for rational cleanup",
                flush=True,
            )
            expr = expr.xreplace(to_dummy)

        num, den = _split_numerator_denominator(expr)
        num = _termwise_rational_cleanup(num)
        den = _termwise_rational_cleanup(den)

        cleaned = num / den
        if dummy_symbols:
            cleaned = sp.collect(cleaned, dummy_symbols, exact=False)
            cleaned = cleaned.xreplace(from_dummy)

        cleaned = simplify_transcendental_args(cleaned)
        cleaned = sp.logcombine(cleaned, force=True)
        if _safe_count_ops(cleaned) <= 4000:
            try:
                cleaned = sp.exptrigsimp(cleaned)
            except Exception:
                pass
        cleaned = sp.powsimp(cleaned, force=True)
        cleaned = sp.factor_terms(cleaned)
        return _common_factor_cleanup(physical_domain_simplify(cleaned))
    except Exception as e:
        _solver_log(f"[SIMPLIFY] rational_transcendental_simplify skipped: {e}")
        return expr


def _split_numerator_denominator(expr: sp.Expr) -> Tuple[sp.Expr, sp.Expr]:
    """Split a rational expression without forcing a deep global simplify."""
    try:
        ops = sp.count_ops(expr)
        if ops <= 3000:
            expr = sp.together(expr)
    except Exception:
        pass
    try:
        return sp.fraction(expr)
    except Exception:
        return expr, sp.Integer(1)


def _termwise_rational_cleanup(expr: sp.Expr) -> sp.Expr:
    """Simplify an Add expression term-by-term instead of as one giant object."""
    def clean_piece(piece):
        try:
            piece = sp.powsimp(piece, force=True)
            piece = sp.cancel(piece)
            return sp.factor_terms(piece)
        except Exception:
            return piece

    try:
        if isinstance(expr, sp.Add):
            return sp.Add(*(clean_piece(arg) for arg in expr.args), evaluate=False)
        return clean_piece(expr)
    except Exception:
        return expr


def solve_final_cleanup(expr: sp.Expr) -> sp.Expr:
    """Final cleanup for solved expressions before returning to pipeline."""
    if expr is None:
        return None
    try:
        expr = bounded_fraction_cleanup(expr)
        # Only attempt matter-variable factoring when the expression contains
        # exactly one matter unknown; global sp.factor() across rho/P_r/P_t
        # can map two equations onto the same expression when geometric
        # prefactors cancel symmetrically (the "same rho and P" bug).
        matter_syms = [sp.Symbol('rho'), sp.Symbol('P_r'), sp.Symbol('P_t'),
                       sp.Symbol('p')]
        present = [s for s in matter_syms if expr.has(s)]
        if len(present) == 1:
            expr = _factor_matter_common_terms(expr)
        return expr
    except Exception:
        return expr


def _factor_matter_common_terms(expr: sp.Expr) -> sp.Expr:
    """Factor common terms from matter expressions to reduce length."""
    if expr is None:
        return None
    try:
        # Try basic factoring first
        factored = sp.factor(expr)
        
        # If factoring worked, apply additional cleanup
        if factored != expr:
            factored = sp.powsimp(factored, force=True)
            factored = sp.cancel(factored)
            return factored
        
        # For complex expressions with repeated patterns, try advanced factoring
        if isinstance(expr, sp.Add) and len(expr.args) > 10:
            factored = _factor_complex_patterns(expr)
            if factored != expr:
                return sp.powsimp(factored, force=True)
        
        return expr
    except Exception:
        return expr


def _factor_complex_patterns(expr: sp.Expr) -> sp.Expr:
    """Factor complex repeated patterns like a0^6 h r^4 |t|^{6h}."""
    if expr is None or not isinstance(expr, sp.Add):
        return expr
    try:
        # Look for common patterns in additive terms
        terms = list(expr.args)
        
        # Find common factors across terms
        common_factors = set()
        term_patterns = []
        
        for term in terms:
            # Extract pattern from term
            pattern = _extract_term_pattern(term)
            if pattern:
                term_patterns.append((term, pattern))
        
        if len(term_patterns) < 2:
            return expr
        
        # Find common sub-patterns
        all_factors = []
        for term, pattern in term_patterns:
            all_factors.extend(pattern['factors'])
        
        # Find factors that appear in multiple terms
        factor_counts = {}
        for factor in all_factors:
            factor_counts[factor] = factor_counts.get(factor, 0) + 1
        
        common_factors = {f for f, count in factor_counts.items() if count >= 2}
        
        if common_factors:
            # Try to factor out common terms
            simplified = expr
            for factor in sorted(common_factors, key=str, reverse=True):
                try:
                    # Try to factor out this common factor
                    temp = sp.factor(simplified, factor)
                    if temp != simplified:
                        simplified = temp
                        _solver_log(f"[PATTERN] Factored out common factor: {factor}")
                except Exception:
                    pass
            
            return simplified
        
        return expr
    except Exception:
        return expr


def _extract_term_pattern(term: sp.Expr) -> Optional[Dict]:
    """Extract pattern factors from a complex term."""
    if term is None:
        return None
    
    try:
        # Look for patterns like a0^6 h r^4 |t|^{6h}
        factors = []
        
        # Extract symbolic factors
        if isinstance(term, sp.Mul):
            for arg in term.args:
                if isinstance(arg, sp.Pow):
                    # Check for powers like a0^6, r^4, |t|^{6h}
                    base = arg.base
                    if hasattr(base, 'name'):
                        factors.append(str(base))
                    elif isinstance(base, sp.Symbol):
                        factors.append(str(base))
                elif isinstance(arg, sp.Symbol):
                    factors.append(str(arg))
                elif isinstance(arg, sp.Abs):
                    # Handle |t|^{6h} pattern
                    inner = arg.args[0] if arg.args else None
                    if inner and hasattr(inner, 'func'):
                        if inner.func == sp.Abs and inner.args:
                            inner_abs = inner.args[0]
                            if isinstance(inner_abs, sp.Pow):
                                # Pattern like |t|^{6h}
                                factors.append(f"|{inner_abs.base}|^{inner_abs.exp}")
        
        return {'factors': factors} if factors else None
    except Exception:
        return None


def _first_complete_solution(result, unknowns: List[sp.Symbol]) -> Optional[Dict[sp.Symbol, sp.Expr]]:
    """Return a pressure solution only if it resolves all requested unknowns."""
    if not result:
        return None
    if isinstance(result, dict):
        candidates = [result]
    elif isinstance(result, (list, tuple)):
        candidates = list(result)
    else:
        return None

    for candidate in candidates:
        if isinstance(candidate, dict):
            sol = candidate
        elif isinstance(candidate, (list, tuple)) and len(candidate) >= len(unknowns):
            sol = {sym: candidate[i] for i, sym in enumerate(unknowns)}
        else:
            continue

        if any(sym not in sol for sym in unknowns):
            continue
        if any(sol[sym] is None for sym in unknowns):
            continue
        if any(sol[sym].has(*unknowns) for sym in unknowns):
            continue
        return {
            sym: fast_simplify(sol[sym])
            for sym in unknowns
        }
    return None


def _is_hard_solve_system(equations: List[sp.Eq]) -> bool:
    """Avoid unbounded generic sp.solve on giant/transcendental systems."""
    try:
        exprs = [
            eq.lhs - eq.rhs
            for eq in equations
            if hasattr(eq, 'lhs') and hasattr(eq, 'rhs')
        ]
        if not exprs:
            return False
        ops = sum(sp.count_ops(expr) for expr in exprs)
        has_trans = any(
            expr.has(sp.exp, sp.log, sp.sin, sp.cos, sp.tan, sp.sinh, sp.cosh, sp.tanh)
            for expr in exprs
        )
        return ops > 8000 or (has_trans and ops > 1800)
    except Exception:
        return True


def _is_heavy_transcendental_expr(expr: sp.Expr, ops_limit: int = 1800) -> bool:
    """Detect expressions where cleanup/substitution should stay purely structural."""
    try:
        root_like = any(
            isinstance(pow_expr.exp, sp.Rational) and pow_expr.exp.q != 1
            for pow_expr in expr.atoms(sp.Pow)
        )
        has_hard_shape = expr.has(
            sp.exp, sp.log, sp.sin, sp.cos, sp.tan,
            sp.sinh, sp.cosh, sp.tanh, sp.Derivative,
        ) or root_like
        if not has_hard_shape:
            return False
        return sp.count_ops(expr) > ops_limit
    except Exception:
        return False


def _has_symbolic_power(expr: sp.Expr) -> bool:
    """True for model powers such as Q**n where the exponent is symbolic."""
    try:
        return any(
            pow_expr.exp.free_symbols
            for pow_expr in expr.atoms(sp.Pow)
        )
    except Exception:
        return False


def _filter_ansatz_subs(expr: sp.Expr, substitutions: Dict[sp.Expr, sp.Expr]) -> Tuple[Dict, Dict]:
    """Split substitutions into exact function/derivative nodes and composites."""
    exact = {}
    composite = {}
    composite_funcs = (
        sp.exp, sp.log, sp.sin, sp.cos, sp.tan,
        sp.sinh, sp.cosh, sp.tanh, sp.sqrt,
    )
    for key, value in substitutions.items():
        try:
            if not expr.has(key):
                continue
            if isinstance(key, sp.Derivative) or (
                isinstance(key, sp.Expr) and key.is_Function and key.func not in composite_funcs
            ):
                exact[key] = value
            else:
                composite[key] = value
        except Exception:
            continue
    return exact, composite


def _apply_filtered_subs(expr: sp.Expr, exact_subs: Dict, composite_subs: Dict) -> sp.Expr:
    """Apply ansatz substitutions cheaply: exact replacements first, composites second."""
    result = expr
    if exact_subs:
        try:
            result = result.xreplace(exact_subs)
        except Exception:
            result = result.subs(exact_subs)

    # Exact replacements can expose composites such as exp(2*Phi)->1.
    if composite_subs:
        filtered_composites = {}
        for key, value in composite_subs.items():
            try:
                if result.has(key):
                    filtered_composites[key] = value
            except Exception:
                pass
        if filtered_composites:
            result = result.xreplace(filtered_composites)
            residual = {
                key: value
                for key, value in filtered_composites.items()
                if result.has(key)
            }
            if residual:
                result = result.subs(residual)

    return sp.powsimp(result, force=True)


def _apply_heavy_ansatz_subs(expr: sp.Expr, substitutions: Dict[sp.Expr, sp.Expr]) -> sp.Expr:
    """
    One-pass ansatz replacement for very large solved expressions.

    The normal path filters each key with expr.has(key), which is fine for
    moderate expressions but costly for Bianchi/fRTLm density expressions.
    Here we keep only exact function/derivative nodes and let xreplace do a
    single tree walk.
    """
    exact = {
        key: value
        for key, value in substitutions.items()
        if isinstance(key, sp.Derivative)
        or (isinstance(key, sp.Expr) and (key.is_Function or key.has(sp.Derivative)))
    }
    if not exact:
        return expr
    try:
        return expr.xreplace(exact)
    except Exception:
        return expr.subs(exact, simultaneous=True)


def _ansatz_result_simplify(expr: sp.Expr) -> sp.Expr:
    """Notebook-style balanced cleanup after ansatz has collapsed geometry."""
    if expr is None:
        return None
    try:
        ops = sp.count_ops(expr)
        if ops > 300:
            return bounded_fraction_cleanup(expr)
        if _has_symbolic_power(expr) and sp.count_ops(expr) > 500:
            return bounded_fraction_cleanup(expr)
        if _is_heavy_transcendental_expr(expr, ops_limit=450):
            return bounded_fraction_cleanup(expr)
        if expr.has(sp.exp, sp.log, sp.sin, sp.cos, sp.tan, sp.sinh, sp.cosh, sp.tanh):
            return _common_factor_cleanup(rational_transcendental_simplify(expr))
        return _common_factor_cleanup(fast_simplify(expr))
    except Exception:
        return expr


def _build_transcendental_subs(expr: sp.Expr, max_blocks: int = 24):
    """Replace repeated large transcendental atoms by dummies during cleanup."""
    trans_funcs = (
        sp.exp, sp.log, sp.sin, sp.cos, sp.tan,
        sp.sinh, sp.cosh, sp.tanh,
    )
    try:
        atoms = []
        for atom in expr.atoms(*trans_funcs):
            if sp.count_ops(atom) >= 2:
                atoms.append(atom)
        if not atoms:
            return {}, {}, []
        atoms = sorted(set(atoms), key=lambda item: (-sp.count_ops(item), str(item)))
        atoms = atoms[:max_blocks]
        dummies = sp.symbols(f'Xans0:{len(atoms)}')
        to_dummy = dict(zip(atoms, dummies))
        from_dummy = dict(zip(dummies, atoms))
        return to_dummy, from_dummy, list(dummies)
    except Exception:
        return {}, {}, []


class FieldEquationSolver:
    """Solver for modified gravity field equations."""
    
    def __init__(self, ctx: 'MetricContext'):
        self.ctx = ctx

    def _solve_linear_fast(
        self,
        equations: List[sp.Eq],
        unknowns: List[sp.Symbol]
    ) -> Optional[Dict[sp.Symbol, sp.Expr]]:
        """Direct notebook-style linsolve path for linear matter systems."""
        if not ENABLE_LINEAR_FAST_SOLVE or not equations or not unknowns:
            return None
        component_solution = self._solve_component_linear_fast(equations, unknowns)
        if component_solution is not None:
            return component_solution
        try:
            selected = [
                eq for eq in equations[:len(unknowns)]
                if hasattr(eq, 'lhs') and hasattr(eq, 'rhs')
            ]
            if len(selected) < len(unknowns):
                return None

            print(f"[SOLVE_LINEAR] Trying direct linsolve for {[str(u) for u in unknowns]}", flush=True)
            solset = sp.linsolve(selected, unknowns)
            if solset is sp.EmptySet:
                return None
            sol_tuple = next(iter(solset), None)
            if sol_tuple is None:
                return None

            solution = _first_complete_solution([tuple(sol_tuple)], unknowns)
            if solution is None:
                return None

            print(f"[SOLVE_LINEAR] Direct linsolve succeeded for {[str(u) for u in unknowns]}", flush=True)
            return {
                sym: solve_final_cleanup(solution[sym])
                for sym in unknowns
            }
        except Exception as e:
            _solver_log(f"[SOLVE_LINEAR] Direct linsolve skipped: {e}")
            return None

    def _solve_component_linear_fast(
        self,
        equations: List[sp.Eq],
        unknowns: List[sp.Symbol],
    ) -> Optional[Dict[sp.Symbol, sp.Expr]]:
        """
        Isolate diagonal matter variables component-by-component.

        Diagonal systems often have tt -> rho, rr -> p/P_r, and angular ->
        P_t.  Solving those as independent one-variable equations avoids
        building a large linsolve system.  Coupled systems return None and use
        the existing linsolve fallback.
        """
        if not ENABLE_COMPONENT_LINEAR_SOLVE:
            return None
        if len(equations) < len(unknowns):
            return None

        try:
            print(
                f"[SOLVE_COMPONENT] Trying component isolation for {[str(u) for u in unknowns]}",
                flush=True,
            )
            solutions = {}
            for eq, sym in zip(equations[:len(unknowns)], unknowns):
                if not hasattr(eq, 'lhs') or not hasattr(eq, 'rhs'):
                    return None

                sol = self._solve_single_separated_linear(eq, sym)
                if sol is None:
                    sol = self._solve_single_linear(eq, sym)
                if sol is None:
                    _solver_log(f"[SOLVE_COMPONENT] no isolated solution for {sym}")
                    return None
                if any(sol.has(other) for other in unknowns):
                    _solver_log(f"[SOLVE_COMPONENT] coupled solution for {sym}")
                    return None

                solutions[sym] = solve_final_cleanup(sol)

            print("[SOLVE_COMPONENT] Component isolation succeeded", flush=True)
            return solutions
        except Exception as e:
            _solver_log(f"[SOLVE_COMPONENT] skipped: {e}")
            return None

    def _solve_single(self, eq: sp.Eq, sym: sp.Symbol) -> sp.Expr:
        """Try to solve a single equation for a single unknown with fallback."""
        separated_sol = self._solve_single_separated_linear(eq, sym)
        if separated_sol is not None:
            print(f"[SOLVE_SINGLE] side-linear isolation succeeded for {sym}", flush=True)
            return separated_sol

        linear_sol = self._solve_single_linear(eq, sym)
        if linear_sol is not None:
            _solver_log(f"[SOLVE_SINGLE] linear isolation succeeded for {sym}")
            return linear_sol

        # Try sympy solve only after cheap linear isolation fails.
        sols = sp.solve(eq, sym, simplify=False)
        _solver_log(f"[SOLVE_SINGLE] sp.solve result: {sols}")
        if sols:
            return solve_light_simplify(sols[0])

        # Fallback: manual linear isolation
        _solver_log(f"[SOLVE_SINGLE] Trying manual isolation for {sym}")
        try:
            # Move everything to one side: lhs - rhs = 0
            lhs_expr = eq.lhs - eq.rhs
            _solver_log_ops("[SOLVE_SINGLE] lhs_expr ops", lhs_expr)

            # Optimization 5: collect on target symbol BEFORE expanding
            if ENABLE_COLLECT_FIRST:
                _solver_log(f"[SOLVE_SINGLE] Using collect-first optimization")
                collected_dict = sp.collect(lhs_expr, sym, evaluate=False)
                _solver_log(f"[SOLVE_SINGLE] collected_dict keys: {list(collected_dict.keys())}")
                
                # Extract coefficient of sym^1 and remainder from dict
                sym_key = sym
                coeff = collected_dict.get(sym_key, 0)
                # Remainder is everything else (sym^0 and higher powers if any)
                remainder_parts = [collected_dict[k] for k in collected_dict if k != sym_key]
                rest = -sp.Add(*remainder_parts) if remainder_parts else 0
                
                _solver_log_ops("[SOLVE_SINGLE] coeff ops", coeff)
                _solver_log_ops("[SOLVE_SINGLE] rest ops", rest)
                
                if coeff != 0:
                    # Expand only the coefficient and rest, not the whole expression
                    coeff_expanded = sp.expand(coeff)
                    rest_expanded = sp.expand(rest)
                    result = solve_light_simplify(rest_expanded / coeff_expanded)
                    _solver_log_ops("[SOLVE_SINGLE] collect-first result ops", result)
                    return result
                
                # If collect-first didn't work, fall through to expand approach
                _solver_log(f"[SOLVE_SINGLE] Collect-first failed (coeff=0), falling back to expand")
            
            # Original approach: collect then get coefficient
            collected = sp.collect(lhs_expr, sym)
            _solver_log_ops("[SOLVE_SINGLE] collected ops", collected)

            # Get coefficient (could be Add with symbol as factor)
            coeff = collected.coeff(sym, 1)
            _solver_log_ops("[SOLVE_SINGLE] coeff ops", coeff)

            # If coeff is 0, try alternative: check if symbol appears at all
            if coeff == 0 or coeff is None:
                # Try pattern matching: extract coefficient manually
                expanded = sp.expand(lhs_expr)
                _solver_log_ops("[SOLVE_SINGLE] expanded ops", expanded)

                # Split into terms containing sym and not containing sym
                _solver_log(f"[SOLVE_SINGLE] expanded free symbols: {expanded.free_symbols}")

                if VERBOSE_LOGS:
                    for term in expanded.args:
                        print(f"[SOLVE_SINGLE]   term ops: {sp.count_ops(term)}, has({sym}): {term.has(sym)}")

                # Try both object-based and name-based matching
                terms_with_sym = [term for term in expanded.args if term.has(sym)]
                terms_without_sym = [term for term in expanded.args if not term.has(sym)]

                # Fallback: name-based matching if object-based fails
                if not terms_with_sym and hasattr(sym, 'name'):
                    terms_with_sym = [term for term in expanded.args
                                      if any(s.name == sym.name for s in term.free_symbols)]
                    terms_without_sym = [term for term in expanded.args
                                         if not any(s.name == sym.name for s in term.free_symbols)]
                    _solver_log(f"[SOLVE_SINGLE] using name-based matching for {sym.name}")

                _solver_log(f"[SOLVE_SINGLE] terms with {sym}: {len(terms_with_sym)}")
                _solver_log(f"[SOLVE_SINGLE] terms without {sym}: {len(terms_without_sym)}")

                if len(terms_with_sym) == 1:
                    # Single term with sym: factor it out
                    term_with = terms_with_sym[0]
                    factored = sp.factor(term_with)
                    # Try to extract coefficient
                    if factored.is_Mul:
                        coeff_factors = [f for f in factored.args
                                         if not f.has(sym) and (not hasattr(sym, 'name') or
                                                                not any(s.name == sym.name for s in f.free_symbols))]
                        coeff = sp.Mul(*coeff_factors) if coeff_factors else 1
                        _solver_log_ops("[SOLVE_SINGLE] extracted coeff ops", coeff)
                    else:
                        coeff = 1

                    rest = -sp.Add(*terms_without_sym) if terms_without_sym else 0
                    _solver_log_ops("[SOLVE_SINGLE] rest ops", rest)

                    if coeff != 0:
                        result = solve_light_simplify(rest / coeff)
                        _solver_log_ops("[SOLVE_SINGLE] manual result ops", result)
                        return result
                elif len(terms_with_sym) > 1:
                    # Multiple terms, try to factor out the symbol
                    sum_with = sp.Add(*terms_with_sym)
                    factored = sp.factor(sum_with)
                    _solver_log_ops("[SOLVE_SINGLE] factored sum ops", factored)
                    coeff = factored.coeff(sym, 1)
                    # Fallback: name-based coeff extraction
                    if coeff == 0 or coeff is None and hasattr(sym, 'name'):
                        # Try to extract by factoring out the symbol by name
                        for arg in factored.args if factored.is_Mul else [factored]:
                            if any(s.name == sym.name for s in arg.free_symbols):
                                # Found symbol, remaining factors are the coeff
                                other_factors = [f for f in factored.args if f != arg] if factored.is_Mul else [1]
                                coeff = sp.Mul(*other_factors) if other_factors else 1
                                break
                    _solver_log_ops("[SOLVE_SINGLE] coeff after factoring ops", coeff)
                    if coeff and coeff != 0:
                        rest = -sp.Add(*terms_without_sym) if terms_without_sym else 0
                        result = solve_light_simplify(rest / coeff)
                        _solver_log_ops("[SOLVE_SINGLE] manual result ops", result)
                        return result

            elif coeff != 0:
                # Standard case: symbol appears with coefficient
                rest = -lhs_expr.subs(sym, 0)
                # Fallback for name-based substitution
                if rest.has(sym) and hasattr(sym, 'name'):
                    for s in rest.free_symbols:
                        if s.name == sym.name:
                            rest = rest.subs(s, 0)
                            break
                result = solve_light_simplify(rest / coeff)
                _solver_log_ops("[SOLVE_SINGLE] standard manual result ops", result)
                return result

            _solver_log(f"[SOLVE_SINGLE] Could not isolate {sym}")
        except Exception as e:
            _solver_log(f"[SOLVE_SINGLE] Manual isolation failed: {e}")
        return None

    def _solve_single_separated_linear(self, eq: sp.Eq, sym: sp.Symbol) -> Optional[sp.Expr]:
        """
        Fast path for field-equation components where one side is purely
        geometric and the other side is linear in one matter variable.

        This avoids differentiating or collecting the giant geometric side,
        which is the expensive move for transcendental f(Q)/f(Q,C) models.
        """
        if not hasattr(eq, 'lhs') or not hasattr(eq, 'rhs'):
            return None
        try:
            rhs_has = eq.rhs.has(sym)
            if rhs_has:
                coeff = eq.rhs.coeff(sym, 1)
                if coeff != 0 and not coeff.has(sym):
                    rest = sp.Integer(0) if eq.rhs == coeff * sym else eq.rhs.subs(sym, 0)
                    return (eq.lhs - rest) / coeff

            lhs_has = eq.lhs.has(sym)
            if lhs_has and not rhs_has:
                coeff = eq.lhs.coeff(sym, 1)
                if coeff != 0 and not coeff.has(sym):
                    rest = sp.Integer(0) if eq.lhs == coeff * sym else eq.lhs.subs(sym, 0)
                    return (eq.rhs - rest) / coeff
        except Exception as e:
            _solver_log(f"[SOLVE_SINGLE] side-linear isolation skipped for {sym}: {e}")
        return None

    def _solve_single_linear(self, eq: sp.Eq, sym: sp.Symbol) -> Optional[sp.Expr]:
        """Fast isolation for equations linear in one matter variable."""
        if not hasattr(eq, 'lhs') or not hasattr(eq, 'rhs'):
            return None
        try:
            expr = eq.lhs - eq.rhs
            if not expr.has(sym):
                return None
            coeff = sp.diff(expr, sym)
            if coeff == 0 or coeff.has(sym):
                return None
            rest = expr.subs(sym, 0)
            return solve_light_simplify(-rest / coeff)
        except Exception as e:
            _solver_log(f"[SOLVE_SINGLE] linear isolation skipped for {sym}: {e}")
            return None

    def solve_sequential(
        self,
        equations: List[sp.Eq],
        unknowns: List[sp.Symbol]
    ) -> Dict[sp.Symbol, sp.Expr]:
        """
        Solve strategy for field equations.
        
        Strategy:
        1. Try the direct notebook-style linsolve system path
        2. Fall back to primary/secondary isolation for non-linear systems
        
        Args:
            equations: List of Eq(LHS, RHS)
            unknowns: List of symbols to solve for
            
        Returns:
            Dict mapping symbols to solutions
        """
        if not equations or not unknowns:
            return {}

        fast = self._solve_linear_fast(equations, unknowns)
        if fast:
            return fast
        
        primary = unknowns[0]
        secondary = unknowns[1:]
        
        # Solve primary from first equation
        _solver_log(f"[SOLVE] Solving for primary: {primary}")
        primary_sol = self._solve_single(equations[0], primary)
        if primary_sol is not None:
            _solver_log_ops("[SOLVE] Primary solution ops", primary_sol)
        else:
            _solver_log("[SOLVE] Primary solution ops: none")

        if primary_sol is None:
            raise ValueError(f"Could not solve for {primary} in equation: {equations[0]}")

        primary_sol = fast_simplify(primary_sol)

        solutions = {primary: primary_sol}

        if secondary and len(equations) > 1:
            # Substitute into remaining equations
            remaining = [
                sp.Eq(eq.lhs.subs(primary, primary_sol), eq.rhs.subs(primary, primary_sol))
                for eq in equations[1:]
                if hasattr(eq, 'lhs') and hasattr(eq, 'rhs')
            ]

            # Solve for secondary unknowns
            secondary_sols = None
            if not _is_hard_solve_system(remaining):
                secondary_sols = sp.solve(remaining, secondary, dict=True, simplify=False)
            complete_secondary = _first_complete_solution(secondary_sols, secondary)
            if complete_secondary is not None:
                secondary_sols = complete_secondary

            # Fallback: try sequential solve for each secondary unknown
            if not secondary_sols and len(remaining) >= len(secondary):
                _solver_log(f"[SOLVE] Fallback for secondary: {secondary}")
                secondary_sols = {}
                for i, sym in enumerate(secondary):
                    if i < len(remaining):
                        sym_sol = self._solve_single(remaining[i], sym)
                        if sym_sol:
                            secondary_sols[sym] = fast_simplify(sym_sol)
                            # Substitute into remaining equations
                            remaining = [
                                sp.Eq(eq.lhs.subs(sym, sym_sol), eq.rhs.subs(sym, sym_sol))
                                for eq in remaining
                                if hasattr(eq, 'lhs') and hasattr(eq, 'rhs')
                            ]

            if secondary_sols:
                if isinstance(secondary_sols, dict):
                    solutions.update(secondary_sols)
                elif isinstance(secondary_sols, list) and len(secondary_sols) > 0:
                    # List of solution tuples
                    for i, sym in enumerate(secondary):
                        solutions[sym] = secondary_sols[0][i]
                else:
                    raise SolveError(f"Unexpected solve result: {secondary_sols}")

                secondary_solution = {sym: solutions[sym] for sym in secondary if sym in solutions}
                if secondary_solution and any(primary_sol.has(sym) for sym in secondary_solution):
                    solutions[primary] = fast_simplify(primary_sol.subs(secondary_solution))
        
        return {
            sym: solve_final_cleanup(val)
            for sym, val in solutions.items()
        }

    def solve_anisotropic_notebook(
        self,
        equations: List[sp.Eq],
        unknowns: List[sp.Symbol]
    ) -> Dict[sp.Symbol, sp.Expr]:
        """
        Notebook-style anisotropic solve for rho, P_r, P_t:
        linsolve([eq1, eq2, eq3], [rho, P_r, P_t]), then fast_simplify
        each component. Ansatz substitution is handled by apply_ansatz().
        """
        if len(equations) < 3 or len(unknowns) < 3:
            return self.solve_sequential(equations, unknowns)

        direct_sols = self._solve_linear_fast(equations[:3], unknowns[:3])
        if direct_sols is not None:
            return direct_sols

        fallback = self._solve_anisotropic_side_linear(equations[:3], unknowns[:3])
        if fallback is not None:
            return fallback

        raise SolveError(
            "Direct anisotropic linsolve did not produce a complete rho/P_r/P_t solution"
        )

    def _solve_anisotropic_side_linear(
        self,
        equations: List[sp.Eq],
        unknowns: List[sp.Symbol],
    ) -> Optional[Dict[sp.Symbol, sp.Expr]]:
        """Bounded component-wise fallback for diagonal anisotropic systems."""
        if len(equations) < 3 or len(unknowns) < 3:
            return None
        solutions = {}
        current_equations = list(equations)
        print("[SOLVE_ANISO] Direct linsolve failed; trying bounded side-linear isolation", flush=True)
        for i, sym in enumerate(unknowns):
            eq = current_equations[i]
            if not hasattr(eq, 'lhs') or not hasattr(eq, 'rhs'):
                return None
            try:
                # Substitute all previously solved symbols into this equation
                # so each component is decoupled before isolation.
                if solutions:
                    eq = sp.Eq(
                        eq.lhs.subs(solutions),
                        eq.rhs.subs(solutions),
                        evaluate=False,
                    )
                sol = self._solve_single_separated_linear(eq, sym)
                if sol is None:
                    sol = self._solve_single_linear(eq, sym)
                if sol is None:
                    return None
                sol = solve_final_cleanup(sol)
                # Reject if the solution still carries another unknown
                if any(sol.has(other) for other in unknowns):
                    return None
                solutions[sym] = sol
            except Exception as e:
                _solver_log(f"[SOLVE_ANISO] bounded fallback skipped for {sym}: {e}")
                return None

        if any(sym not in solutions for sym in unknowns):
            return None
        print("[SOLVE_ANISO] Bounded side-linear isolation succeeded", flush=True)

        # Bug D fix: warn (symbolically) if any two solutions are identical.
        # str() comparison misses mathematically equal but textually different exprs.
        items = list(solutions.items())
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                sym_a, val_a = items[i]
                sym_b, val_b = items[j]
                try:
                    if sp.simplify(val_a - val_b) == 0:
                        print(
                            f"[SOLVE_ANISO] WARNING: {sym_a} and {sym_b} are symbolically identical — "
                            "check that index_pairs ordering matches unknowns ordering"
                        )
                        for sym, val in solutions.items():
                            print(f"[SOLVE_ANISO]   {sym}: {val}")
                        break
                except Exception:
                    if str(val_a) == str(val_b):
                        print(
                            f"[SOLVE_ANISO] WARNING: {sym_a} and {sym_b} appear identical (str fallback)"
                        )

        return solutions
    
    def apply_ansatz(
        self,
        solutions: Dict[sp.Symbol, sp.Expr],
        ansatz_subs: Dict[sp.Symbol, sp.Expr],
        compute_derivatives: bool = True,
        progress_callback: Optional[Any] = None,
    ) -> Tuple[Dict[sp.Symbol, sp.Expr], Optional[Dict[sp.Symbol, sp.Expr]]]:
        """
        Apply ansatz substitutions to solutions.
        
        Also handles derivatives of the metric function by computing
        them analytically from the ansatz expression.
        
        Args:
            solutions: Dict of solved expressions
            ansatz_subs: Dict mapping metric functions to ansatz expressions
            
        Returns:
            Tuple of (simplified_solutions, solution_derivatives)
            solution_derivatives is a dict mapping each symbol to its derivative
            w.r.t. the independent coordinate (for deferred diff optimization)
        """
        # Build extended substitution dict including derivatives
        extended_subs = dict(ansatz_subs)
        
        # For each function substitution, also compute its derivatives
        t = self.ctx.independent_coord
        for func, ansatz_expr in ansatz_subs.items():
            if isinstance(func, sp.FunctionClass):
                func_applied = func(t)
                extended_subs[func_applied] = ansatz_expr
                # First derivative of applied function
                d1 = sp.Derivative(func_applied, t)
                d1_val = sp.diff(ansatz_expr, t)
                extended_subs[d1] = d1_val
                # Second derivative  
                d2 = sp.Derivative(func_applied, t, 2)
                d2_val = sp.diff(ansatz_expr, t, 2)
                extended_subs[d2] = d2_val
            elif isinstance(func, sp.Expr) and func.has(sp.Function):
                # First derivative
                d1 = sp.Derivative(func, t)
                d1_val = sp.diff(ansatz_expr, t)
                extended_subs[d1] = d1_val
                # Second derivative  
                d2 = sp.Derivative(func, t, 2)
                d2_val = sp.diff(ansatz_expr, t, 2)
                extended_subs[d2] = d2_val
        
        _solver_log(f"[APPLY_ANSATZ] extended_subs keys: {len(extended_subs)}")
        final = {}
        solution_derivatives = {} if (ENABLE_DEFERRED_DIFF and compute_derivatives) else None
        
        for sym, expr in solutions.items():
            print(f"[SOLVER] Processing {sym}", flush=True)
            heavy_expr = _is_heavy_transcendental_expr(expr)
            if progress_callback:
                progress_callback(sym, 'substitution')
            _solver_log_ops("[SOLVER] Before subs ops", expr)
            _solver_log(f"[SOLVER] Type of expr: {type(expr)}")
            _solver_log(f"[SOLVER] Expr has Derivative: {expr.has(sp.Derivative)}")
            if heavy_expr:
                substituted = _apply_heavy_ansatz_subs(expr, extended_subs)
            else:
                exact_subs, composite_subs = _filter_ansatz_subs(expr, extended_subs)
                _solver_log(
                    f"[SOLVER] needed substitutions: exact={len(exact_subs)}, "
                    f"composite={len(composite_subs)}"
                )

                # Check if keys match what's in the expression
                if VERBOSE_LOGS:
                    for key in exact_subs:
                        print(f"[SOLVER]   Found exact substitution key")

                # Substitute ansatz including derivatives
                substituted = _apply_filtered_subs(expr, exact_subs, composite_subs)
            _solver_log_ops("[SOLVER] After subs ops", substituted)
            
            # Check for remaining Derivative objects
            remaining_derivs = substituted.atoms(sp.Derivative)
            if remaining_derivs:
                _solver_log(f"[SOLVER] Remaining derivative count: {len(remaining_derivs)}")
                if VERBOSE_LOGS:
                    for d in remaining_derivs:
                        print(f"[SOLVER]   Trying to eval derivative")
            
            # Evaluate any remaining unevaluated derivatives only when needed.
            if progress_callback:
                progress_callback(sym, 'derivatives' if remaining_derivs else 'cleanup')
            evaluated = substituted.doit()
            if not (heavy_expr or _is_heavy_transcendental_expr(evaluated)):
                evaluated = physical_domain_simplify(evaluated)
            _solver_log_ops("[SOLVER] After doit ops", evaluated)
            
            # Optimization 3: Compute derivatives BEFORE final simplification
            if ENABLE_DEFERRED_DIFF and solution_derivatives is not None and t is not None:
                if progress_callback:
                    progress_callback(sym, 'stability derivatives')
                # Compute derivative of the substituted (but not yet simplified) expression
                if _is_heavy_transcendental_expr(evaluated, ops_limit=450):
                    raw_derivative = sp.Derivative(evaluated, t, evaluate=False)
                else:
                    raw_derivative = sp.diff(evaluated, t)
                solution_derivatives[sym] = raw_derivative
                _solver_log_ops(f"[SOLVER] Deferred diff: d{sym}/d{t} ops", raw_derivative)
            
            # Simplify
            if progress_callback:
                progress_callback(sym, 'simplification')
            simplified = _ansatz_result_simplify(evaluated)
            _solver_log_ops("[SOLVER] After simplify ops", simplified)
            final[sym] = simplified
            
        return final, solution_derivatives
    
    def _compute_energy_conditions_impl(
        self,
        rho: sp.Expr,
        Pr: sp.Expr,
        Pt: sp.Expr = None
    ) -> Dict[str, sp.Expr]:
        """Internal implementation for energy conditions computation."""
        _solver_log("[DIAGNOSTICS] Computing energy-condition expressions")
        Pt = Pt or Pr

        if FAST_DIAGNOSTICS:
            NEC_r = lightweight_diagnostic_simplify(rho + Pr)
            NEC_t = lightweight_diagnostic_simplify(rho + Pt)
            WEC = rho
            SEC = lightweight_diagnostic_simplify(rho + Pr + 2 * Pt)
            DEC_r = rho - sp.Abs(Pr)
            DEC_t = rho - sp.Abs(Pt)
        else:
            NEC_r = diagnostic_simplify(rho + Pr)
            NEC_t = diagnostic_simplify(rho + Pt)
            WEC = diagnostic_simplify(rho)
            SEC = diagnostic_simplify(rho + Pr + 2 * Pt)

            def _dec_expr(p_expr):
                p_s = diagnostic_simplify(p_expr)
                try:
                    dec_pos = sp.refine(rho - sp.Abs(p_s), sp.Q.positive(p_s))
                    if not dec_pos.has(sp.Abs):
                        return diagnostic_simplify(dec_pos)
                    dec_neg = sp.refine(rho - sp.Abs(p_s), sp.Q.negative(p_s))
                    if not dec_neg.has(sp.Abs):
                        return diagnostic_simplify(dec_neg)
                except Exception:
                    pass
                try:
                    abs_form = diagnostic_simplify(rho - sp.Abs(p_s))
                    if not abs_form.has(sp.Abs):
                        return abs_form
                except Exception:
                    pass
                plus = diagnostic_simplify(rho + p_s)
                minus = diagnostic_simplify(rho - p_s)
                return sp.Piecewise((minus, sp.Ge(p_s, 0)), (plus, True))

            DEC_r = _dec_expr(Pr)
            DEC_t = _dec_expr(Pt)

        return {
            'NEC_r': NEC_r,
            'NEC_t': NEC_t,
            'WEC': WEC,
            'SEC': SEC,
            'DEC_r': DEC_r,
            'DEC_t': DEC_t,
        }
    
    def compute_energy_conditions(
        self,
        rho: sp.Expr,
        Pr: sp.Expr,
        Pt: sp.Expr = None,
        lazy: bool = None
    ) -> Dict[str, Any]:
        """
        Compute energy conditions from density and pressures.
        
        Args:
            rho: Energy density (final, ansatz-substituted)
            Pr: Radial pressure (final)
            Pt: Tangential pressure (final, optional)
            lazy: If True, return LazyResult objects. Defaults to ENABLE_LAZY_DERIVED.
            
        Returns:
            Dict with NEC, WEC, SEC, DEC expressions (or LazyResults if lazy=True)
        """
        if lazy is None:
            lazy = ENABLE_LAZY_DERIVED
        
        if lazy:
            # Compute all energy conditions in a single deferred call to avoid
            # calling the full _impl N times (once per field).
            def _ec_all(): return self._compute_energy_conditions_impl(rho, Pr, Pt)
            _ec_all.__name__ = 'energy_conditions'
            all_ecs = LazyResult(_ec_all)
            def _nec_r(): return all_ecs.evaluate()['NEC_r']
            def _nec_t(): return all_ecs.evaluate()['NEC_t']
            def _wec():   return all_ecs.evaluate()['WEC']
            def _sec():   return all_ecs.evaluate()['SEC']
            def _dec_r(): return all_ecs.evaluate()['DEC_r']
            def _dec_t(): return all_ecs.evaluate()['DEC_t']
            _nec_r.__name__ = 'NEC_r'; _nec_t.__name__ = 'NEC_t'
            _wec.__name__   = 'WEC';   _sec.__name__   = 'SEC'
            _dec_r.__name__ = 'DEC_r'; _dec_t.__name__ = 'DEC_t'
            return {
                'NEC_r': LazyResult(_nec_r),
                'NEC_t': LazyResult(_nec_t),
                'WEC':   LazyResult(_wec),
                'SEC':   LazyResult(_sec),
                'DEC_r': LazyResult(_dec_r),
                'DEC_t': LazyResult(_dec_t),
            }
        else:
            return self._compute_energy_conditions_impl(rho, Pr, Pt)
    
    def _compute_eos_impl(
        self,
        rho: sp.Expr,
        Pr: sp.Expr,
        Pt: sp.Expr = None
    ) -> Dict[str, sp.Expr]:
        """Internal implementation for EoS computation with lightweight direct construction."""
        _solver_log("[DIAGNOSTICS] Computing equation-of-state expressions")
        Pt = Pt or Pr

        P_eff = sp.Rational(1, 3) * (Pr + 2 * Pt)

        if rho == sp.S.Zero:
            return {
                'omega_r': None,
                'omega_t': None,
                'omega_eff': None,
            }

        if FAST_DIAGNOSTICS:
            omega_r = lightweight_diagnostic_simplify(Pr / rho, is_ratio=True)
            omega_t = lightweight_diagnostic_simplify(Pt / rho, is_ratio=True)
            omega_eff = lightweight_diagnostic_simplify(P_eff / rho, is_ratio=True)
        elif ENABLE_SHARED_DENOM:
            rho_factored = diagnostic_simplify(rho)
            _solver_log(
                f"[EOS] Shared denominator: rho simplified (ops before: {sp.count_ops(rho)}, after: {sp.count_ops(rho_factored)})"
            )
            omega_r = diagnostic_simplify(Pr / rho_factored)
            omega_t = diagnostic_simplify(Pt / rho_factored)
            omega_eff = diagnostic_simplify(P_eff / rho_factored)
        else:
            omega_r = diagnostic_simplify(Pr / rho)
            omega_t = diagnostic_simplify(Pt / rho)
            omega_eff = diagnostic_simplify(P_eff / rho)

        return {
            'omega_r': omega_r,
            'omega_t': omega_t,
            'omega_eff': omega_eff,
        }
    
    def compute_eos(
        self,
        rho: sp.Expr,
        Pr: sp.Expr,
        Pt: sp.Expr = None,
        lazy: bool = None
    ) -> Dict[str, Any]:
        """
        Compute equation of state parameters.
        
        Args:
            rho: Energy density (final, ansatz-substituted)
            Pr: Radial pressure (final)
            Pt: Tangential pressure (final, optional)
            lazy: If True, return LazyResult objects. Defaults to ENABLE_LAZY_DERIVED.
            
        Returns:
            Dict with ω_r, ω_t, ω_eff expressions (or LazyResults if lazy=True)
        """
        if lazy is None:
            lazy = ENABLE_LAZY_DERIVED
        
        if lazy:
            def _eos_all(): return self._compute_eos_impl(rho, Pr, Pt)
            _eos_all.__name__ = 'eos'
            all_eos = LazyResult(_eos_all)
            def _omega_r():   return all_eos.evaluate()['omega_r']
            def _omega_t():   return all_eos.evaluate()['omega_t']
            def _omega_eff(): return all_eos.evaluate()['omega_eff']
            _omega_r.__name__   = 'omega_r'
            _omega_t.__name__   = 'omega_t'
            _omega_eff.__name__ = 'omega_eff'
            return {
                'omega_r':   LazyResult(_omega_r),
                'omega_t':   LazyResult(_omega_t),
                'omega_eff': LazyResult(_omega_eff),
            }
        else:
            return self._compute_eos_impl(rho, Pr, Pt)
    
    

def _force_eval_partials(expr: Optional[sp.Expr]) -> Optional[sp.Expr]:
    """Evaluate remaining partial derivatives aggressively but safely."""
    if expr is None:
        return None
    try:
        if expr.has(sp.Derivative):
            expr = expr.doit()
    except Exception:
        pass
    try:
        derivs = list(expr.atoms(sp.Derivative))
        if derivs:
            repl = {}
            for d in derivs:
                try:
                    repl[d] = d.doit()
                except Exception:
                    continue
            if repl:
                expr = expr.xreplace(repl)
    except Exception:
        pass
    return expr


def _compute_speed_of_sound_impl(
        self,
        rho: sp.Expr,
        Pr: sp.Expr,
        Pt: sp.Expr = None,
        independent_coord: sp.Symbol = None,
        matter_derivatives: Optional[Dict[sp.Symbol, sp.Expr]] = None
    ) -> Dict[str, sp.Expr]:
        """
        Internal implementation for speed of sound with deferred diff optimization.
        
        Optimization 3: If matter_derivatives is provided, use pre-computed derivatives
        from apply_ansatz() instead of differentiating the simplified expressions again.
        """
        _solver_log("[DIAGNOSTICS] Simplifying speed-of-sound expressions")
        Pt = Pt or Pr
        coord = independent_coord or self.ctx.independent_coord
        
        if coord is None:
            return {'cs2_r': None, 'cs2_t': None}

        # Vacuum / fully vanishing matter branch: stability is undefined.
        if rho == sp.S.Zero and Pr == sp.S.Zero and Pt == sp.S.Zero:
            return {'cs2_r': None, 'cs2_t': None}
        
        # Find the symbols for rho and Pr in the derivatives dict
        rho_sym = None
        pr_sym = None
        pt_sym = None
        
        # Try to find matching symbols in matter_derivatives keys
        if matter_derivatives:
            for sym in matter_derivatives:
                if str(sym) == 'rho':
                    rho_sym = sym
                elif str(sym) == 'p' or str(sym) == 'P_r' or str(sym) == 'Pr':
                    pr_sym = sym
                elif str(sym) == 'P_t' or str(sym) == 'Pt':
                    pt_sym = sym
        
        # Optimization 3: Use pre-computed derivatives if available
        if ENABLE_DEFERRED_DIFF and matter_derivatives and rho_sym and pr_sym:
            print(f"[SPEED_OF_SOUND] Using deferred differentiation (pre-computed derivatives)")
            dPr = matter_derivatives.get(pr_sym)
            # F6 fix: do NOT fall back to dPr when pt_sym is None — that silently
            # equates cs2_t to cs2_r for any anisotropic run where the P_t symbol
            # name didn't match.  Keep dPt as None; cs2_t will be set to cs2_r only
            # below if dPt is genuinely absent (i.e., isotropic fluid).
            dPt = matter_derivatives.get(pt_sym) if pt_sym is not None else None
            dRho = matter_derivatives.get(rho_sym)
            
            if dPr is not None and dRho is not None:
                # Always evaluate remaining partials, even in fast mode.
                dPr = _force_eval_partials(dPr)
                dPt = _force_eval_partials(dPt) if dPt is not None else None
                dRho = _force_eval_partials(dRho)
                if dRho == sp.Integer(0) or dRho == sp.S.Zero:
                    print(f"[SPEED_OF_SOUND] dρ/d{coord} = 0 — speed of sound undefined (static rho)")
                    return {'cs2_r': None, 'cs2_t': None}
                cs2_r = diagnostic_simplify(_force_eval_partials(dPr / dRho))
                cs2_t = diagnostic_simplify(_force_eval_partials(dPt / dRho)) if dPt is not None else cs2_r
                return {'cs2_r': cs2_r, 'cs2_t': cs2_t}
            else:
                print(f"[SPEED_OF_SOUND] Deferred diff missing derivatives, falling back to direct")
        
        # Fallback: Direct differentiation (original behavior)
        print(f"[SPEED_OF_SOUND] Using direct differentiation")
        dRho_direct = _force_eval_partials(sp.diff(rho, coord))
        # F9 fix: guard against dRho = 0 (e.g. static ansatz where ρ is constant in coord).
        # Returning None instead of zoo prevents complex-infinity from propagating to the UI.
        if dRho_direct == sp.Integer(0) or dRho_direct == sp.S.Zero:
            print(f"[SPEED_OF_SOUND] dρ/d{coord} = 0 — speed of sound undefined (static rho)")
            return {'cs2_r': None, 'cs2_t': None}
        dPr_direct = _force_eval_partials(sp.diff(Pr, coord))
        dPt_direct = _force_eval_partials(sp.diff(Pt, coord))
        cs2_r = diagnostic_simplify(_force_eval_partials(dPr_direct / dRho_direct))
        cs2_t = diagnostic_simplify(_force_eval_partials(dPt_direct / dRho_direct))
        
        return {'cs2_r': cs2_r, 'cs2_t': cs2_t}
    
def compute_speed_of_sound(
    self,
    rho: sp.Expr,
    Pr: sp.Expr,
    Pt: sp.Expr = None,
    independent_coord: sp.Symbol = None,
    matter_derivatives: Optional[Dict[sp.Symbol, sp.Expr]] = None,
    lazy: bool = None
) -> Dict[str, Any]:
    """
    Compute speed of sound squared: c_s² = dp/dρ.

    Uses parametric differentiation: (dp/dt) / (dρ/dt)

    Args:
        rho: Energy density (final, ansatz-substituted)
        Pr: Radial pressure (final)
        Pt: Tangential pressure (final, optional)
        independent_coord: Independent coordinate symbol (t or r)
        matter_derivatives: Optional dict of pre-computed derivatives from apply_ansatz()
        lazy: If True, return LazyResult objects. Defaults to ENABLE_LAZY_DERIVED.

    Returns:
        Dict with cs²_r, cs²_t expressions (or LazyResults if lazy=True)
    """
    if lazy is None:
        lazy = ENABLE_LAZY_DERIVED

    if lazy:
        def _cs2_all():
            return self._compute_speed_of_sound_impl(rho, Pr, Pt, independent_coord, matter_derivatives)
        _cs2_all.__name__ = 'speed_of_sound'
        all_cs2 = LazyResult(_cs2_all)

        def _cs2_r():
            return all_cs2.evaluate()['cs2_r']

        def _cs2_t():
            return all_cs2.evaluate()['cs2_t']

        _cs2_r.__name__ = 'cs2_r'
        _cs2_t.__name__ = 'cs2_t'
        return {
            'cs2_r': LazyResult(_cs2_r),
            'cs2_t': LazyResult(_cs2_t),
        }

    return self._compute_speed_of_sound_impl(rho, Pr, Pt, independent_coord, matter_derivatives)


FieldEquationSolver._compute_speed_of_sound_impl = _compute_speed_of_sound_impl
FieldEquationSolver.compute_speed_of_sound = compute_speed_of_sound


# ─── Enhanced Simplification Helper Functions (moved from display_simplify.py) ────




def _enhanced_cancel_simplify(expr: sp.Expr) -> sp.Expr:
    """Enhanced cancellation with readability improvements."""
    try:
        result = sp.cancel(expr)
        result = sp.powsimp(result, force=True)
        result = _collect_model_params(result)
        result = _factor_common_display_terms(result)
        return result
    except Exception:
        return expr




def _collect_model_params(expr: sp.Expr) -> sp.Expr:
    # Shim — delegates to the canonical copy in display_simplify.py.
    from core.display_simplify import _collect_model_params as _ds_collect
    return _ds_collect(expr)


def _factor_common_display_terms(expr: sp.Expr) -> sp.Expr:
    # Thin shim — delegates to the canonical implementation in display_simplify.py
    # which uses kernel-substitution and LaTeX ranking.  This shim exists solely
    # to avoid renaming all call-sites inside this module.
    from core.display_simplify import _factor_common_display_terms as _ds_factor
    return _ds_factor(expr)


def _factor_common_additive_part(expr: sp.Expr) -> sp.Expr:
    # Shim — delegates to the canonical implementation in display_simplify.py.
    from core.display_simplify import _factor_common_additive_part as _ds_additive
    return _ds_additive(expr)


# ─── Simplified Partition Strategy Functions ───────────────────────────────────

def _partition_expression(expr: sp.Expr, target_ops: int) -> list:
    """Partition expression into manageable pieces."""
    if isinstance(expr, sp.Add):
        return _partition_add_terms(list(expr.args), target_ops)
    return []


def _partition_add_terms(terms: list, target_ops: int) -> list:
    """Partition additive terms by operation count."""
    pieces = []
    current = []
    current_ops = 0
    for term in terms:
        term_ops = sp.count_ops(term)
        if current and current_ops + term_ops > target_ops:
            pieces.append(sp.Add(*current, evaluate=False))
            current = []
            current_ops = 0
        current.append(term)
        current_ops += term_ops
    if current:
        pieces.append(sp.Add(*current, evaluate=False))
    return pieces


def _build_partition_tasks(pieces: list, target_ops: int) -> list:
    """Build tasks for partition simplification."""
    tasks = []
    for piece_idx, piece in enumerate(pieces):
        tasks.append({
            "key": (piece_idx, "whole", 0),
            "piece_idx": piece_idx,
            "role": "whole",
            "kind": "plain",
            "part_idx": 0,
            "expr": piece,
        })
    return tasks


def _rebuild_partition_pieces(pieces: list, tasks: list, completed: dict) -> list:
    """Rebuild pieces from simplified tasks."""
    rebuilt = []
    for piece_idx, original in enumerate(pieces):
        piece_tasks = [task for task in tasks if task["piece_idx"] == piece_idx]
        if not piece_tasks:
            rebuilt.append(original)
            continue
        
        ordered = sorted(piece_tasks, key=lambda task: task["part_idx"])
        parts = [completed.get(task["key"], task["expr"]) for task in ordered]
        if len(parts) == 1:
            rebuilt.append(parts[0])
        else:
            rebuilt.append(sp.Add(*parts, evaluate=False))
    return rebuilt


def _join_partitioned_expression(original: sp.Expr, pieces: list) -> sp.Expr:
    """Join partitioned pieces back together."""
    return sp.Add(*pieces, evaluate=False)


def _final_join_pass(expr: sp.Expr) -> sp.Expr:
    """Final cleanup pass for joined expressions."""
    try:
        ops = sp.count_ops(expr)
        expr = _strip_physical_abs(sp.powsimp(expr, force=True))
        if ops <= 18000:
            expr = sp.factor_terms(expr)
        if ops <= 9000:
            expr = _collect_model_params(expr)
        expr = _factor_common_display_terms(expr)
    except Exception:
        pass
    return expr





def _simplify_fraction_parts(expr: sp.Expr) -> sp.Expr:
    """Simplify numerator and denominator separately."""
    num, den = sp.fraction(expr)
    num = _simplify_piece(num)
    den = _simplify_piece(den)
    return num / den


def _simplify_fraction_together(expr: sp.Expr) -> sp.Expr:
    """Simplify with together then separate piece simplification."""
    num, den = sp.fraction(sp.together(expr))
    num = _simplify_piece_heavy(num)
    den = _simplify_piece_heavy(den)
    return sp.factor_terms(num / den)


def _simplify_additive_chunks(expr: sp.Expr, together: bool = False) -> sp.Expr:
    """Simplify additive expressions in chunks."""
    if not isinstance(expr, sp.Add):
        return _simplify_piece_heavy(expr) if together else _simplify_piece(expr)
    
    terms = list(expr.args)
    chunks = []
    current = []
    current_ops = 0
    for term in terms:
        term_ops = sp.count_ops(term)
        if current and current_ops + term_ops > (950 if together else 650):
            chunks.append(current)
            current = []
            current_ops = 0
        current.append(term)
        current_ops += term_ops
    if current:
        chunks.append(current)
    
    cleaner = _simplify_piece_heavy if together else _simplify_piece
    cleaned = [cleaner(sp.Add(*chunk, evaluate=False)) for chunk in chunks]
    return sp.Add(*cleaned, evaluate=False)


def _simplify_piece(piece: sp.Expr) -> sp.Expr:
    """Light simplification for expression pieces."""
    ops = sp.count_ops(piece)
    piece = sp.powsimp(piece, force=True)
    if ops <= 700:
        piece = sp.cancel(piece)
    if ops <= 1500:
        piece = sp.factor_terms(piece)
    return _collect_model_params(piece)


def _simplify_piece_heavy(piece: sp.Expr) -> sp.Expr:
    """Heavy simplification for expression pieces."""
    ops = sp.count_ops(piece)
    piece = sp.powsimp(piece, force=True)
    if ops <= 2200:
        piece = sp.cancel(sp.together(piece))
    elif ops <= 4500:
        piece = sp.cancel(piece)
    piece = sp.factor_terms(piece)
    return _collect_model_params(piece)


# ─── Structure-Aware Simplification Helpers ───────────────────────────────────────

def _build_symbolic_power_subs(expr: sp.Expr, max_blocks: int = 24, min_ops: int = 6) -> tuple:
    """
    Replace large symbolic-power atoms like (huge_expr)**n by dummy symbols.
    
    This protects model kernels such as Q**n, C**n, (Q + alpha*C)**n,
    or repeated f(Q,C) model arguments.
    
    Returns:
        tuple: (to_dummy_subs, from_dummy_subs, dummy_symbols)
    """
    if expr is None:
        return {}, [], []
    
    power_atoms = expr.atoms(sp.Pow)
    symbolic_powers = []
    
    for pow_expr in power_atoms:
        base, exp = pow_expr.base, pow_expr.exp
        
        # Check if exponent has free symbols (symbolic power)
        if not exp.free_symbols:
            continue
            
        # Check if base or full power is nontrivial
        base_ops = sp.count_ops(base)
        full_ops = sp.count_ops(pow_expr)
        
        if base_ops >= min_ops or full_ops >= min_ops * 2:
            symbolic_powers.append(pow_expr)
    
    if not symbolic_powers or len(symbolic_powers) > max_blocks:
        return {}, [], []
    
    # Create dummy symbols for protection
    dummy_symbols = []
    to_dummy_subs = {}
    from_dummy_subs = []
    
    for i, pow_expr in enumerate(symbolic_powers):
        dummy = sp.Symbol(f"_XPow{i}")
        dummy_symbols.append(dummy)
        to_dummy_subs[pow_expr] = dummy
        from_dummy_subs.append((dummy, pow_expr))
    
    return to_dummy_subs, from_dummy_subs, dummy_symbols


def _build_repeated_denominator_subs(expr: sp.Expr, min_count: int = 2, min_ops: int = 8) -> tuple:
    """
    Detect repeated nontrivial denominator bases and replace them by dummy symbols.
    
    Returns:
        tuple: (to_dummy_subs, from_dummy_subs, dummy_symbols)
    """
    if expr is None:
        return {}, [], []
    
    # Walk expression to find denominator bases
    denominator_bases = {}
    
    for node in sp.preorder_traversal(expr):
        if isinstance(node, sp.Pow) and node.exp.is_Number and node.exp < 0:
            # This is a denominator (negative exponent)
            base = node.base
            base_ops = sp.count_ops(base)
            
            if base_ops >= min_ops:
                if base not in denominator_bases:
                    denominator_bases[base] = 0
                denominator_bases[base] += 1
    
    # Find repeated denominator bases
    repeated_bases = {base: count for base, count in denominator_bases.items() 
                   if count >= min_count}
    
    if not repeated_bases:
        return {}, [], []
    
    # Create dummy symbols for repeated denominators
    dummy_symbols = []
    to_dummy_subs = {}
    from_dummy_subs = []
    
    for i, (base, count) in enumerate(repeated_bases.items()):
        dummy = sp.Symbol(f"_Den{i}")
        dummy_symbols.append(dummy)
        to_dummy_subs[base] = dummy
        from_dummy_subs.append((dummy, base))
    
    return to_dummy_subs, from_dummy_subs, dummy_symbols


def geometric_kernel_cleanup(expr: sp.Expr) -> sp.Expr:
    """
    Compress common geometric factors such as k*r**2 - 1.
    Useful for FRW_curved and spherical expressions.
    """
    if expr is None:
        return None
    
    try:
        free_symbols = expr.free_symbols
        
        # Look for k and r symbols
        k_sym = None
        r_sym = None
        
        for sym in free_symbols:
            if str(sym) == 'k':
                k_sym = sym
            elif str(sym) == 'r':
                r_sym = sym
        
        if k_sym is None or r_sym is None:
            return expr
        
        # Define geometric kernel G = k*r**2 - 1
        G_geom = sp.Symbol("_Ggeom")
        G_kernel = k_sym * r_sym**2 - 1
        
        # Apply pattern matching and compression
        def compress_geometric_powers(sub_expr):
            if not isinstance(sub_expr, sp.Add):
                return sub_expr
            
            # Try to match patterns like k**n*r**(2n) - n*k**(n-1)*r**(2(n-1)) + ... 
            # This is (k*r**2 - 1)**n expansion
            
            # For now, handle simple cases
            if sub_expr.has(G_kernel):
                # Replace G_kernel with dummy, compress, then restore
                compressed = sub_expr.xreplace({G_kernel: G_geom})
                compressed = sp.factor_terms(compressed)
                
                # Try to factor powers of G_geom
                if compressed.has(G_geom):
                    try:
                        # Extract power of G_geom if possible
                        if isinstance(compressed, sp.Pow) and compressed.base == G_geom:
                            return compressed
                        elif isinstance(compressed, sp.Mul):
                            # Look for G_geom**n factors
                            factors = list(compressed.args)
                            geom_factors = [f for f in factors if isinstance(f, sp.Pow) and f.base == G_geom]
                            if geom_factors:
                                return compressed
                    except Exception:
                        pass
                
                return compressed.xreplace({G_geom: G_kernel})
            
            return sub_expr
        
        # Apply compression recursively
        result = expr
        for _ in range(3):  # Apply a few times to catch nested patterns
            result = compress_geometric_powers(result)
            if result == expr:
                break
            expr = result
        
        return result
        
    except Exception:
        return expr


def _cost_balanced_chunks(terms: list, max_chunk_ops: int = 1800) -> list:
    """
    Split Add terms into chunks based on operation cost, not fixed term count.
    """
    if not terms:
        return []
    
    chunks = []
    current_chunk = []
    current_ops = 0
    
    for term in terms:
        term_ops = sp.count_ops(term)
        
        # If adding this term would exceed limit, start new chunk
        if current_chunk and current_ops + term_ops > max_chunk_ops:
            if current_chunk:
                chunks.append(sp.Add(*current_chunk, evaluate=False))
            current_chunk = [term]
            current_ops = term_ops
        else:
            current_chunk.append(term)
            current_ops += term_ops
    
    # Add final chunk
    if current_chunk:
        chunks.append(sp.Add(*current_chunk, evaluate=False))
    
    return chunks


def _accept_if_better(original: sp.Expr, candidate: sp.Expr, max_growth: float = 1.15) -> sp.Expr:
    """
    Return candidate only if it is not significantly more complex than original
    and passes a fast numerical equivalence check.
    """
    if candidate is None:
        return original

    if candidate == original:
        return original

    try:
        original_ops = sp.count_ops(original)
        candidate_ops = sp.count_ops(candidate)

        growth_ratio = candidate_ops / original_ops if original_ops > 0 else float('inf')
        abs_growth = candidate_ops - original_ops
        acceptable_growth = candidate_ops <= original_ops or growth_ratio <= max_growth or abs_growth < 50
        if not acceptable_growth:
            return original

        if not _numerically_equivalent(original, candidate):
            return original

        return candidate
    except Exception:
        return original

def model_kernel_structured_cleanup(expr: sp.Expr, *, display: bool = False) -> sp.Expr:
    """
    Best path for f(Q), f(Q,C), f(R), and symbolic-power models.
    Protects symbolic powers, repeated denominators, and large geometric kernels.
    """
    if expr is None:
        return None
    
    original = expr
    
    try:
        # Save original
        # sympify expression (already sympy)
        
        # Apply basic cleanup
        expr = physical_domain_simplify(expr)
        expr = _strip_physical_abs(expr)
        expr = sp.powsimp(expr, force=True)
        
        # Apply geometric kernel cleanup
        expr = geometric_kernel_cleanup(expr)
        
        # Protect symbolic powers
        power_subs, power_from_subs, power_dummies = _build_symbolic_power_subs(expr)
        if power_subs:
            expr = expr.xreplace(power_subs)
        
        # Protect repeated denominators
        denom_subs, denom_from_subs, denom_dummies = _build_repeated_denominator_subs(expr)
        if denom_subs:
            expr = expr.xreplace(denom_subs)
        
        # Split using sp.fraction()
        num, den = sp.fraction(expr)
        
        # Simplify numerator and denominator separately
        def clean_piece(piece):
            if piece is None or piece.is_Number:
                return piece
            
            if isinstance(piece, sp.Add):
                chunks = _cost_balanced_chunks(list(piece.args))
                cleaned_chunks = []
                for chunk in chunks:
                    # Apply bounded local cleanup to each chunk
                    chunk_ops = sp.count_ops(chunk)
                    if chunk_ops <= 800:
                        cleaned_chunk = sp.cancel(sp.together(chunk))
                        cleaned_chunk = sp.powsimp(cleaned_chunk, force=True)
                    else:
                        # For large chunks, apply lighter cleanup
                        cleaned_chunk = sp.powsimp(chunk, force=True)
                        if chunk_ops <= 1500:
                            cleaned_chunk = sp.cancel(cleaned_chunk)
                    cleaned_chunks.append(cleaned_chunk)
                return sp.Add(*cleaned_chunks, evaluate=False)
            else:
                # For non-Add pieces, apply standard cleanup
                piece_ops = sp.count_ops(piece)
                if piece_ops <= 800:
                    piece = sp.cancel(sp.together(piece))
                piece = sp.powsimp(piece, force=True)
                if piece_ops <= 1500:
                    piece = sp.cancel(piece)
                return piece
        
        num = clean_piece(num)
        den = clean_piece(den)
        
        # Recombine as num / den
        expr = num / den
        
        # Collect by protected dummy symbols
        all_dummies = power_dummies + denom_dummies
        if all_dummies:
            expr = sp.collect(expr, all_dummies, evaluate=True)
        
        # Restore denominator dummies
        for dummy, original in denom_from_subs:
            expr = expr.xreplace({dummy: original})
        
        # Restore symbolic-power dummies
        for dummy, original in power_from_subs:
            expr = expr.xreplace({dummy: original})
        
        # Apply final cleanup
        expr = sp.powsimp(expr, force=True)
        expr = geometric_kernel_cleanup(expr)
        expr = _collect_model_params(expr)
        
        # Only apply factor_terms() when safe
        final_ops = sp.count_ops(expr)
        if final_ops <= 3000:
            expr = sp.factor_terms(expr)
        
        expr = physical_domain_simplify(expr)
        
        # Apply acceptance check
        return _accept_if_better(original, expr)
        
    except Exception as e:
        print(f"[MODEL_KERNEL_CLEANUP] failed: {e}")
        return original


class SolveError(Exception):
    """Raised when field equation solving fails."""
    pass
