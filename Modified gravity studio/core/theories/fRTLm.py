"""
f(R,T,Lm) Gravity Theory Module

Field equation:
fR R_{μν}  +  g_{μν} □fR  −  ∇_μ∇_ν fR  −  ½ f g_{μν}
−  ½(fL + 2fT)(T_{μν} − Lm g_{μν})  =  8π T_{μν}

Scalars displayed: R, T_scalar, Lm
"""

import builtins
import sys
if not hasattr(builtins, 'display'):
    builtins.display = lambda *args, **kwargs: None

from typing import Tuple, Dict, Any, NamedTuple, List
import sympy as sp


# ─── Model Classification for Non-linear Coupling Detection ─────────────────

class ModelClass(NamedTuple):
    """Classification result for f(R,T,Lm) model."""
    is_nonlinear: bool
    reason: str
    coupling_terms: List[sp.Expr]


def classify_model(model_expr: sp.Expr, R_sym: sp.Symbol, Ts_sym: sp.Symbol, L_sym: sp.Symbol) -> ModelClass:
    """
    Classify f(R,T,Lm) model for non-linear matter couplings.

    MUST be called on raw unsubstituted dummy-symbol expression only.
    A model is non-linear in matter if and only if at least one condition A-D is true:

    Condition A — Cross coupling: any term contains both Ts_sym and L_sym as factors
    Condition B — Non-linear power: degree > 1 in Ts_sym or L_sym (polynomial check)
    Condition C — Transcendental wrapping: matter variable inside exp/log/sin/cos/tan/sqrt
    Condition D — fL non-constant: second derivative w.r.t L is not zero

    Args:
        model_expr: Raw f(R,T,Lm) as sympy expression (no R substitution yet)
        R_sym: Ricci scalar dummy symbol
        Ts_sym: T_mat (trace of SET) dummy symbol
        L_sym: Matter Lagrangian Lm dummy symbol

    Returns:
        ModelClass with is_nonlinear flag, reason, and coupling terms
    """
    coupling_terms = []
    reasons = []

    # Expand to analyze individual terms
    expanded = sp.expand(model_expr)
    terms = expanded.args if expanded.is_Add else [expanded]

    # Condition A — Cross coupling: Ts_sym * L_sym factors
    for term in terms:
        if term.has(Ts_sym) and term.has(L_sym):
            # Check if both appear as factors in the same term
            term_factors = sp.Mul.make_args(term) if term.is_Mul else [term]
            has_T_factor = any(factor.has(Ts_sym) for factor in term_factors)
            has_L_factor = any(factor.has(L_sym) for factor in term_factors)
            if has_T_factor and has_L_factor:
                coupling_terms.append(term)
                reasons.append("Cross coupling T×L — non-linear")
                break

    # Condition B — Non-linear power of matter variable.
    # F12 fix: use sp.Poly on the *expanded* expression (not the raw model_expr) and
    # also use sp.degree() as a fallback, so that non-linear L dependence buried inside
    # a Mul (e.g. R*(1+L**2)) is correctly detected after expansion.
    if not reasons:
        for sym, label in [(Ts_sym, "Quadratic trace — non-linear"),
                           (L_sym, "Quadratic matter — non-linear")]:
            detected = False
            # Primary: Poly on expanded expression
            try:
                deg = sp.Poly(expanded, sym).degree()
                if deg > 1:
                    coupling_terms.append(expanded)
                    reasons.append(label)
                    detected = True
            except sp.PolynomialError:
                pass
            if not detected:
                # Fallback: check individual terms in case Poly fails on the full expr
                for term in terms:
                    try:
                        deg = sp.Poly(term, sym).degree()
                        if deg > 1:
                            coupling_terms.append(term)
                            reasons.append(label)
                            detected = True
                            break
                    except sp.PolynomialError:
                        pass
            if detected:
                break

    # Condition C — Transcendental wrapping of matter variable
    if not reasons:
        transcendental_funcs = (sp.exp, sp.log, sp.sin, sp.cos, sp.tan, sp.sqrt)
        for func in transcendental_funcs:
            for subexpr in model_expr.atoms(sp.Function):
                if subexpr.func == func:
                    arg = subexpr.args[0] if subexpr.args else None
                    if arg and (arg.has(Ts_sym) or arg.has(L_sym)):
                        coupling_terms.append(subexpr)
                        func_name = func.__name__ if hasattr(func, '__name__') else str(func)
                        reasons.append(f"Transcendental matter coupling {func_name} — non-linear")
                        break
            if reasons:
                break

    # Condition D — fL non-constant in L
    if not reasons:
        fLL = sp.diff(model_expr, L_sym, 2)
        if sp.simplify(fLL) != 0:
            coupling_terms.append(fLL)
            reasons.append("Non-constant fL — non-linear")

    # Final classification
    is_nonlinear = len(reasons) > 0
    reason = reasons[0] if reasons else ""

    return ModelClass(is_nonlinear=is_nonlinear, reason=reason, coupling_terms=coupling_terms)


# Lm choice options
LM_CHOICES = {
    'rho':      'rho',
    'neg_rho':  '-rho',
    'p':        'p',
    'T_mat':    'T_mat',
}


def compute_model_derivatives(
    model_expr: sp.Expr,
    R: sp.Symbol,
    Ts: sp.Symbol,
    L: sp.Symbol,
) -> Tuple[sp.Expr, sp.Expr, sp.Expr, sp.Expr]:
    """
    Compute f(R,T_mat,Lm) and its derivatives fR, fT, fL.

    Args:
        model_expr: f as sympy expression in R, Ts (trace T), L (Lm)
        R:  dummy Ricci scalar symbol
        Ts: dummy T_mat (SET trace) symbol
        L:  dummy Lm (matter Lagrangian) symbol

    Returns:
        (f, fR, fT, fL)
    """
    f  = model_expr
    fR = sp.diff(f, R)
    fT = sp.diff(f, Ts)
    fL = sp.diff(f, L)
    return f, fR, fT, fL


def get_Lm_expression(lm_choice: str, rho_sym: sp.Symbol, p_sym: sp.Symbol, T_mat_val: sp.Expr) -> sp.Expr:
    """
    Resolve the matter Lagrangian Lm from the user's choice string.
    """
    choices = {
        'rho':      rho_sym,
        'neg_rho':  -rho_sym,
        'p':        p_sym,
        'T_mat':    T_mat_val,
        'T_scalar': T_mat_val,  # Backward compatibility
    }
    return choices.get(lm_choice, rho_sym)


def assemble_field_equations(
    f_actual: sp.Expr,
    fR_actual: sp.Expr,
    fT_actual: sp.Expr,
    fL_actual: sp.Expr,
    R_actual: sp.Expr,
    T_SET: Any,
    Lm_actual: sp.Expr,
    geometry_cache: Any,
    ctx: Any,
) -> Any:
    """
    Assemble f(R,T,Lm) field equation LHS tensor.

    LHSfRTLm_μν = fR R_μν + g_μν □fR − ∇_μ∇_ν fR − ½f g_μν
                  − ½(fL + 2fT)(T_μν − Lm g_μν)
    """
    import pytearcat as pt
    from core.theories.utils import track_tensor_names

    Ric = geometry_cache.ricci_tensor
    g   = geometry_cache.metric_tensor_obj  # will be set by pipeline

    print("[fRTLm] Caching Hessian of f_R", flush=True)
    HfR = pt.ten('HfRLm', 2)
    HfR.assign(
        pt.C(pt.C(fR_actual, '_j'), '_i'),
        '_i,_j'
    )
    BOX_fR = g('^i,^j') * HfR('_i,_j')

    LHSfRTLm = pt.ten('LHSfRTLm', 2)
    track_tensor_names(geometry_cache, ['HfRLm', 'LHSfRTLm'])
    LHSfRTLm.assign(
        fR_actual * Ric('_i,_j')
        + g('_i,_j') * BOX_fR
        - HfR('_i,_j')
        - sp.Rational(1, 2) * f_actual * g('_i,_j')
        - sp.Rational(1, 2) * (fL_actual + 2 * fT_actual) * (T_SET('_i,_j') - Lm_actual * g('_i,_j')),
        '_i,_j'
    )
    return LHSfRTLm


def extract_components(LHS: Any, T_SET: Any, index_pairs: list, ctx: Any) -> Tuple[list, list]:
    """
    Extract covariant components for f(R,T,Lm).
    Uses 8π convention on the RHS.
    """
    from core.theories.utils import simplify_selected_component

    lhs_comps = []
    rhs_comps = []

    kappa = 8 * sp.pi

    for (i_str, j_str) in index_pairs:
        i = ctx.coord_index[i_str]
        j = ctx.coord_index[j_str]

        # Only extract diagonal components (i == j)
        if i != j:
            continue

        lhs_comp = simplify_selected_component(LHS.tensor[0][i][j], f"f(R,L,T) ({i_str},{j_str})")
        rhs_comp = kappa * T_SET.tensor[0][i][j]

        lhs_comps.append(lhs_comp)
        rhs_comps.append(rhs_comp)

    return lhs_comps, rhs_comps


# ─── Classifier Test Cases ─────────────────────────────────────────────

CLASSIFIER_TEST_CASES = {
    # (expr_string, expected_is_nonlinear, description)
    'GR_matter':         ('R + lam*L',                          False, 'Linear matter coupling'),
    'starobinsky':       ('R + alpha*R**2',                      False, 'Starobinsky — no matter terms'),
    'starobinsky_T':     ('R + alpha*R**2 + beta*T_mat',         False, 'Starobinsky + linear T coupling'),
    'full_linear':       ('R + beta*T_mat + lam*L',              False, 'Full linear — all three scalars'),
    'nonmin_curv':       ('R*(1 + gamma*L)',                     False, 'Non-minimal curvature-matter — linear in L'),
    'cross_coupling':    ('T_mat*L',                             True,  'Cross coupling T×L — non-linear'),
    'L_squared':         ('L**2',                                True,  'Quadratic matter — non-linear'),
    'exp_L':             ('exp(L)',                              True,  'Exponential matter coupling'),
    'log_T':             ('log(T_mat)',                          True,  'Log of trace — non-linear'),
    'R_T_L_product':     ('R*T_mat*L',                           True,  'Triple product — non-linear'),
    'sqrt_L':            ('R + sqrt(L)',                         True,  'Square root matter coupling'),
    'T_squared':         ('R + T_mat**2',                        True,  'Quadratic trace — non-linear'),
}


def run_classifier_tests():
    """Run all classifier test cases and report PASS/FAIL results."""
    print("Running f(R,T,Lm) classifier tests...")
    print("=" * 60)
    
    # Create dummy symbols for testing
    R_sym, Ts_sym, L_sym = sp.symbols('R_sym Ts_sym L_sym')
    
    passed = 0
    failed = 0
    
    for test_name, (expr_str, expected, description) in CLASSIFIER_TEST_CASES.items():
        try:
            # Parse expression with dummy symbols
            local_map = {
                'R': R_sym,
                'R_sym': R_sym,
                'T_mat': Ts_sym,
                'T_scalar': Ts_sym,
                'Ts_sym': Ts_sym,
                'L': L_sym,
                'L_sym': L_sym,
            }
            for param in ('alpha', 'beta', 'gamma', 'delta', 'lam', 'n'):
                local_map[param] = sp.Symbol(param)
            model_expr = sp.sympify(expr_str, locals=local_map)
            
            # Run classifier
            result = classify_model(model_expr, R_sym, Ts_sym, L_sym)
            
            # Check result
            if result.is_nonlinear == expected:
                print(f"PASS: {test_name:25} - {description}")
                passed += 1
            else:
                print(f"FAIL: {test_name:25} - {description}")
                print(f"      Expected: {expected}, Got: {result.is_nonlinear}")
                print(f"      Reason: {result.reason}")
                failed += 1
                
        except Exception as e:
            print(f"ERROR: {test_name:25} - {description}")
            print(f"      Exception: {e}")
            failed += 1
    
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("All tests PASSED ✓")
    else:
        print(f"{failed} tests FAILED ✗")
    
    return failed == 0


# ─── Scalar Name Map for Frontend ─────────────────────────────────

SCALAR_NAME_MAP = {
    # User types    Internal symbol   Display name            LaTeX
    'R':           ('R_sym',          'Ricci scalar',          r'\mathcal{R}'),
    'T_mat':       ('Ts_sym',         'Trace of SET',          r'T_{\text{SET}}'),
    'T':           ('T_tor_sym',      'Torsion scalar',        r'T_{\text{tor}}'),  # Torsion scalar for f(T), f(T,B)
    'L':           ('L_sym',          'Matter Lagrangian',     r'\mathcal{L}_m'),
    'Lm':          ('L_sym',          'Matter Lagrangian (alias)', r'\mathcal{L}_m'),
    'lam':         (None,             'Coupling constant λ',   r'\lambda'),
    'alpha':       (None,             'Model parameter α',     r'\alpha'),
    'beta':        (None,             'Model parameter β',     r'\beta'),
    'gamma':       (None,             'Model parameter γ',     r'\gamma'),
    'n':           (None,             'Power index n',         r'n'),
}


if __name__ == '__main__':
    run_classifier_tests()
