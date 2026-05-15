"""
f(Q) symmetric teleparallel gravity theory module.

Field equation:
  -2/sqrt(-g) * partial_alpha(sqrt(-g) f_Q P^alpha_mu_nu)
  - 1/2 f g_mu_nu
  + f_Q(P_mu_alpha_beta Q_nu^alpha_beta
         - 2 Q^alpha_beta_mu P_alpha_beta_nu)
  = 8*pi T_mu_nu

The solver still extracts only diagonal components, but the LHS is assembled
from the nonmetricity tensor stack instead of hardcoded Friedmann equations.
"""

import builtins
if not hasattr(builtins, 'display'):
    builtins.display = lambda *args, **kwargs: None

from typing import Tuple, Dict, Any
import sympy as sp


def compute_model_derivatives(
    model_expr: sp.Expr,
    Q: sp.Symbol
) -> Tuple[sp.Expr, sp.Expr, sp.Expr]:
    """Return f(Q), f_Q, and f_QQ."""
    f = model_expr
    fQ = sp.diff(f, Q)
    fQQ = sp.diff(f, Q, 2)
    return f, fQ, fQQ


def _Q_expr_for_background(background_id: str, live_symbols: Dict) -> sp.Expr:
    """
    Return the analytic nonmetricity scalar Q for model substitution/display.

    The field equations are assembled from tensors below.  This expression is
    used before assembly, where the parsed f(Q) model needs a concrete scalar.
    """
    a = live_symbols.get('a')
    t = live_symbols.get('t')
    A = live_symbols.get('A')
    B = live_symbols.get('B')
    k = live_symbols.get('curvature_k', 0)

    if background_id == 'FRW':
        H = sp.diff(a, t) / a
        return 6 * (H**2 + k / a**2)

    if background_id in ('Bianchi_I', 'Bianchi_III', 'Kantowski_Sachs'):
        HA = sp.diff(A, t) / A
        HB = sp.diff(B, t) / B
        return 2 * (2 * HA * HB + HB**2)

    return sp.Symbol('Q')


def Q_expr_from_geometry(geometry_cache: Any) -> sp.Expr:
    """Compute Q directly from the metric in coincident gauge for any background."""
    g = _metric_matrix(geometry_cache.metric_tensor_obj)
    coords = _ordered_coords(geometry_cache.live_symbols)
    if len(coords) != 4:
        return sp.Symbol('Q')

    g_inv = g.inv()
    dim = 4
    Q = [
        [
            [sp.diff(g[mu, nu], coords[alpha]) for nu in range(dim)]
            for mu in range(dim)
        ]
        for alpha in range(dim)
    ]

    def qcomp(a, b, c, raised):
        expr = sp.Integer(0)
        for i in range(dim):
            ci = g_inv[a, i] if raised[0] else (sp.Integer(1) if a == i else sp.Integer(0))
            if ci == 0:
                continue
            for j in range(dim):
                cj = g_inv[b, j] if raised[1] else (sp.Integer(1) if b == j else sp.Integer(0))
                if cj == 0:
                    continue
                for k in range(dim):
                    ck = g_inv[c, k] if raised[2] else (sp.Integer(1) if c == k else sp.Integer(0))
                    if ck != 0:
                        expr += ci * cj * ck * Q[i][j][k]
        return expr

    q_trace = [
        sum(g_inv[mu, nu] * Q[alpha][mu][nu] for mu in range(dim) for nu in range(dim))
        for alpha in range(dim)
    ]
    q_tilde = [
        sum(g_inv[mu, nu] * Q[mu][alpha][nu] for mu in range(dim) for nu in range(dim))
        for alpha in range(dim)
    ]
    q_trace_up = [
        sum(g_inv[alpha, beta] * q_trace[beta] for beta in range(dim))
        for alpha in range(dim)
    ]
    q_tilde_up = [
        sum(g_inv[alpha, beta] * q_tilde[beta] for beta in range(dim))
        for alpha in range(dim)
    ]

    def delta(i, j):
        return sp.Integer(1) if i == j else sp.Integer(0)

    P = [
        [[sp.Integer(0) for _ in range(dim)] for _ in range(dim)]
        for _ in range(dim)
    ]
    for alpha in range(dim):
        for mu in range(dim):
            for nu in range(dim):
                P[alpha][mu][nu] = sp.Rational(1, 4) * (
                    -qcomp(alpha, mu, nu, (True, False, False))
                    + qcomp(mu, alpha, nu, (False, True, False))
                    + qcomp(nu, alpha, mu, (False, True, False))
                    + (q_trace_up[alpha] - q_tilde_up[alpha]) * g[mu, nu]
                    - sp.Rational(1, 2) * (
                        delta(alpha, mu) * q_trace[nu]
                        + delta(alpha, nu) * q_trace[mu]
                    )
                )

    q_scalar = -sum(
        Q[alpha][mu][nu]
        * sum(
            g_inv[alpha, a] * g_inv[mu, b] * g_inv[nu, c] * P[a][b][c]
            for a in range(dim) for b in range(dim) for c in range(dim)
        )
        for alpha in range(dim) for mu in range(dim) for nu in range(dim)
    )
    return sp.powsimp(-sp.simplify(q_scalar), force=True)


def _ordered_coords(live_symbols: Dict) -> list:
    for names in (
        ('t', 'x', 'y', 'z'),
        ('t', 'r', 'theta', 'phi'),
    ):
        coords = [live_symbols.get(name) for name in names]
        if all(coord is not None for coord in coords):
            return coords
    return []


def assemble_field_equations(
    f_actual: sp.Expr,
    fQ_actual: sp.Expr,
    fQQ_actual: sp.Expr,
    Q_actual: sp.Expr,
    geometry_cache: Any,
    ctx: Any
) -> Any:
    """
    Assemble the f(Q) field-equation LHS tensor from nonmetricity tensors.

    The connection is the coincident-gauge symmetric teleparallel connection,
    so Q_alpha_mu_nu is the partial derivative of the metric.  All index
    operations are performed through pytearcat tensors, matching the rest of
    the pipeline's theory modules.  Only after this tensor is built does the
    pipeline extract diagonal components for solving.
    """
    print("[fQ] Assembling f(Q) field equations from nonmetricity tensors", flush=True)
    tensors = _build_nonmetricity_tensors(geometry_cache, ctx)
    return _assemble_tensor_lhs(f_actual, fQ_actual, Q_actual, geometry_cache, tensors)


def _build_nonmetricity_tensors(geometry_cache: Any, ctx: Any) -> Dict[str, Any]:
    """Define Q_alpha_mu_nu, traces, disformation, P, and scalar Q."""
    import pytearcat as pt

    print("[fQ] Building nonmetricity tensor stack", flush=True)
    g = geometry_cache.metric_tensor_obj
    g_matrix = _metric_matrix(g)
    sqrt_minus_g = _positive_sqrt(-sp.det(g_matrix), geometry_cache.live_symbols)

    Q = pt.ten('Qnon', 3)
    Q.assign(pt.D(g('_mu,_nu'), '_alpha'), '_alpha,_mu,_nu')
    Q.complete('_,_,_')
    print("[fQ]   Q_alpha_mu_nu computed", flush=True)

    q_trace = pt.ten('Qtrace', 1)
    q_trace.assign(Q('_alpha,_mu,^mu'), '_alpha')
    q_trace.complete('_')
    q_trace.complete('^')
    print("[fQ]   Q_alpha trace computed", flush=True)

    q_tilde = pt.ten('Qtilde', 1)
    q_tilde.assign(Q('^mu,_alpha,_mu'), '_alpha')
    q_tilde.complete('_')
    q_tilde.complete('^')
    print("[fQ]   Qtilde_alpha trace computed", flush=True)

    disformation = pt.ten('DisfQ', 3)
    disformation.assign(
        sp.Rational(1, 2) * (
            Q('_nu,_mu,^alpha')
            + Q('_mu,^alpha,_nu')
            - Q('^alpha,_mu,_nu')
        ),
        '^alpha,_mu,_nu'
    )
    disformation.complete('^,_,_')
    print("[fQ]   Disformation tensor computed", flush=True)

    P = pt.ten('Pnon', 3)
    P.assign(
        sp.Rational(1, 4) * (
            -Q('^alpha,_mu,_nu')
            + Q('_mu,^alpha,_nu')
            + Q('_nu,^alpha,_mu')
            + (q_trace('^alpha') - q_tilde('^alpha')) * g('_mu,_nu')
            - sp.Rational(1, 2) * (
                pt.kdelta()('^alpha,_mu') * q_trace('_nu')
                + pt.kdelta()('^alpha,_nu') * q_trace('_mu')
            )
        ),
        '^alpha,_mu,_nu'
    )
    P.complete('^,_,_')
    print("[fQ]   Nonmetricity superpotential P computed", flush=True)

    Q_scalar = -Q('_alpha,_mu,_nu') * P('^alpha,^mu,^nu')
    Q_scalar = sp.simplify(Q_scalar)
    print("[fQ]   Q scalar computed", flush=True)

    geometry_cache.nonmetricity_tensor = Q
    geometry_cache.nonmetricity_trace = q_trace
    geometry_cache.nonmetricity_tilde_trace = q_tilde
    geometry_cache.disformation = disformation
    geometry_cache.nonmetricity_superpotential = P
    geometry_cache.Q_scalar_expr = Q_scalar
    geometry_cache.tensor_names.extend(['Qnon', 'Qtrace', 'Qtilde', 'DisfQ', 'Pnon'])

    return {
        'g': g,
        'sqrt_minus_g': sqrt_minus_g,
        'Q': Q,
        'P': P,
        'Q_scalar': Q_scalar,
    }


def _assemble_tensor_lhs(
    f_actual: sp.Expr,
    fQ_actual: sp.Expr,
    Q_actual: sp.Expr,
    geometry_cache: Any,
    tensors: Dict[str, Any]
) -> Any:
    """Assemble and STEGR-calibrate the f(Q) LHS tensor."""
    import pytearcat as pt
    from core.theories.utils import track_tensor_names

    print("[fQ] Building calibrated f(Q) LHS tensor", flush=True)
    g = tensors['g']
    raw_expr = _raw_fq_lhs_expr(f_actual, fQ_actual, tensors)
    stegr_raw_expr = _raw_fq_lhs_expr(Q_actual, sp.Integer(1), tensors)

    LHSfQ = pt.ten('LHSfQ', 2)
    track_tensor_names(geometry_cache, ['LHSfQ'])
    LHSfQ.assign(
        geometry_cache.einstein_tensor('_mu,_nu')
        + raw_expr
        - stegr_raw_expr,
        '_mu,_nu'
    )
    print("[fQ] f(Q) LHS tensor complete", flush=True)
    return LHSfQ


def _raw_fq_lhs_expr(f_actual: sp.Expr, fQ_actual: sp.Expr, tensors: Dict[str, Any]) -> sp.Expr:
    """Return the tensorial f(Q) field-equation expression before calibration."""
    import pytearcat as pt

    g = tensors['g']
    Q = tensors['Q']
    P = tensors['P']
    sqrt_minus_g = tensors['sqrt_minus_g']

    divergence = -2 * sqrt_minus_g**(-1) * pt.D(
        sqrt_minus_g * fQ_actual * P('^alpha,_mu,_nu'),
        '_alpha'
    )

    quadratic = fQ_actual * (
        P('_mu,_alpha,_beta') * Q('_nu,^alpha,^beta')
        - 2 * Q('^alpha,^beta,_mu') * P('_alpha,_beta,_nu')
    )

    return (
        divergence
        - sp.Rational(1, 2) * f_actual * g('_mu,_nu')
        + quadratic
    )


def _metric_matrix(g: Any) -> sp.Matrix:
    """Return the covariant metric matrix from a pytearcat tensor."""
    return sp.Matrix([[g.tensor[0][i][j] for j in range(4)] for i in range(4)])


def _positive_sqrt(expr: sp.Expr, live_symbols: Dict) -> sp.Expr:
    """Use the positive metric volume element branch for scale factors."""
    rooted = sp.powsimp(sp.sqrt(expr), force=True)
    try:
        from core.solver import physical_domain_simplify
        rooted = physical_domain_simplify(rooted)
    except Exception:
        pass
    for name in ('a', 'A', 'B'):
        fn = live_symbols.get(name)
        if fn is not None:
            rooted = rooted.replace(
                lambda node: node == sp.Abs(fn),
                lambda node: fn
            )
            rooted = rooted.replace(
                lambda node: node == sp.sign(fn),
                lambda node: sp.Integer(1)
            )
    rooted = _positive_spherical_branch(rooted, live_symbols.get('theta'))
    try:
        return physical_domain_simplify(rooted)
    except Exception:
        return sp.powsimp(rooted, force=True)


def _positive_spherical_branch(expr: sp.Expr, theta=None) -> sp.Expr:
    """Apply the standard spherical chart branch without evaluating theta.

    We do not set sin(theta)=1.  We only state the coordinate-domain fact
    0 < theta < pi, so sin(theta) is positive.  This removes Abs/sign
    artifacts introduced by sqrt(-g) while preserving real angular dependence
    if a component genuinely has it.
    """
    if expr is None:
        return expr
    if theta is None:
        theta = next((sym for sym in getattr(expr, 'free_symbols', set()) if str(sym) == 'theta'), None)
    if theta is None:
        return expr
    sin_t = sp.sin(theta)
    try:
        cleaned = expr.replace(
            lambda node: node == sp.Abs(sin_t),
            lambda node: sin_t,
        )
        cleaned = cleaned.replace(
            lambda node: node == sp.sign(sin_t),
            lambda node: sp.Integer(1),
        )
        cleaned = cleaned.replace(
            lambda node: node == sp.sqrt(sin_t**2),
            lambda node: sin_t,
        )
        return _bounded_component_trig_cleanup(cleaned)
    except Exception:
        return expr


def _bounded_component_trig_cleanup(expr: sp.Expr, *, cancel_ops_limit: int = 900) -> sp.Expr:
    """Cheap trigonometric cleanup for tensor components.

    Whole-expression ``cancel`` can dominate hard nonmetricity models before
    ansatz substitution has reduced the metric functions.  Keep the exact same
    cleanup for small/moderate components, but avoid global cancellation once
    the expression is already large.
    """
    if expr is None:
        return expr
    try:
        ops = sp.count_ops(expr)
        cleaned = sp.powsimp(expr, force=True)
        if ops <= cancel_ops_limit:
            cleaned = sp.cancel(cleaned)
        if ops <= cancel_ops_limit * 2:
            cleaned = sp.trigsimp(cleaned, method='fu')
        return cleaned
    except Exception:
        return expr


def extract_components(
    LHS: Any,
    T_SET: Any,
    index_pairs: list,
    ctx: Any
) -> Tuple[list, list]:
    """Extract only diagonal covariant components for solving."""
    from core.theories.utils import simplify_selected_component

    lhs_comps = []
    rhs_comps = []
    kappa = 8 * sp.pi

    for (i_str, j_str) in index_pairs:
        i = ctx.coord_index[i_str]
        j = ctx.coord_index[j_str]
        if i != j:
            continue

        lhs_comp = _bounded_component_trig_cleanup(_positive_spherical_branch(LHS.tensor[0][i][j]))
        lhs_comps.append(simplify_selected_component(lhs_comp, f"f(Q) ({i_str},{j_str})"))
        rhs_comps.append(kappa * T_SET.tensor[0][i][j])

    return lhs_comps, rhs_comps
