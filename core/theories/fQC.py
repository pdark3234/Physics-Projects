"""
f(Q,C) nonmetricity-boundary gravity.

C is the symmetric-teleparallel boundary scalar defined here as C = R - Q.
Equivalently, f(Q,C) is treated as F(R,Q) = f(Q, R - Q), so
F_R = f_C and F_Q = f_Q - f_C.
"""

import builtins
if not hasattr(builtins, 'display'):
    builtins.display = lambda *args, **kwargs: None

from typing import Any, Tuple
import sympy as sp


def compute_model_derivatives(
    model_expr: sp.Expr,
    Q: sp.Symbol,
    C: sp.Symbol,
) -> Tuple[sp.Expr, sp.Expr, sp.Expr, sp.Expr, sp.Expr]:
    """Return f(Q,C), f_Q, f_C, f_QQ, and f_CC."""
    f = model_expr
    fQ = sp.diff(f, Q)
    fC = sp.diff(f, C)
    fQQ = sp.diff(f, Q, 2)
    fCC = sp.diff(f, C, 2)
    return f, fQ, fC, fQQ, fCC


def assemble_field_equations(
    f_actual: sp.Expr,
    fQ_actual: sp.Expr,
    fC_actual: sp.Expr,
    Q_actual: sp.Expr,
    C_actual: sp.Expr,
    geometry_cache: Any,
    ctx: Any,
) -> Any:
    """Assemble the covariant f(Q,C) LHS tensor."""
    import pytearcat as pt
    from core.theories.utils import track_tensor_names

    print("[fQC] Assembling f(Q,C) field equations", flush=True)
    g = geometry_cache.metric_tensor_obj
    actual_R = geometry_cache.ricci_scalar
    Ei = geometry_cache.einstein_tensor

    print("[fQC]   Caching Hessian of f_C", flush=True)
    HfC = pt.ten('HfC', 2)
    HfC.assign(
        pt.C(pt.C(fC_actual, '_j'), '_i'),
        '_i,_j',
    )

    print("[fQC]   Computing Box(f_C)", flush=True)
    Box_fC = g('^i,^j') * HfC('_i,_j')

    LHSfQC = pt.ten('LHSfQC', 2)
    track_tensor_names(geometry_cache, ['HfC', 'LHSfQC'])
    print("[fQC]   Assigning compact f(Q,C) LHS tensor", flush=True)
    LHSfQC.assign(
        -(fQ_actual + fC_actual) * Ei('_i,_j')
        - sp.Rational(1, 2) * g('_i,_j') * (
            f_actual - actual_R * fC_actual - Q_actual * fQ_actual
        )
        + HfC('_i,_j')
        - g('_i,_j') * Box_fC,
        '_i,_j',
    )
    print("[fQC] f(Q,C) LHS tensor complete", flush=True)
    return LHSfQC


def extract_components(LHS: Any, T_SET: Any, index_pairs: list, ctx: Any) -> Tuple[list, list]:
    """Extract diagonal covariant components for solving."""
    from core.theories.utils import simplify_selected_component_transcendental

    lhs_comps = []
    rhs_comps = []
    kappa = 8 * sp.pi

    for (i_str, j_str) in index_pairs:
        i = ctx.coord_index[i_str]
        j = ctx.coord_index[j_str]
        if i != j:
            continue
        from core.theories.fQ import _bounded_component_trig_cleanup, _positive_spherical_branch
        lhs_comp = _bounded_component_trig_cleanup(_positive_spherical_branch(LHS.tensor[0][i][j]))
        lhs_comps.append(
            simplify_selected_component_transcendental(
                lhs_comp,
                f"f(Q,C) ({i_str},{j_str})",
            )
        )
        rhs_comps.append(kappa * T_SET.tensor[0][i][j])

    return lhs_comps, rhs_comps
