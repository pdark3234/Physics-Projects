"""
f(T,B) Gravity Theory Module

Field equation (mixed form — from reference notebook):
2e(□fB) δ^λ_ν  −  2e ∇^λ∇_ν fB  +  e B fB δ^λ_ν
+  4e(∂_μ fB + ∂_μ fT) S_ν^{μλ}
+  4 eᵃ_ν ∂_μ(e Sₐ^{μλ}) fT
−  4e fT T^σ_{μν} S_σ^{λμ}
−  e f δ^λ_ν  =  16π e T^λ_ν

Scalars displayed: T, B, T−B
"""

import builtins
import sys
if not hasattr(builtins, 'display'):
    builtins.display = lambda *args, **kwargs: None

from typing import Tuple, Dict, Any
import sympy as sp


def compute_model_derivatives(
    model_expr: sp.Expr,
    T: sp.Symbol,
    B: sp.Symbol
) -> Tuple[sp.Expr, sp.Expr, sp.Expr]:
    """
    Compute f(T,B) and its partial derivatives fT, fB.

    Returns:
        (f, fT, fB)
    """
    f  = model_expr
    fT = sp.diff(f, T)
    fB = sp.diff(f, B)
    return f, fT, fB


def assemble_field_equations(
    f_actual: sp.Expr,
    fT_actual: sp.Expr,
    fB_actual: sp.Expr,
    T_actual: sp.Expr,
    B_actual: sp.Expr,
    geometry_cache: Any,
    ctx: Any
) -> Any:
    """
    Assemble f(T,B) field equation LHS tensor.

    Uses staged simplification (§10.5) to avoid combinatorial blowup:
    1. trigsimp per term before summation
    2. powsimp after summation
    3. simplify on the assembled tensor before .complete()
    """
    import pytearcat as pt
    from core.theories.utils import track_tensor_names

    e, det_e = geometry_cache.vierbein
    T_tens   = geometry_cache.torsion_tensor
    S        = geometry_cache.superpotential
    KD       = pt.kdelta()

    # D'Alembertian of fB: □fB = g^ij ∇_i ∇_j fB
    # Use Levi-Civita connection (cached from torsion for f(T,B))
    g = geometry_cache.metric_tensor_obj if hasattr(geometry_cache, 'metric_tensor_obj') and geometry_cache.metric_tensor_obj else None

    # Covariant derivatives using pt.C (Levi-Civita in torsion context)
    # Use the inverse metric from the geometry cache
    if g is not None:
        print("[fTB] Caching Hessian of fB...")
        gup = pt.ten('gup', 2)
        gup.assign(g('^i,^j'), '^i,^j')
        HfB = pt.ten('HfB', 2)
        HfB.assign(
            pt.C(pt.C(fB_actual, '_j'), '_i'),
            '_i,_j'
        )
        BOX_fB = gup('^i,^j') * HfB('_i,_j')
        print("[fTB] Assembling field equation terms...")
    else:
        HfB = None
        BOX_fB = sp.Integer(0)

    # Assemble value using the spec formula directly
    # term5 must follow the same flat-index contraction as f(T):
    #   4 e^{-1} eⁱ_ν ∂_μ(e eᵢ^α S_α^{μλ}) fT
    # (matches f(T) first term with indices relabeled μ↔ν, ρ→μ, ν→λ)
    term1 = 2 * det_e * KD('^lam,_nu') * BOX_fB
    term2 = -2 * det_e * (
        gup('^lam,^alpha') * HfB('_alpha,_nu')
        if HfB is not None else sp.Integer(0)
    )
    term3 = det_e * B_actual * fB_actual * KD('^lam,_nu')
    term4 = 4 * det_e * (pt.D(fB_actual, '_mu') + pt.D(fT_actual, '_mu')) * S('_nu,^mu,^lam')
    term5 = 4 * e('^i,_nu') * pt.D(det_e * e('_i,^alpha') * S('_alpha,^mu,^lam'), '_mu') * fT_actual
    term6 = -4 * det_e * fT_actual * T_tens('^sig,_mu,_nu') * S('_sig,^lam,^mu')
    term7 = -det_e * f_actual * KD('^lam,_nu')

    value = term1 + term2 + term3 + term4 + term5 + term6 + term7

    print("[fTB] Creating LHS tensor...")
    LHSfTB = pt.ten('LHSfTB', 2)
    track_tensor_names(geometry_cache, ['gup', 'HfB', 'LHSfTB'])
    LHSfTB.assign(value, '^lam,_nu')
    LHSfTB.complete('^,_')
    LHSfTB._mgs_det_e = det_e
    print("[fTB] LHS assembly complete")
    return LHSfTB


def extract_components(LHS: Any, T_SET: Any, index_pairs: list, ctx: Any) -> Tuple[list, list]:
    """
    Extract mixed-index components for f(T,B) LHS and SET.
    RHS factor is 16π after dividing the density-form equation by det(e).
    Components extracted as ^λ_ν mixed.
    """
    from core.theories.utils import simplify_selected_component

    lhs_comps = []
    rhs_comps = []

    kappa = 16 * sp.pi
    det_e = getattr(LHS, "_mgs_det_e", None)

    for (i_str, j_str) in index_pairs:
        i = ctx.coord_index[i_str]
        j = ctx.coord_index[j_str]

        # Only extract diagonal components (i == j)
        if i != j:
            continue

        lhs_raw = LHS.tensor[1][i][j]
        if det_e is not None:
            lhs_raw = sp.cancel(lhs_raw / det_e)
            lhs_raw = sp.trigsimp(lhs_raw)
        lhs_comp = simplify_selected_component(lhs_raw, f"f(T,B) ({i_str},{j_str})")
        rhs_comp = kappa * T_SET.tensor[1][i][j]

        lhs_comps.append(lhs_comp)
        rhs_comps.append(rhs_comp)

    return lhs_comps, rhs_comps
