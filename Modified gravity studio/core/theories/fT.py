"""
f(T) Gravity Theory Module

Field equation (mixed form):
e⁻¹ eⁱ_μ ∂_ρ(e eᵢ^α S_α^{νρ}) fT  +  S^ν^λ_α T^α_{λμ} fT
−  S^ν^ρ_μ ∂_ρT fTT  +  ¼ δ^ν_μ f  =  4π T^ν_μ

Scalars displayed: T
"""

import builtins
import sys
if not hasattr(builtins, 'display'):
    builtins.display = lambda *args, **kwargs: None

from typing import Tuple, Dict, Any
import sympy as sp


def compute_model_derivatives(model_expr: sp.Expr, T: sp.Symbol) -> Tuple[sp.Expr, sp.Expr, sp.Expr]:
    """
    Compute f(T) and its derivatives fT, fTT.

    Returns:
        (f, fT, fTT)
    """
    f   = model_expr
    fT  = sp.diff(f, T)
    fTT = sp.diff(f, T, 2)
    return f, fT, fTT


def assemble_field_equations(
    f_actual: sp.Expr,
    fT_actual: sp.Expr,
    fTT_actual: sp.Expr,
    T_actual: sp.Expr,
    geometry_cache: Any,
    ctx: Any
) -> Any:
    """
    Assemble f(T) field equation LHS tensor.

    LHS^ν_μ = e⁻¹ eⁱ_μ ∂_ρ(e eᵢ^α S_α^{νρ}) fT
              + S^ν^λ_α T^α_{λμ} fT
              − S^ν^ρ_μ ∂_ρT fTT
              + ¼ δ^ν_μ f
    """
    import pytearcat as pt
    from core.theories.utils import track_tensor_names

    e, det_e = geometry_cache.vierbein
    T_tens   = geometry_cache.torsion_tensor
    S        = geometry_cache.superpotential
    Slor     = geometry_cache.lorentz_superpotential
    omega    = geometry_cache.spin_connection
    KD       = pt.kdelta()

    lorentz_div_S = (
        pt.D(det_e * Slor('_a,^rho,^nu'), '_rho')
        - det_e * omega('^b,_a,_rho') * Slor('_b,^rho,^nu')
    )

    LHSfT = pt.ten('LHSfT', 2)
    track_tensor_names(geometry_cache, ['LHSfT'])
    LHSfT.assign(
        det_e**(-1) * e('_mu,^a') * lorentz_div_S * fT_actual
        + S('^nu,^lambda,_alpha') * T_tens('^alpha,_lambda,_mu') * fT_actual
        - S('^nu,^rho,_mu') * pt.D(T_actual, '_rho') * fTT_actual
        + sp.Rational(1, 4) * KD('^nu,_mu') * f_actual,
        '^nu,_mu'
    )
    LHSfT.complete('^,_')
    return LHSfT


def extract_components(LHS: Any, T_SET: Any, index_pairs: list, ctx: Any) -> Tuple[list, list]:
    """
    Extract mixed-index components ^ν_μ for f(T) LHS and SET.
    f(T) field equation is in mixed form, so T_SET must also be in mixed form.
    RHS factor is 4π (not 8π) because the field equation uses ¼δ (not ½δ).
    """
    from core.theories.utils import simplify_selected_component

    lhs_comps = []
    rhs_comps = []

    kappa = 4 * sp.pi  # f(T) field equation convention

    for (i_str, j_str) in index_pairs:
        i = ctx.coord_index[i_str]   # upper index (nu)
        j = ctx.coord_index[j_str]   # lower index (mu)

        # Only extract diagonal components (i == j)
        if i != j:
            continue

        lhs_comp = sp.trigsimp(sp.cancel(LHS.tensor[1][i][j]))
        lhs_comp = simplify_selected_component(lhs_comp, f"f(T) ({i_str},{j_str})")
        rhs_comp = kappa * T_SET.tensor[1][i][j]

        lhs_comps.append(lhs_comp)
        rhs_comps.append(rhs_comp)

    return lhs_comps, rhs_comps
