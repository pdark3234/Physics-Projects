"""
f(R) Gravity Theory Module

Field equations for f(R) gravity:
fR * R_μν - ½ f g_μν - ∇_μ ∇_ν fR + g_μν □ fR = 8π T_μν

where fR = df/dR
"""

# Mock IPython display before pytearcat imports
import builtins
import sys
if not hasattr(builtins, 'display'):
    builtins.display = lambda *args, **kwargs: None

from typing import Tuple, Dict, Any
import sympy as sp


def compute_model_derivatives(model_expr: sp.Expr, R: sp.Symbol) -> Tuple[sp.Expr, sp.Expr]:
    """
    Compute f(R) and its derivatives.
    
    Args:
        model_expr: f(R) as sympy expression
        R: Ricci scalar symbol
        
    Returns:
        (f, fR) where fR = df/dR
    """
    f = model_expr
    fR = sp.diff(f, R)
    
    return f, fR


def assemble_field_equations(
    f: sp.Expr,
    fR: sp.Expr,
    ricci_scalar: sp.Expr,
    ricci_tensor: Any,
    metric: Any,
    geometry_cache: 'GeometryCache',
    ctx: 'MetricContext'
) -> Any:
    """
    Assemble f(R) field equation LHS tensor.
    
    LHS_μν = fR * R_μν - ½ f g_μν - ∇_μ ∇_ν fR + g_μν □ fR
    
    Args:
        f: Model function value
        fR: df/dR
        ricci_scalar: Computed R value
        ricci_tensor: Computed R_μν tensor
        metric: Metric tensor g
        geometry_cache: Cached geometric objects
        ctx: MetricContext with symbols
        
    Returns:
        LHS tensor (pytearcat)
    """
    import pytearcat as pt
    from core.theories.utils import track_tensor_names

    print("[fR] Caching Hessian of f_R", flush=True)
    HfR = pt.ten('HfR', 2)
    HfR.assign(
        pt.C(pt.C(fR, '_j'), '_i'),
        '_i,_j'
    )
    
    # Create LHS tensor
    LHS = pt.ten('LHSfR', 2)
    track_tensor_names(geometry_cache, ['HfR', 'LHSfR'])
    
    # D'Alembertian of fR: □fR = g^ij ∇_i ∇_j fR
    DALAMBERT_fR = metric('^i,^j') * HfR('_i,_j')
    
    # Assemble LHS components
    # fR * R_μν - ½ f g_μν - ∇_μ ∇_ν fR + g_μν □ fR
    LHS.assign(
        fR * ricci_tensor('_i,_j')
        - sp.Rational(1, 2) * f * metric('_i,_j')
        - HfR('_i,_j')
        + metric('_i,_j') * DALAMBERT_fR,
        '_i,_j'
    )
    
    return LHS


def extract_components(LHS: Any, T_SET: Any, index_pairs: list, ctx: 'MetricContext') -> Tuple[list, list]:
    """
    Extract components from LHS and SET tensors at given index pairs.
    
    Args:
        LHS: Field equation LHS tensor
        T_SET: Stress-energy tensor
        index_pairs: List of (i, j) string pairs
        ctx: MetricContext with coord_index mapping
        
    Returns:
        (lhs_components, rhs_components) as sympy expressions
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

        # Extract components from tensor arrays
        lhs_comp = simplify_selected_component(LHS.tensor[0][i][j], f"f(R) ({i_str},{j_str})")
        rhs_comp = kappa * T_SET.tensor[0][i][j]

        lhs_comps.append(lhs_comp)
        rhs_comps.append(rhs_comp)

    return lhs_comps, rhs_comps


def solve_field_equations(
    lhs_comps: list,
    rhs_comps: list,
    unknowns: list
) -> Dict[str, sp.Expr]:
    """
    Solve field equations for unknowns.
    
    Strategy:
    1. Solve first equation for primary unknown (usually rho)
    2. Substitute into remaining equations
    3. Solve for secondary unknowns (p, etc.)
    
    Args:
        lhs_comps: LHS tensor components
        rhs_comps: RHS tensor components (8πT components)
        unknowns: List of unknown symbols to solve for
        
    Returns:
        Dict mapping unknown names to solutions
    """
    # Build equations LHS = RHS
    equations = [sp.Eq(lhs, rhs) for lhs, rhs in zip(lhs_comps, rhs_comps)]
    
    # Sequential solve strategy
    primary_unknown = unknowns[0]
    secondary_unknowns = unknowns[1:]
    
    # Solve primary from first equation
    primary_solutions = sp.solve(equations[0], primary_unknown)
    if not primary_solutions:
        raise ValueError(f"Could not solve for {primary_unknown}")
    primary_solution = primary_solutions[0]
    
    # Substitute into remaining equations
    remaining_eqs = [
        eq.subs(primary_unknown, primary_solution)
        for eq in equations[1:]
    ]
    
    # Solve for secondary unknowns
    if secondary_unknowns:
        secondary_solution = sp.solve(remaining_eqs, secondary_unknowns)
        if not secondary_solution:
            raise ValueError(f"Could not solve for {secondary_unknowns}")
    else:
        secondary_solution = {}
    
    # Combine solutions
    solutions = {primary_unknown: primary_solution}
    if secondary_solution:
        if isinstance(secondary_solution, dict):
            solutions.update(secondary_solution)
        else:
            # Handle list of solutions
            for i, unk in enumerate(secondary_unknowns):
                solutions[unk] = secondary_solution[0][i] if secondary_solution else None
    
    return solutions
