"""Tolman-Oppenheimer-Volkoff diagnostics for static spherical backgrounds."""

from typing import Any, Dict, Optional

import sympy as sp

from core.solver import lightweight_diagnostic_simplify


TOV_BACKGROUNDS = {"SS_wormhole", "SS_blackhole"}


def supports_tov(background_id: str, stress_tensor: str) -> bool:
    """Return whether a run can expose static spherical TOV diagnostics."""
    return background_id in TOV_BACKGROUNDS and stress_tensor in {
        "perfect_fluid",
        "anisotropic",
    }


def _metric_expr(name: str, geom: Any, extended_subs: Dict) -> Optional[sp.Expr]:
    live = getattr(geom, "live_symbols", {}) or {}
    fn = live.get(name)
    if fn is None:
        return None
    return extended_subs.get(fn, fn)


def _clean(expr: Optional[sp.Expr]) -> Optional[sp.Expr]:
    if expr is None:
        return None
    try:
        return lightweight_diagnostic_simplify(expr)
    except Exception:
        return expr


def compute_tov_diagnostics(
    *,
    background_id: str,
    stress_tensor: str,
    rho: sp.Expr,
    radial_pressure: sp.Expr,
    tangential_pressure: Optional[sp.Expr],
    geom: Any,
    extended_subs: Dict,
    independent_coord: sp.Symbol,
) -> Dict[str, Optional[sp.Expr]]:
    """
    Build static spherical TOV force-balance diagnostics.

    The residual is -dP_r/dr - (rho + P_r) Phi' + 2(P_t - P_r)/r.
    Equilibrium corresponds to residual = 0.
    """
    if not supports_tov(background_id, stress_tensor) or independent_coord is None:
        return {}

    r = independent_coord
    Pt = tangential_pressure if tangential_pressure is not None else radial_pressure

    if background_id == "SS_wormhole":
        shape_expr = _metric_expr("b", geom, extended_subs)
        redshift_expr = _metric_expr("Phi", geom, extended_subs)
        mass_expr = shape_expr / 2 if shape_expr is not None else None
        redshift_gradient = sp.diff(redshift_expr, r) if redshift_expr is not None else None
    elif background_id == "SS_blackhole":
        nu_expr = _metric_expr("nu_bh", geom, extended_subs)
        lam_expr = _metric_expr("lam_bh", geom, extended_subs)
        mass_expr = r * (1 - 1 / lam_expr) / 2 if lam_expr is not None else None
        redshift_gradient = sp.diff(sp.log(nu_expr), r) / 2 if nu_expr is not None else None
    else:
        return {}

    pressure_gradient = sp.diff(radial_pressure, r)
    hydrostatic_force = -pressure_gradient
    gravitational_force = (
        -(rho + radial_pressure) * redshift_gradient
        if redshift_gradient is not None else None
    )
    anisotropic_force = 2 * (Pt - radial_pressure) / r

    residual = None
    if gravitational_force is not None:
        residual = hydrostatic_force + gravitational_force + anisotropic_force

    mass_derivative = sp.diff(mass_expr, r) if mass_expr is not None else None
    mass_continuity_residual = (
        mass_derivative - 4 * sp.pi * r**2 * rho
        if mass_derivative is not None else None
    )
    compactness = 2 * mass_expr / r if mass_expr is not None else None

    return {
        "mass": _clean(mass_expr),
        "compactness": _clean(compactness),
        "redshift_gradient": _clean(redshift_gradient),
        "pressure_gradient": _clean(pressure_gradient),
        "hydrostatic_force": _clean(hydrostatic_force),
        "gravitational_force": _clean(gravitational_force),
        "anisotropic_force": _clean(anisotropic_force),
        "tov_residual": _clean(residual),
        "mass_continuity_residual": _clean(mass_continuity_residual),
    }
