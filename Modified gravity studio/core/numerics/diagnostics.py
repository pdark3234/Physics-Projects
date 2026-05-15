"""Numerical diagnostics derived from pointwise matter solutions."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

import sympy as sp


_SYM_FUNCS = {
    "exp": sp.exp,
    "log": sp.log,
    "sin": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
    "sinh": sp.sinh,
    "cosh": sp.cosh,
    "tanh": sp.tanh,
    "sqrt": sp.sqrt,
    "Abs": sp.Abs,
    "pi": sp.pi,
    "E": sp.E,
}


def safe_divide(num, den):
    import numpy as np

    num_arr, den_arr = np.broadcast_arrays(np.asarray(num, dtype=float), np.asarray(den, dtype=float))
    out = np.full_like(num_arr, np.nan, dtype=float)
    finite_den = np.abs(den_arr[np.isfinite(den_arr)])
    scale = float(np.nanmax(finite_den)) if finite_den.size else 1.0
    threshold = max(1e-8, 1e-6 * scale)
    mask = np.isfinite(num_arr) & np.isfinite(den_arr) & (np.abs(den_arr) > threshold)
    out[mask] = num_arr[mask] / den_arr[mask]
    return out


def _list_from_array(values):
    import numpy as np

    arr = np.asarray(values, dtype=float)
    return [float(v) if np.isfinite(v) else None for v in arr]


def _solution_array(solutions: Dict[str, List[Any]], *names: str):
    import numpy as np

    for name in names:
        if name in solutions:
            return np.asarray([np.nan if v is None else v for v in solutions[name]], dtype=float)
    return None


def _local_dict(variable: str, parameters: Sequence[str]) -> Dict[str, Any]:
    local = dict(_SYM_FUNCS)
    for name in [variable, *parameters]:
        local[name] = sp.Symbol(name)
    return local


def _compile_expr(expr_str: str, variable: str, parameters: Sequence[str]):
    local = _local_dict(variable, parameters)
    expr = sp.sympify(expr_str, locals=local)
    var_sym = local[variable]
    param_syms = tuple(local[name] for name in parameters)
    return sp.lambdify((var_sym, *param_syms), expr, modules=["numpy"])


def _eval_metric_expr(expr_str: str, x_values, variable: str, parameter_names, param_values):
    import numpy as np

    try:
        fn = _compile_expr(expr_str, variable, parameter_names)
        raw = fn(x_values, *param_values)
        arr = np.asarray(raw, dtype=np.complex128)
        if arr.shape == ():
            arr = np.full_like(x_values, arr, dtype=np.complex128)
        if np.max(np.abs(arr.imag)) > 1e-7:
            return None
        out = arr.real.astype(float)
        out[~np.isfinite(out)] = np.nan
        return out
    except Exception:
        return None


def compute_numeric_diagnostics(x_values, solutions: Dict[str, List[Any]], stress_tensor: str) -> Dict[str, List[Any]]:
    import numpy as np

    rho = _solution_array(solutions, "rho")
    if rho is None:
        return {}
    Pr = _solution_array(solutions, "P_r", "Pr", "p")
    Pt = _solution_array(solutions, "P_t", "Pt")
    if Pr is None:
        return {}
    if Pt is None:
        Pt = Pr

    omega_r = safe_divide(Pr, rho)
    omega_t = safe_divide(Pt, rho)
    omega_eff = safe_divide((Pr + 2 * Pt) / 3.0, rho)
    NEC_r = rho + Pr
    NEC_t = rho + Pt
    WEC = rho
    SEC = rho + Pr + 2 * Pt
    DEC_r = rho - np.abs(Pr)
    DEC_t = rho - np.abs(Pt)

    drho = np.gradient(rho, x_values)
    cs2_r = safe_divide(np.gradient(Pr, x_values), drho)
    cs2_t = safe_divide(np.gradient(Pt, x_values), drho)

    if stress_tensor == "anisotropic" or not np.allclose(Pt, Pr, equal_nan=True):
        out = {
            "omega_r": omega_r,
            "omega_t": omega_t,
            "omega_eff": omega_eff,
            "NEC_r": NEC_r,
            "NEC_t": NEC_t,
            "WEC": WEC,
            "SEC": SEC,
            "DEC_r": DEC_r,
            "DEC_t": DEC_t,
            "cs2_r": cs2_r,
            "cs2_t": cs2_t,
        }
    else:
        out = {
            "omega": omega_r,
            "NEC": NEC_r,
            "WEC": WEC,
            "SEC": SEC,
            "DEC": DEC_r,
            "cs2": cs2_r,
        }
    return {key: _list_from_array(value) for key, value in out.items()}


def compute_numeric_tov(
    x_values,
    solutions: Dict[str, List[Any]],
    *,
    background_id: str,
    stress_tensor: str,
    metric_functions: Dict[str, str],
    variable: str,
    parameter_names: Sequence[str],
    param_values: Sequence[float],
) -> Dict[str, List[Any]]:
    import numpy as np

    if background_id not in {"SS_wormhole", "SS_blackhole"}:
        return {}
    if stress_tensor not in {"perfect_fluid", "anisotropic"}:
        return {}

    rho = _solution_array(solutions, "rho")
    Pr = _solution_array(solutions, "P_r", "Pr", "p")
    Pt = _solution_array(solutions, "P_t", "Pt")
    if rho is None or Pr is None:
        return {}
    if Pt is None:
        Pt = Pr

    mass = None
    redshift_gradient = None
    if background_id == "SS_wormhole":
        b_expr = metric_functions.get("b")
        phi_expr = metric_functions.get("Phi")
        b = _eval_metric_expr(b_expr, x_values, variable, parameter_names, param_values) if b_expr else None
        phi = _eval_metric_expr(phi_expr, x_values, variable, parameter_names, param_values) if phi_expr else None
        if b is not None:
            mass = b / 2.0
        if phi is not None:
            redshift_gradient = np.gradient(phi, x_values)
    elif background_id == "SS_blackhole":
        nu_expr = metric_functions.get("nu_bh")
        lam_expr = metric_functions.get("lam_bh")
        nu = _eval_metric_expr(nu_expr, x_values, variable, parameter_names, param_values) if nu_expr else None
        lam = _eval_metric_expr(lam_expr, x_values, variable, parameter_names, param_values) if lam_expr else None
        if lam is not None:
            mass = x_values * (1.0 - safe_divide(1.0, lam)) / 2.0
        if nu is not None:
            redshift_gradient = np.gradient(np.log(np.abs(nu)), x_values) / 2.0

    pressure_gradient = np.gradient(Pr, x_values)
    hydrostatic_force = -pressure_gradient
    anisotropic_force = safe_divide(2.0 * (Pt - Pr), x_values)
    out = {
        "pressure_gradient": pressure_gradient,
        "hydrostatic_force": hydrostatic_force,
        "anisotropic_force": anisotropic_force,
    }
    if redshift_gradient is not None:
        gravitational_force = -(rho + Pr) * redshift_gradient
        out["redshift_gradient"] = redshift_gradient
        out["gravitational_force"] = gravitational_force
        out["residual"] = hydrostatic_force + gravitational_force + anisotropic_force
    if mass is not None:
        out["mass"] = mass
        out["compactness"] = safe_divide(2.0 * mass, x_values)
        out["mass_continuity_residual"] = np.gradient(mass, x_values) - 4.0 * np.pi * x_values**2 * rho
    return {key: _list_from_array(value) for key, value in out.items()}
