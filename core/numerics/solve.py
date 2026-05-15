"""Pointwise numerical solves for exported field-equation residuals."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Sequence

import math

import sympy as sp

from core.numerics.diagnostics import compute_numeric_diagnostics, compute_numeric_tov


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


@dataclass(frozen=True)
class NumericSpec:
    residuals: tuple[str, ...]
    variable: str
    unknowns: tuple[str, ...]
    parameters: tuple[str, ...]


def _coerce_float(value: Any, name: str) -> float:
    try:
        out = float(sp.N(sp.sympify(value)))
    except Exception as exc:
        raise ValueError(f"Parameter '{name}' must be numeric") from exc
    if not math.isfinite(out):
        raise ValueError(f"Parameter '{name}' must be finite")
    return out


def _local_dict(variable: str, unknowns: Sequence[str], parameters: Sequence[str]) -> Dict[str, Any]:
    local = dict(_SYM_FUNCS)
    for name in [variable, *unknowns, *parameters]:
        local[name] = sp.Symbol(name)
    return local


def _symbol_names(exprs: Iterable[sp.Expr]) -> set[str]:
    names: set[str] = set()
    for expr in exprs:
        names.update(str(sym) for sym in expr.free_symbols)
    return names


@lru_cache(maxsize=64)
def _compile_residuals(spec: NumericSpec):
    import numpy as np

    local = _local_dict(spec.variable, spec.unknowns, spec.parameters)
    exprs = tuple(sp.sympify(expr, locals=local) for expr in spec.residuals)
    var_sym = local[spec.variable]
    unknown_syms = tuple(local[name] for name in spec.unknowns)
    param_syms = tuple(local[name] for name in spec.parameters)
    args = (var_sym, *unknown_syms, *param_syms)
    fn = sp.lambdify(args, exprs, modules=["numpy"])
    required = _symbol_names(exprs) - {spec.variable, *spec.unknowns}
    return fn, tuple(sorted(required))


def _evaluate_residuals(fn, x_value: float, y_values, param_values):
    import numpy as np

    try:
        raw = fn(x_value, *list(y_values), *list(param_values))
        arr = np.asarray(raw, dtype=np.complex128).reshape(-1)
    except Exception:
        return None
    if arr.size == 0:
        return None
    if not np.all(np.isfinite(arr.real)) or not np.all(np.isfinite(arr.imag)):
        return None
    if np.max(np.abs(arr.imag)) > 1e-7:
        return None
    return arr.real.astype(float)


def solve_residual_system(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Solve a residual system over a 1D domain using continuation.

    The system is solved pointwise with scipy.optimize.least_squares.  The
    previous successful point becomes the next initial guess, which is the key
    stabilizer for non-linear modified-gravity residuals.
    """
    import numpy as np
    from scipy.optimize import least_squares

    residuals = tuple(str(item) for item in payload.get("residuals", []) if str(item).strip())
    variable = str(payload.get("variable") or "t")
    unknowns = tuple(str(item) for item in payload.get("unknowns", []) if str(item).strip())
    if not residuals:
        raise ValueError("No residual equations were supplied")
    if not unknowns:
        raise ValueError("No matter unknowns were supplied")

    domain = payload.get("domain") or {}
    x_min = _coerce_float(domain.get("min", 0.1), "domain.min")
    x_max = _coerce_float(domain.get("max", 10.0), "domain.max")
    points = int(domain.get("points", 120))
    points = max(8, min(points, 500))
    if x_min == x_max:
        raise ValueError("Domain min and max must differ")
    if x_min > x_max:
        x_min, x_max = x_max, x_min

    raw_params = payload.get("parameters") or {}
    parameter_names = tuple(sorted(str(name) for name in raw_params))
    spec = NumericSpec(residuals=residuals, variable=variable, unknowns=unknowns, parameters=parameter_names)
    fn, required_params = _compile_residuals(spec)

    missing = [name for name in required_params if name not in raw_params]
    if missing:
        raise ValueError(f"Missing numeric parameter values: {', '.join(missing)}")

    param_values = [_coerce_float(raw_params[name], name) for name in parameter_names]
    raw_guesses = payload.get("initial_guesses") or {}
    current_guess = np.array(
        [_coerce_float(raw_guesses.get(name, 1.0), f"initial guess {name}") for name in unknowns],
        dtype=float,
    )

    x_values = np.linspace(x_min, x_max, points)
    solutions: Dict[str, List[Any]] = {name: [] for name in unknowns}
    residual_norm: List[Any] = []
    success: List[bool] = []
    messages: List[str] = []

    def residual_vec(y, x_value):
        values = _evaluate_residuals(fn, float(x_value), y, param_values)
        if values is None:
            return np.full(len(residuals), 1e12, dtype=float)
        return values

    for x_value in x_values:
        try:
            result = least_squares(
                residual_vec,
                current_guess,
                args=(float(x_value),),
                max_nfev=250,
                xtol=1e-10,
                ftol=1e-10,
                gtol=1e-10,
            )
            norm = float(np.linalg.norm(result.fun))
            ok = bool(result.success and math.isfinite(norm) and norm < 1e-5)
            if ok:
                current_guess = result.x.astype(float)
                for name, value in zip(unknowns, result.x):
                    solutions[name].append(float(value))
                residual_norm.append(norm)
                success.append(True)
            else:
                for name in unknowns:
                    solutions[name].append(None)
                residual_norm.append(norm if math.isfinite(norm) else None)
                success.append(False)
                messages.append(f"x={float(x_value):.6g}: residual norm {norm:.3g}")
        except Exception as exc:
            for name in unknowns:
                solutions[name].append(None)
            residual_norm.append(None)
            success.append(False)
            messages.append(f"x={float(x_value):.6g}: {exc}")

    finite_count = int(sum(success))
    warnings = []
    if finite_count < len(x_values):
        warnings.append(f"{len(x_values) - finite_count} of {len(x_values)} points did not converge.")
    if messages:
        warnings.extend(messages[:5])

    diagnostics = compute_numeric_diagnostics(
        x_values,
        solutions,
        str(payload.get("stress_tensor") or ""),
    )
    tov = compute_numeric_tov(
        x_values,
        solutions,
        background_id=str(payload.get("background_id") or ""),
        stress_tensor=str(payload.get("stress_tensor") or ""),
        metric_functions=payload.get("metric_functions") or {},
        variable=variable,
        parameter_names=parameter_names,
        param_values=param_values,
    )

    return {
        "variable": variable,
        "x": [float(v) for v in x_values],
        "solutions": solutions,
        "diagnostics": diagnostics,
        "tov": tov,
        "residual_norm": residual_norm,
        "success": success,
        "warnings": warnings,
        "metadata": {
            "points": len(x_values),
            "converged_points": finite_count,
            "unknowns": list(unknowns),
            "parameters": list(parameter_names),
            "residual_count": len(residuals),
        },
    }
