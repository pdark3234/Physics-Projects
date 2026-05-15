"""Numerical evaluation of symbolic plot expressions."""

from __future__ import annotations

from typing import Any, Dict

import math

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


def _coerce_float(value: Any, name: str) -> float:
    try:
        out = float(sp.N(sp.sympify(value)))
    except Exception as exc:
        raise ValueError(f"Parameter '{name}' must be numeric") from exc
    if not math.isfinite(out):
        raise ValueError(f"Parameter '{name}' must be finite")
    return out


def _local_dict(variable: str, params: Dict[str, Any]) -> Dict[str, Any]:
    local = dict(_SYM_FUNCS)
    local[variable] = sp.Symbol(variable)
    for name in params:
        local[str(name)] = sp.Symbol(str(name))
    return local


def _evaluate_expr(expr_str: str, x_values, variable: str, params: Dict[str, float]):
    import numpy as np

    local = _local_dict(variable, params)
    expr = sp.sympify(expr_str, locals=local)
    symbols = {str(sym) for sym in expr.free_symbols}
    missing = sorted(symbols - {variable, *params.keys()})
    if missing:
        raise ValueError(f"Missing numeric parameter values: {', '.join(missing)}")
    args = [local[variable], *[local[name] for name in params]]
    fn = sp.lambdify(args, expr, modules=["numpy"])
    raw = fn(x_values, *[params[name] for name in params])
    arr = np.asarray(raw, dtype=np.complex128)
    if arr.shape == ():
        arr = np.full_like(x_values, arr, dtype=np.complex128)
    out = []
    for value in arr.reshape(-1):
        if not np.isfinite(value.real) or not np.isfinite(value.imag) or abs(value.imag) > 1e-7:
            out.append(None)
        else:
            out.append(float(value.real))
    return out


def evaluate_plot_series(payload: Dict[str, Any]) -> Dict[str, Any]:
    import numpy as np

    variable = str(payload.get("variable") or "t")
    domain = payload.get("domain") or {}
    x_min = _coerce_float(domain.get("min", 0.1), "domain.min")
    x_max = _coerce_float(domain.get("max", 10.0), "domain.max")
    points = max(8, min(int(domain.get("points", 300)), 1000))
    if x_min == x_max:
        raise ValueError("Domain min and max must differ")
    if x_min > x_max:
        x_min, x_max = x_max, x_min

    raw_params = payload.get("parameters") or {}
    params = {str(name): _coerce_float(value, str(name)) for name, value in raw_params.items()}
    groups = payload.get("groups") or {}
    if not isinstance(groups, dict) or not groups:
        raise ValueError("No plot expressions were supplied")

    x_values = np.linspace(x_min, x_max, points)
    evaluated: Dict[str, Dict[str, Any]] = {}
    warnings = []
    for group_name, series in groups.items():
        if not isinstance(series, dict):
            continue
        evaluated[group_name] = {}
        for label, expr_str in series.items():
            try:
                evaluated[group_name][str(label)] = _evaluate_expr(str(expr_str), x_values, variable, params)
            except Exception as exc:
                warnings.append(f"{group_name}.{label}: {exc}")

    return {
        "variable": variable,
        "x": [float(v) for v in x_values],
        "groups": evaluated,
        "warnings": warnings,
        "metadata": {
            "points": len(x_values),
            "parameters": list(params.keys()),
        },
    }
