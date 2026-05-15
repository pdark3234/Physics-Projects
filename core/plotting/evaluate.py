"""Numerical evaluation of symbolic plot expressions."""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, Iterable, List, Tuple

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


@lru_cache(maxsize=128)
def _compile_single_expr(expr_str: str, variable: str, parameter_names: Tuple[str, ...]):
    local = _local_dict(variable, {name: None for name in parameter_names})
    expr = sp.sympify(expr_str, locals=local)
    allowed = {variable, *parameter_names}
    missing = tuple(sorted(str(sym) for sym in expr.free_symbols if str(sym) not in allowed))
    args = [local[variable], *[local[name] for name in parameter_names]]
    return sp.lambdify(args, expr, modules=["numpy"]), missing


def _evaluate_expr(expr_str: str, x_values, variable: str, params: Dict[str, float]):
    import numpy as np

    parameter_names = tuple(params.keys())
    fn, missing = _compile_single_expr(expr_str, variable, parameter_names)
    if missing:
        raise ValueError(f"Missing numeric parameter values: {', '.join(missing)}")
    raw = fn(x_values, *[params[name] for name in parameter_names])
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


def _coerce_int(value: Any, name: str, *, min_value: int = 1, max_value: int = 101) -> int:
    try:
        out = int(value)
    except Exception as exc:
        raise ValueError(f"'{name}' must be an integer") from exc
    return max(min_value, min(max_value, out))


def _linspace_values(min_value: float, max_value: float, steps: int) -> List[float]:
    if steps <= 1:
        return [float(min_value)]
    step = (max_value - min_value) / float(steps - 1)
    return [float(min_value + i * step) for i in range(steps)]


def _flatten_groups(groups: Dict[str, Any]) -> List[Tuple[str, str, str]]:
    flat: List[Tuple[str, str, str]] = []
    for group_name, series in (groups or {}).items():
        if not isinstance(series, dict):
            continue
        for label, expr_str in series.items():
            flat.append((str(group_name), str(label), str(expr_str)))
    return flat


@lru_cache(maxsize=32)
def _compile_plot_expressions_cached(
    flat_exprs: Tuple[Tuple[str, str, str], ...],
    variable: str,
    parameter_names: Tuple[str, ...],
):
    local = _local_dict(variable, {name: None for name in parameter_names})
    args = [local[variable], *[local[name] for name in parameter_names]]
    compiled = []
    for group, label, expr_str in flat_exprs:
        expr = sp.sympify(expr_str, locals=local)
        missing = sorted(str(sym) for sym in expr.free_symbols if str(sym) not in {variable, *parameter_names})
        if missing:
            raise ValueError(f"{group}.{label} is missing parameter ranges for: {', '.join(missing)}")
        compiled.append((group, label, sp.lambdify(args, expr, modules=["numpy"])))
    return compiled


def _compile_plot_expressions(
    flat_exprs: Iterable[Tuple[str, str, str]],
    variable: str,
    parameter_names: List[str],
):
    return _compile_plot_expressions_cached(tuple(flat_exprs), variable, tuple(parameter_names))


def _numeric_array(raw, x_values):
    import numpy as np

    arr = np.asarray(raw, dtype=np.complex128)
    if arr.shape == ():
        arr = np.full_like(x_values, arr, dtype=np.complex128)
    arr = arr.reshape(-1)
    valid = np.isfinite(arr.real) & np.isfinite(arr.imag) & (np.abs(arr.imag) <= 1e-7)
    out = np.full(arr.shape, np.nan, dtype=float)
    out[valid] = arr.real[valid]
    return out


def _array_stats(values) -> Dict[str, Any]:
    import numpy as np

    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {"finite_fraction": 0.0, "min": None, "max": None, "max_abs": None}
    return {
        "finite_fraction": float(finite.size / max(1, arr.size)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "max_abs": float(np.max(np.abs(finite))),
    }


def _score_scan_point(series_stats: Dict[str, Dict[str, Dict[str, Any]]], constraints: Dict[str, Any]) -> Dict[str, Any]:
    tol = abs(float(constraints.get("tolerance", 1e-8)))
    tov_tol = abs(float(constraints.get("tov_tolerance", 1.0)))
    min_finite = float(constraints.get("min_finite_fraction", 0.95))
    checks = []
    penalties = []

    def add_check(name: str, passed: bool, penalty: float = 0.0):
        checks.append({"name": name, "passed": bool(passed)})
        if not passed:
            penalties.append(float(max(0.0, penalty)))

    all_stats = [
        stat
        for group in series_stats.values()
        for stat in group.values()
    ]
    if constraints.get("finite", True):
        finite_fraction = min((stat["finite_fraction"] for stat in all_stats), default=0.0)
        add_check("finite", finite_fraction >= min_finite, (min_finite - finite_fraction) * 100.0)

    matter = series_stats.get("matter", {})
    rho = matter.get("rho")
    if constraints.get("rho_positive", True) and rho:
        rho_min = rho.get("min")
        add_check("rho >= 0", rho_min is not None and rho_min >= -tol, abs(min(0.0, rho_min or 0.0)) * 10.0)

    if constraints.get("energy_conditions", True):
        ec_stats = series_stats.get("energy_conditions", {})
        if ec_stats:
            ec_min = min((stat["min"] for stat in ec_stats.values() if stat["min"] is not None), default=None)
            add_check("energy conditions >= 0", ec_min is not None and ec_min >= -tol, abs(min(0.0, ec_min or 0.0)) * 10.0)

    if constraints.get("stability", True):
        stability_stats = series_stats.get("stability", {})
        if stability_stats:
            st_min = min((stat["min"] for stat in stability_stats.values() if stat["min"] is not None), default=None)
            st_max = max((stat["max"] for stat in stability_stats.values() if stat["max"] is not None), default=None)
            stable = st_min is not None and st_max is not None and st_min >= -tol and st_max <= 1.0 + tol
            penalty = abs(min(0.0, st_min or 0.0)) * 20.0 + max(0.0, (st_max or 0.0) - 1.0) * 20.0
            add_check("0 <= c_s^2 <= 1", stable, penalty)

    if constraints.get("tov_residual", False):
        residual = series_stats.get("tov", {}).get("residual")
        if residual:
            max_abs = residual.get("max_abs")
            passed = max_abs is not None and max_abs <= tov_tol
            penalty = 0.0 if passed else min(100.0, 20.0 * ((max_abs or tov_tol * 10.0) / max(tov_tol, 1e-12)))
            add_check("|TOV residual|", passed, penalty)

    passed = all(check["passed"] for check in checks)
    score = max(0.0, 100.0 - sum(penalties))
    return {"passed": passed, "score": float(score), "checks": checks}


def scan_parameter_ranges(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Scan numeric parameter ranges against solved diagnostic expressions."""
    import itertools
    import numpy as np

    variable = str(payload.get("variable") or "t")
    domain = payload.get("domain") or {}
    x_min = _coerce_float(domain.get("min", 0.1), "domain.min")
    x_max = _coerce_float(domain.get("max", 10.0), "domain.max")
    points = max(8, min(int(domain.get("points", 160)), 400))
    if x_min == x_max:
        raise ValueError("Domain min and max must differ")
    if x_min > x_max:
        x_min, x_max = x_max, x_min

    ranges = payload.get("parameter_ranges") or {}
    if not isinstance(ranges, dict) or not ranges:
        raise ValueError("No parameter ranges were supplied")

    parameter_names = [str(name) for name in ranges.keys()]
    grids: Dict[str, List[float]] = {}
    total_points = 1
    for name in parameter_names:
        spec = ranges.get(name) or {}
        lo = _coerce_float(spec.get("min", 0.0), f"{name}.min")
        hi = _coerce_float(spec.get("max", lo), f"{name}.max")
        steps = _coerce_int(spec.get("steps", 7), f"{name}.steps", min_value=1, max_value=41)
        grids[name] = _linspace_values(lo, hi, steps)
        total_points *= len(grids[name])
    if total_points > 12000:
        raise ValueError("Parameter grid is too large. Reduce steps so total samples stay below 12000.")

    groups = payload.get("groups") or {}
    flat_exprs = _flatten_groups(groups)
    if not flat_exprs:
        raise ValueError("No plot expressions were supplied")

    compiled = _compile_plot_expressions(flat_exprs, variable, parameter_names)
    x_values = np.linspace(x_min, x_max, points)
    constraints = payload.get("constraints") or {}

    accepted = []
    samples = []
    best = None
    heatmap_accumulator: Dict[Tuple[float, float], Dict[str, Any]] = {}
    heatmap_params = parameter_names[:2]

    for values in itertools.product(*(grids[name] for name in parameter_names)):
        params = dict(zip(parameter_names, values))
        series_stats: Dict[str, Dict[str, Dict[str, Any]]] = {}
        for group, label, fn in compiled:
            try:
                raw = fn(x_values, *[params[name] for name in parameter_names])
                arr = _numeric_array(raw, x_values)
                stat = _array_stats(arr)
            except Exception:
                stat = {"finite_fraction": 0.0, "min": None, "max": None, "max_abs": None}
            series_stats.setdefault(group, {})[label] = stat

        scored = _score_scan_point(series_stats, constraints)
        sample = {
            "parameters": {name: float(params[name]) for name in parameter_names},
            "score": scored["score"],
            "passed": scored["passed"],
            "checks": scored["checks"],
        }
        samples.append(sample)
        if scored["passed"]:
            accepted.append(sample)
        if best is None or sample["score"] > best["score"]:
            best = sample

        if len(heatmap_params) >= 2:
            cell = (float(params[heatmap_params[0]]), float(params[heatmap_params[1]]))
            current = heatmap_accumulator.get(cell)
            if current is None or sample["score"] > current["score"]:
                heatmap_accumulator[cell] = {"score": sample["score"], "passed": sample["passed"]}

    ranges_out = {}
    for name in parameter_names:
        vals = [sample["parameters"][name] for sample in accepted]
        ranges_out[name] = {
            "min": float(min(vals)) if vals else None,
            "max": float(max(vals)) if vals else None,
            "accepted_values": sorted(set(float(v) for v in vals))[:80],
        }

    samples_sorted = sorted(samples, key=lambda item: item["score"], reverse=True)
    heatmap = None
    if len(heatmap_params) >= 2:
        heatmap = {
            "x_param": heatmap_params[0],
            "y_param": heatmap_params[1],
            "x_values": grids[heatmap_params[0]],
            "y_values": grids[heatmap_params[1]],
            "cells": [
                {
                    "x": x,
                    "y": y,
                    "score": heatmap_accumulator.get((float(x), float(y)), {}).get("score", 0.0),
                    "passed": heatmap_accumulator.get((float(x), float(y)), {}).get("passed", False),
                }
                for y in grids[heatmap_params[1]]
                for x in grids[heatmap_params[0]]
            ],
        }

    return {
        "variable": variable,
        "total": int(total_points),
        "accepted": len(accepted),
        "acceptance_fraction": float(len(accepted) / max(1, total_points)),
        "best": best,
        "accepted_ranges": ranges_out,
        "top_samples": samples_sorted[:20],
        "heatmap": heatmap,
        "metadata": {
            "domain": {"min": x_min, "max": x_max, "points": points},
            "parameters": parameter_names,
        },
    }
