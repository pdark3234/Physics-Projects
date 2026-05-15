"""Shared helpers for theory assembly modules."""

from typing import Any, Iterable


def track_tensor_names(geometry_cache: Any, names: Iterable[str]) -> None:
    """Track pytearcat tensor names for cleanup without growing duplicates."""
    if geometry_cache is None or not hasattr(geometry_cache, 'tensor_names'):
        return
    for name in names:
        if name not in geometry_cache.tensor_names:
            geometry_cache.tensor_names.append(name)


def simplify_selected_component(expr: Any, label: str) -> Any:
    """Log and simplify one selected field-equation component."""
    import sympy as sp
    from core.solver import bounded_fraction_cleanup, fast_simplify

    try:
        ops = sp.count_ops(expr)
        print(f"[SIMPLIFY] Simplifying selected component {label} (ops={ops})", flush=True)
    except Exception:
        ops = 0
        print(f"[SIMPLIFY] Simplifying selected component {label}", flush=True)

    try:
        symbolic_power = any(
            pow_expr.exp.free_symbols
            for pow_expr in expr.atoms(sp.Pow)
        )
    except Exception:
        symbolic_power = False
    try:
        hard_transcendental = ops > 900 and expr.has(
            sp.exp, sp.log, sp.sin, sp.cos, sp.tan,
            sp.sinh, sp.cosh, sp.tanh, sp.sqrt,
        )
    except Exception:
        hard_transcendental = False

    if (symbolic_power and ops > 700) or hard_transcendental or ops > 2400:
        simplified = bounded_fraction_cleanup(expr)
    else:
        simplified = fast_simplify(expr)
    simplified = _keep_if_not_exploded(expr, simplified, ops, label)

    try:
        print(
            f"[SIMPLIFY] Selected component {label} simplified (ops={sp.count_ops(simplified)})",
            flush=True,
        )
    except Exception:
        print(f"[SIMPLIFY] Selected component {label} simplified", flush=True)
    return simplified


def simplify_selected_component_transcendental(expr: Any, label: str) -> Any:
    """Log and simplify a component while preserving transcendental atoms."""
    import sympy as sp
    from core.solver import bounded_fraction_cleanup, rational_transcendental_simplify

    try:
        ops = sp.count_ops(expr)
        print(f"[SIMPLIFY] Rational-transcendental cleanup for {label} (ops={ops})", flush=True)
    except Exception:
        ops = 0
        print(f"[SIMPLIFY] Rational-transcendental cleanup for {label}", flush=True)

    try:
        hard = ops > 2500 and expr.has(
            sp.exp, sp.log, sp.sin, sp.cos, sp.tan,
            sp.sinh, sp.cosh, sp.tanh,
        )
    except Exception:
        hard = False

    if hard:
        simplified = bounded_fraction_cleanup(expr)
    else:
        simplified = rational_transcendental_simplify(expr)
    simplified = _keep_if_not_exploded(expr, simplified, ops, label)

    try:
        print(
            f"[SIMPLIFY] Rational-transcendental {label} done (ops={sp.count_ops(simplified)})",
            flush=True,
        )
    except Exception:
        print(f"[SIMPLIFY] Rational-transcendental {label} done", flush=True)
    return simplified


def _keep_if_not_exploded(original: Any, candidate: Any, original_ops: int, label: str) -> Any:
    """Reject component cleanup that greatly increases downstream complexity."""
    import sympy as sp

    try:
        candidate_ops = sp.count_ops(candidate)
        if candidate_ops > max(original_ops * 3, original_ops + 3500):
            print(
                f"[SIMPLIFY] Keeping original {label}; candidate grew "
                f"{original_ops}->{candidate_ops}",
                flush=True,
            )
            return original
    except Exception:
        pass
    return candidate
