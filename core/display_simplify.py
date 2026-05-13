"""Parallel display simplification helpers for large SymPy expressions."""

from __future__ import annotations

import base64
import pickle
import subprocess
import sys
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

import sympy as sp

DISPLAY_TOURNAMENT_MIN_OPS = 1000
DISPLAY_TOURNAMENT_MAX_OPS = 14000
DISPLAY_TOURNAMENT_TIMEOUT = 30.0
DISPLAY_TOURNAMENT_MAX_WORKERS = 8
DISPLAY_PARTITION_MIN_OPS = 3500
DISPLAY_PARTITION_MAX_OPS = 120000
DISPLAY_PARTITION_TARGET_OPS = 1800
DISPLAY_PARTITION_TIMEOUT = 45.0
DISPLAY_PARTITION_MAX_WORKERS = 8

_STRATEGIES = (
    "powsimp_factor",
    "fraction_parts",
    "collect_params",
    "chunk_terms",
    "cancel_factor",
    "together_factor",
    "fraction_together",
    "chunk_together",
)


def simplify_for_display(expr: sp.Expr, ops: Optional[int] = None,
                         label: str = "expr") -> sp.Expr:
    """Public entry point for display-focused simplification.

    Previously this function contained a duplicated subprocess-tournament
    implementation that was never called from any code path (the real pipeline
    lived in ``results._readable_display_simplify`` →
    ``results._parallel_display_simplify``).

    This version delegates to the canonical pipeline so there is exactly one
    code path for display simplification.  Callers that want to drive the
    pipeline directly should import from ``core.results`` instead.
    """
    if expr is None:
        return None
    try:
        ops = int(ops if ops is not None else sp.count_ops(expr))
    except Exception:
        return expr
    try:
        from core.results import _readable_display_simplify, _parallel_display_simplify
        expr = _readable_display_simplify(expr, ops)
        expr = _parallel_display_simplify(expr, ops, label)
        return expr
    except Exception:
        # Graceful degradation: cheap factor pass only
        factored = _factor_common_display_terms(expr)
        return factored if factored is not None else expr


def _select_strategies(expr: sp.Expr) -> Iterable[str]:
    try:
        if _has_symbolic_power(expr):
            return (
                "powsimp_factor",
                "fraction_parts",
                "collect_params",
                "chunk_terms",
                "cancel_factor",
                "together_factor",
                "fraction_together",
                "chunk_together",
            )
        # Sqrt-dominated (anisotropic wormhole / NEC/DEC forms with radicals):
        # prefer fraction_parts first so the num/den decomposition can expose
        # collapsible radical pairs before expensive together/cancel calls.
        # Extended to handle larger anisotropic expressions and include parameter collection.
        sqrt_atoms = [
            pw for pw in expr.atoms(sp.Pow)
            if pw.exp == sp.Rational(1, 2)
        ]
        if sqrt_atoms and (len(sqrt_atoms) >= 2 or sp.count_ops(expr) >= 1500):
            return (
                "fraction_parts",
                "powsimp_factor",
                "collect_params",
                "cancel_factor",
                "together_factor",
                "fraction_together",
            )
        if isinstance(expr, sp.Add):
            return (
                "powsimp_factor",
                "fraction_parts",
                "chunk_terms",
                "cancel_factor",
                "together_factor",
                "chunk_together",
            )
        return (
            "powsimp_factor",
            "fraction_parts",
            "collect_params",
            "cancel_factor",
            "together_factor",
        )
    except Exception:
        return _STRATEGIES


def _decode_worker_result(raw: bytes) -> Optional[Tuple[int, int, sp.Expr]]:
    if not raw:
        return None
    try:
        data = pickle.loads(base64.b64decode(raw.strip()))
        expr = data["expr"]
        return int(data["ops"]), int(data["latex_len"]), expr
    except Exception:
        return None


def _parallel_partition_simplify(expr: sp.Expr, ops: int) -> Optional[sp.Expr]:
    """
    Split a large additive expression into independent pieces and simplify each
    piece in its own helper process before joining the expression back together.
    """
    if ops < DISPLAY_PARTITION_MIN_OPS or ops > DISPLAY_PARTITION_MAX_OPS:
        return None

    pieces = _partition_expression(expr, DISPLAY_PARTITION_TARGET_OPS)
    start = time.perf_counter()
    tasks = _build_partition_tasks(pieces, DISPLAY_PARTITION_TARGET_OPS)
    if len(tasks) < 2:
        return None

    jobs = []
    for task in tasks:
        if time.perf_counter() - start >= DISPLAY_PARTITION_TIMEOUT:
            break
        proc = _start_worker("partition_piece")
        if proc is None:
            continue
        task_expr = task["expr"]
        payload = _encode_worker_payload({
            "mode": "partition_piece",
            "expr": task_expr,
            "strategy": "piece_heavy" if _count_ops(task_expr) > 900 else "piece",
        })
        jobs.append((task, proc, payload))
        if len(jobs) >= DISPLAY_PARTITION_MAX_WORKERS:
            break

    if not jobs:
        return None

    completed: Dict[Tuple[int, str, int], sp.Expr] = {}
    for task, proc, payload in jobs:
        original_piece = task["expr"]
        remaining = max(0.1, DISPLAY_PARTITION_TIMEOUT - (time.perf_counter() - start))
        try:
            out, _ = proc.communicate(input=payload, timeout=remaining)
        except subprocess.TimeoutExpired:
            _kill_process(proc)
            completed[task["key"]] = original_piece
            continue
        except Exception:
            _kill_process(proc)
            completed[task["key"]] = original_piece
            continue

        candidate = _decode_worker_result(out)
        if candidate is None:
            completed[task["key"]] = original_piece
            continue

        cand_ops, latex_len, cand_expr = candidate
        if _is_reasonable_piece_candidate(original_piece, cand_expr, cand_ops, latex_len):
            completed[task["key"]] = cand_expr
        else:
            completed[task["key"]] = original_piece

    for task in tasks:
        completed.setdefault(task["key"], task["expr"])

    joined_pieces = _rebuild_partition_pieces(pieces, tasks, completed)
    joined = _join_partitioned_expression(expr, joined_pieces)
    joined = _final_join_pass(joined)
    joined_ops = _count_ops(joined)
    if not _is_reasonable_joined_candidate(expr, ops, joined, joined_ops):
        return None

    print(
        f"[DISPLAY_SIMPLIFY] partition helpers {len(jobs)}/{len(tasks)} tasks "
        f"across {len(pieces)} chunks, "
        f"ops {ops}->{joined_ops}",
        flush=True,
    )
    return joined


def _partition_expression(expr: sp.Expr, target_ops: int) -> List[sp.Expr]:
    if isinstance(expr, sp.Add):
        return _partition_add_terms(list(expr.args), target_ops)

    num, den = _safe_fraction(expr)
    if den != 1:
        return [expr]

    if isinstance(expr, sp.Derivative) and _count_ops(expr.expr) > target_ops:
        return [expr]

    return []


def _partition_add_terms(terms: List[sp.Expr], target_ops: int) -> List[sp.Expr]:
    pieces = []
    current = []
    current_ops = 0
    for term in terms:
        term_ops = _count_ops(term)
        if current and current_ops + term_ops > target_ops:
            pieces.append(sp.Add(*current, evaluate=False))
            current = []
            current_ops = 0
        current.append(term)
        current_ops += term_ops
    if current:
        pieces.append(sp.Add(*current, evaluate=False))
    return pieces


def _build_partition_tasks(pieces: List[sp.Expr], target_ops: int) -> List[Dict[str, Any]]:
    tasks: List[Dict[str, Any]] = []
    for piece_idx, piece in enumerate(pieces):
        if isinstance(piece, sp.Derivative) and _count_ops(piece.expr) > target_ops:
            _append_nested_expr_tasks(tasks, piece_idx, "whole", piece.expr, target_ops)
            continue

        num, den = _safe_fraction(piece)
        if den != 1 and (_count_ops(num) > target_ops or _count_ops(den) > target_ops):
            _append_nested_expr_tasks(tasks, piece_idx, "num", num, target_ops)
            _append_nested_expr_tasks(tasks, piece_idx, "den", den, target_ops)
        else:
            tasks.append({
                "key": (piece_idx, "whole", 0),
                "piece_idx": piece_idx,
                "role": "whole",
                "kind": "plain",
                "part_idx": 0,
                "expr": piece,
            })
    return tasks


def _append_nested_expr_tasks(
    tasks: List[Dict[str, Any]],
    piece_idx: int,
    role: str,
    expr: sp.Expr,
    target_ops: int,
) -> None:
    if isinstance(expr, sp.Derivative) and _count_ops(expr.expr) > target_ops:
        inner_num, inner_den = _safe_fraction(expr.expr)
        if inner_den != 1:
            num_parts = _partition_fraction_side(inner_num, target_ops)
            den_parts = _partition_fraction_side(inner_den, target_ops)
            for part_idx, part in enumerate(num_parts):
                tasks.append({
                    "key": (piece_idx, role, part_idx),
                    "piece_idx": piece_idx,
                    "role": role,
                    "kind": "derivative_fraction",
                    "side": "num",
                    "part_idx": part_idx,
                    "expr": part,
                    "variables": expr.variable_count,
                })
            offset = len(num_parts)
            for part_idx, part in enumerate(den_parts):
                tasks.append({
                    "key": (piece_idx, role, offset + part_idx),
                    "piece_idx": piece_idx,
                    "role": role,
                    "kind": "derivative_fraction",
                    "side": "den",
                    "part_idx": offset + part_idx,
                    "expr": part,
                    "variables": expr.variable_count,
                })
            return

        parts = _partition_fraction_side(expr.expr, target_ops)
        for part_idx, part in enumerate(parts):
            tasks.append({
                "key": (piece_idx, role, part_idx),
                "piece_idx": piece_idx,
                "role": role,
                "kind": "derivative",
                "part_idx": part_idx,
                "expr": part,
                "variables": expr.variable_count,
            })
        return

    parts = _partition_fraction_side(expr, target_ops)
    for part_idx, part in enumerate(parts):
        tasks.append({
            "key": (piece_idx, role, part_idx),
            "piece_idx": piece_idx,
            "role": role,
            "kind": "plain",
            "part_idx": part_idx,
            "expr": part,
        })


def _partition_fraction_side(expr: sp.Expr, target_ops: int) -> List[sp.Expr]:
    if isinstance(expr, sp.Derivative):
        return _partition_fraction_side(expr.expr, target_ops)
    num, den = _safe_fraction(expr)
    if den != 1:
        return [expr]
    if isinstance(expr, sp.Add):
        return _partition_add_terms(list(expr.args), target_ops)
    if isinstance(expr, sp.Mul):
        return _partition_mul_factors(expr, target_ops)
    return [expr]


def _partition_mul_factors(expr: sp.Expr, target_ops: int) -> List[sp.Expr]:
    factors = list(expr.args)
    pieces = []
    current = []
    current_ops = 0
    for factor in factors:
        factor_ops = _count_ops(factor)
        if current and current_ops + factor_ops > target_ops:
            pieces.append(sp.Mul(*current, evaluate=False))
            current = []
            current_ops = 0
        current.append(factor)
        current_ops += factor_ops
    if current:
        pieces.append(sp.Mul(*current, evaluate=False))
    return pieces


def _rebuild_partition_pieces(
    pieces: List[sp.Expr],
    tasks: List[Dict[str, Any]],
    completed: Dict[Tuple[int, str, int], sp.Expr],
) -> List[sp.Expr]:
    by_piece: Dict[int, List[Dict[str, Any]]] = {}
    for task in tasks:
        by_piece.setdefault(task["piece_idx"], []).append(task)

    rebuilt = []
    for piece_idx, original in enumerate(pieces):
        piece_tasks = by_piece.get(piece_idx, [])
        if not piece_tasks:
            rebuilt.append(original)
            continue
        if all(task["role"] == "whole" for task in piece_tasks):
            rebuilt.append(_rebuild_role_expr(
                piece_tasks,
                completed,
                original,
                wrap_derivative=isinstance(original, sp.Derivative),
            ))
            continue

        num_tasks = sorted(
            (task for task in piece_tasks if task["role"] == "num"),
            key=lambda task: task["part_idx"],
        )
        den_tasks = sorted(
            (task for task in piece_tasks if task["role"] == "den"),
            key=lambda task: task["part_idx"],
        )
        if not num_tasks or not den_tasks:
            rebuilt.append(original)
            continue

        num = _rebuild_role_expr(num_tasks, completed, original)
        den = _rebuild_role_expr(den_tasks, completed, original)
        rebuilt.append(num / den)
    return rebuilt


def _rebuild_role_expr(
    tasks: List[Dict[str, Any]],
    completed: Dict[Tuple[int, str, int], sp.Expr],
    fallback: sp.Expr,
    wrap_derivative: bool = False,
) -> sp.Expr:
    if not tasks:
        return fallback
    ordered = sorted(tasks, key=lambda task: task["part_idx"])
    first = ordered[0]
    if first.get("kind") == "derivative_fraction":
        num_parts = [
            completed.get(task["key"], task["expr"])
            for task in ordered
            if task.get("side") == "num"
        ]
        den_parts = [
            completed.get(task["key"], task["expr"])
            for task in ordered
            if task.get("side") == "den"
        ]
        expr = _join_fraction_side(num_parts) / _join_fraction_side(den_parts)
    else:
        parts = [completed.get(task["key"], task["expr"]) for task in ordered]
        expr = _join_fraction_side(parts)
    if wrap_derivative or first.get("kind") in ("derivative", "derivative_fraction"):
        variables = first.get("variables")
        if variables:
            try:
                return sp.Derivative(expr, *variables, evaluate=False)
            except Exception:
                return sp.Derivative(expr, evaluate=False)
    return expr


def _join_fraction_side(parts: List[sp.Expr]) -> sp.Expr:
    if not parts:
        return sp.Integer(1)
    if len(parts) == 1:
        return parts[0]
    if any(isinstance(part, sp.Add) for part in parts):
        return sp.Add(*parts, evaluate=False)
    return sp.Mul(*parts, evaluate=False)


def _join_partitioned_expression(original: sp.Expr, pieces: List[sp.Expr]) -> sp.Expr:
    return sp.Add(*pieces, evaluate=False)


def _safe_fraction(expr: sp.Expr) -> Tuple[sp.Expr, sp.Expr]:
    try:
        return sp.fraction(expr)
    except Exception:
        return expr, sp.Integer(1)


def _start_worker(mode: str) -> Optional[subprocess.Popen]:
    try:
        return subprocess.Popen(
            [sys.executable, "-m", "core.display_simplify", mode],
            cwd=_project_root(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return None


def _encode_worker_payload(payload: Any) -> bytes:
    return base64.b64encode(pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))


def _is_reasonable_piece_candidate(
    original: sp.Expr,
    candidate: sp.Expr,
    candidate_ops: int,
    latex_len: int,
) -> bool:
    if candidate is None or latex_len <= 0 or candidate_ops <= 1:
        return False
    if candidate_ops > max(_count_ops(original) * 4, _count_ops(original) + 2500):
        return False
    try:
        if not candidate.free_symbols.issubset(original.free_symbols):
            return False
    except Exception:
        return False
    return True


def _is_reasonable_joined_candidate(
    original: sp.Expr,
    original_ops: int,
    candidate: sp.Expr,
    candidate_ops: int,
) -> bool:
    if candidate is None or candidate == original:
        return False
    if candidate_ops > max(original_ops * 3, original_ops + 5000):
        return False
    try:
        if original.free_symbols != candidate.free_symbols:
            return False
    except Exception:
        return False
    return candidate_ops < original_ops or _joined_latex_is_better(original, candidate)


def _joined_latex_is_better(original: sp.Expr, candidate: sp.Expr) -> bool:
    try:
        cand_len = len(sp.latex(candidate))
        original_len = len(sp.latex(original))
        return cand_len < int(original_len * 0.97)
    except Exception:
        return False


def _is_reasonable_candidate(
    original: sp.Expr,
    original_ops: int,
    candidate: sp.Expr,
    candidate_ops: int,
    latex_len: int,
) -> bool:
    if candidate is None or latex_len <= 0:
        return False
    if candidate == original:
        return False
    if not getattr(candidate, "free_symbols", None):
        return False
    if candidate_ops <= 1:
        return False
    if candidate_ops > max(original_ops * 3, original_ops + 2500):
        return False
    try:
        if original.free_symbols != candidate.free_symbols:
            return False
    except Exception:
        return False
    return True


def _kill_process(proc: subprocess.Popen) -> None:
    try:
        proc.kill()
    except Exception:
        pass
    try:
        proc.communicate(timeout=1)
    except Exception:
        pass


def _project_root() -> str:
    from pathlib import Path

    return str(Path(__file__).resolve().parents[1])


def _worker_main(strategy: str) -> int:
    try:
        payload = sys.stdin.buffer.read()
        data = pickle.loads(base64.b64decode(payload))
        if isinstance(data, dict):
            expr = data["expr"]
            mode = data.get("mode", strategy)
            result = _run_worker_mode(expr, mode, data.get("strategy"))
        else:
            expr = data
            result = _run_strategy(expr, strategy)
        result = _final_light_pass(result)
        out = {
            "expr": result,
            "ops": int(sp.count_ops(result)),
            "latex_len": len(sp.latex(result)),
        }
        sys.stdout.buffer.write(base64.b64encode(pickle.dumps(out, protocol=pickle.HIGHEST_PROTOCOL)))
        return 0
    except Exception:
        return 2


def _run_worker_mode(expr: sp.Expr, mode: str, strategy: Optional[str] = None) -> sp.Expr:
    if mode == "partition_piece":
        if strategy == "piece_heavy":
            return _simplify_piece_heavy(expr)
        return _simplify_piece(expr)
    return _run_strategy(expr, strategy or mode)


def _run_strategy(expr: sp.Expr, strategy: str) -> sp.Expr:
    expr = _strip_physical_abs(sp.powsimp(expr, force=True))
    if strategy == "powsimp_factor":
        return sp.factor_terms(expr)
    if strategy == "fraction_parts":
        return _simplify_fraction_parts(expr)
    if strategy == "collect_params":
        return _collect_model_params(sp.factor_terms(expr))
    if strategy == "chunk_terms":
        return _simplify_additive_chunks(expr)
    if strategy == "cancel_factor":
        return _cancel_factor(expr)
    if strategy == "together_factor":
        return _together_factor(expr)
    if strategy == "fraction_together":
        return _simplify_fraction_together(expr)
    if strategy == "chunk_together":
        return _simplify_additive_chunks(expr, together=True)
    return expr


def _simplify_fraction_parts(expr: sp.Expr) -> sp.Expr:
    num, den = sp.fraction(expr)
    # For expressions with sqrt factors, try sqrtdenest before the standard
    # piece simplification so paired radicals collapse inside num/den.
    has_sqrt = any(
        pw.exp == sp.Rational(1, 2)
        for pw in expr.atoms(sp.Pow)
    )
    if has_sqrt:
        try:
            num = sp.sqrtdenest(sp.powsimp(num, force=True))
            den = sp.sqrtdenest(sp.powsimp(den, force=True))
        except Exception:
            pass
    num = _simplify_piece(num)
    den = _simplify_piece(den)
    return num / den


def _simplify_fraction_together(expr: sp.Expr) -> sp.Expr:
    num, den = sp.fraction(sp.together(expr))
    num = _simplify_piece_heavy(num)
    den = _simplify_piece_heavy(den)
    return sp.factor_terms(num / den)


def _cancel_factor(expr: sp.Expr) -> sp.Expr:
    return sp.factor_terms(sp.cancel(expr))


def _together_factor(expr: sp.Expr) -> sp.Expr:
    return sp.factor_terms(sp.cancel(sp.together(expr)))


def _simplify_additive_chunks(expr: sp.Expr, together: bool = False) -> sp.Expr:
    if not isinstance(expr, sp.Add):
        return _simplify_piece_heavy(expr) if together else _simplify_piece(expr)
    terms = list(expr.args)
    chunks = []
    current = []
    current_ops = 0
    for term in terms:
        term_ops = _count_ops(term)
        if current and current_ops + term_ops > (950 if together else 650):
            chunks.append(current)
            current = []
            current_ops = 0
        current.append(term)
        current_ops += term_ops
    if current:
        chunks.append(current)
    cleaner = _simplify_piece_heavy if together else _simplify_piece
    cleaned = [cleaner(sp.Add(*chunk, evaluate=False)) for chunk in chunks]
    return sp.Add(*cleaned, evaluate=False)


def _simplify_piece(piece: sp.Expr) -> sp.Expr:
    ops = _count_ops(piece)
    piece = sp.powsimp(piece, force=True)
    if ops <= 700:
        piece = sp.cancel(piece)
    if ops <= 1500:
        piece = sp.factor_terms(piece)
    return _collect_model_params(piece)


def _simplify_piece_heavy(piece: sp.Expr) -> sp.Expr:
    ops = _count_ops(piece)
    piece = sp.powsimp(piece, force=True)
    if ops <= 2200:
        piece = sp.cancel(sp.together(piece))
    elif ops <= 4500:
        piece = sp.cancel(piece)
    piece = sp.factor_terms(piece)
    return _collect_model_params(piece)


def _collect_model_params(expr: sp.Expr) -> sp.Expr:
    try:
        params = [
            sp.Symbol(name)
            for name in ("alpha", "beta", "gamma", "lam", "lamc", "lambda", "n", "m", "h", "k")
            if sp.Symbol(name) in expr.free_symbols
        ]
        if params:
            return sp.collect(expr, params, evaluate=True)
    except Exception:
        pass
    return expr


def _final_light_pass(expr: sp.Expr) -> sp.Expr:
    ops = _count_ops(expr)
    expr = _strip_physical_abs(sp.powsimp(expr, force=True))
    if ops <= 3000:
        expr = sp.factor_terms(expr)
    expr = _factor_common_display_terms(expr)
    return expr


def _final_join_pass(expr: sp.Expr) -> sp.Expr:
    ops = _count_ops(expr)
    try:
        expr = _strip_physical_abs(sp.powsimp(expr, force=True))
        if ops <= 18000:
            expr = sp.factor_terms(expr)
        if ops <= 9000:
            expr = _collect_model_params(expr)
        expr = _evaluate_display_derivatives(expr)
        expr = _factor_common_display_terms(expr)
    except Exception:
        pass
    return expr


def _factor_common_display_terms(expr: sp.Expr) -> sp.Expr:
    """
    Pull common display kernels out of long sums.

    SymPy often misses common factors when they contain repeated radicals,
    exponentials, or symbolic powers. Replacing those kernels with temporary
    symbols lets factor_terms see the shared algebraic structure, then we
    restore the original kernels.
    """
    if expr is None:
        return None
    try:
        ops = _count_ops(expr)
        if ops > 45000:
            return expr

        num, den = _safe_fraction(expr)
        num = _factor_common_additive_part(num)
        den = _factor_common_additive_part(den)
        rebuilt = num / den
        rebuilt = sp.powsimp(rebuilt, force=True)
        if _count_ops(rebuilt) <= 22000:
            rebuilt = sp.factor_terms(rebuilt)
        return rebuilt
    except Exception:
        return expr


def _factor_common_additive_part(expr: sp.Expr) -> sp.Expr:
    if expr is None:
        return expr
    try:
        if isinstance(expr, sp.Add):
            return _factor_add_with_kernels(expr)
        if isinstance(expr, sp.Mul):
            return sp.Mul(
                *(_factor_common_additive_part(arg) for arg in expr.args),
                evaluate=False,
            )
        return expr
    except Exception:
        return expr


def _factor_add_with_kernels(expr: sp.Expr) -> sp.Expr:
    try:
        if _count_ops(expr) > 22000:
            return expr
        kernels = _common_display_kernels(expr)
        if not kernels:
            return sp.factor_terms(expr)

        replacements = {
            kernel: sp.Symbol(f"_K{i}")
            for i, kernel in enumerate(kernels)
        }
        inverse = {v: k for k, v in replacements.items()}
        compressed = expr.xreplace(replacements)
        compressed = sp.factor_terms(sp.powsimp(compressed, force=True))
        compressed = _collect_kernel_symbols(compressed, list(inverse.keys()))
        restored = compressed.xreplace(inverse)
        restored = sp.powsimp(restored, force=True)
        return sp.factor_terms(restored)
    except Exception:
        return expr


def _common_display_kernels(expr: sp.Expr) -> List[sp.Expr]:
    try:
        terms = list(expr.args) if isinstance(expr, sp.Add) else [expr]
        if len(terms) < 2:
            return []
        counts: Dict[sp.Expr, int] = {}
        for term in terms:
            seen = set(_display_kernel_atoms(term))
            for atom in seen:
                counts[atom] = counts.get(atom, 0) + 1
        min_repeats = 2 if len(terms) <= 4 else max(2, len(terms) // 3)
        kernels = [
            atom for atom, count in counts.items()
            if count >= min_repeats and _count_ops(atom) >= 1
        ]
        kernels = sorted(
            kernels,
            key=lambda item: (-counts[item], -_count_ops(item), str(item)),
        )
        return kernels[:12]
    except Exception:
        return []


def _display_kernel_atoms(expr: sp.Expr) -> List[sp.Expr]:
    atoms = []
    try:
        atoms.extend(expr.atoms(sp.exp, sp.log, sp.sin, sp.cos, sp.tan, sp.sinh, sp.cosh, sp.tanh))
        for pow_expr in expr.atoms(sp.Pow):
            exp = pow_expr.exp
            # Both sqrt (1/2) and symbolic-power kernels are first-class.
            if exp == sp.Rational(1, 2) or getattr(exp, "free_symbols", set()):
                atoms.append(pow_expr)
        for name in ("alpha", "beta", "gamma", "lam", "lamc", "lambda", "n", "m", "h"):
            sym = sp.Symbol(name)
            if sym in expr.free_symbols:
                atoms.append(sym)
    except Exception:
        pass
    return atoms


def _collect_kernel_symbols(expr: sp.Expr, kernels: List[sp.Symbol]) -> sp.Expr:
    try:
        if kernels:
            return sp.collect(expr, kernels, evaluate=True)
    except Exception:
        pass
    return expr


def _evaluate_display_derivatives(expr: sp.Expr) -> sp.Expr:
    """Evaluate derivative nodes only after their arguments are small enough."""
    try:
        replacements = {}
        for deriv in sorted(expr.atoms(sp.Derivative), key=lambda item: -_count_ops(item.expr)):
            inner_ops = _count_ops(deriv.expr)
            if inner_ops > 3200:
                continue
            try:
                evaluated = deriv.doit()
                evaluated_ops = _count_ops(evaluated)
                if evaluated_ops > max(inner_ops * 4, inner_ops + 4500):
                    continue
                evaluated = _strip_physical_abs(sp.powsimp(evaluated, force=True))
                if evaluated_ops <= 5000:
                    evaluated = sp.factor_terms(evaluated)
                replacements[deriv] = evaluated
            except Exception:
                continue
        if replacements:
            return expr.xreplace(replacements)
    except Exception:
        pass
    return expr


def _strip_physical_abs(expr: sp.Expr) -> sp.Expr:
    try:
        replacements = {}
        for node in expr.atoms(sp.Abs):
            if node.args and _is_physical_positive(node.args[0]):
                replacements[node] = node.args[0]
        if replacements:
            return expr.xreplace(replacements)
    except Exception:
        pass
    return expr


def _is_physical_positive(expr: sp.Expr) -> bool:
    if expr is None:
        return False
    if expr.is_positive is True:
        return True
    if isinstance(expr, sp.Symbol):
        return expr.name in {"t", "r", "a0", "A0", "B0", "h", "r0"}
    if expr.is_Pow:
        return _is_physical_positive(expr.base)
    if expr.is_Mul:
        return all(_is_physical_positive(arg) for arg in expr.args)
    if expr.func == sp.sin and expr.args and str(expr.args[0]) in ("theta", "th"):
        return True
    return False


def _has_symbolic_power(expr: sp.Expr) -> bool:
    try:
        return any(pow_expr.exp.free_symbols for pow_expr in expr.atoms(sp.Pow))
    except Exception:
        return False


def _count_ops(expr: sp.Expr) -> int:
    try:
        return int(sp.count_ops(expr))
    except Exception:
        return 10**9


if __name__ == "__main__":
    sys.exit(_worker_main(sys.argv[1] if len(sys.argv) > 1 else "powsimp_factor"))
