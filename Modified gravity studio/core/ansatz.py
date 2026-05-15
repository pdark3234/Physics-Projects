"""Metric-function ansatz parsing and substitution helpers."""

from typing import Dict

import re
import sympy as sp


ENABLE_ANSATZ_PRECOMPUTE = True


def _coerce_param_values(raw_params):
    values = {}
    if not isinstance(raw_params, dict):
        return values
    for key, value in raw_params.items():
        if value in (None, ''):
            continue
        try:
            values[str(key)] = sp.nsimplify(value, rational=True)
        except Exception:
            try:
                values[str(key)] = sp.sympify(value)
            except Exception:
                continue
    return values


def build_ansatz_subs(inp, ctx, log_debug=lambda _msg: None) -> Dict:
    """Parse ansatz expressions and build substitution dict."""
    t = ctx.independent_coord
    coord_name = ctx.independent_coord_name
    local_dict = {
        coord_name: t,
        'exp': sp.exp,
        'log': sp.log,
        'sin': sp.sin,
        'cos': sp.cos,
        'tan': sp.tan,
        'sinh': sp.sinh,
        'cosh': sp.cosh,
        'tanh': sp.tanh,
        'sqrt': sp.sqrt,
        'pi': sp.pi,
        'E': sp.E,
    }

    ansatz_param_values = _coerce_param_values(getattr(inp, 'ansatz_params', {}))

    ansatz_subs = {}
    for fn_name, expr_str in inp.ansatz.items():
        fn_sym = ctx.metric_fns.get(fn_name)
        if fn_sym is None:
            for k, v in ctx.metric_fns.items():
                if k.lower() == fn_name.lower():
                    fn_sym = v
                    break
        if fn_sym is None:
            continue

        reserved = {coord_name, 't', 'r', 'theta', 'phi', 'x', 'y', 'z'}
        tokens = set(re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', expr_str))
        for tok in tokens:
            if tok in ansatz_param_values:
                local_dict[tok] = ansatz_param_values[tok]
            elif tok not in reserved and tok not in local_dict:
                local_dict[tok] = sp.Symbol(tok)

        expr = sp.sympify(expr_str, locals=local_dict)
        expr = sp.nsimplify(expr, rational=True)

        if ENABLE_ANSATZ_PRECOMPUTE and expr.has(sp.exp, sp.log, sp.sqrt, sp.sin, sp.cos):
            log_debug(f"[ANSATZ] Transcendental detected in {fn_name} - pre-computing derivatives to order 3")
            d1 = sp.powsimp(sp.diff(expr, t), force=True)
            d2 = sp.powsimp(sp.diff(d1, t), force=True)
            d3 = sp.powsimp(sp.diff(d2, t), force=True)
            ansatz_subs[fn_sym] = expr
            ansatz_subs[sp.Derivative(fn_sym, t)] = d1
            ansatz_subs[sp.Derivative(fn_sym, t, 2)] = d2
            ansatz_subs[sp.Derivative(fn_sym, t, 3)] = d3
        else:
            ansatz_subs[fn_sym] = expr

    if inp.background_id == 'SS_wormhole':
        coord = ctx.independent_coord
        for fn_name, expr_str in inp.ansatz.items():
            fn_sym = ctx.metric_fns.get(fn_name)
            if fn_sym is None:
                continue

            expr = sp.sympify(expr_str, locals=local_dict)
            if expr.is_number:
                c = expr
                log_debug(f"[ANSATZ] Constant {fn_name}={c} - injecting composite exp() substitutions for SS background")

                ansatz_subs[sp.exp(2 * fn_sym)] = sp.exp(2 * c)
                ansatz_subs[sp.exp(4 * fn_sym)] = sp.exp(4 * c)
                ansatz_subs[sp.exp(-2 * fn_sym)] = sp.exp(-2 * c)

                if fn_name in ctx.metric_fns:
                    base_fn = ctx.metric_fns[fn_name]
                    ansatz_subs[sp.exp(2 * base_fn)] = sp.exp(2 * c)
                    ansatz_subs[sp.exp(4 * base_fn)] = sp.exp(4 * c)
                    ansatz_subs[sp.exp(-2 * base_fn)] = sp.exp(-2 * c)

                    coord_symbol = coord if coord is not None else sp.Symbol('r')
                    wildcard_fn = sp.Function(fn_name)(coord_symbol)
                    ansatz_subs[sp.exp(2 * wildcard_fn)] = sp.exp(2 * c)
                    ansatz_subs[sp.exp(4 * wildcard_fn)] = sp.exp(4 * c)
                    ansatz_subs[sp.exp(-2 * wildcard_fn)] = sp.exp(-2 * c)

                    if inp.background_id.startswith('FRW'):
                        coord_symbol = coord if coord is not None else sp.Symbol('t')
                        frw_fn = sp.Function(fn_name)(coord_symbol)
                        ansatz_subs[frw_fn**2] = c**2
                        ansatz_subs[frw_fn**3] = c**3
                        ansatz_subs[frw_fn**4] = c**4

                if coord is not None:
                    ansatz_subs[sp.Derivative(fn_sym, coord)] = sp.Integer(0)
                    ansatz_subs[sp.Derivative(fn_sym, coord, 2)] = sp.Integer(0)
                    ansatz_subs[sp.Derivative(fn_sym, coord, 3)] = sp.Integer(0)

    return ansatz_subs


def build_extended_subs(ansatz_subs: Dict, t) -> Dict:
    """Extend ansatz substitutions to include first and second derivatives."""
    extended = dict(ansatz_subs)
    if t is None:
        return extended
    for func, ansatz_expr in ansatz_subs.items():
        if isinstance(func, sp.Expr) and func.has(sp.Function):
            d1 = sp.Derivative(func, t)
            d2 = sp.Derivative(func, t, 2)
            extended[d1] = sp.diff(ansatz_expr, t)
            extended[d2] = sp.diff(ansatz_expr, t, 2)
    return extended
