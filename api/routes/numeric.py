"""Numerical solve routes."""

from flask import Blueprint, jsonify, request

from core.numerics import solve_residual_system
from api.routes._logging import clean_error, quiet_route_output, route_log, summarize_numeric_payload


numeric_bp = Blueprint("numeric", __name__, url_prefix="/api/numeric")


@numeric_bp.route("/solve", methods=["POST"])
def solve_numeric_residuals():
    """Solve an exported non-linear residual system over a 1D domain."""
    try:
        payload = request.get_json() or {}
        route_log(f"[NUMERIC] solve start: {summarize_numeric_payload(payload)}")
        with quiet_route_output():
            result = solve_residual_system(payload)
        meta = result.get("metadata") or {}
        route_log(
            f"[NUMERIC] solve complete: converged={meta.get('converged_points', 0)}/"
            f"{meta.get('points', 0)}, diagnostics={len(result.get('diagnostics') or {})}, "
            f"tov={len(result.get('tov') or {})}, warnings={len(result.get('warnings') or [])}"
        )
        return jsonify(result)
    except Exception as exc:
        route_log(f"[NUMERIC] solve error: {clean_error(exc)}")
        return jsonify({"error": clean_error(exc)}), 400
