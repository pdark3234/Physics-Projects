"""Plot evaluation routes."""

from flask import Blueprint, jsonify, request

from core.plotting import evaluate_plot_series
from api.routes._logging import clean_error, quiet_route_output, route_log, summarize_plot_payload


plotting_bp = Blueprint("plotting", __name__, url_prefix="/api/plot")


@plotting_bp.route("/evaluate", methods=["POST"])
def evaluate_symbolic_plot():
    """Evaluate solved symbolic expressions over a 1D domain."""
    try:
        payload = request.get_json() or {}
        route_log(f"[PLOT] evaluate symbolic: {summarize_plot_payload(payload)}")
        with quiet_route_output():
            result = evaluate_plot_series(payload)
        meta = result.get("metadata") or {}
        route_log(
            f"[PLOT] complete symbolic: points={meta.get('points', 0)}, "
            f"warnings={len(result.get('warnings') or [])}"
        )
        return jsonify(result)
    except Exception as exc:
        route_log(f"[PLOT] error symbolic: {clean_error(exc)}")
        return jsonify({"error": clean_error(exc)}), 400
