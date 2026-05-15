"""Small route logging helpers that avoid dumping symbolic payloads."""

from __future__ import annotations

import sys
import contextlib
import io
from typing import Any, Dict


def _count_series(groups: Dict[str, Any]) -> int:
    return sum(len(series) for series in groups.values() if isinstance(series, dict))


def summarize_plot_payload(payload: Dict[str, Any]) -> str:
    domain = payload.get("domain") or {}
    groups = payload.get("groups") or {}
    group_names = ", ".join(str(name) for name in groups.keys()) if isinstance(groups, dict) else "none"
    return (
        f"variable={payload.get('variable') or 'x'}, "
        f"domain=[{domain.get('min', '?')}, {domain.get('max', '?')}], "
        f"points={domain.get('points', '?')}, "
        f"groups={group_names}, "
        f"series={_count_series(groups) if isinstance(groups, dict) else 0}"
    )


def summarize_numeric_payload(payload: Dict[str, Any]) -> str:
    domain = payload.get("domain") or {}
    return (
        f"background={payload.get('background_id') or '?'}, "
        f"stress={payload.get('stress_tensor') or '?'}, "
        f"variable={payload.get('variable') or 'x'}, "
        f"domain=[{domain.get('min', '?')}, {domain.get('max', '?')}], "
        f"points={domain.get('points', '?')}, "
        f"unknowns={len(payload.get('unknowns') or [])}, "
        f"residuals={len(payload.get('residuals') or [])}, "
        f"parameters={len((payload.get('parameters') or {}).keys())}"
    )


def clean_error(exc: Exception) -> str:
    message = str(exc)
    if len(message) <= 300:
        return message
    return f"{message[:300]}... [truncated]"


def route_log(message: str) -> None:
    print(message, file=sys.__stdout__, flush=True)


@contextlib.contextmanager
def quiet_route_output():
    """Keep numerical plotting routes from echoing symbolic internals."""
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        yield
