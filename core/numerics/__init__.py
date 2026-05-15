"""Numerical helpers for residual-based matter solves."""

from .diagnostics import compute_numeric_diagnostics, compute_numeric_tov
from .solve import solve_residual_system

__all__ = [
    "compute_numeric_diagnostics",
    "compute_numeric_tov",
    "solve_residual_system",
]
