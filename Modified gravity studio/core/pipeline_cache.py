"""Shared in-memory caches for the symbolic pipeline."""

from typing import Any, Callable, Dict, Tuple

import copy
import os
import threading

import sympy as sp


ENABLE_REDUCED_EQUATION_CACHE = True
ENABLE_ANSATZ_CACHE = True
ENABLE_LHS_CACHE = True
ENABLE_COMPONENT_CACHE = True
ENABLE_SCALAR_CACHE = True  # Set False to bypass scalar cache during debugging

MAX_REDUCED_CACHE = max(2, int(os.getenv("MGS_MAX_REDUCED_CACHE", "8")))
MAX_ANSATZ_CACHE = max(2, int(os.getenv("MGS_MAX_ANSATZ_CACHE", "16")))
MAX_LHS_CACHE = max(2, int(os.getenv("MGS_MAX_LHS_CACHE", "8")))
MAX_COMPONENT_CACHE = max(4, int(os.getenv("MGS_MAX_COMPONENT_CACHE", "48")))
MAX_SCALAR_CACHE = max(2, int(os.getenv("MGS_MAX_SCALAR_CACHE", "16")))

_REDUCED_EQUATION_CACHE: Dict[Any, Any] = {}
_ANSATZ_CACHE: Dict[Any, Any] = {}
_LHS_CACHE: Dict[Any, Any] = {}
_COMPONENT_CACHE: Dict[Any, Any] = {}
_SCALAR_CACHE: Dict[Any, Any] = {}
_cache_lock = threading.Lock()


def flush_sympy_caches(log_debug: Callable[[str], None] = lambda _msg: None):
    """Clear SymPy internal caches to prevent memory bloat across runs."""
    try:
        if hasattr(sp, 'cache') and hasattr(sp.cache, 'clear_cache'):
            sp.cache.clear_cache()

        from sympy.core.basic import Basic
        if hasattr(Basic, '_assumptions_orig_cache'):
            Basic._assumptions_orig_cache.clear()

        log_debug("[PIPELINE] SymPy caches flushed")
    except Exception as e:
        log_debug(f"[PIPELINE] Failed to flush SymPy caches: {e}")


def _bounded_store(cache: Dict[Any, Any], key: Tuple, value: Any, max_size: int) -> None:
    cache[key] = value
    while len(cache) > max_size:
        oldest_key = next(iter(cache))
        cache.pop(oldest_key, None)


def prune_all_caches(log_debug: Callable[[str], None] = lambda _msg: None):
    """Keep pipeline caches bounded across long sessions."""
    with _cache_lock:
        before = (
            len(_REDUCED_EQUATION_CACHE),
            len(_ANSATZ_CACHE),
            len(_LHS_CACHE),
            len(_COMPONENT_CACHE),
            len(_SCALAR_CACHE),
        )
        while len(_REDUCED_EQUATION_CACHE) > MAX_REDUCED_CACHE:
            _REDUCED_EQUATION_CACHE.pop(next(iter(_REDUCED_EQUATION_CACHE)), None)
        while len(_ANSATZ_CACHE) > MAX_ANSATZ_CACHE:
            _ANSATZ_CACHE.pop(next(iter(_ANSATZ_CACHE)), None)
        while len(_LHS_CACHE) > MAX_LHS_CACHE:
            _LHS_CACHE.pop(next(iter(_LHS_CACHE)), None)
        while len(_COMPONENT_CACHE) > MAX_COMPONENT_CACHE:
            _COMPONENT_CACHE.pop(next(iter(_COMPONENT_CACHE)), None)
        while len(_SCALAR_CACHE) > MAX_SCALAR_CACHE:
            _SCALAR_CACHE.pop(next(iter(_SCALAR_CACHE)), None)
        after = (
            len(_REDUCED_EQUATION_CACHE),
            len(_ANSATZ_CACHE),
            len(_LHS_CACHE),
            len(_COMPONENT_CACHE),
            len(_SCALAR_CACHE),
        )
    if before != after:
        log_debug(f"[CACHE] Pipeline caches pruned {before} -> {after}")


def get_or_compute_lhs(key: Tuple, compute_fn: Callable[[], Any],
                       log_debug: Callable[[str], None]) -> Any:
    """Return a cached LHS tensor or compute and store it."""
    if not ENABLE_LHS_CACHE:
        return compute_fn()

    with _cache_lock:
        cached = _LHS_CACHE.get(key)
    if cached is not None:
        log_debug(f"[CACHE] LHS cache hit: {key[1:5]}")
        return cached

    value = compute_fn()
    with _cache_lock:
        _bounded_store(_LHS_CACHE, key, value, MAX_LHS_CACHE)
    log_debug(f"[CACHE] LHS cache store: {key[1:5]}")
    return value


def get_or_compute_ansatz(key: Tuple, compute_fn: Callable[[], Dict],
                          log_debug: Callable[[str], None]) -> Dict:
    """Return cached ansatz substitutions or compute and store them."""
    if not ENABLE_ANSATZ_CACHE:
        return compute_fn()

    with _cache_lock:
        cached = _ANSATZ_CACHE.get(key)
    if cached is not None:
        log_debug(f"[CACHE] Ansatz cache hit: {key[1:]}")
        return copy.deepcopy(cached)

    value = compute_fn()
    with _cache_lock:
        _bounded_store(_ANSATZ_CACHE, key, copy.deepcopy(value), MAX_ANSATZ_CACHE)
    log_debug(f"[CACHE] Ansatz cache store: {key[1:]}")
    return value


def get_or_compute_reduced(key: Tuple, compute_fn: Callable[[], Any],
                           log_debug: Callable[[str], None]) -> Any:
    """Return cached reduced equations or compute and store a deep-copyable payload."""
    if ENABLE_REDUCED_EQUATION_CACHE:
        with _cache_lock:
            cached = _REDUCED_EQUATION_CACHE.get(key)
        if cached is not None:
            log_debug(f"[CACHE] Reduced equation cache hit: {key[1:5]}")
            return copy.deepcopy(cached)

    value = compute_fn()
    if ENABLE_REDUCED_EQUATION_CACHE:
        with _cache_lock:
            _bounded_store(_REDUCED_EQUATION_CACHE, key, copy.deepcopy(value), MAX_REDUCED_CACHE)
        log_debug(f"[CACHE] Reduced equation cache store: {key[1:5]}")
    return value


def get_component_cache(key: Tuple) -> Any:
    """Fetch one cached extracted field-equation component pair."""
    if not ENABLE_COMPONENT_CACHE:
        return None
    with _cache_lock:
        cached = _COMPONENT_CACHE.get(key)
    if cached is None:
        return None
    return copy.deepcopy(cached)


def set_component_cache(key: Tuple, value: Any,
                        log_debug: Callable[[str], None]) -> None:
    """Store one extracted field-equation component pair."""
    if not ENABLE_COMPONENT_CACHE:
        return
    with _cache_lock:
        _bounded_store(_COMPONENT_CACHE, key, copy.deepcopy(value), MAX_COMPONENT_CACHE)
    log_debug(f"[CACHE] Component cache store: {key[1:6]}")


def get_scalar_cache(key: Tuple) -> Any:
    """Fetch a scalar-cache entry. Returns None when cache is disabled."""
    if not ENABLE_SCALAR_CACHE:
        return None
    with _cache_lock:
        return _SCALAR_CACHE.get(key)


def set_scalar_cache(key: Tuple, value: Dict[str, Any],
                     log_debug: Callable[[str], None]) -> None:
    """Store a scalar-cache entry."""
    if not ENABLE_SCALAR_CACHE:
        return
    with _cache_lock:
        _bounded_store(_SCALAR_CACHE, key, value, MAX_SCALAR_CACHE)
    log_debug(f"[CACHE] Scalar cache store: {key[1:3]}")


def clear_all_pipeline_caches(log_debug: Callable[[str], None] = lambda _msg: None) -> None:
    """Clear all in-memory pipeline caches immediately."""
    with _cache_lock:
        before = (
            len(_REDUCED_EQUATION_CACHE),
            len(_ANSATZ_CACHE),
            len(_LHS_CACHE),
            len(_COMPONENT_CACHE),
            len(_SCALAR_CACHE),
        )
        _REDUCED_EQUATION_CACHE.clear()
        _ANSATZ_CACHE.clear()
        _LHS_CACHE.clear()
        _COMPONENT_CACHE.clear()
        _SCALAR_CACHE.clear()
    flush_sympy_caches(log_debug)
    log_debug(f"[CACHE] Pipeline caches cleared {before} -> (0, 0, 0, 0, 0)")
