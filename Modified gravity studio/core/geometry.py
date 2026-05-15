"""
Geometry Engine with Caching (All Backgrounds)

Computes and caches geometric objects that depend only on the background metric.
Supports curvature path (f(R), f(R,T,Lm)) and teleparallel path (f(T), f(T,B)).
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Tuple, Optional, List
import sympy as sp
import threading
import builtins
import io
import sys
import os
import pickle

# Disk cache configuration
MGS_DISK_CACHE = os.environ.get('MGS_DISK_CACHE', 'false').lower() == 'true'
GEOMETRY_CACHE_DIR = os.path.join(os.path.dirname(__file__), '.geometry_cache')
CACHE_VERSION = 4  # Increment whenever geometry structure, vierbein entries, or
                   # LazyResult pickle contract changes.  Stale disk caches are
                   # silently ignored and recomputed — never lower this value.

# ─── Environment mocking ─────────────────────────────────────────────────────
if not hasattr(builtins, 'display'):
    builtins.display = lambda *args, **kwargs: None

if 'IPython' not in sys.modules:
    class _MockDisplay:
        @staticmethod
        def display(*a, **kw): pass
        class Math:
            def __init__(self, *a, **kw): pass
        class Latex:
            def __init__(self, *a, **kw): pass
        class IProgress:
            pass
    class _MockIPython:
        display = _MockDisplay()
    sys.modules['IPython'] = _MockIPython()
    sys.modules['IPython.display'] = _MockDisplay()
    sys.modules['ipywidgets'] = _MockDisplay()

import tqdm
tqdm.tqdm_notebook = tqdm.tqdm
import types as _types
_fake_notebook = _types.ModuleType('tqdm.notebook')
_fake_notebook.tqdm_notebook = tqdm.tqdm
sys.modules['tqdm.notebook'] = _fake_notebook

# ─── Cache dataclass ─────────────────────────────────────────────────────────

@dataclass
class GeometryCache:
    """Cached geometric objects for a specific background."""

    # Curvature path
    christoffel: Any = None
    riemann: Any = None
    ricci_tensor: Any = None
    ricci_scalar: Optional[sp.Expr] = None
    einstein_tensor: Any = None

    # Teleparallel path
    vierbein: Optional[Tuple[Any, sp.Expr]] = None   # (e tensor, det_e)
    torsion_tensor: Any = None
    contorsion: Any = None
    superpotential: Any = None
    T_scalar_expr: Optional[sp.Expr] = None
    B_scalar_expr: Optional[sp.Expr] = None
    lc_from_torsion: Any = None

    # Metric tensor reference (for assembling LHS)
    metric_tensor_obj: Any = None

    # Live symbols (populated during _compute_geometry)
    live_symbols: Dict[str, Any] = field(default_factory=dict)   # e.g. {'a': a_fun, 't': t_sym}

    # Pytearcat module object for Stage 3 optimization (stored on cache hit)
    pt_module: Any = None

    # Tensor names created during this run (for cleanup)
    tensor_names: List[str] = field(default_factory=list)


# ─── Cache storage ────────────────────────────────────────────────────────────

GEOMETRY_CACHE: Dict[Tuple[str, int], GeometryCache] = {}
_cache_lock = threading.Lock()


def _get_cache_path(background_id: str, curvature_k: int) -> str:
    """Get disk cache file path for a background."""
    cache_file = f"{background_id}_{curvature_k}.pkl"
    return os.path.join(GEOMETRY_CACHE_DIR, cache_file)


def _load_from_disk(background_id: str, curvature_k: int) -> Optional[GeometryCache]:
    """Load geometry cache from disk if available and valid."""
    if not MGS_DISK_CACHE:
        return None
    
    cache_path = _get_cache_path(background_id, curvature_k)
    if not os.path.exists(cache_path):
        return None
    
    try:
        with open(cache_path, 'rb') as f:
            data = pickle.load(f)
        
        # Check cache version
        if data.get('version') != CACHE_VERSION:
            print(f"[GEOMETRY] Disk cache version mismatch for {background_id}_{curvature_k}, recomputing...")
            return None
        
        print(f"[GEOMETRY] Loaded from disk cache: {background_id}_{curvature_k}")
        return data.get('cache')
    except Exception as e:
        print(f"[GEOMETRY] Failed to load disk cache: {e}")
        return None


def _save_to_disk(background_id: str, curvature_k: int, cache: GeometryCache) -> None:
    """Save geometry cache to disk."""
    if not MGS_DISK_CACHE:
        return
    
    try:
        os.makedirs(GEOMETRY_CACHE_DIR, exist_ok=True)
        cache_path = _get_cache_path(background_id, curvature_k)
        
        with open(cache_path, 'wb') as f:
            pickle.dump({'version': CACHE_VERSION, 'cache': cache}, f)
        
        print(f"[GEOMETRY] Saved to disk cache: {background_id}_{curvature_k}")
    except Exception as e:
        print(f"[GEOMETRY] Failed to save disk cache: {e}")


def get_geometry(background_id: str, curvature_k: int = 0) -> GeometryCache:
    """
    Return cached geometry for background, computing if necessary.
    Thread-safe. Checks disk cache before computing.
    """
    key = (background_id, curvature_k)
    
    # Check memory cache first
    with _cache_lock:
        if key in GEOMETRY_CACHE:
            return GEOMETRY_CACHE[key]
    
    # Try disk cache before computing
    disk_cache = _load_from_disk(background_id, curvature_k)
    if disk_cache is not None:
        with _cache_lock:
            GEOMETRY_CACHE[key] = disk_cache
        return disk_cache
    
    # Compute geometry (expensive!)
    cache = _compute_geometry(background_id, curvature_k)
    
    # Store in memory cache
    with _cache_lock:
        GEOMETRY_CACHE[key] = cache
    
    # Save to disk cache
    _save_to_disk(background_id, curvature_k, cache)
    
    return cache


# ─── Auto-confirm decorator ───────────────────────────────────────────────────

def _with_auto_confirm(func):
    """Patch sys.stdin to auto-answer 'y' to pytearcat overwrite prompts."""
    def wrapper(*args, **kwargs):
        orig = sys.stdin
        sys.stdin = io.StringIO('y\n')
        try:
            return func(*args, **kwargs)
        finally:
            sys.stdin = orig
    return wrapper


# ─── Per-background geometry computation ─────────────────────────────────────

@_with_auto_confirm
def _compute_geometry(background_id: str, curvature_k: int) -> GeometryCache:
    """Compute all geometric objects for the given background."""
    import pytearcat as pt

    cache = GeometryCache()

    if background_id == 'FRW':
        _compute_FRW(pt, cache, curvature_k)
    elif background_id == 'Bianchi_I':
        _compute_Bianchi_I(pt, cache)
    elif background_id == 'Bianchi_III':
        _compute_Bianchi_III(pt, cache)
    elif background_id == 'Kantowski_Sachs':
        _compute_Kantowski_Sachs(pt, cache)
    elif background_id == 'SS_wormhole':
        _compute_SS_wormhole(pt, cache)
    elif background_id == 'SS_blackhole':
        _compute_SS_blackhole(pt, cache)
    else:
        raise ValueError(f"Unknown background: {background_id!r}")

    # Store background identity so downstream modules (e.g. fQ) can look it up
    cache.live_symbols['_background_id'] = background_id

    return cache


def _curvature_path(pt: Any, cache: GeometryCache):
    """Compute Christoffel → Riemann → Ricci → Ricci scalar → Einstein."""
    cache.christoffel   = pt.christoffel()
    cache.riemann       = pt.riemann()
    ricci               = pt.ricci()
    ricci.complete('_,_')
    cache.ricci_tensor  = ricci
    cache.ricci_scalar  = pt.riccis()
    einstein            = pt.einstein()
    einstein.complete('_,_')
    cache.einstein_tensor = einstein
    # Track tensor names for cleanup
    cache.tensor_names.extend(['Christoffel', 'Riemann', 'Ricci', 'Einstein'])


def _teleparallel_path(pt: Any, cache: GeometryCache, vierbein_matrix, vierbein_inv):
    """Compute vierbein → torsion → contorsion → superpotential → T, B scalars."""
    e = pt.ten('e', 2)
    e.assign(vierbein_matrix, '_,^')
    e.assign(vierbein_inv,    '^,_')
    e.complete('_,^')
    e.complete('^,_')

    det_e = sp.det(sp.Matrix(vierbein_matrix))
    cache.vierbein = (e, det_e)

    # Weitzenböck connection
    GammaW = pt.ten('GammaW', 3)
    GammaW.assign(e('^d,_i') * pt.D(e('_u,^i'), '_v'), '^d,_u,_v')
    GammaW.complete('^,_,_')

    # Torsion tensor T^λ_μν
    Ttens = pt.ten('Ttens', 3)
    Ttens.assign(GammaW('^d,_v,_u') - GammaW('^d,_u,_v'), '^d,_u,_v')
    Ttens.complete('^,_,_')
    cache.torsion_tensor = Ttens

    # Contorsion K^ρ_μν = ½(T_μ^ρ_ν + T_ν^ρ_μ − T^ρ_μν)
    K = pt.ten('K', 3)
    K.assign(
        sp.Rational(1, 2) * (
            Ttens('_mu,^roh,_nu') +
            Ttens('_nu,^roh,_mu') -
            Ttens('^roh,_mu,_nu')
        ),
        '^roh,_mu,_nu'
    )
    K.complete('^,_,_')
    cache.contorsion = K

    # Superpotential S_ρ^μν
    KD = pt.kdelta()
    S = pt.ten('S', 3)
    S.assign(
        K('^mu,^nu,_roh')
        - KD('^mu,_roh') * Ttens('_sigma,^sigma,^nu')
        + KD('^nu,_roh') * Ttens('_sigma,^sigma,^mu'),
        '_roh,^mu,^nu'
    )
    S.complete('_,^,^')
    cache.superpotential = S

    # Torsion scalar T = ½ S_ρ^μν T^ρ_μν
    T_scalar = sp.Rational(1, 2) * S('_roh,^mu,^nu') * Ttens('^roh,_mu,_nu')
    cache.T_scalar_expr = sp.simplify(T_scalar)

    # Boundary term B = 2e⁻¹ ∂_μ(e T^ν_ν^μ)
    B_scalar = 2 * det_e**(-1) * pt.D(det_e * Ttens('^mu,_mu,^nu'), '_nu')
    cache.B_scalar_expr = sp.simplify(B_scalar)

    # Track tensor names for cleanup
    cache.tensor_names.extend(['e', 'GammaW', 'Ttens', 'K', 'KD', 'S'])

    # Levi-Civita Christoffel from torsion (for f(T,B))
    cache.lc_from_torsion = pt.christoffel(Ttens('^rho,_mu,_nu'))


# ─── Individual background implementations ────────────────────────────────────


def _compute_FRW(pt, cache, k):
    t, r, theta, phi = pt.coords('t,r,theta,phi')
    a = pt.fun('a', 't')
    k_sym = pt.con('k')
    metric_str = 'ds2 = -dt**2 + a**2/(1 - k*r**2)*dr**2 + a**2*r**2*dtheta**2 + a**2*r**2*sin(theta)**2*dphi**2'
    g = pt.metric(metric_str)
    g.complete('_,_')
    cache.metric_tensor_obj = g
    cache.live_symbols = {'t': t, 'r': r, 'theta': theta, 'phi': phi, 'a': a,
                          'curvature_k': k_sym}
    cache.tensor_names.append('g')  # Track metric tensor
    _curvature_path(pt, cache)
    # Use the symbolic radial vierbein factor matching g_rr.
    sqrt_k = sp.sqrt(1 - k_sym * r**2)
    vierbein_m = [
        [1,          0,              0,                         0],
        [0,  a/sqrt_k,              0,                         0],
        [0,          0,          a*r,                         0],
        [0,          0,              0,        a*r*sp.sin(theta)],
    ]
    vierbein_i = [
        [1,                0,          0,                            0],
        [0,  sqrt_k/a,              0,                            0],
        [0,                0,      1/(a*r),                       0],
        [0,                0,          0,  1/(a*r*sp.sin(theta))],
    ]
    _teleparallel_path(pt, cache, vierbein_m, vierbein_i)


def _compute_Bianchi_I(pt, cache):
    t, x, y, z = pt.coords('t,x,y,z')
    A = pt.fun('A', 't')
    B = pt.fun('B', 't')
    g = pt.metric('ds2 = -dt**2 + A**2*dx**2 + B**2*dy**2 + B**2*dz**2')
    g.complete('_,_')
    cache.metric_tensor_obj = g
    cache.live_symbols = {'t': t, 'x': x, 'y': y, 'z': z, 'A': A, 'B': B}
    cache.tensor_names.append('g')  # Track metric tensor
    _curvature_path(pt, cache)
    vierbein_m = [[1,0,0,0],[0,A,0,0],[0,0,B,0],[0,0,0,B]]
    vierbein_i = [[1,0,0,0],[0,1/A,0,0],[0,0,1/B,0],[0,0,0,1/B]]
    _teleparallel_path(pt, cache, vierbein_m, vierbein_i)
    cache.live_symbols['spatial_vector'] = [0, 1/A, 0, 0]


def _compute_Bianchi_III(pt, cache):
    t, x, y, z = pt.coords('t,x,y,z')
    A = pt.fun('A', 't')
    B = pt.fun('B', 't')
    g = pt.metric('ds2 = -dt**2 + A**2*dx**2 + exp(2*x)*(B**2*dy**2 + B**2*dz**2)')
    g.complete('_,_')
    cache.metric_tensor_obj = g
    cache.live_symbols = {'t': t, 'x': x, 'y': y, 'z': z, 'A': A, 'B': B}
    cache.tensor_names.append('g')  # Track metric tensor
    _curvature_path(pt, cache)
    ex = sp.exp(x)
    vierbein_m = [[1,0,0,0],[0,A,0,0],[0,0,B*ex,0],[0,0,0,B*ex]]
    vierbein_i = [[1,0,0,0],[0,1/A,0,0],[0,0,1/(B*ex),0],[0,0,0,1/(B*ex)]]
    _teleparallel_path(pt, cache, vierbein_m, vierbein_i)
    cache.live_symbols['spatial_vector'] = [0, 1/A, 0, 0]


def _compute_Kantowski_Sachs(pt, cache):
    t, r, theta, phi = pt.coords('t,r,theta,phi')
    A = pt.fun('A', 't')
    B = pt.fun('B', 't')
    g = pt.metric('ds2 = -dt**2 + A**2*dr**2 + B**2*dtheta**2 + B**2*sin(theta)**2*dphi**2')
    g.complete('_,_')
    cache.metric_tensor_obj = g
    cache.live_symbols = {'t': t, 'r': r, 'theta': theta, 'phi': phi, 'A': A, 'B': B}
    cache.tensor_names.append('g')  # Track metric tensor
    _curvature_path(pt, cache)
    vierbein_m = [[1,0,0,0],[0,A,0,0],[0,0,B,0],[0,0,0,B*sp.sin(theta)]]
    vierbein_i = [[1,0,0,0],[0,1/A,0,0],[0,0,1/B,0],[0,0,0,1/(B*sp.sin(theta))]]
    _teleparallel_path(pt, cache, vierbein_m, vierbein_i)
    cache.live_symbols['spatial_vector'] = [0, 1/A, 0, 0]


def _compute_SS_wormhole(pt, cache):
    t, r, theta, phi = pt.coords('t,r,theta,phi')
    b   = pt.fun('b',   'r')
    Phi = pt.fun('Phi', 'r')
    g = pt.metric('ds2 = -exp(2*Phi)*dt**2 + 1/(1-b/r)*dr**2 + r**2*dtheta**2 + r**2*sin(theta)**2*dphi**2')
    g.complete('_,_')
    cache.metric_tensor_obj = g
    cache.live_symbols = {'t': t, 'r': r, 'theta': theta, 'phi': phi, 'b': b, 'Phi': Phi}
    cache.tensor_names.append('g')  # Track metric tensor
    _curvature_path(pt, cache)
    # Wormhole vierbein
    ePhi = sp.exp(Phi)
    sqrt_term = sp.sqrt(1 - b/r)
    vierbein_m = [
        [ePhi,           0,         0,                   0],
        [0,    1/sqrt_term,         0,                   0],
        [0,              0,         r,                   0],
        [0,              0,         0,    r*sp.sin(theta)],
    ]
    vierbein_i = [
        [1/ePhi,      0,      0,                      0],
        [0,    sqrt_term,     0,                      0],
        [0,           0,   1/r,                      0],
        [0,           0,     0, 1/(r*sp.sin(theta))],
    ]
    _teleparallel_path(pt, cache, vierbein_m, vierbein_i)
    # Spatial vector x^μ = (0, √(1-b/r), 0, 0)
    cache.live_symbols['spatial_vector'] = [0, sqrt_term, 0, 0]



def _compute_SS_blackhole(pt, cache):
    t, r, theta, phi = pt.coords('t,r,theta,phi')
    nu_bh = pt.fun('nu_bh', 'r')
    lam_bh = pt.fun('lam_bh', 'r')
    g = pt.metric('ds2 = -nu_bh*dt**2 + lam_bh*dr**2 + r**2*dtheta**2 + r**2*sin(theta)**2*dphi**2')
    g.complete('_,_')
    cache.metric_tensor_obj = g
    cache.live_symbols = {'t': t, 'r': r, 'theta': theta, 'phi': phi, 'nu_bh': nu_bh, 'lam_bh': lam_bh}
    cache.tensor_names.append('g')
    _curvature_path(pt, cache)
    en2 = sp.sqrt(nu_bh)
    el2 = sp.sqrt(lam_bh)
    vierbein_m = [
        [en2, 0,    0,                   0],
        [0,   el2,  0,                   0],
        [0,   0,    r,                   0],
        [0,   0,    0,  r*sp.sin(theta)],
    ]
    vierbein_i = [
        [1/en2, 0,      0,                      0],
        [0,     1/el2,  0,                      0],
        [0,     0,      1/r,                    0],
        [0,     0,      0,  1/(r*sp.sin(theta))],
    ]
    _teleparallel_path(pt, cache, vierbein_m, vierbein_i)
    cache.live_symbols['spatial_vector'] = [0, 1 / sp.sqrt(lam_bh), 0, 0]


# ─── Cache management ─────────────────────────────────────────────────────────

def clear_cache():
    """Clear all cached geometry (useful for testing)."""
    with _cache_lock:
        GEOMETRY_CACHE.clear()
