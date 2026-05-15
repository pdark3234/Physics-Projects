"""
Metric Registry - Background Spacetime Definitions

Each background defines:
- Coordinates and metric functions
- Vierbein (for teleparallel theories)
- Canonical index pairs for field equations
- Unknowns to solve for per SET type
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple, Any, Optional
import sympy as sp
from core.registry.theories import THEORY_REGISTRY


@dataclass
class MetricContext:
    """Container for all metric-related symbols and definitions."""

    coord_index: Dict[str, int] = field(default_factory=dict)
    independent_coord: Any = None          # live sympy Symbol after setup
    independent_coord_name: str = 't'
    metric_fns: Dict[str, Any] = field(default_factory=dict)
    metric_fn_names: List[str] = field(default_factory=list)
    vierbein_matrix: List[List] = field(default_factory=list)
    vierbein_inv: List[List] = field(default_factory=list)
    spatial_vector_contravariant: Any = None  # None → set in geometry; list → literal
    metric_string: str = ''
    canonical_index_pairs: Dict[str, List[Tuple[str, str]]] = field(default_factory=dict)
    unknowns_for_set: Dict[str, List[str]] = field(default_factory=dict)
    has_vierbein: bool = True
    curvature_k: Any = 0
    metric_tensor: Any = None



# ─── FRW ──────────────────────────────────────────────────────────────

def create_FRW(k: int = 0) -> MetricContext:
    ctx = MetricContext()
    ctx.coord_index = {'t': 0, 'r': 1, 'theta': 2, 'phi': 3}
    ctx.independent_coord_name = 't'
    ctx.metric_fn_names = ['a']
    ctx.curvature_k = sp.Integer(k)
    ctx.metric_string = 'ds2 = -dt**2 + a**2/(1 - k*r**2)*dr**2 + a**2*r**2*dtheta**2 + a**2*r**2*sin(theta)**2*dphi**2'
    ctx.canonical_index_pairs = {
        'perfect_fluid': [('t', 't'), ('r', 'r')],
        'dust':          [('t', 't')],
        'radiation':     [('t', 't')],
        'vacuum':        [('t', 't')],
    }
    ctx.unknowns_for_set = {
        'perfect_fluid': ['rho', 'p'],
        'dust':          ['rho'],
        'radiation':     ['rho'],
        'vacuum':        ['Lambda'],
    }
    return ctx


# ─── LRS Bianchi Type-I ──────────────────────────────────────────────────────

def create_Bianchi_I() -> MetricContext:
    ctx = MetricContext()
    ctx.coord_index = {'t': 0, 'x': 1, 'y': 2, 'z': 3}
    ctx.independent_coord_name = 't'
    ctx.metric_fn_names = ['A', 'B']
    ctx.metric_string = 'ds2 = -dt**2 + A**2*dx**2 + B**2*dy**2 + B**2*dz**2'
    ctx.vierbein_matrix = [[1,0,0,0],[0,'A',0,0],[0,0,'B',0],[0,0,0,'B']]
    ctx.vierbein_inv    = [[1,0,0,0],[0,'1/A',0,0],[0,0,'1/B',0],[0,0,0,'1/B']]
    ctx.spatial_vector_contravariant = None
    ctx.canonical_index_pairs = {
        'perfect_fluid': [('t', 't'), ('x', 'x')],
        'dust':          [('t', 't')],
        'radiation':     [('t', 't')],
        'vacuum':        [('t', 't')],
        'anisotropic':   [('t', 't'), ('x', 'x'), ('y', 'y')],
    }
    ctx.unknowns_for_set = {
        'perfect_fluid': ['rho', 'p'],
        'dust':          ['rho'],
        'radiation':     ['rho'],
        'vacuum':        ['Lambda'],
        'anisotropic':   ['rho', 'P_r', 'P_t'],
    }
    return ctx


# ─── LRS Bianchi Type-III ─────────────────────────────────────────────────────

def create_Bianchi_III() -> MetricContext:
    ctx = MetricContext()
    ctx.coord_index = {'t': 0, 'x': 1, 'y': 2, 'z': 3}
    ctx.independent_coord_name = 't'
    ctx.metric_fn_names = ['A', 'B']
    ctx.metric_string = 'ds2 = -dt**2 + A**2*dx**2 + exp(2*x)*(B**2*dy**2 + B**2*dz**2)'
    ctx.vierbein_matrix = [[1,0,0,0],[0,'A',0,0],[0,0,'B*exp(x)',0],[0,0,0,'B*exp(x)']]
    ctx.vierbein_inv    = [[1,0,0,0],[0,'1/A',0,0],[0,0,'1/(B*exp(x))',0],[0,0,0,'1/(B*exp(x))']]
    ctx.spatial_vector_contravariant = None
    ctx.canonical_index_pairs = {
        'perfect_fluid': [('t', 't'), ('x', 'x')],
        'dust':          [('t', 't')],
        'radiation':     [('t', 't')],
        'vacuum':        [('t', 't')],
        'anisotropic':   [('t', 't'), ('x', 'x'), ('y', 'y')],
    }
    ctx.unknowns_for_set = {
        'perfect_fluid': ['rho', 'p'],
        'dust':          ['rho'],
        'radiation':     ['rho'],
        'vacuum':        ['Lambda'],
        'anisotropic':   ['rho', 'P_r', 'P_t'],
    }
    return ctx


# ─── Kantowski-Sachs ─────────────────────────────────────────────────────────

def create_Kantowski_Sachs() -> MetricContext:
    ctx = MetricContext()
    ctx.coord_index = {'t': 0, 'r': 1, 'theta': 2, 'phi': 3}
    ctx.independent_coord_name = 't'
    ctx.metric_fn_names = ['A', 'B']
    ctx.metric_string = 'ds2 = -dt**2 + A**2*dr**2 + B**2*dtheta**2 + B**2*sin(theta)**2*dphi**2'
    ctx.vierbein_matrix = [[1,0,0,0],[0,'A',0,0],[0,0,'B',0],[0,0,0,'B*sin(theta)']]
    ctx.vierbein_inv    = [[1,0,0,0],[0,'1/A',0,0],[0,0,'1/B',0],[0,0,0,'1/(B*sin(theta))']]
    ctx.spatial_vector_contravariant = None
    ctx.canonical_index_pairs = {
        'perfect_fluid': [('t', 't'), ('r', 'r')],
        'dust':          [('t', 't')],
        'radiation':     [('t', 't')],
        'vacuum':        [('t', 't')],
        'anisotropic':   [('t', 't'), ('r', 'r'), ('theta', 'theta')],
    }
    ctx.unknowns_for_set = {
        'perfect_fluid': ['rho', 'p'],
        'dust':          ['rho'],
        'radiation':     ['rho'],
        'vacuum':        ['Lambda'],
        'anisotropic':   ['rho', 'P_r', 'P_t'],
    }
    return ctx


# ─── Static Spherically Symmetric — Wormhole ─────────────────────────────────

def create_SS_wormhole() -> MetricContext:
    ctx = MetricContext()
    ctx.coord_index = {'t': 0, 'r': 1, 'theta': 2, 'phi': 3}
    ctx.independent_coord_name = 'r'
    ctx.metric_fn_names = ['b', 'Phi']
    ctx.metric_string = 'ds2 = -exp(2*Phi)*dt**2 + 1/(1-b/r)*dr**2 + r**2*dtheta**2 + r**2*sin(theta)**2*dphi**2'
    # spatial_vector_contravariant set in geometry after symbols are live
    ctx.spatial_vector_contravariant = None
    ctx.canonical_index_pairs = {
        'perfect_fluid': [('t', 't'), ('r', 'r')],
        'dust':          [('t', 't')],
        'vacuum':        [('t', 't')],
        'anisotropic':   [('t', 't'), ('r', 'r'), ('theta', 'theta')],
    }
    ctx.unknowns_for_set = {
        'perfect_fluid': ['rho', 'p'],
        'dust':          ['rho'],
        'vacuum':        ['Lambda'],
        'anisotropic':   ['rho', 'P_r', 'P_t'],
    }
    return ctx



def create_SS_blackhole() -> MetricContext:
    ctx = MetricContext()
    ctx.coord_index = {'t': 0, 'r': 1, 'theta': 2, 'phi': 3}
    ctx.independent_coord_name = 'r'
    ctx.metric_fn_names = ['nu_bh', 'lam_bh']
    ctx.metric_string = 'ds2 = -nu_bh*dt**2 + lam_bh*dr**2 + r**2*dtheta**2 + r**2*sin(theta)**2*dphi**2'
    ctx.spatial_vector_contravariant = None
    ctx.canonical_index_pairs = {
        'vacuum':        [('t', 't')],
        'perfect_fluid': [('t', 't'), ('r', 'r')],
        'anisotropic':   [('t', 't'), ('r', 'r'), ('theta', 'theta')],
    }
    ctx.unknowns_for_set = {
        'vacuum':        ['Lambda'],
        'perfect_fluid': ['rho', 'p'],
        'anisotropic':   ['rho', 'P_r', 'P_t'],
    }
    return ctx


# ─── Registry ────────────────────────────────────────────────────────────────

METRIC_REGISTRY: Dict[str, Callable[[], MetricContext]] = {
    'FRW':              create_FRW,
    'Bianchi_I':        create_Bianchi_I,
    'Bianchi_III':      create_Bianchi_III,
    'Kantowski_Sachs':  create_Kantowski_Sachs,
    'SS_wormhole':      create_SS_wormhole,
    'SS_blackhole':     create_SS_blackhole,
}

BACKGROUND_NAMES = {
    'FRW':              'FRW',
    'Bianchi_I':        'LRS Bianchi Type-I',
    'Bianchi_III':      'LRS Bianchi Type-III',
    'Kantowski_Sachs':  'Kantowski-Sachs',
    'SS_wormhole':      'Static Spherically Symmetric (Wormhole)',
    'SS_blackhole':     'Static Spherically Symmetric (Black Hole)',
}

BACKGROUND_METRIC_LATEX = {
    'FRW':              r'ds^2 = -dt^2 + \frac{a(t)^2}{1-kr^2}dr^2 + a(t)^2 r^2\,d\Omega^2',
    'Bianchi_I':        r'ds^2 = -dt^2 + A(t)^2\,dx^2 + B(t)^2\!\left(dy^2 + dz^2\right)',
    'Bianchi_III':      r'ds^2 = -dt^2 + A(t)^2\,dx^2 + e^{2x}B(t)^2\!\left(dy^2 + dz^2\right)',
    'Kantowski_Sachs':  r'ds^2 = -dt^2 + A(t)^2\,dr^2 + B(t)^2\!\left(d\theta^2 + \sin^2\!\theta\,d\phi^2\right)',
    'SS_wormhole':      r'ds^2 = -e^{2\Phi(r)}\,dt^2 + \frac{dr^2}{1-b(r)/r} + r^2\,d\Omega^2',
    'SS_blackhole':     r'ds^2 = -\nu_{bh}(r)\,dt^2 + \lambda_{bh}(r)\,dr^2 + r^2\,d\Omega^2',
}

THEORY_BACKGROUNDS = {
    theory_id: spec.supports_backgrounds
    for theory_id, spec in THEORY_REGISTRY.items()
}

_FRW_BACKGROUNDS = {'FRW'}

# ─── Symmetry Group Classification (Lie / Isometry Groups) ───────────────────
BACKGROUND_SYMMETRY_GROUP = {
    'FRW': {
        'isometry_group':   r'G_k \times \mathbb{R}',
        'isometry_dim':     7,
        'geometry_class':   'FRW cosmology',
        'Killing_vectors':  6,
        'spatial_symmetry': r'\Sigma_k',
        'notes': 'Spatial sections are maximally symmetric with selectable curvature k = 1, 0, or -1.',
        'lie_algebra':      'g_k',
    },
    'Bianchi_I': {
        'isometry_group':   r'\mathbb{R}^3 \rtimes SO(2)',
        'isometry_dim':     4,
        'geometry_class':   'LRS Bianchi Type-I (homogeneous, anisotropic)',
        'Killing_vectors':  4,
        'spatial_symmetry': r'\mathbb{R}^3\ \text{with axial SO(2)}',
        'notes': 'Abelian Bianchi group. Flat spatial sections with anisotropic expansion. '
                 'LRS subclass: axial symmetry SO(2) in y-z plane. Reduces to FRW flat when A=B.',
        'lie_algebra':      'b_I (Abelian)',
    },
    'Bianchi_III': {
        'isometry_group':   r'\mathbb{R}^2 \rtimes \mathbb{R}',
        'isometry_dim':     3,
        'geometry_class':   'LRS Bianchi Type-III (homogeneous, anisotropic)',
        'Killing_vectors':  3,
        'spatial_symmetry': r'H^2 \times \mathbb{R}\ \text{structure}',
        'notes': 'Non-Abelian Bianchi group. The exp(2x) factor introduces negative '
                 'curvature in the transverse directions. Used in open anisotropic cosmologies.',
        'lie_algebra':      'b_III (non-Abelian)',
    },
    'Kantowski_Sachs': {
        'isometry_group':   r'\mathbb{R} \times SO(3)',
        'isometry_dim':     4,
        'geometry_class':   'Kantowski-Sachs (homogeneous, non-Bianchi)',
        'Killing_vectors':  4,
        'spatial_symmetry': r'\mathbb{R} \times S^2',
        'notes': 'Not of Bianchi type. Spatial sections ℝ × S². '
                 'Arises naturally inside a Schwarzschild black hole. '
                 'Spherical SO(3) symmetry plus radial translation.',
        'lie_algebra':      'r ⊕ so(3)',
    },
    'SS_wormhole': {
        'isometry_group':   r'\mathbb{R} \times SO(3)',
        'isometry_dim':     4,
        'geometry_class':   'Static spherically symmetric (Morris-Thorne wormhole)',
        'Killing_vectors':  4,
        'spatial_symmetry': r'SO(3)',
        'notes': 'Static (time-translation Killing vector ∂_t) plus spherical SO(3). '
                 'Morris-Thorne traversable wormhole: shape function b(r), redshift Φ(r). '
                 'NEC violation required for traversability.',
        'lie_algebra':      'r ⊕ so(3)',
    },
    'SS_blackhole': {
        'isometry_group':   r'\mathbb{R} \times SO(3)',
        'isometry_dim':     4,
        'geometry_class':   'Static spherically symmetric (Schwarzschild / Reissner-Nordstrom)',
        'Killing_vectors':  4,
        'spatial_symmetry': r'SO(3)',
        'notes': 'Static spherical black-hole metric in direct coefficient form using independent '
                 'functions nu_bh(r), lam_bh(r). Fixed Schwarzschild and charged Reissner-Nordstrom '
                 'presets are provided.',
        'lie_algebra':      'r ⊕ so(3)',
    },
}

# ─── Geometry / cosmology context for each background ────────────────────────
BACKGROUND_GEOMETRY_INFO = {
    'FRW': {
        'Friedmann_1':  r'H^2 + k/a^2 = \tfrac{8\pi}{3}\rho',
        'Friedmann_2':  r'\dot{H} - k/a^2 = -4\pi(\rho+p)',
        'curvature':    'k = 1, 0, or -1',
        'topology':     r'S^3,\ H^3\ \text{or}\ \mathbb{R}^3',
    },
    'Bianchi_I': {
        'shear':        r'\sigma^2 = \tfrac{1}{2}[(\dot{A}/A-\dot{B}/B)^2]',
        'curvature':    'Flat spatial sections, anisotropic',
    },
    'Bianchi_III': {
        'curvature':    'Negatively curved transverse sections',
    },
    'Kantowski_Sachs': {
        'curvature':    r'S^2 \times \mathbb{R}',
        'notes':        'Interior of Schwarzschild black hole (GR)',
    },
    'SS_wormhole': {
        'throat':       r'b(r_0)=r_0,\; b\'(r_0)<1',
        'NEC':          r'\rho + P_r < 0\ \text{(exotic matter required)}',
    },
    'SS_blackhole': {
        'schwarzschild': r'\nu_{bh}(r)=1-2M/r,\; \lambda_{bh}(r)=1/\nu_{bh}(r)',
        'rn':           r'\nu_{bh}(r)=1-2M/r+Q^2/r^2,\; \lambda_{bh}(r)=1/\nu_{bh}(r)',
        'horizon':      r'\nu_{bh}(r_h)=0',
    },
}


def is_set_allowed(background_id: str, set_type: str) -> bool:
    """Check if SET type is allowed for given background."""
    if background_id in _FRW_BACKGROUNDS and set_type == 'anisotropic':
        return False
    if background_id in ('SS_wormhole', 'SS_blackhole') and set_type == 'radiation':
        return False
    
    # Disable vacuum, dust, radiation for non-cosmology backgrounds
    non_cosmology_backgrounds = {'Bianchi_I', 'Bianchi_III', 'Kantowski_Sachs', 'SS_wormhole'}
    if background_id in non_cosmology_backgrounds and set_type in ('vacuum', 'dust', 'radiation'):
        return False
    if background_id == 'SS_blackhole' and set_type in ('dust', 'radiation'):
        return False
    
    return True


# ─── Lm Compatibility Registry for fRTLm Theory ────────────────────────────────

LM_SET_COMPATIBILITY = {
    # (set_type, matter_lag) -> (allowed: bool, reason: str)
    ('perfect_fluid', 'neg_rho'):  (True,  ''),
    ('perfect_fluid', 'p'):        (True,  ''),
    ('perfect_fluid', 'rho'):      (True,  ''),
    ('perfect_fluid', 'T_mat'):    (True,  ''),
    ('anisotropic',   'neg_rho'):  (True,  ''),
    ('anisotropic',   'p'):        (False, 'Lm = p is only defined for isotropic pressure. '
                                           'Anisotropic fluids have two independent pressures P_r and P_t — '
                                           'no unique scalar pressure exists. Use Lm = −ρ instead, '
                                           'which gives τ_μν = 0 and is the standard choice in the literature '
                                           'for anisotropic wormhole studies in f(R,T,Lm) gravity.'),
    ('anisotropic',   'rho'):      (True,  ''),
    ('anisotropic',   'T_mat'):    (False, 'Lm = T (trace) for anisotropic fluid gives T = −ρ + P_r + 2P_t, '
                                           'a composite quantity with non-trivial metric dependence. '
                                           'The τ_μν term does not vanish in this case and the current '
                                           'pipeline assumes τ_μν = 0. Use Lm = −ρ for a well-defined result.'),
    ('dust',          'neg_rho'):  (True,  ''),
    ('dust',          'p'):        (True,  ''),
    ('dust',          'rho'):      (True,  ''),
    ('dust',          'T_mat'):    (True,  ''),
    ('radiation',     'neg_rho'):  (True,  ''),
    ('radiation',     'p'):        (True,  ''),
    ('radiation',     'rho'):      (True,  ''),
    ('radiation',     'T_mat'):    (True,  ''),
    ('vacuum',        'neg_rho'):  (True,  ''),
    ('vacuum',        'p'):        (True,  ''),
    ('vacuum',        'rho'):      (True,  ''),
    ('vacuum',        'T_mat'):    (True,  ''),
}

def check_lm_set_compatibility(set_type: str, matter_lag: str) -> tuple:
    """Returns (allowed: bool, reason: str). Only relevant for fRTLm theory."""
    key = (set_type, matter_lag)
    return LM_SET_COMPATIBILITY.get(key, (True, ''))


def get_metric_context(background_id: str, curvature_k: int = 0) -> MetricContext:
    """Factory to create metric context for given background."""
    if background_id not in METRIC_REGISTRY:
        raise ValueError(f"Unknown background: {background_id}")
    if background_id == 'FRW':
        return METRIC_REGISTRY[background_id](curvature_k)
    return METRIC_REGISTRY[background_id]()
