"""
Stress-energy tensor library.

Each matter model exposes one assemble(...) API. The caller chooses the index
form needed by the theory:
  - '_,_' for curvature and nonmetricity theories
  - '^,_' for torsion theories
"""

import builtins
from typing import Any, List, Optional

import sympy as sp

from core.solver import get_matter_symbols

if not hasattr(builtins, 'display'):
    builtins.display = lambda *args, **kwargs: None


def _contravariant_time_component(g: Any) -> sp.Expr:
    """Return the normalized timelike component u^t for a diagonal metric."""
    g00 = sp.sympify(g.tensor[0][0][0])
    return sp.sqrt(-sp.Integer(1) / g00)


def _contravariant_radial_component(g: Any) -> sp.Expr:
    """Return the normalized radial spacelike component x^r for a diagonal metric."""
    g11 = sp.sympify(g.tensor[1][1][0])
    return sp.sqrt(sp.Integer(1) / g11)


def _unit_time_vector(pt, g: Any, name: str):
    """Build a metric-normalized comoving four-velocity u^mu."""
    u = pt.ten(name, 1)
    u.assign([_contravariant_time_component(g), 0, 0, 0], '^')
    u.complete('^')
    u.complete('_')
    return u


def _unit_spatial_vector(pt, g: Any, name: str, spatial_vector: Optional[List] = None):
    """Build a metric-normalized radial spacelike vector x^mu."""
    x = pt.ten(name, 1)
    vector = spatial_vector if spatial_vector is not None else [0, _contravariant_radial_component(g), 0, 0]
    x.assign(vector, '^')
    x.complete('^')
    x.complete('_')
    return x


class PerfectFluid:
    """Perfect fluid with rho and isotropic pressure p."""

    def __init__(self):
        symbols = get_matter_symbols()
        self.rho = symbols['rho']
        self.p = symbols['p']
        self.unknowns = [self.rho, self.p]

    def assemble(self, g: Any, index_form: str = '_,_') -> Any:
        import pytearcat as pt

        u = _unit_time_vector(pt, g, 'upf')
        if index_form == '^,_':
            KD = pt.kdelta()
            TSET = pt.ten('TSETpfm', 2)
            TSET.assign(
                (self.rho + self.p) * u('^nu') * u('^alpha') * g('_alpha,_mu')
                + self.p * KD('^nu,_mu'),
                '^nu,_mu',
            )
            TSET.simplify()
            TSET.complete('^,_')
            return TSET

        TSET = pt.ten('TSETpf', 2)
        TSET.assign(
            (self.rho + self.p) * u('_mu') * u('_nu') + self.p * g('_mu,_nu'),
            '_mu,_nu',
        )
        TSET.simplify()
        TSET.complete('_,_')
        return TSET

    @property
    def trace(self) -> sp.Expr:
        return -self.rho + 3 * self.p


class AnisotropicFluid:
    """Anisotropic fluid with rho, radial pressure P_r, and tangential pressure P_t."""

    def __init__(self):
        symbols = get_matter_symbols()
        self.rho = symbols['rho']
        self.P_r = symbols['P_r']
        self.P_t = symbols['P_t']
        self.unknowns = [self.rho, self.P_r, self.P_t]

    def assemble(self, g: Any, spatial_vector: List, index_form: str = '_,_') -> Any:
        import pytearcat as pt

        u = _unit_time_vector(pt, g, 'uani')
        x = _unit_spatial_vector(pt, g, 'xani', spatial_vector)

        if index_form == '^,_':
            KD = pt.kdelta()
            TSET = pt.ten('TSETanim', 2)
            TSET.assign(
                (self.rho + self.P_t) * u('^nu') * u('^alpha') * g('_alpha,_mu')
                + self.P_t * KD('^nu,_mu')
                + (self.P_r - self.P_t) * x('^nu') * x('^alpha') * g('_alpha,_mu'),
                '^nu,_mu',
            )
            TSET.simplify()
            TSET.complete('^,_')
            return TSET

        TSET = pt.ten('TSETani', 2)
        TSET.assign(
            (self.rho + self.P_t) * u('_mu') * u('_nu')
            + self.P_t * g('_mu,_nu')
            + (self.P_r - self.P_t) * x('_mu') * x('_nu'),
            '_mu,_nu',
        )
        TSET.simplify()
        TSET.complete('_,_')
        return TSET

    @property
    def trace(self) -> sp.Expr:
        return -self.rho + self.P_r + 2 * self.P_t


class Dust:
    """Pressureless fluid."""

    def __init__(self):
        symbols = get_matter_symbols()
        self.rho = symbols['rho']
        self.p = sp.Integer(0)
        self.unknowns = [self.rho]

    def assemble(self, g: Any, index_form: str = '_,_', **kwargs) -> Any:
        import pytearcat as pt

        u = _unit_time_vector(pt, g, 'udust')
        if index_form == '^,_':
            TSET = pt.ten('TSETdustm', 2)
            TSET.assign(
                self.rho * u('^nu') * u('^alpha') * g('_alpha,_mu'),
                '^nu,_mu',
            )
            TSET.simplify()
            TSET.complete('^,_')
            return TSET

        TSET = pt.ten('TSETdust', 2)
        TSET.assign(self.rho * u('_mu') * u('_nu'), '_mu,_nu')
        TSET.simplify()
        TSET.complete('_,_')
        return TSET

    @property
    def trace(self) -> sp.Expr:
        return -self.rho


class Radiation:
    """Radiation fluid with p = rho / 3."""

    def __init__(self):
        symbols = get_matter_symbols()
        self.rho = symbols['rho']
        self.p = self.rho / 3
        self.unknowns = [self.rho]

    def assemble(self, g: Any, index_form: str = '_,_') -> Any:
        import pytearcat as pt

        u = _unit_time_vector(pt, g, 'urad')
        if index_form == '^,_':
            KD = pt.kdelta()
            TSET = pt.ten('TSETradm', 2)
            TSET.assign(
                (self.rho + self.p) * u('^nu') * u('^alpha') * g('_alpha,_mu')
                + self.p * KD('^nu,_mu'),
                '^nu,_mu',
            )
            TSET.simplify()
            TSET.complete('^,_')
            return TSET

        TSET = pt.ten('TSETrad', 2)
        TSET.assign(
            (self.rho + self.p) * u('_mu') * u('_nu') + self.p * g('_mu,_nu'),
            '_mu,_nu',
        )
        TSET.simplify()
        TSET.complete('_,_')
        return TSET

    @property
    def trace(self) -> sp.Expr:
        return sp.Integer(0)


class Vacuum:
    """Vacuum stress tensor with cosmological constant Lambda."""

    def __init__(self):
        symbols = get_matter_symbols()
        self.Lambda = symbols['Lambda']
        self.unknowns = [self.Lambda]

    def assemble(self, g: Any, index_form: str = '_,_') -> Any:
        import pytearcat as pt

        if index_form == '^,_':
            KD = pt.kdelta()
            TSET = pt.ten('TSETvacm', 2)
            TSET.assign(-self.Lambda * KD('^nu,_mu'), '^nu,_mu')
            TSET.simplify()
            TSET.complete('^,_')
            return TSET

        TSET = pt.ten('TSETvac', 2)
        TSET.assign(-self.Lambda * g('_mu,_nu'), '_mu,_nu')
        TSET.simplify()
        TSET.complete('_,_')
        return TSET

    @property
    def trace(self) -> sp.Expr:
        return -4 * self.Lambda


def create_stress_energy(set_type: str) -> Any:
    """Factory for stress-energy tensor handlers."""
    mapping = {
        'perfect_fluid': PerfectFluid,
        'anisotropic': AnisotropicFluid,
        'dust': Dust,
        'radiation': Radiation,
        'vacuum': Vacuum,
    }
    if set_type not in mapping:
        raise ValueError(f"Unknown stress-energy type: {set_type!r}. Valid: {list(mapping.keys())}")
    return mapping[set_type]()
