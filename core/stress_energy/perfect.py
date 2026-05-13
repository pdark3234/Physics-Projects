"""
Perfect Fluid Stress-Energy Tensor

T_μν = (ρ + p) u_μ u_ν + p g_μν

For (-,+,+,+) signature.
"""

# Mock IPython display before pytearcat imports
import builtins
import sys
if not hasattr(builtins, 'display'):
    builtins.display = lambda *args, **kwargs: None

from typing import Any, Tuple
import sympy as sp


class PerfectFluid:
    """Perfect fluid stress-energy tensor."""
    
    def __init__(self, rho: sp.Symbol = None, p: sp.Symbol = None):
        """
        Initialize perfect fluid.
        
        Args:
            rho: Energy density symbol (created if None)
            p: Pressure symbol (created if None)
        """
        self.rho = rho or sp.Symbol('rho')
        self.p = p or sp.Symbol('p')
        self.unknowns = [self.rho, self.p]
        
    def assemble(self, metric: Any) -> Any:
        """
        Assemble perfect fluid SET tensor.
        
        Args:
            metric: pytearcat metric tensor g
            
        Returns:
            T_SET tensor (pytearcat)
        """
        import pytearcat as pt
        
        # Four-velocity for comoving observer: u^μ = (1, 0, 0, 0)
        u = pt.ten('u', 1)
        u.assign([1, 0, 0, 0], '^')
        u.complete('^')
        
        # T_μν = (ρ + p) u_μ u_ν + p g_μν
        TSET = pt.ten('TSET', 2)
        TSET.assign(
            (self.rho + self.p) * u('_mu') * u('_nu') + self.p * metric('_mu,_nu'),
            '_mu,_nu'
        )
        
        TSET.simplify()
        TSET.complete('_,_')
        
        return TSET
    
    def get_trace(self, metric: Any, TSET: Any) -> sp.Expr:
        """
        Compute trace T = g^μν T_μν.
        
        For perfect fluid: T = -ρ + 3p (for -+++ signature)
        """
        trace = metric('^mu,^nu') * TSET('_mu,_nu')
        return sp.simplify(trace)


class Dust:
    """Dust (pressureless perfect fluid): p = 0."""
    
    def __init__(self, rho: sp.Symbol = None):
        self.rho = rho or sp.Symbol('rho')
        self.p = 0
        self.unknowns = [self.rho]
        
    def assemble(self, metric: Any) -> Any:
        import pytearcat as pt
        
        u = pt.ten('udust', 1)
        u.assign([1, 0, 0, 0], '^')
        u.complete('^')
        
        TSET = pt.ten('Tdust', 2)
        TSET.assign(self.rho * u('_mu') * u('_nu'), '_mu,_nu')
        
        TSET.simplify()
        TSET.complete('_,_')
        
        return TSET


class Radiation:
    """Radiation fluid: p = ρ/3."""
    
    def __init__(self, rho: sp.Symbol = None):
        self.rho = rho or sp.Symbol('rho')
        self.p = self.rho / 3
        self.unknowns = [self.rho]
        
    def assemble(self, metric: Any) -> Any:
        import pytearcat as pt
        
        u = pt.ten('urad', 1)
        u.assign([1, 0, 0, 0], '^')
        u.complete('^')
        
        TSET = pt.ten('Trad', 2)
        # T = (4/3 ρ) u⊗u + (ρ/3) g
        TSET.assign(
            (self.rho + self.p) * u('_mu') * u('_nu') + self.p * metric('_mu,_nu'),
            '_mu,_nu'
        )
        
        TSET.simplify()
        TSET.complete('_,_')
        
        return TSET


class Vacuum:
    """Vacuum energy (cosmological constant)."""
    
    def __init__(self, Lambda: sp.Symbol = None):
        self.Lambda = Lambda or sp.Symbol('Lambda')
        self.unknowns = [self.Lambda]
        
    def assemble(self, metric: Any) -> Any:
        import pytearcat as pt
        
        TSET = pt.ten('Tvac', 2)
        TSET.assign(-self.Lambda * metric('_mu,_nu'), '_mu,_nu')
        
        TSET.simplify()
        TSET.complete('_,_')
        
        return TSET


def create_stress_energy(set_type: str) -> Any:
    """
    Factory for stress-energy tensors.
    
    Args:
        set_type: 'perfect_fluid', 'dust', 'radiation', 'vacuum'
        
    Returns:
        Stress-energy tensor handler
    """
    if set_type == 'perfect_fluid':
        return PerfectFluid()
    elif set_type == 'dust':
        return Dust()
    elif set_type == 'radiation':
        return Radiation()
    elif set_type == 'vacuum':
        return Vacuum()
    else:
        raise ValueError(f"Unknown stress-energy type: {set_type}")
