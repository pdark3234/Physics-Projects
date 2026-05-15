"""Theory capability registry.

This keeps UI metadata, backend validation, and pipeline capability hints in
one place.  The symbolic implementation still lives in each theory module.
"""

from dataclasses import asdict, dataclass, field
from typing import Dict, List


@dataclass(frozen=True)
class TheorySpec:
    id: str
    name: str
    geometry_class: str
    geometry_label: str
    latex: str
    scalars: List[str]
    model_symbols: List[str]
    supports_backgrounds: List[str]
    analytical_solve: str = "symbolic"
    equation_export: bool = True
    derived_quantities: str = "available"
    notes: str = ""
    flags: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return asdict(self)


ALL_BACKGROUNDS = [
    'FRW',
    'Bianchi_I',
    'Bianchi_III',
    'Kantowski_Sachs',
    'SS_wormhole',
    'SS_blackhole',
]

NONMETRICITY_BACKGROUNDS = [
    'FRW',
    'Bianchi_I',
    'Bianchi_III',
    'Kantowski_Sachs',
    'SS_wormhole',
    'SS_blackhole',
]


THEORY_REGISTRY: Dict[str, TheorySpec] = {
    'fR': TheorySpec(
        id='fR',
        name='f(R) Gravity',
        geometry_class='curvature',
        geometry_label='Curvature',
        latex=r'f_R R_{\mu\nu} - \tfrac{1}{2}f g_{\mu\nu} - \nabla_\mu\nabla_\nu f_R + g_{\mu\nu}\square f_R = 8\pi T_{\mu\nu}',
        scalars=['R'],
        model_symbols=['R'],
        supports_backgrounds=ALL_BACKGROUNDS,
    ),
    'fRTLm': TheorySpec(
        id='fRTLm',
        name='f(R,T,Lm) Gravity',
        geometry_class='curvature',
        geometry_label='Curvature',
        latex=r'f_R R_{\mu\nu} + g_{\mu\nu}\square f_R - \nabla_\mu\nabla_\nu f_R - \tfrac{1}{2}f g_{\mu\nu} - \tfrac{1}{2}(f_L + 2f_T)(T_{\mu\nu} - \mathcal{L}_m g_{\mu\nu}) = 8\pi T_{\mu\nu}',
        scalars=['R', 'T_scalar', 'Lm'],
        model_symbols=['R', 'T_scalar', 'T_mat', 'L'],
        supports_backgrounds=ALL_BACKGROUNDS,
        analytical_solve='linear-matter-or-export',
        derived_quantities='deferred-for-heavy-anisotropic',
        notes='Nonlinear matter couplings export reduced equations instead of forcing a symbolic solve.',
    ),
    'fT': TheorySpec(
        id='fT',
        name='f(T) Teleparallel Gravity',
        geometry_class='torsion',
        geometry_label='Torsion',
        latex=r'e^{-1} e^i_{\ \mu}\partial_\rho(e\,e_i^{\ \alpha} S_\alpha^{\ \nu\rho}) f_T + S^{\nu\lambda}_{\ \ \alpha} T^\alpha_{\ \lambda\mu} f_T - S^{\nu\rho}_{\ \ \mu}\partial_\rho T\,f_{TT} + \tfrac{1}{4}\delta^\nu_\mu f = 4\pi T^\nu_{\ \mu}',
        scalars=['T'],
        model_symbols=['T'],
        supports_backgrounds=ALL_BACKGROUNDS,
    ),
    'fTB': TheorySpec(
        id='fTB',
        name='f(T,B) Gravity',
        geometry_class='torsion',
        geometry_label='Torsion',
        latex=r'2e\,\square f_B\,\delta^\lambda_\nu - 2e\nabla^\lambda\nabla_\nu f_B + eB f_B \delta^\lambda_\nu + 4e(\partial_\mu f_B + \partial_\mu f_T)S_\nu^{\ \mu\lambda} + 4e^a_{\ \nu}\partial_\mu(e S_a^{\ \mu\lambda})f_T - 4e f_T T^\sigma_{\ \mu\nu}S_\sigma^{\ \lambda\mu} - ef\delta^\lambda_\nu = 16\pi e T^\lambda_{\ \nu}',
        scalars=['T', 'B', 'T-B'],
        model_symbols=['T', 'B'],
        supports_backgrounds=ALL_BACKGROUNDS,
        derived_quantities='expensive-for-mixed-models',
    ),
    'fQ': TheorySpec(
        id='fQ',
        name='f(Q) Symmetric Teleparallel Gravity',
        geometry_class='nonmetricity',
        geometry_label='Non-metricity',
        latex=r'-\frac{2}{\sqrt{-g}}\nabla_\alpha(\sqrt{-g}\,f_Q P^\alpha{}_{\mu\nu}) - \tfrac{1}{2}f g_{\mu\nu} + f_Q(P_{\mu\alpha\beta}Q_\nu{}^{\alpha\beta} - 2Q^{\alpha\beta}{}_\mu P_{\alpha\beta\nu}) = 8\pi T_{\mu\nu}',
        scalars=['Q'],
        model_symbols=['Q'],
        supports_backgrounds=NONMETRICITY_BACKGROUNDS,
        notes='Nonmetricity tensor stack is built during LHS assembly.',
    ),
    'fQC': TheorySpec(
        id='fQC',
        name='f(Q,C) Nonmetricity Boundary Gravity',
        geometry_class='nonmetricity',
        geometry_label='Non-metricity',
        latex=r'f(Q,C),\quad C=\mathcal{R}-Q,\quad F_R=f_C,\quad F_Q=f_Q-f_C',
        scalars=['Q', 'C'],
        model_symbols=['Q', 'C'],
        supports_backgrounds=NONMETRICITY_BACKGROUNDS,
        notes='Uses C = R - Q and chain-rule derivatives F_R=f_C, F_Q=f_Q-f_C.',
    ),
}


def get_theory_spec(theory_id: str) -> TheorySpec:
    return THEORY_REGISTRY[theory_id]


def theory_ids() -> set:
    return set(THEORY_REGISTRY)


def theory_payload() -> List[Dict]:
    return [spec.to_dict() for spec in THEORY_REGISTRY.values()]
